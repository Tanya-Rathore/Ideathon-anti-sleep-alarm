
import argparse
import io
import math
import os
from pathlib import Path
import struct
import sys
import time
import urllib.request
import wave

from state import ClosureTimer

ROOT = Path(__file__).resolve().parent
MODEL_URL = ('https://storage.googleapis.com/mediapipe-models/'
             'face_landmarker/face_landmarker/float16/1/face_landmarker.task')


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seconds', type=float, default=2.0, help='Continuous closure delay (default: 2)')
    p.add_argument('--camera', type=int, default=0, help='Webcam index (default: 0)')
    p.add_argument('--threshold', type=float, default=0.50, help='Blink score to mark closed')
    p.add_argument('--open-threshold', type=float, default=0.35, help='Blink score to mark reopened')
    p.add_argument('--model', type=Path, default=ROOT / 'models' / 'face_landmarker.task')
    p.add_argument('--test-alarm', action='store_true', help='Play alarm for 3 seconds, without camera')
    a = p.parse_args()
    if not math.isfinite(a.seconds) or a.seconds <= 0:
        p.error('--seconds must be a finite positive number')
    if not 0 <= a.open_threshold < a.threshold <= 1:
        p.error('Use 0 <= --open-threshold < --threshold <= 1')
    if a.camera < 0:
        p.error('--camera must be non-negative')
    return a


def ensure_model(path):
    if path.is_file():
        return
    print('Downloading the face model once from Google...')
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.download')
    try:
        with urllib.request.urlopen(MODEL_URL, timeout=60) as response, temp.open('wb') as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        if temp.stat().st_size < 1_000_000:
            raise RuntimeError('Downloaded model is unexpectedly small.')
        temp.replace(path)
    except Exception as exc:
        temp.unlink(missing_ok=True)
        raise RuntimeError(f'Model download failed: {exc}\nDownload {MODEL_URL}\nSave it as {path}') from exc


class Alarm:
    """Loop a generated beep/pause pattern asynchronously; no external audio file."""
    def __init__(self):
        os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')
        import pygame
        self.pg = pygame
        try:
            pygame.mixer.init(frequency=44100, size=-16, channels=1, buffer=512)
            audio = io.BytesIO()
            with wave.open(audio, 'wb') as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(44100)
                samples = bytearray()
                for i in range(26460):  # 0.30 seconds beep + 0.30 seconds silence
                    t = i / 44100
                    envelope = max(0, min(1, t / 0.01, (0.30 - t) / 0.01))
                    sample = int(10000 * envelope * math.sin(2 * math.pi * 1000 * t))
                    samples.extend(struct.pack('<h', sample))
                wav.writeframes(samples)
            audio.seek(0)
            self.sound = pygame.mixer.Sound(file=audio)
        except Exception:
            pygame.mixer.quit()
            raise
        self.channel = None

    def set(self, enabled):
        if enabled and self.channel is None:
            self.channel = self.sound.play(loops=-1)
            if self.channel is None:
                raise RuntimeError('Cannot play alarm: no available audio channel.')
        elif not enabled and self.channel is not None:
            self.channel.stop()
            self.channel = None

    def close(self):
        self.set(False)
        self.pg.mixer.quit()


def main():
    args = arguments()
    alarm = Alarm()  # Fail visibly if sound is unavailable.
    cap = None
    cv2 = None
    try:
        if args.test_alarm:
            print('Testing alarm for 3 seconds. Check your speaker volume.')
            alarm.set(True)
            time.sleep(3)
            return
        import cv2
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        ensure_model(args.model)
        options = vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(args.model)),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            output_face_blendshapes=True,
            min_face_detection_confidence=0.6,
            min_face_presence_confidence=0.6,
            min_tracking_confidence=0.6,
        )
        cap = cv2.VideoCapture(args.camera)
        if not cap.isOpened():
            raise RuntimeError('Cannot open webcam. Close other camera apps, enable camera permission, or try --camera 1.')
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        timer = ClosureTimer(args.seconds, args.threshold, args.open_threshold)
        title = 'Eye Closure Alarm | Q: quit | T: test sound'
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        last_timestamp = -1
        test_until = 0.0
        print('Keep one face visible, facing the camera. Q/Esc quits; T tests the sound.')
        with vision.FaceLandmarker.create_from_options(options) as detector:
            while True:
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError('Webcam stopped providing frames. Alarm stopped; restart after checking camera.')
                captured_at = time.monotonic()
                timestamp = max(last_timestamp + 1, int(captured_at * 1000))
                last_timestamp = timestamp
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = detector.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), timestamp)
                scores = None
                if result.face_blendshapes:
                    values = {c.category_name: c.score for c in result.face_blendshapes[0]}
                    if 'eyeBlinkLeft' in values and 'eyeBlinkRight' in values:
                        scores = values['eyeBlinkLeft'], values['eyeBlinkRight']
                        if not all(math.isfinite(s) for s in scores):
                            scores = None
                # Discard stale inference instead of sounding on outdated information.
                if time.monotonic() - captured_at > timer.max_gap:
                    scores = None
                status, duration, active = timer.update(captured_at, scores)
                alarm.set(active or time.monotonic() < test_until)
                color = (30, 40, 255) if active else ((0, 210, 255) if timer.closed else (80, 220, 80))
                if scores is None:
                    color = (0, 180, 255)
                if result.face_landmarks:
                    h, w = frame.shape[:2]
                    for idx in (33, 133, 159, 145, 362, 263, 386, 374):
                        point = result.face_landmarks[0][idx]
                        cv2.circle(frame, (int(point.x*w), int(point.y*h)), 2, color, -1)
                lines = [status, f'Closed: {duration:.1f}s / {args.seconds:.1f}s',
                         (f'Blink score L: {scores[0]:.2f} R: {scores[1]:.2f}' if scores else 'Tracking unavailable - timer reset'),
                         'Q: quit | T: test sound']
                cv2.rectangle(frame, (0, 0), (frame.shape[1], 128), (25, 25, 25), -1)
                for i, line in enumerate(lines):
                    cv2.putText(frame, line, (12, 27 + 29*i), cv2.FONT_HERSHEY_SIMPLEX, 0.57, color, 1, cv2.LINE_AA)
                cv2.imshow(title, frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord('q'), ord('Q')):
                    break
                if key in (ord('t'), ord('T')):
                    test_until = time.monotonic() + 1.5
                if cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                    break
    finally:
        alarm.close()
        if cap is not None:
            cap.release()
        if cv2 is not None:
            cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        sys.exit(1)
