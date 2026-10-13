"""Constant-frame-rate overhead recording on the GUI thread."""
from pathlib import Path
import time

import cv2


class MissionVideoRecorder:
    def __init__(self, path, fps=20., clock=time.monotonic):
        self.path = Path(path).expanduser().resolve()
        if self.path.exists():
            raise FileExistsError(f"Video already exists: {self.path}")
        self.fps = fps
        self.clock = clock
        self.writer = None
        self.frame = None
        self.started = None
        self.frames = 0

    def update(self, frame=None):
        now = self.clock()
        # Fill elapsed time with the previous image, never future motion.
        if self.writer is not None:
            due = int((now - self.started) * self.fps) + 1
            while self.frames < due:
                self.writer.write(self.frame)
                self.frames += 1
        if frame is not None:
            self.frame = frame.copy()
            if self.writer is None:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                height, width = frame.shape[:2]
                self.writer = cv2.VideoWriter(str(self.path),
                    cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (width, height))
                if not self.writer.isOpened():
                    self.writer.release()
                    self.writer = None
                    raise RuntimeError(f"Cannot create video: {self.path}")
                self.started = now
                self.writer.write(self.frame)
                self.frames = 1

    def close(self):
        if self.writer is not None:
            self.update()
            # Include the final pose even if the mission just completed.
            self.writer.write(self.frame)
            self.writer.release()
            self.writer = None
            print(f"Saved mission video: {self.path}")
