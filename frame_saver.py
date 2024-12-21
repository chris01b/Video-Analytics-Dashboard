import cv2
from pathlib import Path
from threading import Thread
from queue import Queue

class FrameSaver:
    def __init__(self, save_dir='drift_frames', quality=90):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.queue = Queue()
        self.quality = quality
        self.thread = Thread(target=self.worker, daemon=True)
        self.thread.start()

    def save_compressed_frame(self, frame_num, frame):
        save_path = self.save_dir / f"frame_{frame_num}.jpg"
        cv2.imwrite(str(save_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.quality])

    def worker(self):
        while True:
            frame_num, frame = self.queue.get()
            if frame_num is None:
                break
            self.save_compressed_frame(frame_num, frame)
            self.queue.task_done()

    def enqueue_frame(self, frame_num, frame):
        self.queue.put((frame_num, frame))

    def stop(self):
        self.queue.put((None, None))
        self.thread.join()
