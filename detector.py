import torch
from ultralytics import YOLO

class Detector:
    def __init__(self, weights='yolo11n.pt', device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = YOLO(weights).to(device)
        self.device = device
        if device == 'cuda':
            print("Model loaded on GPU.")
        else:
            print("CUDA not available. Model running on CPU. Performance may be slower.")

    @torch.no_grad()
    def track(self, source, conf_thres=0.25, save=False, project='runs/detect', name='exp', persist=True, stream=False):
        return self.model.track(
            source=source,
            conf=conf_thres,
            save=save,
            project=project,
            name=name,
            persist=persist,
            stream=stream,
            verbose=False
        )
