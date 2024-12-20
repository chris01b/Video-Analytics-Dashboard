from pathlib import Path
import cv2
import torch
from ultralytics import YOLO
from graphs import draw_boxes

@torch.no_grad()
def detect(weights='yolo11n.pt',
           source='0', 
           conf_thres=0.25,
           line_thickness=3,
           view_img=False,
           nosave=False,
           hide_labels=False,
           hide_conf=False,
           save_txt=False,
           project='runs/detect',
           name='exp',
           exist_ok=False,
           save_crop=False):
    """
    Perform detection and tracking with YOLOv11 and built-in BoT-SORT.
    """
    # Initialize model
    model = YOLO(weights)

    # Create directory if saving images
    save_dir = Path(project) / name
    save_dir.mkdir(parents=True, exist_ok=True)

    # Run tracking
    results = model.track(source=source, conf=conf_thres, show=view_img, save=not nosave, project=project, name=name,
                          persist=True)  # persist for multi-frame tracking

    # results are generator if source is video/stream
    # They are already saved if save=True. If you want to do something custom here, you can.
    # Each `results` iteration provides a `Result` object with boxes that have IDs.

    # If you need custom handling (draw, txt), you'd do it here. For demonstration:
    for r in results:
        frame = r.orig_img
        boxes = r.boxes.xyxy.cpu()
        track_ids = r.boxes.id.int().cpu().tolist() if r.boxes.id is not None else []
        class_indices = r.boxes.cls.int().cpu().tolist() if r.boxes.cls is not None else []
        # If needed, draw additional boxes/IDs here:
        if len(boxes) > 0:
            draw_boxes(frame, boxes, track_ids)
        if view_img:
            cv2.imshow(str(source), frame)
            if cv2.waitKey(1) == ord('q'):
                break
    cv2.destroyAllWindows()
