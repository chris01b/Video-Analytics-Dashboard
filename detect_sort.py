from pathlib import Path
import cv2
import torch
from ultralytics import YOLO
from graphs import draw_boxes

@torch.no_grad()
def detect(
    weights='yolo11n.pt',
    source='0', 
    conf_thres=0.25,
    view_img=False,
    nosave=False,
    project='runs/detect',
    name='exp'
):
    """
    Perform detection and tracking with YOLOv11 and built-in BoT-SORT.
    """
    # Initialize model
    model = YOLO(weights)

    # Create directory if saving images
    save_dir = Path(project) / name
    save_dir.mkdir(parents=True, exist_ok=True)

    # Run tracking
    results = model.track(
        source=source,
        conf=conf_thres,
        show=view_img,
        save=not nosave,
        project=project,
        name=name,
        persist=True
    )  # persist for multi-frame tracking

    for r in results:
        frame = r.orig_img
        boxes = r.boxes.xyxy.cpu()
        track_ids = r.boxes.id.int().cpu().tolist() if r.boxes.id is not None else []
        class_indices = r.boxes.cls.int().cpu().tolist() if r.boxes.cls is not None else []
        # Draw bounding boxes with IDs
        if len(boxes) > 0:
            draw_boxes(frame, boxes, track_ids)
        if view_img:
            cv2.imshow(str(source), frame)
            if cv2.waitKey(1) == ord('q'):
                break
    cv2.destroyAllWindows()
