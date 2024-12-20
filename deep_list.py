import sys
import time
import cv2
import torch
import psutil
import subprocess
from collections import Counter
from ultralytics import YOLO
from graphs import draw_boxes

def get_gpu_memory():
    try:
        result = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'],
            encoding='utf-8')
        gpu_memory = [int(x) for x in result.strip().split('\n')]
        return gpu_memory[0]
    except:
        return 'NA'

@torch.no_grad()
def detect(weights='yolo11n.pt',
           source='0',
           stframe=None,
           kpi1_text=None,
           kpi2_text=None,
           kpi3_text=None,
           js1_text=None,
           js2_text=None,
           js3_text=None,
           conf_thres=0.25,
           nosave=True,
           display_labels=True,
           conf_thres_drift=0.75,
           save_poor_frame__=False,
           inf_ov_1_text=None,
           inf_ov_2_text=None,
           inf_ov_3_text=None,
           inf_ov_4_text=None,
           fps_warn=None,
           fps_drop_warn_thresh=8,
           stats_update_interval=5,
           frame_update_interval=5):
    model = YOLO(weights)

    # Initialize counters and holders
    prev_time = time.time()
    last_stats_update = time.time()
    global_graph_dict = dict()
    test_drift = []
    poor_perf_frame_counter = 0
    min_FPS = 10000
    max_FPS = -1

    # Use model.track with stream=True to get frame-by-frame results
    results = model.track(source=source, conf=conf_thres, persist=True, stream=True)
    frame_num = -1
    frame_counter = 0  # New counter

    for r in results:
        frame_num += 1
        frame_counter += 1
        im0 = r.orig_img
        boxes = r.boxes
        class_indices = boxes.cls.int().cpu().tolist() if boxes.cls is not None else []
        confs = boxes.conf.cpu().tolist() if boxes.conf is not None else []
        track_ids = boxes.id.int().cpu().tolist() if boxes.id is not None else []

        mapped_ = dict()
        # Count classes in current frame
        if len(class_indices) > 0:
            class_counts = Counter([model.names[c] for c in class_indices])
            mapped_.update(class_counts)
        global_graph_dict = Counter(global_graph_dict) + Counter(mapped_)

        # Drift detection (poor performing frames)
        for c_idx, conf_ in zip(class_indices, confs):
            if conf_ < conf_thres_drift:
                cls_name = model.names[c_idx]
                if cls_name not in test_drift:
                    test_drift.append(cls_name)
                if save_poor_frame__:
                    cv2.imwrite(f"drift_frames/frame_{frame_num}.png", im0)
                    poor_perf_frame_counter += 1

        # Draw boxes and labels if needed
        if len(track_ids) > 0:
            xyxy = boxes.xyxy.cpu()
            # draw boxes with IDs
            draw_boxes(im0, xyxy, track_ids)

            if display_labels:
                for box, cls_idx, conf_ in zip(xyxy, class_indices, confs):
                    x1, y1, x2, y2 = map(int, box)
                    label = f"{model.names[cls_idx]} {conf_:.2f}"
                    cv2.putText(im0, label, (x1, max(y1 - 10, 0)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                # Flush buffer
                sys.stdout.flush()

        # Compute FPS
        curr_time = time.time()
        fps_ = round(1 / (curr_time - prev_time), 1)
        prev_time = curr_time

        # Update system stats at defined intervals
        if (curr_time - last_stats_update) >= stats_update_interval:
            memory_usage = f"{psutil.virtual_memory().percent}%"
            cpu_usage = f"{psutil.cpu_percent()}%"
            gpu_memory = f"{get_gpu_memory()} MB"

            js1_text.write(memory_usage)
            js2_text.write(cpu_usage)
            js3_text.write(gpu_memory)

            last_stats_update = curr_time  # Reset the timer

        # Update Inference Stats at frame intervals
        if frame_counter % frame_update_interval == 0:
            kpi1_text.write(f"{fps_} FPS")
            if fps_ < fps_drop_warn_thresh:
                fps_warn.warning(f"FPS dropped below {fps_drop_warn_thresh}")
            kpi2_text.write(str(mapped_))
            kpi3_text.write(str(global_graph_dict))

            inf_ov_1_text.write(str(test_drift))
            inf_ov_2_text.write(str(poor_perf_frame_counter))

            if fps_ < min_FPS:
                min_FPS = fps_
                inf_ov_3_text.write(str(min_FPS))
            if fps_ > max_FPS:
                max_FPS = fps_
                inf_ov_4_text.write(str(max_FPS))

        # Display frame in Streamlit
        stframe.image(im0, channels="BGR", use_container_width=True)

    # After loop ends
    print("Inference Complete")
