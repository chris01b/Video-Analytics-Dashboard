import sys
import time
import cv2
import torch
import psutil
import subprocess
from collections import defaultdict, deque
import streamlit as st
from ultralytics import YOLO
from graphs import draw_boxes
from pathlib import Path
from threading import Thread
from queue import Queue

# Initialize a thread-safe queue for saving frames
save_queue = Queue()

def save_compressed_frame(frame_num, frame):
    save_path = f"drift_frames/frame_{frame_num}.jpg"
    cv2.imwrite(save_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])  # 90% quality

def save_frames_worker():
    while True:
        frame_num, frame = save_queue.get()
        if frame_num is None:
            break  # Sentinel to stop the thread
        Path("drift_frames").mkdir(parents=True, exist_ok=True)
        save_compressed_frame(frame_num, frame)
        save_queue.task_done()

# Start the background thread for saving frames
save_thread = Thread(target=save_frames_worker, daemon=True)
save_thread.start()

def get_gpu_memory():
    try:
        result = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'],
            encoding='utf-8'
        )
        gpu_memory = [int(x) for x in result.strip().split('\n')]
        return gpu_memory[0]
    except:
        return 'NA'

@torch.no_grad()
def detect(
    weights='yolo11n.pt',
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
    stats_update_interval=5,    # Seconds between system stats updates
    frame_update_interval=10,   # Frames between inference stats updates
    save_interval=10,           # Seconds between saving frames
    save_threshold=3,           # Minimum low-conf detections to trigger saving
    display_interval=1,         # How often to display frames (default every frame)
    project='runs/detect',
    name='exp'
):
    # Initialize model
    model = YOLO(weights)
    
    # Explicitly move model to GPU if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    
    if device == 'cuda':
        print("Model loaded on GPU.")
    else:
        print("CUDA not available. Model running on CPU. Performance may be slower.")
        if st is not None:
            st.warning("CUDA is not available. The model will run on CPU, which may lead to slower performance.")
    
    # Create directory if saving images
    save_dir = Path(project) / name
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize counters and holders
    prev_time = time.time()
    last_stats_update = time.time()
    last_save_time = time.time()
    global_graph_dict = defaultdict(int)
    test_drift = set()
    poor_perf_frame_counter = 0
    min_FPS = float('inf')
    max_FPS = 0
    frame_num = -1
    frame_counter = 0  # New counter
    
    # FPS smoothing
    fps_deque = deque(maxlen=30)  # Adjust window size as needed
    
    # Initialize GPU memory usage
    gpu_memory_usage = get_gpu_memory()
    
    # Use model.track with stream=True to get frame-by-frame results
    results = model.track(
        source=source,
        conf=conf_thres,
        save=not nosave,
        project=project,
        name=name,
        persist=True,
        stream=True
    )
    
    for r in results:
        frame_num += 1
        frame_counter += 1
        im0 = r.orig_img
        
        # Move tensors to CPU in a consolidated manner
        boxes = r.boxes
        if boxes.cls is not None:
            cls = boxes.cls.int()
            conf = boxes.conf
            track_id = boxes.id.int() if boxes.id is not None else None
            cls_cpu = cls.cpu().tolist()
            conf_cpu = conf.cpu().tolist()
            track_id_cpu = track_id.cpu().tolist() if track_id is not None else []
        else:
            cls_cpu, conf_cpu, track_id_cpu = [], [], []
        
        xyxy_cpu = boxes.xyxy.cpu().numpy() if boxes.xyxy is not None else []
        
        # Update class counts
        if len(cls_cpu) > 0:
            for cls_idx in cls_cpu:
                cls_name = model.names[cls_idx]
                global_graph_dict[cls_name] += 1
    
        # Drift detection (poor performing frames)
        low_conf_detections = [model.names[c_idx] for c_idx, conf_ in zip(cls_cpu, conf_cpu)
                                if conf_ < conf_thres_drift]
        if low_conf_detections:
            test_drift.update(low_conf_detections)
            # Implement threshold: save frame only if more than 'save_threshold' low-confidence detections
            current_time = time.time()
            if len(low_conf_detections) >= save_threshold and (current_time - last_save_time) >= save_interval:
                if save_poor_frame__:
                    save_queue.put((frame_num, im0.copy()))
                    poor_perf_frame_counter += 1
                    last_save_time = current_time
    
        # Draw boxes and labels
        if len(track_id_cpu) > 0:
            draw_boxes(im0, xyxy_cpu, track_id_cpu)
    
            if display_labels:
                for box, cls_idx, conf_ in zip(xyxy_cpu, cls_cpu, conf_cpu):
                    x1, y1, x2, y2 = map(int, box)
                    label = f"{model.names[cls_idx]} {conf_:.2f}"
                    cv2.putText(im0, label, (x1, max(y1 - 10, 0)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                # Flush buffer
                sys.stdout.flush()
    
        # Compute FPS
        curr_time = time.time()
        elapsed = curr_time - prev_time
        if elapsed > 0:
            fps_ = 1 / elapsed
        else:
            fps_ = 0.0
        fps_deque.append(fps_)
        prev_time = curr_time
    
        # Update system stats at defined intervals
        if (curr_time - last_stats_update) >= stats_update_interval:
            memory_usage = f"{psutil.virtual_memory().percent}%"
            cpu_usage = f"{psutil.cpu_percent()}%"
            gpu_mem = get_gpu_memory()
            gpu_memory = f"{gpu_mem} MB" if gpu_mem != 'NA' else 'NA'
    
            js1_text.write(memory_usage)
            js2_text.write(cpu_usage)
            js3_text.write(gpu_memory)
    
            last_stats_update = curr_time  # Reset the timer
    
        # Update Inference Stats at frame intervals
        if frame_counter % frame_update_interval == 0:
            avg_fps = sum(fps_deque) / len(fps_deque) if fps_deque else 0.0
            kpi1_text.write(f"{avg_fps:.1f} FPS")
    
            if avg_fps < fps_drop_warn_thresh:
                fps_warn.warning(f"Average FPS below {fps_drop_warn_thresh}")
    
            kpi2_text.write(str(dict(global_graph_dict)))
            kpi3_text.write(str(len(global_graph_dict)))
    
            inf_ov_1_text.write(", ".join(test_drift) if test_drift else "None")
            inf_ov_2_text.write(str(poor_perf_frame_counter))
    
            if avg_fps < min_FPS:
                min_FPS = avg_fps
                inf_ov_3_text.write(f"{min_FPS:.1f} FPS")
    
            if avg_fps > max_FPS:
                max_FPS = avg_fps
                inf_ov_4_text.write(f"{max_FPS:.1f} FPS")
    
            # Reset FPS deque after updating
            fps_deque.clear()
    
        # Display frame in Streamlit based on display_interval
        if stframe is not None and (display_interval == 1 or frame_num % display_interval == 0):
            # Display the frame in its original resolution
            stframe.image(im0, channels="BGR", use_container_width=True)
    
    # After loop ends
    save_queue.put((None, None))  # Signal the thread to exit
    save_thread.join()
    print("Inference Complete")
