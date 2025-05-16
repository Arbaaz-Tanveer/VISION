import cv2
import time
import threading
import os
import numpy as np
from ultralytics import YOLO
# Import groundmapping functionality
from groundmapping import (pixel_to_ground, compute_calibration_params, undistort_image,
                           update_ground_map, CoordinateEstimator)
from utilities import *
# ---------------------------------------------------
# Global calibration variables (to be computed once)
# ---------------------------------------------------
calibration = {}  # Will hold map1, map2, K, D, new_K, estimator

# Global variable to hold the latest ground map.
latest_ground_map = None

# ---------------------------------------------------
# Load the YOLO TensorRT engine
# ---------------------------------------------------
try:
    trt_model = YOLO("yolo11n.engine",verbose=False)
    print("TensorRT model loaded successfully.")
except Exception as e:
    print(f"Error loading TensorRT model: {e}")
    exit()

# ---------------------------------------------------
# Global shared dictionaries and lock
# ---------------------------------------------------
global_lock = threading.Lock()
# Store each frame as a tuple: (frame, timestamp)
frames_dict = {}         # {camera_index: (latest captured frame, frame timestamp)}
results_dict = {}        # {camera_index: latest inference result (with YOLO overlays)}
capture_fps_dict = {}    # {camera_index: latest capture FPS}
# detection_results stores all detections for each camera.
# Each detection dict now includes a 'timestamp' entry.
detection_results = {}   # {camera_index: [detection, detection, ...]}

# ---------------------------------------------------
# Capture thread function (supports optional custom settings)
# ---------------------------------------------------

#camera indices
camera_indices = [None,None,None,None]

def capture_thread_func(camera_index, settings=None):
    if settings:
        cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
    else:
        cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_index}")
        return

    if settings:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings.get('width', 640))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.get('height', 480))
        cap.set(cv2.CAP_PROP_FPS, settings.get('fps', 30))
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*settings.get('fourcc', 'MJPG')))
        cap.set(cv2.CAP_PROP_BUFFERSIZE, settings.get('buffersize', 1))
        cap.set(cv2.CAP_PROP_BRIGHTNESS, settings.get('brightness', 30))
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, settings.get('auto_exposure', 1))
        cap.set(cv2.CAP_PROP_EXPOSURE, settings.get('exposure', 700))
        print(f"Camera {camera_index} opened with custom settings: {settings}")
    else:
        print(f"Camera {camera_index} opened with default settings.")

    frame_count = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame_count += 1
        elapsed = time.time() - start_time
        if elapsed >= 1.0:
            fps_val = frame_count / elapsed
            with global_lock:
                capture_fps_dict[camera_index] = fps_val
            frame_count = 0
            start_time = time.time()

        timestamp = time.time()
        with global_lock:
            frames_dict[camera_index] = (frame.copy(), timestamp)

        time.sleep(0.005)
    cap.release()

# ---------------------------------------------------
# Inference thread function: processes each camera’s frame one by one
# ---------------------------------------------------
def inference_thread_func(batch_size=4):
    model_names = trt_model.names if hasattr(trt_model, 'names') else {}

    while True:
        # 1) Grab a consistent snapshot of available frames
        with global_lock:
            items = list(frames_dict.items())  # [(cam_index, (frame, ts)), ...]

        if not items:
            time.sleep(0.01)
            continue

        # 2) Build batches of up to batch_size
        for i in range(0, len(items), batch_size):
            batch_items = items[i:i+batch_size]
            cams, frames_ts = zip(*batch_items)  # cams=[cam_index,…], frames_ts=[(frame,ts),…]
            batch_frames = [frame.copy() for frame, ts in frames_ts]
            batch_timestamps = [ts for frame, ts in frames_ts]

            try:
                # 3) Single batch inference call
                batch_results = trt_model(batch_frames)  # returns a list of Results, len == len(batch_frames)

                # 4) Unpack each result
                for cam_index, timestamp, result in zip(cams, batch_timestamps, batch_results):
                    inferred_frame = result.plot() if hasattr(result, "plot") else batch_frames[cams.index(cam_index)]
                    detections_info = []

                    if hasattr(result, "boxes") and result.boxes is not None:
                        boxes = result.boxes.xyxy.cpu().numpy() if hasattr(result.boxes.xyxy, "cpu") else result.boxes.xyxy
                        classes = result.boxes.cls.cpu().numpy() if hasattr(result.boxes.cls, "cpu") else result.boxes.cls
                        for idx, box in enumerate(boxes):
                            x1, y1, x2, y2 = box
                            center_x, bottom_center_y = (x1 + x2) / 2, y2
                            cls_id = int(classes[idx])
                            label = model_names.get(cls_id, f"Class {cls_id}")

                            # Optional reprojection
                            if calibration:
                                ground_coords = pixel_to_ground(
                                    [(center_x, bottom_center_y)],
                                    calibration["estimator"], calibration["K"],
                                    calibration["D"], calibration["new_K"],
                                    show=False
                                )
                                ground_coord = ground_coords[0] if ground_coords else None
                            else:
                                ground_coord = None

                            detections_info.append({
                                "camera": f"Camera {cam_index}",
                                "object":   label,
                                "pixel_coord": (center_x, bottom_center_y),
                                "ground_coord": ground_coord,
                                "timestamp": timestamp
                            })

                    # 5) Write back to shared dicts
                    with global_lock:
                        results_dict[cam_index]      = inferred_frame
                        detection_results[cam_index]= detections_info

            except Exception as e:
                # In case the whole batch fails, you might want to fallback per‐frame
                print(f"Batch inference error: {e}")
                for cam_index, (frame, ts) in batch_items:
                    with global_lock:
                        results_dict[cam_index]       = frame
                        detection_results[cam_index] = []
        time.sleep(0.001)


# ---------------------------------------------------
# Localisation thread function: update ground map and localise every 2 seconds
# ---------------------------------------------------
def localisation_thread_func():
    """
    This function collects the latest frames from defined cameras,
    undistorts them, updates a common ground map, and then uses a Localizer
    instance to compute the robot's position. It updates a global variable 
    with the new ground map so that it can be displayed in the main loop.
    """
    from localisation2 import Localizer
    # We don't call matplotlib's plot here so it does not block.
    
    # Wait until calibration maps and estimator are computed.
    while not ("map1" in calibration and "map2" in calibration and "estimator" in calibration):
        time.sleep(0.1)

    map_size_m = 28
    scale = 40
    map_size_px = int(map_size_m * scale)
    
    # Define camera roles.
    camera_roles = {camera_indices[0]: "front", camera_indices[1]: "right", camera_indices[2]: "back", camera_indices[3]: "left"}

    # localizer = Localizer(gt_path='src/vision_pkg/vision_pkg/maps/test_field.png', num_levels= 3)
    localizer = Localizer(gt_path='src/vision_pkg/vision_pkg/maps/test_field.png', threshold=127)

    while True:
        time.sleep(0)
        common_ground_map = np.zeros((map_size_px, map_size_px), dtype=np.uint8)
        with global_lock:
            available_frames = {cam: frame_ts for cam, frame_ts in frames_dict.items()}

        for cam_index, role in camera_roles.items():
            if cam_index in available_frames:
                frame, ts = available_frames[cam_index]
                undistorted = undistort_image(frame, calibration["map1"], calibration["map2"], show=False)
                common_ground_map = update_ground_map(
                    common_ground_map,
                    undistorted,
                    calibration["estimator"],
                    thresh_val=100,
                    scale=scale,
                    max_distance=15,
                    camera=role,
                    show=False
                )

        # Optionally perform localisation (result printed).
        h_map, w_map = common_ground_map.shape
        center = (w_map // 2, h_map // 2)
        bot_pos = (0,0)
        bot_angle = 0
        pos_range = 1000
        angle_range = 360
        try:
            (tx_cartesian, ty_cartesian, heading, score, time_taken,
            warp_matrix) = localizer.localize(
                common_ground_map,
                num_good_matches=10,
                center=center,
                plot_mode='none',
                bot_pos=bot_pos,
                bot_angle=bot_angle,
                pos_range=pos_range,
                angle_range=angle_range
            )
            # in meters
            tx_cartesian_m = tx_cartesian / scale
            ty_cartesian_m = ty_cartesian / scale

            print(f"Localization result: Position: ({tx_cartesian_m:.2f}, {ty_cartesian_m:.2f}), "
                    f"Heading: {heading:.2f}°, Time taken: {time_taken:.2f}s")
            # Localizer.plot_results(localizer.ground_truth, common_ground_map, warp_matrix, robot_ground, -heading, center, true_angle=15)
        except Exception as e:
            print(f"Error in localisation: {e}")
        
        # Update the global ground map.
        with global_lock:
            global latest_ground_map
            latest_ground_map = common_ground_map.copy()

# ---------------------------------------------------
# Main function: initializes calibration, starts threads, and displays results.
# ---------------------------------------------------
def main():
    # Compute calibration parameters and undistortion maps.
    cam_manager = CameraManager()
    h, w = 480,640
    map1, map2, K, D, new_K = compute_calibration_params(h, w, distortion_param=0.05, show=False)
    with global_lock:
        calibration["map1"] = map1
        calibration["map2"] = map2
        calibration["K"] = K
        calibration["D"] = D
        calibration["new_K"] = new_K
    estimator = CoordinateEstimator(
        image_width=w,
        image_height=h,
        fov_horizontal=95,  # example value, in degrees
        fov_vertical=78,    # example value, in degrees
        camera_height=0.75,  # meters
        camera_tilt=30 
    )
    with global_lock:
        calibration["estimator"] = estimator

    # Define camera settings and indices.
    custom_settings = {
        'width': w,
        'height': h,
        'fps': 30,
        'fourcc': 'MJPG',
        'buffersize': 1,
        'brightness': 20,
        'auto_exposure': 1,
        'exposure': 350
    }
    camera_indices[0] = cam_manager.get_camera_index("front")
    camera_indices[1] = cam_manager.get_camera_index("right")
    camera_indices[2] = cam_manager.get_camera_index("back")
    camera_indices[3] = cam_manager.get_camera_index("left")
    camera_roles = {camera_indices[0]: "front", camera_indices[1]: "right", camera_indices[2]: "back", camera_indices[3]: "left"}

    camera_configs = [{'camera_index': idx, 'settings': custom_settings} for idx in camera_indices]

    capture_threads = []
    for config in camera_configs:
        cam_idx = config['camera_index']
        settings = config['settings']
        temp_backend = cv2.CAP_V4L2 if settings else 0
        cap_temp = cv2.VideoCapture(cam_idx)
        if not cap_temp.isOpened():
            print(f"Camera {cam_idx} not available.")
            continue
        cap_temp.release()
        t = threading.Thread(target=capture_thread_func, args=(cam_idx, settings), daemon=True)
        t.start()
        capture_threads.append(t)

    if not capture_threads:
        print("No cameras available, exiting.")
        return

    infer_thread = threading.Thread(target=inference_thread_func, daemon=True)
    infer_thread.start()

    # localise_thread = threading.Thread(target=localisation_thread_func, daemon=True)
    # localise_thread.start()

    # Main display loop: show inference frames and the latest ground map.
    while True:
        with global_lock:
            for cam_index, detections in detection_results.items():
                print(f"Detections from Camera {cam_index}:")
                for det in detections:
                    print(f"  Object: {det['object']} | Pixel: {det['pixel_coord']} | "
                          f"Ground: {det['ground_coord']} | Timestamp: {det['timestamp']:.3f} | {det['camera']}")
        with global_lock:
            current_results = {cam: result.copy() for cam, result in results_dict.items() if result is not None}
            current_fps = capture_fps_dict.copy()
            ground_map_to_show = latest_ground_map.copy() if latest_ground_map is not None else None

        # Display each camera's inference result.
        for cam_index, frame in current_results.items():
            fps_val = current_fps.get(cam_index, 0.0)
            cv2.putText(frame, f"FPS: {fps_val:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            window_name = f"Camera {camera_roles[cam_index]} Inference"
            cv2.imshow(window_name, frame)

        # Display the ground map if available.
        if ground_map_to_show is not None:
            cv2.imshow("Common Ground Map", ground_map_to_show)
            
        # A short waitKey allows the OpenCV windows to refresh.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
