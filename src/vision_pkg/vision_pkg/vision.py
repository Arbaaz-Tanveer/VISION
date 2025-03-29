import cv2
import time
import threading
import os
import numpy as np
from ultralytics import YOLO
# Import groundmapping functionality
from groundmapping import (pixel_to_ground, compute_calibration_params, undistort_image,
                           update_ground_map, CoordinateEstimator)

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
    trt_model = YOLO("yolo11n.pt",verbose=False)
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
        cap.set(cv2.CAP_PROP_EXPOSURE, settings.get('exposure', 500))
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
def inference_thread_func():
    model_names = trt_model.names if hasattr(trt_model, 'names') else {}
    while True:
        with global_lock:
            current_frames = {cam: (frame.copy(), ts) for cam, (frame, ts) in frames_dict.items()}
        if not current_frames:
            time.sleep(0.01)
            continue

        for cam_index, (frame, frame_timestamp) in current_frames.items():
            detections_info = []
            try:
                results = trt_model(frame)
                inferred_frame = None
                for r in results:
                    inferred_frame = r.plot()
                    if hasattr(r, "boxes") and r.boxes is not None:
                        boxes = r.boxes.xyxy.cpu().numpy() if hasattr(r.boxes.xyxy, "cpu") else r.boxes.xyxy
                        classes = r.boxes.cls.cpu().numpy() if hasattr(r.boxes.cls, "cpu") else r.boxes.cls
                        for idx, box in enumerate(boxes):
                            x1, y1, x2, y2 = box
                            center_x = (x1 + x2) / 2
                            bottom_center_y = y2 
                            pixel_coord = (center_x, bottom_center_y)
                            
                            cls_id = int(classes[idx])
                            obj_label = model_names.get(cls_id, f"Class {cls_id}")
                            
                            if calibration:
                                ground_coords = pixel_to_ground([pixel_coord],
                                                                calibration["estimator"],
                                                                calibration["K"],
                                                                calibration["D"],
                                                                calibration["new_K"],
                                                                show=False)
                                ground_coord = ground_coords[0] if ground_coords else None
                            else:
                                ground_coord = None
                            
                            detections_info.append({
                                "camera": f"Camera {cam_index}",
                                "object": obj_label,
                                "pixel_coord": pixel_coord,
                                "ground_coord": ground_coord,
                                "timestamp": frame_timestamp
                            })
                if inferred_frame is None:
                    inferred_frame = frame
            except Exception as e:
                print(f"Error during inference on camera {cam_index}: {e}")
                inferred_frame = frame
                detections_info = []

            with global_lock:
                results_dict[cam_index] = inferred_frame
                detection_results[cam_index] = detections_info

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
    from localisation import Localizer
    # We don't call matplotlib's plot here so it does not block.
    
    # Wait until calibration maps and estimator are computed.
    while not ("map1" in calibration and "map2" in calibration and "estimator" in calibration):
        time.sleep(0.1)

    map_size_m = 28
    scale = 40
    map_size_px = int(map_size_m * scale)
    
    # Define camera roles.
    camera_roles = {0: "back", 6: "front", 3: "left", 2: "right"}

    localizer = Localizer(gt_path='src/vision_pkg/vision_pkg/maps/test_field.png', num_levels= 3)
    # localiser = Localizer(gt_path='src/vision_pkg/vision_pkg/maps/test_field.png', threshold=127)

    while True:
        time.sleep(2)
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
                    thresh_val=210,
                    scale=scale,
                    max_distance=15,
                    camera=role,
                    show=False
                )

        # Optionally perform localisation (result printed).
        h_map, w_map = common_ground_map.shape
        center = (w_map // 2, h_map // 2)
        try:
            (tx_cartesian, ty_cartesian, heading, cc, time_taken,
             warp_matrix, robot_ground) = localizer.localize(
                common_ground_map,
                approx_angle=15,
                approx_x_cartesian=-360,
                approx_y_cartesian=-200,
                angle_range=100,
                trans_range=100,
                center=center,
                num_starts=150
            )
            # (tx_cartesian, ty_cartesian, heading, cc, time_taken,
            #  warp_matrix, robot_ground) = localizer.localize(
            #     common_ground_map, num_good_matches=10, center=center, plot_mode='best'
            # )
            #in meters
            tx_cartesian_m = tx_cartesian/scale      
            ty_cartesian_m = ty_cartesian/scale

            print(f"Localization result: Position: ({tx_cartesian_m:.2f}, {ty_cartesian_m:.2f}), "
                  f"Heading: {heading:.2f}°, Time taken: {time_taken:.2f}s")
            Localizer.plot_results(localizer.ground_truth, common_ground_map, warp_matrix, robot_ground, -heading, center, true_angle=15)
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
    h, w = 960, 1280
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
        'exposure': 45
    }
    camera_indices = [3,4,6,0]
    camera_configs = [{'camera_index': idx, 'settings': custom_settings} for idx in camera_indices]

    capture_threads = []
    for config in camera_configs:
        cam_idx = config['camera_index']
        settings = config['settings']
        temp_backend = cv2.CAP_V4L2 if settings else 0
        cap_temp = cv2.VideoCapture(cam_idx, temp_backend)
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

    localise_thread = threading.Thread(target=localisation_thread_func, daemon=True)
    localise_thread.start()

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
            window_name = f"Camera {cam_index} Inference"
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
