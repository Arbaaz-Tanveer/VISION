import cv2
import time
import threading
import os
import numpy as np
from ultralytics import YOLO
# Import groundmapping functionality
from vision_pkg.groundmapping import (pixel_to_ground, compute_calibration_params, undistort_image,
                           update_ground_map, CoordinateEstimator)
from vision_pkg.utilities import *

# ROS2 imports
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from std_msgs.msg import Float32MultiArray, MultiArrayDimension, Int16MultiArray
from functools import partial
from localisation_pkg.srv import Localisation

# ---------------------------------------------------
# Global calibration variables (to be computed once)
# ---------------------------------------------------
calibration = {}  # Will hold map1, map2, K, D, new_K, estimator

# Global variable to hold the latest ground map.
latest_ground_map = None
latest_localiser_img = None
latest_dotted_img = None
map_scale = 100
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
bot_pos = [0.0,1.6, np.pi] #x,y,theta
obstacles = [] #no.of obstacles,x1,y1,timestamp1.....
ball_pos = [] #x,y,timestamps
odo_buffer = OdometryBuffer(capacity=2000)    #buffer for storing odometry records
contour_images = {}

x_abs_delta_since_last_localisation = 0
y_abs_delta_since_last_localisation = 0
rotation_since_last_localisation = 0

# ROS2 node and publishers/subscribers
ros_node = None
bot_pos_pub = None
ball_pos_pub = None
obstacles_pub = None
command_sub = None
localisation_client = None
executor = None
executor_localisation = None


# ---------------------------------------------------
# Capture thread function (supports optional custom settings)
# ---------------------------------------------------

#camera indices
camera_indices = [None,None,None,None]
odom_calls = 0
odom_time_last = time.time()
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
        cap.set(cv2.CAP_PROP_BRIGHTNESS, settings.get('brightness', 20))
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
# ROS2 Callback for command subscriber
# ---------------------------------------------------
def command_callback(msg):
    """
    Callback function that processes commands received from the command topic
    The message is expected to be a Float32MultiArray with custom format
    """
    global bot_pos, odom_calls, odom_time_last, x_abs_delta_since_last_localisation, y_abs_delta_since_last_localisation, rotation_since_last_localisation
    try:
        command_data = msg.data
        # print(f"Received command: {command_data}")
        # Process the command based on its format
        # Example: If first value is command ID followed by parameters
        if len(command_data) > 0:
            with global_lock:
                odo_buffer.add_record(command_data[0], command_data[1], command_data[2], command_data[3])
                odom_calls += 1
                if (time.time() - odom_time_last) > 1.0:
                    print(f"Odometry calls per second: {odom_calls/(time.time()-odom_time_last)}")
                    odom_calls = 0
                    odom_time_last = time.time()
                theta = bot_pos[2]
        
        dx,dy,dtheta = command_data[1], command_data[2], command_data[3]
        cos_angle = math.cos(theta)
        sin_angle = math.sin(theta)
        global_dx = dx * cos_angle - dy * sin_angle
        global_dy = dx * sin_angle + dy * cos_angle

    

        with global_lock:
            bot_pos[0] += global_dx
            bot_pos[1] += global_dy
            bot_pos[2] += dtheta
            x_abs_delta_since_last_localisation += abs(dx)
            y_abs_delta_since_last_localisation += abs(dy)            
            rotation_since_last_localisation += abs(dtheta)
    
    except Exception as e:
        print(f"Error processing command: {e}")


# ---------------------------------------------------
# Create Float32MultiArray message helper function
# ---------------------------------------------------
def create_float32_array(data, label=""):
    """Helper function to create a Float32MultiArray message with proper layout"""
    msg = Float32MultiArray()
    msg.layout.dim.append(MultiArrayDimension())
    msg.layout.dim[0].label = label
    msg.layout.dim[0].size = len(data)
    msg.layout.dim[0].stride = len(data)
    msg.layout.data_offset = 0
    # Ensure all data is explicitly converted to float32
    msg.data = [float(val) for val in data]
    return msg

# Note: We removed the separate publisher thread function since publishing
# will be done directly in the main loop at the inference rate

# ---------------------------------------------------
# Inference thread function: processes each camera's frame one by one
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

class LocalisationClient(Node):

    def __init__(self):
        super().__init__('localisation_client')
        self.cli = self.create_client(Localisation, 'localise')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.req = Localisation.Request()

    def send_request(self, flattened_points, bounds):
        points = Int16MultiArray()
        points.data = flattened_points
        dim0 = MultiArrayDimension()
        dim0.label = 'points'
        dim0.size = len(flattened_points)//2
        dim0.stride = len(flattened_points)
        dim1 = MultiArrayDimension()
        dim1.label = 'coords'
        dim1.size = 2
        dim1.stride = 2
        points.layout.dim = [dim0, dim1]
        points.layout.data_offset = 0
        self.req.points = points

        bounds_msg = Float32MultiArray()
        bounds_msg.data = bounds
        dim0 = MultiArrayDimension()
        dim0.label = 'axes'
        dim0.size = 3
        dim0.stride = 6
        dim1 = MultiArrayDimension()
        dim1.label = 'bounds'
        dim1.size = 2
        dim1.stride = 2
        bounds_msg.layout.dim = [dim0, dim1]
        bounds_msg.layout.data_offset = 0
        self.req.bounds = bounds_msg
        
        self.get_logger().info("Sending request to localisation server")
        return self.cli.call_async(self.req)

# ---------------------------------------------------
# Localisation thread function: update ground map and localise every 2 seconds
# ---------------------------------------------------
def localisation_thread_func():
    """
    This function collects the latest frames from defined cameras,
    undistorts them, updates a common ground map, and then uses a Localizer
    instance to compute the robot's position. It updates a global variable 
    with the new ground map so that it can be displayed in the main loop.
    
    Now includes timing measurements to track latency between frame capture and localization result.
    """
    global bot_pos, x_abs_delta_since_last_localisation, y_abs_delta_since_last_localisation, rotation_since_last_localisation
    odometry_weight = 0.8

    show_localisation_result = True
    
    # Wait until calibration maps and estimator are computed.
    while not ("map1" in calibration and "map2" in calibration and "estimator" in calibration):
        time.sleep(0.1)

    map_size_m = 6
    with global_lock:
        scale = map_scale
    map_size_px = int(map_size_m * scale)
    
    # Define camera roles.
    camera_roles = {camera_indices[0]: "front", camera_indices[1]: "right", camera_indices[2]: "back", camera_indices[3]: "left"}

    # localizer = Localizer(gt_path='src/vision_pkg/vision_pkg/maps/test_field.png', num_levels= 3)
    image_processor = ImageProcessing()
    
    # Define estimated camera frame processing latency in seconds
    # This is the time between when a frame is captured in real life and when it's available in the system
    estimated_camera_latency = 0.1  # 50ms - adjust based on your hardware
    
    while True:
        time.sleep(0)
        common_ground_map = np.zeros((map_size_px, map_size_px), dtype=np.uint8)
        
        # Collect frame timestamps
        frame_timestamps = []
        with global_lock:
            available_frames = {cam: frame_ts for cam, frame_ts in frames_dict.items()}
            past_bot_pos = bot_pos
        
        # First pass: collect all timestamps
        for cam_index in camera_roles:
            if cam_index in available_frames:
                _, ts = available_frames[cam_index]
                frame_timestamps.append(ts)
        
        # Calculate average timestamp if we have frames
        if frame_timestamps:
            avg_frame_timestamp = sum(frame_timestamps) / len(frame_timestamps)
            # Subtract latency to get estimated real-world capture time
            estimated_real_capture_time = avg_frame_timestamp - estimated_camera_latency
            print(f"Camera frames timestamps: {[f'{ts:.6f}' for ts in frame_timestamps]}")
            print(f"Average frame timestamp: {avg_frame_timestamp:.6f}, Estimated real capture time: {estimated_real_capture_time:.6f}")
        else:
            estimated_real_capture_time = None
            print("No camera frames available for timestamp calculation")
        ground_map_start = time.time()
        avg_update_time = 0
        # Process the frames for localization
        for cam_index, role in camera_roles.items():
            if cam_index in available_frames:
                frame, ts = available_frames[cam_index]
                binary_image = image_processor.white_threshold(frame)
                process_start_time = time.time()
                undistorted = undistort_image(binary_image, calibration["map1"], calibration["map2"], show=False)
                avg_update_time += time.time()-process_start_time
                binary_image = image_processor.process_map(undistorted)
                with global_lock:
                    contour_images[cam_index] = binary_image
                common_ground_map = update_ground_map(
                    common_ground_map,
                    binary_image,
                    calibration["estimator"],
                    thresh_val=120,
                    scale=scale,
                    max_distance=map_size_m//2,
                    camera=role,
                    show=False
                )
        avg_update_time/=4
        # print(f"Ground Map undistort time avg: {avg_update_time:.6f}s")
        # common_ground_map = image_processor.process_map(common_ground_map)
        
        # Optionally perform localisation (result printed).
        
        # Record the start time for localization
        ground_map_end = time.time()
        try:
            dotted_img, points = skeletonizer(common_ground_map, grid_spacing = 3, make_dotted_img = False)
            skeletonizer_end = time.time()
            with global_lock:
                curr_pos = bot_pos
            map_pos = [(bot_pos[0] + 1.4)*map_scale, (-bot_pos[1] + 1.2)*map_scale, 2*np.pi - bot_pos[2]]
            bound_size = [0.7 * map_scale, 0.7 * map_scale, np.pi/4]

            future = localisation_client.send_request(points, 
                                                      [map_pos[0] - bound_size[0], 
                                                       map_pos[0] + bound_size[0], 
                                                       map_pos[1] - bound_size[1], 
                                                       map_pos[1] + bound_size[1], 
                                                       map_pos[2] - bound_size[2], 
                                                       map_pos[2] + bound_size[2]])
            executor_localisation.spin_until_future_complete(future, 0.5)
            response = future.result()
            if response is None:
                localisation_client.get_logger().error("Request timed out")
                continue
            localisation_client.get_logger().info(
                'Response received: X = %f Y = %f Theta = %f' %
                (response.transform.data[0], response.transform.data[1], response.transform.data[2]))

            tx_cartesian = response.transform.data[0] / map_scale - 1.4
            ty_cartesian = -(response.transform.data[1] / map_scale - 1.2)
            heading = principal_value_radians(2*np.pi - response.transform.data[2])
            

            with global_lock:
                curr_pos = bot_pos
                sq_dist_delta = x_abs_delta_since_last_localisation*x_abs_delta_since_last_localisation + y_abs_delta_since_last_localisation*y_abs_delta_since_last_localisation
                angle_delta = rotation_since_last_localisation
            square_distance_tolerance = 0.1 + sq_dist_delta * sq_dist_delta * 0.01 #max dist tolerance scales like dist squared 
            angle_tolerance = 0.35 + angle_delta * 0.01 #+ sq_dist_delta * 0.05
            localisation_client.get_logger().info(f"Current sq dist tolerance = {square_distance_tolerance:.3f} angle tolerance = {angle_tolerance:.3f}")
            if (tx_cartesian - curr_pos[0])*(tx_cartesian - curr_pos[0]) + (ty_cartesian - curr_pos[1])*(ty_cartesian - curr_pos[1]) > square_distance_tolerance or abs((heading - curr_pos[2] + np.pi) % (2 * np.pi) - np.pi) > angle_tolerance:
                localisation_client.get_logger().info("Position too far, rejecting data")
                continue

            with global_lock:
                x_abs_delta_since_last_localisation = 0
                y_abs_delta_since_last_localisation = 0
                rotation_since_last_localisation = 0

            visualiser_start = time.time()
            if show_localisation_result:
                localiser_result_img = visualise_localisation_result(common_ground_map, response.transform.data[2], response.transform.data[0], response.transform.data[1])
            else:
                localiser_result_img = None
            # Record when localization finished
            localization_end_time = time.time()
        
            pos_localisation = [tx_cartesian,ty_cartesian,heading]
            
            # Calculate and print timing information
            if estimated_real_capture_time is not None:
                total_latency = localization_end_time - estimated_real_capture_time
                print(f"Total latency (capture to localization result): {total_latency:.6f}s, Ground map creation: {ground_map_end - ground_map_start:.6f}s, Skeletonizer: {skeletonizer_end - ground_map_end:.6f}s, Genetic: {visualiser_start - skeletonizer_end:.6f}s, Visualiser: {localization_end_time-visualiser_start:.6f}s")
            
                # print(f"Breakdown: Estimated camera latency: {estimated_camera_latency:.6f}s, Processing time: {time_taken:.6f}s,Processing time thorugh time calcilation = {localization_end_time - localization_start_time}")
            
            with global_lock:
                curr_pos = bot_pos
            pos_localisation = odo_buffer.integrate_with_initial(pos_localisation, time_window_ms=total_latency*1000)   #pos at the time the frame was captured with which localisation was done 

            final_pos = [
            odometry_weight * curr_pos[0] + (1 - odometry_weight) * pos_localisation[0],
            odometry_weight * curr_pos[1] + (1 - odometry_weight) * pos_localisation[1],
            odometry_weight * ((curr_pos[2] - pos_localisation[2] + np.pi) % (2 * np.pi) - np.pi) + pos_localisation[2]
            ]

            print(f"the current position of the bot is {final_pos}")
            with global_lock:
                bot_pos = [final_pos[0],final_pos[1],final_pos[2]]
            # Update global bot position
            # with global_lock:                                #--------------------------bot position updated here 
            #     bot_pos = [tx_cartesian_m, ty_cartesian_m, heading]
                
            # Localizer.plot_results(localizer.ground_truth, common_ground_map, warp_matrix, robot_ground, -heading, center, true_angle=15)
        except Exception as e:
            # localisation_client.get_logger().error(f"Error in localisation")
            logging.exception("Error in localisation")
            # print(f"Error in localisation: {e}")
        
        # Update the global ground map.
        with global_lock:
            global latest_ground_map, latest_localiser_img, latest_dotted_img
            latest_ground_map = common_ground_map.copy() if common_ground_map is not None else None
            latest_localiser_img = localiser_result_img.copy() if localiser_result_img is not None else None
            latest_dotted_img = dotted_img.copy() if dotted_img is not None else None

# ---------------------------------------------------
# Initialize ROS2 Node and publishers/subscribers
# ---------------------------------------------------
def init_ros2():
    global ros_node, bot_pos_pub, ball_pos_pub, obstacles_pub, command_sub, localisation_client, executor, executor_localisation
    
    # Initialize ROS2
    rclpy.init()
    
    # Create ROS2 node
    ros_node = Node('vision_system')
    
    # Create publishers
    bot_pos_pub = ros_node.create_publisher(
        Float32MultiArray, 
        'robot_position', 
        10
    )
    
    ball_pos_pub = ros_node.create_publisher(
        Float32MultiArray, 
        'ball_position', 
        10
    )
    
    obstacles_pub = ros_node.create_publisher(
        Float32MultiArray, 
        'obstacles_position', 
        10
    )
    
    # Create subscriber
    command_sub = ros_node.create_subscription(
        Float32MultiArray,
        'odom_delta',
        command_callback,
        10
    )
    
    localisation_client = LocalisationClient()

    executor = MultiThreadedExecutor(1)
    executor_localisation = MultiThreadedExecutor(1)
    executor.add_node(ros_node)
    executor_localisation.add_node(localisation_client)
    
    # Start a thread to spin the ROS node
    threading.Thread(target=lambda: executor.spin(), daemon=True).start()
    
    print("ROS2 node initialized successfully")

# ---------------------------------------------------
# Main function: initializes calibration, starts threads, and displays results.
# ---------------------------------------------------
def main():
    # Initialize ROS2
    init_ros2()
    
    # Compute calibration parameters and undistortion maps.
    

    cam_manager = CameraManager()
    h, w = 480,640
    combined_frames = np.zeros((2*h, 2*w, 3), dtype=np.uint8)
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
        'fps': 18,
        'fourcc': 'MJPG',
        'buffersize': 1,
        'brightness': 20,
        'auto_exposure': 1,
        'exposure': 400
    }
    camera_indices[0] = cam_manager.get_camera_index("front")
    camera_indices[1] = cam_manager.get_camera_index("right")
    camera_indices[2] = cam_manager.get_camera_index("back")
    camera_indices[3] = cam_manager.get_camera_index("left")
    camera_roles = {camera_indices[0]: "front", camera_indices[1]: "right", camera_indices[2]: "back", camera_indices[3]: "left"}

    camera_configs = [{'camera_index': idx, 'settings': custom_settings} for idx in camera_indices]

    # Camera Display Settings
    show_contours = True
    contour_alpha = 0.5
    show_individual_cameras = False

    capture_threads = []
    for config in camera_configs:
        cam_idx = config['camera_index']
        settings = config['settings']
        temp_backend = cv2.CAP_V4L2 if settings else 0
        cap_temp = cv2.VideoCapture(cam_idx,temp_backend)
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
    localisation_client.get_logger().info("Started Localisation thread")

                # Main display loop: show inference frames and the latest ground map.
    try:
        while True:
            with global_lock:
                global ball_pos, obstacles
                ball_pos = None
                obstacles = []
                for cam_index, detections in detection_results.items():
                    for det in detections:
                        if det['ground_coord'][0] is not None or det['ground_coord'][1] is not None:
                            if det['object'] == 'sports ball' and det['ground_coord'] is not None:
                                ball_pos = det['ground_coord'] 
                                ball_pos = list(compute_observed_bot_position(camera_roles[cam_index], ball_pos, bot_pos))
                                # Add timestamp
                                ball_pos.append(det['timestamp'])
                            else:
                                obs_pos = compute_observed_bot_position(camera_roles[cam_index], det['ground_coord'], bot_pos)
                                obstacles.append(list(obs_pos))
                
                # Publish to ROS topics directly in the main loop
                # 1. Robot position
                current_bot_pos = bot_pos.copy() if isinstance(bot_pos, list) else [0, 0, 0]
                bot_msg = create_float32_array(current_bot_pos, "bot_position")
                bot_pos_pub.publish(bot_msg)
                
                # 2. Ball position
                if ball_pos and len(ball_pos) >= 2:
                    # Ensure we have x, y, timestamp format
                    current_ball_pos = ball_pos.copy()
                    if len(current_ball_pos) == 2:  # If only x,y are available
                        current_ball_pos.append(time.time())  # Add timestamp
                    ball_msg = create_float32_array(current_ball_pos, "ball_position")
                    ball_pos_pub.publish(ball_msg)
                
                # 3. Obstacles
                if obstacles:
                    obstacles_data = [len(obstacles)]  # Start with number of obstacles
                    for obs in obstacles:
                        if obs and len(obs) >= 2:  # Ensure obstacle has at least x,y
                            obstacles_data.extend([obs[0], obs[1]])  # Add x,y coordinates
                    obstacles_msg = create_float32_array(obstacles_data, "obstacles")
                    obstacles_pub.publish(obstacles_msg)
                else:
                    # Publish empty obstacle list (just the count of 0)
                    obstacles_msg = create_float32_array([0], "obstacles")
                    obstacles_pub.publish(obstacles_msg)
                    
                #the odometry integration
                new_pose = odo_buffer.integrate_with_initial([0.0, 0.0, 0.0], time_window_ms=10000)
                print(f"Updated pose over the last 10 seconds: x={new_pose[0]:.3f}, y={new_pose[1]:.3f}, theta={new_pose[2]:.3f}")
                # print(f"Bot Position: x={bot_pos[0]:.3f} y={bot_pos[1]:.3f} theta={bot_pos[2]:.3f}")
                    
            with global_lock:
                current_results = {cam: result.copy() for cam, result in results_dict.items() if result is not None}
                contour_images_to_show = {cam: result.copy() for cam, result in contour_images.items() if result is not None}
                current_fps = capture_fps_dict.copy()
                ground_map_to_show = latest_ground_map.copy() if latest_ground_map is not None else None
                localiser_img_to_show = latest_localiser_img.copy() if latest_localiser_img is not None else None
                dotted_img_to_show = latest_dotted_img.copy() if latest_dotted_img is not None else None
            
            # Display each camera's inference result.
            for cam_index, frame in current_results.items():
                fps_val = current_fps.get(cam_index, 0.0)
                if show_contours:
                    contour_img = contour_images_to_show.get(cam_index, None)
                    if contour_img is not None:
                        contour_img = cv2.cvtColor(contour_img, cv2.COLOR_GRAY2BGR)
                        frame = cv2.addWeighted(frame, 1, contour_img, contour_alpha, 0)
                cv2.putText(frame, f"FPS: {fps_val:.1f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                if show_individual_cameras:
                    window_name = f"Camera {camera_roles[cam_index]} Inference"
                    cv2.imshow(window_name, frame)
                else:
                    cv2.putText(frame, f"{camera_roles[cam_index]}", (200, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    x_offset = ((cam_index//2)%2)*w
                    y_offset = ((cam_index//4))*h
                    combined_frames[y_offset:y_offset+h, x_offset:x_offset+w, :] = frame
            if not show_individual_cameras:
                cv2.imshow("Combined Frames", combined_frames)

            # Display the ground map if available.
            if ground_map_to_show is not None:
                cv2.imshow("Common Ground Map", ground_map_to_show)

            if localiser_img_to_show is not None:
                cv2.imshow("Localiser Result", localiser_img_to_show)

            if dotted_img_to_show is not None:
                cv2.imshow("Dotted Ground Map", dotted_img_to_show)
                
            # A short waitKey allows the OpenCV windows to refresh.
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        print("Shutting down gracefully...")
    finally:
        # Cleanup
        cv2.destroyAllWindows()
        if ros_node is not None:
            ros_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()