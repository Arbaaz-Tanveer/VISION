import cv2
import numpy as np
import time

#############################################
# Coordinate Estimator and Ground Map Code  #
#############################################

class CoordinateEstimator:
    def __init__(self, image_width, image_height, fov_horizontal, fov_vertical, camera_height, camera_tilt=0):
        """
        Initialize the coordinate estimator.
        """
        self.image_width = image_width
        self.image_height = image_height
        self.fov_h = np.radians(fov_horizontal)
        self.fov_v = np.radians(fov_vertical)
        self.camera_height = camera_height
        # Camera tilt is defined as the angle down from horizontal.
        self.camera_tilt = np.radians(camera_tilt)
        
        # Calculate focal lengths in pixels
        self.focal_length_x = (image_width / 2) / np.tan(self.fov_h / 2)
        self.focal_length_y = (image_height / 2) / np.tan(self.fov_v / 2)
        
        # Image center
        self.cx = image_width / 2
        self.cy = image_height / 2
        
        # Precompute world coordinates lookup table for each pixel.
        self.world_coords = self.precompute_world_coords()

    def precompute_world_coords(self):
        """
        Precompute the intersection of camera rays with the ground plane (y=0)
        for every pixel in the image. Returns an array of shape (H, W, 2) containing
        the (world_x, world_z) coordinates.
        """
        xs, ys = np.meshgrid(np.arange(self.image_width), np.arange(self.image_height))
        x_norm = (xs - self.cx) / self.focal_length_x
        y_norm = (ys - self.cy) / self.focal_length_y
        rays = np.stack((x_norm, y_norm, np.ones_like(x_norm)), axis=2)
        norms = np.linalg.norm(rays, axis=2, keepdims=True)
        rays_normalized = rays / norms

        # Rotation matrix for camera tilt (rotating from camera to world coordinates)
        ct = np.cos(self.camera_tilt)
        st = np.sin(self.camera_tilt)
        R = np.array([
            [1,    0,   0],
            [0,   ct,  st],
            [0,  -st,  ct]
        ])
        
        rays_flat = rays_normalized.reshape(-1, 3).T  # shape: (3, H*W)
        rays_world_flat = R @ rays_flat
        rays_world = rays_world_flat.T.reshape(self.image_height, self.image_width, 3)
        
        # Calculate scale factor 't' for intersection with ground (y=0)
        with np.errstate(divide='ignore', invalid='ignore'):
            t = -self.camera_height / rays_world[..., 1]
            t = np.where(np.abs(rays_world[..., 1]) < 1e-6, np.nan, t)
            t = np.where(t >= 0, np.nan, t)
        
        world_x = t * rays_world[..., 0]
        world_z = t * rays_world[..., 2]
        world_coords = np.stack((world_x, world_z), axis=2)
        return world_coords

    def estimate_world_position(self, pixel_x, pixel_y):
        """
        Estimate the ground-plane (world) coordinates of a pixel.
        """
        world_point = self.world_coords[int(pixel_y), int(pixel_x)]
        return np.array([world_point[0], 0, world_point[1]])
    
    def draw_world_grid(self, image, grid_size=1.0, max_distance=10.0):
        """
        Draw a world-space grid on the image.
        """
        result = image.copy()
        x_range = np.arange(-max_distance, max_distance + grid_size, grid_size)
        z_range = np.arange(0, max_distance + grid_size, grid_size)
        for x in x_range:
            points = []
            for z in z_range:
                world_point = np.array([x, 0, z])
                ct = np.cos(-self.camera_tilt)
                st = np.sin(-self.camera_tilt)
                R_inv = np.array([
                    [1, 0, 0],
                    [0, ct, st],
                    [0, -st, ct]
                ])
                camera_pos = np.array([0, self.camera_height, 0])
                point_cam = R_inv @ (world_point - camera_pos)
                if point_cam[2] > 0:
                    pixel_x = int((point_cam[0] / point_cam[2]) * self.focal_length_x + self.cx)
                    pixel_y = int(self.cy - (point_cam[1] / point_cam[2]) * self.focal_length_y)
                    if 0 <= pixel_x < self.image_width and 0 <= pixel_y < self.image_height:
                        points.append((pixel_x, pixel_y))
            if len(points) >= 2:
                for i in range(len(points) - 1):
                    cv2.line(result, points[i], points[i + 1], (0, 255, 0), 2)
                    
        for z in z_range:
            points = []
            for x in x_range:
                world_point = np.array([x, 0, z])
                ct = np.cos(-self.camera_tilt)
                st = np.sin(-self.camera_tilt)
                R_inv = np.array([
                    [1, 0, 0],
                    [0, ct, st],
                    [0, -st, ct]
                ])
                camera_pos = np.array([0, self.camera_height, 0])
                point_cam = R_inv @ (world_point - camera_pos)
                if point_cam[2] > 0:
                    pixel_x = int((point_cam[0] / point_cam[2]) * self.focal_length_x + self.cx)
                    pixel_y = int(self.cy - (point_cam[1] / point_cam[2]) * self.focal_length_y)
                    if 0 <= pixel_x < self.image_width and 0 <= pixel_y < self.image_height:
                        points.append((pixel_x, pixel_y))
            if len(points) >= 2:
                for i in range(len(points) - 1):
                    cv2.line(result, points[i], points[i + 1], (0, 255, 0), 2)
                mid_point = points[len(points) // 2]
                cv2.putText(result, f"{z:.1f}m", 
                            (mid_point[0], mid_point[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        return result

def initialize_estimator(desired_width,desired_height):
    """
    Initialize and return a CoordinateEstimator instance with desired parameters.
    Adjust these parameters as needed for your setup.
    """
    return CoordinateEstimator(
        image_width=desired_width,
        image_height=desired_height,
        fov_horizontal=92,  # example value, in degrees
        fov_vertical=80,    # example value, in degrees
        camera_height=0.68,  # meters
        camera_tilt=30      # degrees down from horizontal
    )

def create_ground_map(map_size_m=40, scale=20):
    """
    Create an empty ground map.
    """
    map_size_px = int(map_size_m * scale)
    return np.zeros((map_size_px, map_size_px), dtype=np.uint8)

def update_ground_map(ground_map, image, estimator, scale=20, max_distance=20, camera=None):
    """
    Update the ground map with detections from the image.
    """
    map_size_px = ground_map.shape[0]
    map_center = (map_size_px // 2, map_size_px // 2)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret_thresh, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    ys, xs = np.where(binary > 0)
    world_points = estimator.world_coords[ys, xs]  # (N,2) array: [:,0]=world_x, [:,1]=world_z
    world_x = world_points[:, 0]
    world_z = world_points[:, 1]
    # Filter points within a maximum distance (here, based on z-distance)
    within_range = world_z <= max_distance
    world_x = world_x[within_range]
    world_z = world_z[within_range]

    if camera == "front":
        map_x = (map_center[0] - world_x * scale).astype(np.int32)
        map_y = (map_center[1] + world_z * scale).astype(np.int32)
    elif camera == "back":
        map_x = (map_center[0] + world_x * scale).astype(np.int32)
        map_y = (map_center[1] - world_z * scale).astype(np.int32)
    elif camera == "right":
        map_x = (map_center[0] - world_z * scale).astype(np.int32)
        map_y = (map_center[1] - world_x * scale).astype(np.int32)
    elif camera == "left":
        map_x = (map_center[0] + world_z * scale).astype(np.int32)
        map_y = (map_center[1] + world_x * scale).astype(np.int32)
    else:
        map_x = (map_center[0] - world_x * scale).astype(np.int32)
        map_y = (map_center[1] + world_z * scale).astype(np.int32)
    
    valid = (map_x >= 0) & (map_x < map_size_px) & (map_y >= 0) & (map_y < map_size_px)
    map_x = map_x[valid]
    map_y = map_y[valid]
    
    ground_map[map_y, map_x] = 255
    return ground_map

#############################################
# Multi-Camera Live Feed and Processing     #
#############################################

def open_camera(index, width, height, fps=60):
    """
    Open a camera using OpenCV. Returns the capture if successful, else None.
    """
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    if not cap.isOpened():
        print(f"Camera with index {index} could not be opened.")
        return None

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    # Optional camera settings
    cap.set(cv2.CAP_PROP_BRIGHTNESS, 20)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
    cap.set(cv2.CAP_PROP_EXPOSURE, 1700)
    return cap

def multi_camera_preview():
    """
    Opens up to four cameras (front, back, left, right) and processes their live feed.
    The code is robust so that if some cameras are not connected, it continues with available ones.
    """
    # Define camera indices for different orientations.
    # Adjust the indices to match your system.
    camera_indices = {
        "front": 2,
        "back": 1,
        "left": 0,
        "right": 3
    }
    desired_width = 1644
    desired_height = 1256
    desired_fps = 60

    cameras = {}
    for orientation, idx in camera_indices.items():
        cap = open_camera(idx, desired_width, desired_height, fps=desired_fps)
        if cap is not None:
            cameras[orientation] = cap
        else:
            print(f"Skipping {orientation} camera.")

    if not cameras:
        print("No cameras available. Exiting.")
        return

    # Prepare undistortion parameters.
    # For simplicity, we use the same calibration for all cameras.
    # You can adjust these for each camera if needed.
    distortion_k1 = (105 - 100) / 100.0  # 0.05
    D = np.array([[distortion_k1], [0.0], [0.0], [0.0]], dtype=np.float32)

    # Initialize the coordinate estimator (parameters may be adjusted for your use-case)
    estimator = initialize_estimator(desired_width,desired_height)
    ground_map = create_ground_map(map_size_m=10, scale=100)

    # Setup FPS counters per camera
    fps_counters = {orientation: {"frame_count": 0, "start_time": time.time(), "fps": 0} for orientation in cameras}

    print("Press 'q' to exit, 's' to save the latest frame from any camera.")
    latest_frames = {orientation: None for orientation in cameras}

    while True:
        # Reset ground map each frame if you prefer fresh detections.
        ground_map[:] = 0

        for orientation, cap in cameras.items():
            ret, frame = cap.read()
            if not ret:
                print(f"Warning: Couldn't retrieve frame from {orientation} camera.")
                continue

            # Undistort the frame.
            h, w = frame.shape[:2]
            K = np.array([[w / 2, 0, w / 2],
                          [0, w / 2, h / 2],
                          [0, 0, 1]], dtype=np.float32)
            new_w, new_h = w, h
            new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                        K, D, (w, h), np.eye(3), balance=1, new_size=(new_w, new_h))
            map1, map2 = cv2.fisheye.initUndistortRectifyMap(
                        K, D, np.eye(3), new_K, (new_w, new_h), cv2.CV_16SC2)
            undistorted = cv2.remap(frame, map1, map2,
                                    interpolation=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_CONSTANT)
            latest_frames[orientation] = undistorted

            # Update FPS counter.
            counter = fps_counters[orientation]
            counter["frame_count"] += 1
            elapsed = time.time() - counter["start_time"]
            if elapsed > 1:
                counter["fps"] = counter["frame_count"] / elapsed
                counter["frame_count"] = 0
                counter["start_time"] = time.time()

            cv2.putText(undistorted, f"{orientation} FPS: {counter['fps']:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Update the ground map with detections from this camera.
            # The 'camera' parameter tells the mapping function how to orient the points.
            ground_map = update_ground_map(ground_map, undistorted, estimator,
                                           scale=100, max_distance=20, camera=orientation)

            # Display each camera feed.
            cv2.imshow(f"{orientation} Feed", undistorted)

        # Display the ground map.
        cv2.imshow("Ground Map (40x40m)", ground_map)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Save the latest available frame from each camera.
            for orientation, frame in latest_frames.items():
                if frame is not None:
                    save_path = f'undistorted_{orientation}_saved.jpg'
                    cv2.imwrite(save_path, frame)
                    print(f"Image saved to {save_path}")

    # Release all cameras and close windows.
    for cap in cameras.values():
        cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    multi_camera_preview()
