import cv2
import numpy as np
import time

class CoordinateEstimator:
    def __init__(self, image_width, image_height, fov_horizontal, fov_vertical, camera_height, camera_tilt=0):
        """
        Initialize the coordinate estimator.
        
        Args:
            image_width (int): Width of the camera image in pixels
            image_height (int): Height of the camera image in pixels
            fov_horizontal (float): Horizontal field of view in degrees
            fov_vertical (float): Vertical field of view in degrees
            camera_height (float): Height of camera from ground in meters
            camera_tilt (float): Camera tilt angle from vertical in degrees (default 0)
        """
        self.image_width = image_width
        self.image_height = image_height
        self.fov_h = np.radians(fov_horizontal)
        self.fov_v = np.radians(fov_vertical)
        self.camera_height = camera_height
        self.camera_tilt = np.radians(camera_tilt)
        
        # Calculate focal length in pixels
        self.focal_length_x = (image_width / 2) / np.tan(self.fov_h / 2)
        self.focal_length_y = (image_height / 2) / np.tan(self.fov_v / 2)
        
        # Image center
        self.cx = image_width / 2
        self.cy = image_height / 2
        
        # Create camera matrix
        self.camera_matrix = np.array([
            [self.focal_length_x, 0, self.cx],
            [0, self.focal_length_y, self.cy],
            [0, 0, 1]
        ])
        
        self.calculate_horizon_line()
        
    def calculate_horizon_line(self):
        """Calculate the horizon line in image coordinates."""
        # The horizon line is where the rotated ray's y-component becomes 0.
        horizon_angle = self.camera_tilt - np.pi/2
        self.horizon_y = self.cy - self.focal_length_y * np.tan(horizon_angle)
        
    def is_above_horizon(self, pixel_y):
        """Check if a pixel is above the horizon line."""
        return pixel_y < self.horizon_y
    
    def pixel_to_ray(self, pixel_x, pixel_y):
        """
        Convert pixel coordinates to a ray direction in camera space.
        """
        x = (pixel_x - self.cx) / self.focal_length_x
        y = (pixel_y - self.cy) / self.focal_length_y
        ray = np.array([x, y, 1.0])
        return ray / np.linalg.norm(ray)
    
    def draw_world_grid(self, image, grid_size=1.0, max_distance=10.0):
        """
        Draw a world-space grid on the image with corrected vertical direction.
        
        Args:
            image (numpy.ndarray): Input image.
            grid_size (float): Size of grid squares in meters.
            max_distance (float): Maximum distance to draw grid.
            
        Returns:
            numpy.ndarray: Image with grid drawn.
        """
        result = image.copy()
        # Draw the horizon line
        cv2.line(result, 
                 (0, int(self.horizon_y)), 
                 (self.image_width, int(self.horizon_y)), 
                 (255, 0, 0), 2)  # Blue horizon line
        
        # Define ranges for longitudinal (x) and lateral (z) lines
        x_range = np.arange(-max_distance, max_distance + grid_size, grid_size)
        z_range = np.arange(0, max_distance + grid_size, grid_size)
        
        # Draw longitudinal grid lines (parallel to camera's forward direction)
        for x in x_range:
            points = []
            for z in z_range:
                # World point in camera coordinates: (x, -camera_height, z)
                point_cam = np.array([x, -self.camera_height, z])
                # Apply inverse camera tilt (rotation about x-axis)
                rotation_matrix = np.array([
                    [1, 0, 0],
                    [0, np.cos(-self.camera_tilt), -np.sin(-self.camera_tilt)],
                    [0, np.sin(-self.camera_tilt),  np.cos(-self.camera_tilt)]
                ])
                point_cam = rotation_matrix @ point_cam
                
                if point_cam[2] > 0:
                    pixel_x = int((point_cam[0] / point_cam[2]) * self.focal_length_x + self.cx)
                    pixel_y = int(self.cy - (point_cam[1] / point_cam[2]) * self.focal_length_y)
                    if 0 <= pixel_x < self.image_width and 0 <= pixel_y < self.image_height:
                        points.append((pixel_x, pixel_y))
            if len(points) >= 2:
                for i in range(len(points) - 1):
                    cv2.line(result, points[i], points[i+1], (0, 255, 0), 1)
        
        # Draw lateral grid lines (perpendicular to camera's forward direction)
        for z in z_range:
            points = []
            for x in x_range:
                point_cam = np.array([x, -self.camera_height, z])
                rotation_matrix = np.array([
                    [1, 0, 0],
                    [0, np.cos(-self.camera_tilt), -np.sin(-self.camera_tilt)],
                    [0, np.sin(-self.camera_tilt),  np.cos(-self.camera_tilt)]
                ])
                point_cam = rotation_matrix @ point_cam
                
                if point_cam[2] > 0:
                    pixel_x = int((point_cam[0] / point_cam[2]) * self.focal_length_x + self.cx)
                    pixel_y = int(self.cy - (point_cam[1] / point_cam[2]) * self.focal_length_y)
                    if 0 <= pixel_x < self.image_width and 0 <= pixel_y < self.image_height:
                        points.append((pixel_x, pixel_y))
            if len(points) >= 2:
                for i in range(len(points) - 1):
                    cv2.line(result, points[i], points[i+1], (0, 255, 0), 1)
            # Add distance labels
            if len(points) > 0:
                mid_point = points[len(points) // 2]
                cv2.putText(result, f"{z:.1f}m", 
                            (mid_point[0], mid_point[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        return result

def live_grid_feed(width=1280, height=960, camera_index=4, scale=1.0):
    # Open the camera using V4L2 (Linux) - adjust if necessary
    cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, 60)  # Request 60 FPS
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimal buffering

    # Optional camera settings (adjust as needed)
    cap.set(cv2.CAP_PROP_BRIGHTNESS, 20)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # Disable auto exposure (if applicable)
    cap.set(cv2.CAP_PROP_EXPOSURE, 700)
    
    print("Camera reported FPS:", cap.get(cv2.CAP_PROP_FPS))
    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    # Read one frame to determine dimensions
    ret, frame = cap.read()
    if not ret:
        print("Error: Couldn't read frame from camera")
        cap.release()
        return

    h, w = frame.shape[:2]

    # Set up fisheye undistortion parameters.
    # Define the approximate camera matrix assuming the optical center is at the image center.
    K = np.array([[w / 2, 0, w / 2],
                  [0, w / 2, h / 2],
                  [0, 0, 1]], dtype=np.float32)

    # Fixed distortion coefficients:
    k1 = (105 - 100) / 100.0  # 0.05
    k2 = (100 - 100) / 100.0  # 0.0
    k3 = (100 - 100) / 100.0  # 0.0
    k4 = (100 - 100) / 100.0  # 0.0
    D = np.array([[k1], [k2], [k3], [k4]], dtype=np.float32)

    # Compute new dimensions based on scale factor.
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                K, D, (w, h), np.eye(3), balance=1, new_size=(new_w, new_h))
    
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
                    K, D, np.eye(3), new_K, (new_w, new_h), cv2.CV_16SC2)

    # Initialize CoordinateEstimator with desired parameters.
    # Adjust these parameters based on your camera's configuration.
    estimator = CoordinateEstimator(
        image_width=new_w,
        image_height=new_h,
        fov_horizontal=95,  # example value, in degrees
        fov_vertical=78,    # example value, in degrees
        camera_height=0.75,  # meters
        camera_tilt=30      # degrees down from horizontal
    )
    
    # Variables for FPS calculation
    frame_count = 0
    start_time = time.time()
    fps = 0

    print("Press 'q' to exit or 's' to save the current frame.")
    latest_frame = None
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Couldn't retrieve frame")
                break

            # Undistort the frame
            undistorted = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            latest_frame = undistorted.copy()

            # Draw the grid overlay on the undistorted frame
            grid_frame = estimator.draw_world_grid(undistorted, grid_size=0.5, max_distance=20)

            # FPS calculation
            frame_count += 1
            elapsed_time = time.time() - start_time
            if elapsed_time > 1:
                fps = frame_count / elapsed_time
                frame_count = 0
                start_time = time.time()

            # Overlay the FPS text
            cv2.putText(grid_frame, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow('Live Grid Feed', grid_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                save_path = 'grid_frame_saved.jpg'
                cv2.imwrite(save_path, grid_frame)
                print(f"Image saved to {save_path}")
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    live_grid_feed()