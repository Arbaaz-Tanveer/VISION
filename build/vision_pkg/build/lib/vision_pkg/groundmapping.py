import cv2
import numpy as np

# --------------------------
# Calibration Utilities
# --------------------------
def compute_calibration_params(h,w, balance=1, distortion_param=0.05, show=False):
    # h, w = image.shape[:2]
    K = np.array([[w/2, 0, w/2],
                  [0, w/2, h/2],
                  [0, 0, 1]], dtype=np.float32)
    # Using a fixed distortion parameter for all coefficients.
    D = np.array([[distortion_param], [0.0], [0.0], [0.0]], dtype=np.float32)
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, (w, h), np.eye(3), balance=balance)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), new_K, (w, h), cv2.CV_16SC2)
    # if show:
    #     undistorted_preview = cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    #     cv2.imshow("Undistorted Preview", undistorted_preview)
    #     cv2.waitKey(0)
    #     cv2.destroyWindow("Undistorted Preview")
    return map1, map2, K, D, new_K

# --------------------------
# Coordinate Estimator Class
# --------------------------
class CoordinateEstimator:
    def __init__(self, image_width, image_height, fov_horizontal, fov_vertical, camera_height, camera_tilt=0):
        self.image_width = image_width
        self.image_height = image_height
        self.fov_h = np.radians(fov_horizontal)
        self.fov_v = np.radians(fov_vertical)
        self.camera_height = camera_height
        self.camera_tilt = np.radians(camera_tilt)
        
        # Compute focal lengths in pixels.
        self.focal_length_x = (image_width / 2) / np.tan(self.fov_h / 2)
        self.focal_length_y = (image_height / 2) / np.tan(self.fov_v / 2)
        
        # Image center.
        self.cx = image_width / 2
        self.cy = image_height / 2
        
        # Precompute the lookup table mapping each pixel in the undistorted image to its world coordinate.
        self.world_coords = self.precompute_world_coords()

    def precompute_world_coords(self):
        """
        Precompute intersections of rays with the ground plane (y=0) for each pixel.
        Returns:
            world_coords (np.ndarray): Array of shape (H, W, 2) with (world_x, world_z) coordinates.
        """
        xs, ys = np.meshgrid(np.arange(self.image_width), np.arange(self.image_height))
        x_norm = (xs - self.cx) / self.focal_length_x
        y_norm = (ys - self.cy) / self.focal_length_y
        rays = np.stack((x_norm, y_norm, np.ones_like(x_norm)), axis=2)
        norms = np.linalg.norm(rays, axis=2, keepdims=True)
        rays_normalized = rays / norms

        # Rotation matrix for camera tilt.
        ct = np.cos(self.camera_tilt)
        st = np.sin(self.camera_tilt)
        R = np.array([
            [1, 0, 0],
            [0, ct, st],
            [0, -st, ct]
        ])
        
        rays_flat = rays_normalized.reshape(-1, 3).T
        rays_world_flat = R @ rays_flat
        rays_world = rays_world_flat.T.reshape(self.image_height, self.image_width, 3)
        
        # Compute scaling factor t so that y=0 (ground plane) is intersected.
        with np.errstate(divide='ignore', invalid='ignore'):
            t = -self.camera_height / rays_world[..., 1]
            t = np.where(np.abs(rays_world[..., 1]) < 1e-6, np.nan, t)
            t = np.where(t >= 0, np.nan, t)
            
        world_x = t * rays_world[..., 0]
        world_z = t * rays_world[..., 2]
        world_coords = np.stack((world_x, world_z), axis=2)
        return world_coords

# --------------------------
# Image Processing Functions
# --------------------------
def undistort_image(image, map1, map2, show=False):
    """
    Undistort the input image using precomputed undistortion maps.
    Args:
        image (np.ndarray): Input distorted image.
        map1, map2: Precomputed undistortion maps.
        show (bool): If True, display the undistorted image.
    Returns:
        undistorted (np.ndarray): The undistorted image.
    """
    undistorted = cv2.remap(image, map1, map2,
                             interpolation=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT)
    if show:
        cv2.imshow("Undistorted Image", undistorted)
        cv2.waitKey(0)
        cv2.destroyWindow("Undistorted Image")
    return undistorted

def white_threshold(image, thresh_val=230, show=False):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
    if show:
        cv2.imshow("White Threshold", binary)
        cv2.waitKey(0)
        cv2.destroyWindow("White Threshold")
    return binary

def update_ground_map(ground_map, binary, estimator, thresh_val=230, scale=15, max_distance=15, camera="front", show=False, shift_z = 0.05, shift_x = 0):
    map_size_px = ground_map.shape[0]
    map_center = (map_size_px // 2, map_size_px // 2)
    
    # Apply white thresholding.
    # binary = white_threshold(image, thresh_val, show=False)
    ys, xs = np.where(binary > 0)
    
    # Retrieve corresponding world coordinates from the precomputed lookup.
    world_points = estimator.world_coords[ys, xs]
    world_x = world_points[:, 0]
    world_z = world_points[:, 1]
    
    # Filter out points beyond max_distance.
    within_range = np.abs(world_z) <= max_distance
    world_x = world_x[within_range]
    world_z = world_z[within_range]
    
    # Transform world coordinates to ground map pixel indices based on the camera view.
    if camera == "front":
        map_x = (map_center[0] - (world_x - shift_x) * scale).astype(np.int32)
        map_y = (map_center[1] + (world_z - shift_z) * scale).astype(np.int32) 
    elif camera == "back":
        map_x = (map_center[0] + (world_x - shift_x) * scale).astype(np.int32)
        map_y = (map_center[1] - (world_z - shift_z) * scale).astype(np.int32)
    elif camera == "right":
        map_x = (map_center[0] - (world_z - shift_z) * scale).astype(np.int32)
        map_y = (map_center[1] - (world_x - shift_x) * scale).astype(np.int32)
    elif camera == "left":
        map_x = (map_center[0] + (world_z - shift_z) * scale).astype(np.int32)
        map_y = (map_center[1] + (world_x - shift_x) * scale).astype(np.int32)
    else:
        print("not valid camera")
    
    valid = (map_x >= 0) & (map_x < map_size_px) & (map_y >= 0) & (map_y < map_size_px)
    map_x = map_x[valid]
    map_y = map_y[valid]
    
    # Update the ground map by marking the projected points.
    ground_map[map_y, map_x] = 255
    if show:
        cv2.imshow("Updated Ground Map", ground_map)
        cv2.waitKey(0)
        cv2.destroyWindow("Updated Ground Map")
    return ground_map

def undistort_pixel(pixel_coords, K, D, new_K, show=False):
   #converts pixel in distorted to pixels in unidstorted
    pts = np.array(pixel_coords, dtype=np.float32).reshape(-1, 1, 2)
    undistorted = cv2.fisheye.undistortPoints(pts, K, D, None, new_K)
    undistorted = undistorted.reshape(-1, 2)
    undistorted_pixels = [tuple(pt) for pt in undistorted]
    if show:
        for orig, und in zip(pixel_coords, undistorted_pixels):
            print(f"Distorted pixel {orig} -> Undistorted pixel {und}")
    return undistorted_pixels

def pixel_to_ground(pixel_coords, estimator, K, D, new_K, show=False):
    #caluculate the posiition of pixel in ground coordinates
    undistorted_pixels = undistort_pixel(pixel_coords, K, D, new_K, show=show)
    ground_coords = []
    for (x, y) in undistorted_pixels:
        ix = int(round(x))
        iy = int(round(y))
        if 0 <= ix < estimator.image_width and 0 <= iy < estimator.image_height:
            world_x, world_z = -estimator.world_coords[iy, ix]
            ground_coords.append((world_x, world_z))
        else:
            ground_coords.append((None, None))
    if show:
        for orig, und, world in zip(pixel_coords, undistorted_pixels, ground_coords):
            print(f"Distorted pixel {orig} -> Undistorted {und} -> Ground Coordinates: {world}")
    return ground_coords
