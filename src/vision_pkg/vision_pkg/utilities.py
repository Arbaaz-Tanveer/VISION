import collections
import math
import logging
from typing import Deque, NamedTuple, Tuple, List
import pyudev
import os
import cv2
import numpy as np
import matplotlib as plt
from skimage.morphology import skeletonize
import time

# Configure logging (if not already configured in the main file)
logging.basicConfig(level=logging.INFO)

class OdometryRecord(NamedTuple):
    timestamp: int  # Timestamp in milliseconds
    dx: float       # Incremental x displacement in robot frame (forward)
    dy: float       # Incremental y displacement in robot frame (right)
    dtheta: float   # Incremental rotation (radians)

class OdometryBuffer:
    def __init__(self, capacity: int = 1000):
        self.capacity = capacity
        self.buffer: Deque[OdometryRecord] = collections.deque(maxlen=capacity)
        self.last_timestamp: int = None

    def add_record(self, timestamp: int, dx: float, dy: float, dtheta: float):
        if self.last_timestamp is not None:
            if timestamp < self.last_timestamp:
                logging.warning("Detected timestamp decrease (possible microcontroller restart). "
                                "Clearing buffer to avoid invalid integration.")
                self.buffer.clear()
        self.last_timestamp = timestamp

        record = OdometryRecord(timestamp, dx, dy, dtheta)
        self.buffer.append(record)

    def integrate_with_initial(self, initial_pose: Tuple[float, float, float],
                               time_window_ms: int) -> Tuple[float, float, float]:
        """
        Integrate forward in time from an initial pose over the past time_window_ms.
        """
        if not self.buffer:
            # logging.warning("No odometry records available. Returning the initial pose.")
            return initial_pose

        latest_timestamp = self.buffer[-1].timestamp
        earliest_timestamp = self.buffer[0].timestamp
        available_window = latest_timestamp - earliest_timestamp

        if available_window < time_window_ms:
            logging.warning("Requested integration window of %d ms exceeds available data (%d ms). "
                            "Integrating over the maximum available window.", time_window_ms, available_window)
            records = list(self.buffer)
        else:
            start_time = latest_timestamp - time_window_ms
            records = [record for record in self.buffer if record.timestamp >= start_time]

        # Integration is performed in chronological order.
        x, y, theta = initial_pose

        for record in records:
            # Rotate the incremental displacement from the robot's frame to the global frame.
            cos_angle = math.cos(theta)
            sin_angle = math.sin(theta)
            global_dx = record.dx * cos_angle - record.dy * sin_angle
            global_dy = record.dx * sin_angle + record.dy * cos_angle

            # Update the global pose.
            x += global_dx
            y += global_dy
            theta += record.dtheta

        return (x, y, theta)

    def integrate_backward(self, final_pose: Tuple[float, float, float],
                             time_window_ms: int) -> Tuple[float, float, float]:
        """
        Integrate odometry records backward in time over the specified time window,
        starting from the final_pose to estimate the initial pose.
        
        The reversal is achieved by processing the records in reverse chronological order
        and inverting the incremental transformation for each record.
        """
        if not self.buffer:
            logging.warning("No odometry records available. Returning the final pose as the initial pose.")
            return final_pose

        latest_timestamp = self.buffer[-1].timestamp
        # Use records over the last time_window_ms from the final timestamp.
        start_time = latest_timestamp - time_window_ms
        # Filter records within the window.
        records = [record for record in self.buffer if record.timestamp >= start_time]

        if not records:
            logging.warning("No odometry records in the specified time window. Returning the final pose.")
            return final_pose

        # Process records in reverse chronological order.
        x, y, theta = final_pose
        # Reverse the order since we are "undoing" the motion.
        for record in reversed(records):
            # Undo the rotation: theta_prev = theta_next - dtheta.
            theta_prev = theta - record.dtheta

            # To undo the translation, note that in forward integration:
            #   x_next = x_prev + (dx * cos(theta_prev) - dy * sin(theta_prev))
            # Therefore, to retrieve x_prev:
            x_prev = x - (record.dx * math.cos(theta - record.dtheta) - record.dy * math.sin(theta - record.dtheta))
            y_prev = y - (record.dx * math.sin(theta - record.dtheta) + record.dy * math.cos(theta - record.dtheta))

            # Update the pose.
            x, y, theta = x_prev, y_prev, theta_prev

        return (x, y, theta)

class CameraManager:
    def __init__(self):
        # Define your mapping from camera name to target ID path.
        #to get these paths use in termianl "ls -l /dev/v4l/by-path"
        #or "udevadm info --query=property --name=/dev/video0"  for specific camera index
        self.latency_ms = 150  
        self.camera_mapping = {
            "front": "platform-3610000.usb-usb-0:2.3.2:1.0",
            "right": "platform-3610000.usb-usb-0:2.3.1:1.0",
            "back": "platform-3610000.usb-usb-0:2.3.4:1.0",
            "left": "platform-3610000.usb-usb-0:2.3.3:1.0"
        }

    def list_video_devices(self):
        context = pyudev.Context()
        devices = []
        for device in context.list_devices(subsystem='video4linux'):
            devnode = device.device_node
            id_path = device.get('ID_PATH') or "N/A"
            
            # Find by-path symlinks corresponding to this device node.
            by_path_dir = "/dev/v4l/by-path"
            by_path_links = []
            if os.path.exists(by_path_dir):
                for entry in os.listdir(by_path_dir):
                    full_path = os.path.join(by_path_dir, entry)
                    if os.path.realpath(full_path) == devnode:
                        by_path_links.append(full_path)
            
            devices.append({
                'devnode': devnode,
                'id_path': id_path,
                'by_path_links': by_path_links,
            })
        return devices

    def get_device_by_id_path(self, target_id_path):
        devices = self.list_video_devices()
        for dev in devices:
            if target_id_path in dev['id_path']:
                return dev
        return None

    def get_camera_index(self, camera_name):
        """
        Returns the camera index (integer) for the given camera name.
        The index is derived from the device node (e.g. '/dev/video0' -> 0).
        """
        target_id_path = self.camera_mapping.get(camera_name.lower())
        if not target_id_path:
            raise ValueError(f"Camera name '{camera_name}' is not defined.")
        
        device = self.get_device_by_id_path(target_id_path)
        if not device:
            raise RuntimeError(f"No device found with ID_PATH matching '{target_id_path}'.")

        # Option 1: If by-path links are available, use the first one.
        # Option 2: Otherwise, use the devnode.
        if device['by_path_links']:
            # Resolve the by-path symlink to the real device.
            devnode = os.path.realpath(device['by_path_links'][0])
        else:
            devnode = device['devnode']

        # Now, assuming devnode is something like "/dev/video0",
        # extract the numeric index by splitting on "video".
        basename = os.path.basename(devnode)  # e.g., "video0"
        if "video" in basename:
            try:
                index = int(basename.replace("video", ""))
                return index
            except ValueError:
                raise RuntimeError(f"Could not extract camera index from device node: {devnode}")
        else:
            # Fallback: Filter digits (may not be accurate)
            try:
                index = int(''.join(filter(str.isdigit, devnode)))
                return index
            except ValueError:
                raise RuntimeError(f"Could not extract camera index from device node: {devnode}")
        
def compute_observed_bot_position(camera: str,
                                  rel_measurement: Tuple[float, float],
                                  observer_pose: Tuple[float, float, float]) -> Tuple[float, float]:
    """
    Given a camera name, a relative measurement (x, z) from the camera,
    and an observer's global pose (x, y, theta), compute the observed bot's
    global position.
    """
    x_cam, z_cam = rel_measurement
    # Convert camera measurement to robot frame based on the camera orientation.
    if camera == 'right':
        # No rotation.
        robot_offset_x = z_cam    
        robot_offset_y = -x_cam    
    elif camera == 'front':
        robot_offset_x = x_cam         
        robot_offset_y = z_cam        
    elif camera == 'left':
        robot_offset_x = -z_cam   
        robot_offset_y = x_cam   
    elif camera == 'back':
        robot_offset_x = -x_cam  
        robot_offset_y = -z_cam    
    else:
        raise ValueError("Camera must be one of 'front', 'right', 'back', or 'left'.")

    obs_x, obs_y, theta = observer_pose

    # Transform the robot offset (in robot frame) to the global field coordinates.
    global_x = obs_x + robot_offset_x * math.cos(theta) - robot_offset_y * math.sin(theta)
    global_y = obs_y + robot_offset_x * math.sin(theta) + robot_offset_y * math.cos(theta)

    return (global_x, global_y)

def skeletonizer(img, grid_spacing = 3, make_dotted_img = False):

    thinned_img = (skeletonize(img) * 255).astype(np.uint8)
    # Create mask for thinned image
    y, x = np.where(thinned_img == 255)

    mask = (y % grid_spacing == 0) & (x % grid_spacing == 0)
    selected_y = y[mask]
    selected_x = x[mask]

    if make_dotted_img:
        dotted_img = np.zeros_like(thinned_img)
        dotted_img[selected_y, selected_x] = 255
    else:
        dotted_img = None

    # Center the coordinates
    x_centered = selected_x - img.shape[1] / 2.0
    y_centered = selected_y - img.shape[0] / 2.0

    flattened_arr = np.stack((x_centered, y_centered), axis=1).flatten().astype(np.int16).tolist()

    return dotted_img, flattened_arr

def visualise_localisation_result(cam, angle, dx, dy):
    # Return None early if no activation map
    if cam is None:
        return None

    # Load the field map
    field_img = cv2.imread(
        '/home/orin/vision_github/VISION/src/vision_pkg/vision_pkg/maps/test_field.png'
    )
    if field_img is None:
        raise FileNotFoundError("Field map image not found")

    # Normalize activation map to uint8
    if cam.dtype != np.uint8:
        cam_norm = cv2.normalize(cam, None, 0, 255, cv2.NORM_MINMAX)
        cam_uint8 = cam_norm.astype(np.uint8)
    else:
        cam_uint8 = cam

    # Compute rotation in radians and its sin/cos
    theta_rad = angle # adjust zero-reference if needed
    cos_t = np.cos(theta_rad)
    sin_t = np.sin(theta_rad)

    # Field center
    h_map, w_map = field_img.shape[:2]

    h_img, w_img = cam.shape[:2]
    ix, iy = w_img / 2.0, h_img / 2.0

    # Combined affine: rotate around center then translate
    # Compute translation terms so rotation is about (cx, cy)
    tx = dx - (cos_t * ix - sin_t * iy)
    ty = dy - (sin_t * ix + cos_t * iy)
    M = np.array([[cos_t, -sin_t, tx],
                  [sin_t,  cos_t, ty]], dtype=np.float32)

    # Warp activation map into field space
    overlay_map = cv2.warpAffine(cam_uint8, M, (w_map, h_map), flags=cv2.INTER_LINEAR)

    # Apply colormap
    cam_color = cv2.applyColorMap(overlay_map, cv2.COLORMAP_JET)

    # Blend with field image
    alpha = 0.5
    output = cv2.addWeighted(field_img, 1 - alpha, cam_color, alpha, 0)

    # Annotate
    rot_deg = (np.rad2deg(theta_rad) % 360) - 180
    text = f"dx: {dx:.2f}, dy: {dy:.2f}, theta: {rot_deg:.2f}"
    cv2.putText(
        output, text, (field_img.shape[0]//100, field_img.shape[1]//30), cv2.FONT_HERSHEY_SIMPLEX,
        field_img.shape[1]/1000, (0, 255, 0), 1, cv2.LINE_AA
    )

    return output

if __name__ == "__main__":
    # Create an odometry buffer with capacity to store 2000 records.
    odo_buffer = OdometryBuffer(capacity=2000)
    cam_manager = CameraManager()
    try:
        front_index = cam_manager.get_camera_index("back")
        print("Front camera index:", front_index)
    except Exception as e:
        print(e)
    
    # Simulate adding odometry data.
    import random
    current_time = 0  # starting timestamp in ms

    # Add records for 5 seconds (every 50ms)
    for _ in range(100):
        current_time += 50
        dx = random.uniform(0.0, 0.05)
        dy = random.uniform(-0.02, 0.02)
        dtheta = random.uniform(-0.01, 0.01)
        odo_buffer.add_record(current_time, dx, dy, dtheta)
    
    # Assume the net pose 2 seconds ago was known:
    initial_pose = (1.0, 1.0, 0.5)  # (x, y, theta)
    
    # Integrate over the past 2000 ms (2 seconds) to get the current pose.
    new_pose = odo_buffer.integrate_with_initial(initial_pose, time_window_ms=2000)
    print(f"Updated pose over the last 2 seconds: x={new_pose[0]:.3f}, y={new_pose[1]:.3f}, theta={new_pose[2]:.3f}")

    # Now, use integrate_backward to recover the initial pose from the final pose.
    recovered_initial_pose = odo_buffer.integrate_backward(new_pose, time_window_ms=2000)
    print(f"Recovered initial pose (backward integration): x={recovered_initial_pose[0]:.3f}, y={recovered_initial_pose[1]:.3f}, theta={recovered_initial_pose[2]:.3f}")

    # Example camera measurement:
    # Suppose the 'back' camera sees an observed bot at (0.2, 1.5) where 0.2 m is to the right
    # and 1.5 m forward in the camera's frame.
    observer_pose = (1, 1, 0)
    camera = 'back'
    rel_measurement = (0.2, 1.5)  # (x, z) in camera coordinates

    observed_bot_global = compute_observed_bot_position(camera, rel_measurement, observer_pose)
    print(f"Observed bot global position (from {camera} camera): x={observed_bot_global[0]:.3f}, y={observed_bot_global[1]:.3f}")


class ImageProcessing:
    def __init__(self):
        pass
    
    def white_threshold(self,image, thresh_val=40, show=False):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, binary = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
        if show:
            cv2.imshow("White Threshold", binary)
            cv2.waitKey(0)
            cv2.destroyWindow("White Threshold")
        return binary
    
    def adaptive_threshold(self, image, block_size=53, C=-10):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # ADAPTIVE_THRESH_MEAN_C computes the mean of the neighborhood area minus C.
        thresholded = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY, block_size, C
        )
        return thresholded

    def process_map(self,binary_image, min_arc_length=50, min_area=200, max_area=15000, min_length=20, max_circularity=0.35,show = False):

        # Apply Gaussian blur to the binary image to reduce noise
        blurred = cv2.GaussianBlur(binary_image, (3, 3), 0)
        
        # # Ensure image is binary by thresholding (in case the blur made values non-binary)
        # _, thresh = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY)
        
        # Find contours in the binary mask
        start = time.time()
        contours, _ = cv2.findContours(blurred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        time_taken = time.time()-start
        print(f"Time for contour detection: {time_taken:.6f}s")
        
        # Create a blank mask to draw the filtered contours
        mask_clean = np.zeros_like(blurred)
        
        for cnt in contours:
            arc_len = cv2.arcLength(cnt, closed=True)
            area = cv2.contourArea(cnt)
            x, y, w, h = cv2.boundingRect(cnt)

            if arc_len == 0:
                continue
            circularity = 4 * np.pi * (area / (arc_len * arc_len))

            # Keep contours that meet the criteria
            if (arc_len >= min_arc_length and area >= min_area and area <= max_area and 
                max(w, h) >= min_length and circularity < max_circularity):
                cv2.drawContours(mask_clean, [cnt], -1, 255, thickness=cv2.FILLED)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
        final_result = cv2.dilate(mask_clean, kernel, iterations=1)
        if(show):
            self.show_intermediate_results(blurred, mask_clean, final_result)
        return mask_clean
    
    def show_intermediate_results(self,blurred, edges, dilated_edges):
        """
        Display intermediate processing results using matplotlib.
        """
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        axs[0].imshow(blurred, cmap='gray')
        axs[0].set_title("Blurred Ground Map")
        axs[0].axis('off')
        
        axs[1].imshow(edges, cmap='gray')
        axs[1].set_title("Canny Edges")
        axs[1].axis('off')
        
        axs[2].imshow(dilated_edges, cmap='gray')
        axs[2].set_title("Dilated Edges")
        axs[2].axis('off')
        
        plt.tight_layout()
        plt.show()