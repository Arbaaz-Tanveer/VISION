import collections
import math
import logging
from typing import Deque, NamedTuple, Tuple, List
import pyudev
import os

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
            logging.warning("No odometry records available. Returning the initial pose.")
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
            "front": "pci-0000:05:00.3-usb-0:3:1.0",
            "back": "pci-0000:05:00.3-usb-0:2:2.0",
            "right": "pci-0000:05:00.3-usb-0:2:3.0",
            "left": "pci-0000:05:00.3-usb-0:2:4.0"
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
    if camera == 'front':
        # No rotation.
        robot_offset_x = z_cam    
        robot_offset_y = -x_cam    
    elif camera == 'left':
        robot_offset_x = x_cam         
        robot_offset_y = z_cam        
    elif camera == 'back':
        robot_offset_x = -z_cam   
        robot_offset_y = x_cam   
    elif camera == 'right':
        robot_offset_x = -x_cam  
        robot_offset_y = -z_cam    
    else:
        raise ValueError("Camera must be one of 'front', 'right', 'back', or 'left'.")

    obs_x, obs_y, theta = observer_pose

    # Transform the robot offset (in robot frame) to the global field coordinates.
    global_x = obs_x + robot_offset_x * math.cos(theta) - robot_offset_y * math.sin(theta)
    global_y = obs_y + robot_offset_x * math.sin(theta) + robot_offset_y * math.cos(theta)

    return (global_x, global_y)

# Example usage (for testing purposes)
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
