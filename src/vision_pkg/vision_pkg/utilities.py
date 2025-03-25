import collections
import math
import logging
from typing import Deque, NamedTuple, Tuple, List

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
            # Rotate the incremental displacement from the robot's frame to the global frame
            cos_angle = math.cos(theta)
            sin_angle = math.sin(theta)
            global_dx = record.dx * cos_angle - record.dy * sin_angle
            global_dy = record.dx * sin_angle + record.dy * cos_angle

            # Update the global pose
            x += global_dx
            y += global_dy
            theta += record.dtheta

        return (x, y, theta)

def compute_observed_bot_position(camera: str,
                                  rel_measurement: Tuple[float, float],
                                  observer_pose: Tuple[float, float, float]) -> Tuple[float, float]:
    
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

    # Example camera measurement:
    # Suppose the 'front' camera sees an observed bot at (0.2, 1.5) where 0.2 m is to the right
    # and 1.5 m forward in the camera's frame.
    observer_pose = (1,1,0)
    camera = 'back'
    rel_measurement = (0.2, 1.5)  # (x, z) in camera coordinates
    # observer_pose = new_pose    # using the integrated pose as observer's current pose

    observed_bot_global = compute_observed_bot_position(camera, rel_measurement, observer_pose)
    print(f"Observed bot global position (from {camera} camera): x={observed_bot_global[0]:.3f}, y={observed_bot_global[1]:.3f}")
