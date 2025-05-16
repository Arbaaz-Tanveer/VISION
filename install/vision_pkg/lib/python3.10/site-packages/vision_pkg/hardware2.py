import math
import logging
import time
import serial

# Import your existing odometry classes and functions from the utilities module.
from utilities import OdometryBuffer

# Configure logging (if not already configured in your utilities module)
logging.basicConfig(level=logging.INFO)

# ---------------------------
# New Kinematics and Serial Communication Classes
# ---------------------------
class OmniKinematics:
    def __init__(self, wheel_radius: float, bot_radius: float):
        self.wheel_radius = wheel_radius
        self.bot_radius = bot_radius

    def compute_motion(self, d_enc1: int, d_enc2: int, d_enc3: int, d_enc4: int, dt: float) -> tuple:
        """
        Given the differences in encoder ticks (d_encX) and the time interval (dt),
        compute the incremental displacement (dx, dy) and rotation (dtheta).
        
        The encoder differences are converted to wheel displacements by multiplying
        with the wheel radius. The omni-drive kinematics model is used:
        
            dx = (w1 + w2 + w3 + w4) / 4
            dy = (-w1 + w2 + w3 - w4) / 4
            dtheta = (-w1 + w2 - w3 + w4) / (4 * bot_radius)
            
        where w1...w4 are the wheel displacements.
        """
        # Convert encoder differences to wheel displacements in meters.
        w1 = d_enc1 * self.wheel_radius
        w2 = d_enc2 * self.wheel_radius
        w3 = d_enc3 * self.wheel_radius
        w4 = d_enc4 * self.wheel_radius

        dx = (w1 + w2 + w3 + w4)*math.sqrt(2) / (4.0)
        dy = (-w1 + w2 + w3 - w4)*math.sqrt(2) / (4.0)
        dtheta = (-w1 - w2 + w3 + w4) / (4.0 * self.bot_radius)
        return dx, dy, dtheta

    def compute_wheel_velocities(self, vx, vy, omega):
        scale = 1/math.sqrt(2)

        v1 = (scale*((vx - vy) - omega * self.bot_radius)) / self.wheel_radius
        v2 = (scale*((vx + vy) - omega * self.bot_radius)) / self.wheel_radius
        v3 = (scale*((vx + vy) + omega * self.bot_radius)) / self.wheel_radius
        v4 = (scale*((vx - vy) + omega * self.bot_radius)) / self.wheel_radius
        
        # Scale the values
        return [v1, v2, v3, v4]
    
class STM32Serial:
    def __init__(self, port: str, baudrate: int = 9600, timeout: float = 0.1):
        self.ser = serial.Serial(port, baudrate, timeout=timeout)
        # Allow some time for the STM32 to reset and establish the connection.
        time.sleep(2)

    def read_data(self) -> str:
        """Read and return a line of data from the STM32 if available."""
        if self.ser.in_waiting > 0:
            line = self.ser.readline().decode('utf-8').strip()
            return line
        return None

    def send_velocity(self, wheel_velocities):
        """
        Send wheel velocities to the STM32 microcontroller.
        
        Args:
            wheel_velocities: List of wheel velocities [v1, v2, v3, v4]
        """
        # Convert velocities to non-negative integers and create binary string for signs
        signs = ['0' if v >= 0 else '1' for v in wheel_velocities]
        wheel_velocities = [abs(round(v)) for v in wheel_velocities]
        binary_string = ''.join(signs)
        
        # Construct message
        message = f"{wheel_velocities[0]},{wheel_velocities[1]},{wheel_velocities[2]},{wheel_velocities[3]},{binary_string}\n"
        
        # Send over serial
        self.ser.write(message.encode('utf-8'))
        logging.info(f"Sent velocity command: {message}")

    def close(self):
        self.ser.close()

# ---------------------------
# Main Loop: STM32 Communication and Odometry Integration
# ---------------------------
def main():
    # Parameters for kinematics and odometry integration.
    wheel_radius = 0.05*6.28/1800   # Example conversion factor: meters per encoder tick
    print(wheel_radius)
    bot_radius = 0.25     # Distance from robot center to wheels in meters
    const_vx = 0.0       # m/s, forward
    const_vy = 0.0       # m/s, no lateral movement
    const_omega = 0.0    # rad/s, no rotation
    # Instantiate the odometry buffer (with a large enough capacity).
    odo_buffer = OdometryBuffer(capacity=5000)
    # Instantiate the kinematics class.
    kinematics = OmniKinematics(wheel_radius, bot_radius)
    
    # Initialize the serial connection (update port as needed, e.g., '/dev/ttyUSB0')
    stm32_serial = STM32Serial(port='/dev/ttyACM0', baudrate=115200)
    logging.info("Connected to STM32 on /dev/ttyUSB0")
    
    # Variables to store previous encoder values and timestamp.
    previous_encoders = None
    previous_timestamp = None

    # Global pose (x, y, theta). You can change the initial pose as needed.
    global_pose = (0.0, 0.0, 0.0)
    
    try:
        while True:
            data = stm32_serial.read_data()
            if data:
                try: 
                    # Expected data format:
                    # encoder1:encoder2:encoder3:encoder4:timestamp:...
                    parts = data.split(':')
                    if len(parts) >= 5:

                        # Parse the absolute encoder positions and the timestamp.
                        enc1, enc2, enc3, enc4 = map(int, parts[:4])
                        timestamp = int(parts[4])
                        logging.info(f"Received encoders: {enc1}, {enc2}, {enc3}, {enc4} at timestamp: {timestamp}")
                        
                        # Initialize previous encoder values if this is the first reading.
                        if previous_encoders is None:
                            previous_encoders = (enc1, enc2, enc3, enc4)
                            previous_timestamp = timestamp
                            continue

                        # Calculate differences in encoder ticks.
                        d_enc1 = enc1 - previous_encoders[0]
                        d_enc2 = enc2 - previous_encoders[1]
                        d_enc3 = enc3 - previous_encoders[2]
                        d_enc4 = enc4 - previous_encoders[3]
                        dt = (timestamp - previous_timestamp) / 1000.0  # Convert milliseconds to seconds

                        # Ensure dt is positive.
                        if dt <= 0:
                            logging.warning("Non-positive dt encountered; skipping update.")
                            continue

                        # Compute incremental motion.
                        dx, dy, dtheta = kinematics.compute_motion(d_enc1, d_enc2, d_enc3, d_enc4, dt)
                        logging.info(f"Delta Motion: dx={dx:.4f}, dy={dy:.4f}, dtheta={dtheta:.4f}")
                        
                        # Add the new odometry record to the buffer.
                        odo_buffer.add_record(timestamp, dx, dy, dtheta)
                        
                        # Update previous encoder values for the next iteration.
                        previous_encoders = (enc1, enc2, enc3, enc4)
                        previous_timestamp = timestamp

                        # Integrate the odometry over the past 50 seconds (50000 milliseconds)
                        integrated_pose = odo_buffer.integrate_with_initial(global_pose, time_window_ms=50000)
                        logging.info(f"Integrated Pose (last 50 sec): x={integrated_pose[0]:.3f}, y={integrated_pose[1]:.3f}, theta={integrated_pose[2]:.3f}")
                        
                        # Optionally update the global_pose to the integrated result.
                        # global_pose = integrated_pose
                        wheel_velocities = kinematics.compute_wheel_velocities(
                                const_vx, const_vy, const_omega
                            )
                            # Send to STM32
                        stm32_serial.send_velocity(wheel_velocities)
                        print(f"sent wheel velocities = {wheel_velocities}")
                        
                except ValueError as e:
                    logging.warning(f"Error parsing data '{data}': {e}")
            # Sleep briefly to avoid high CPU usage.
            # time.sleep(0.00)
                    
    except KeyboardInterrupt:
        logging.info("Terminating STM32 communication loop.")
        stm32_serial.close()

if __name__ == "__main__":
    main()