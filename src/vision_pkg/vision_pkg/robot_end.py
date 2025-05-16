import socket
import json
import threading
import time

class ActualRobot:
    def __init__(self, robot_ip, robot_port, controller_ip, controller_port):
        # Store IP and port details
        self.robot_ip = robot_ip
        self.robot_port = robot_port
        self.controller_addr = (controller_ip, controller_port)
        
        # Create and bind UDP socket
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.bind((self.robot_ip, self.robot_port))
        print(f"Robot listening on {self.robot_ip}:{self.robot_port}")
        
        # Initialize robot state
        self.position = (0, 0)  # (x, y) in meters
        self.orientation = 0    # in degrees, 0 = east
        self.ball_position = None  # (x, y) if detected
        self.obstacles = []     # List of (x, y) positions
        
        # Start sensor simulation thread
        self.sensor_thread = threading.Thread(target=self.update_sensors)
        self.sensor_thread.daemon = True
        self.sensor_thread.start()
        
        # Start status sending thread
        self.send_status_thread = threading.Thread(target=self.send_status_periodically)
        self.send_status_thread.daemon = True
        self.send_status_thread.start()

    def update_sensors(self):
        """Simulate sensor updates every second."""
        while True:
            # Simulate detecting a ball and obstacles
            self.ball_position = (6, 4.5)  # Fixed for simulation
            self.obstacles = [(2, 3), (4, 5)]  # Example obstacles
            time.sleep(1)

    def send_status_periodically(self):
        """Send status updates to controller every 0.1 seconds."""
        while True:
            status = {
                "position": self.position,
                "orientation": self.orientation,
                "ball_position": self.ball_position,
                "obstacles": self.obstacles
            }
            self.socket.sendto(json.dumps(status).encode(), self.controller_addr)
            time.sleep(0.1)

    def run(self):
        """Listen for and process commands from the controller."""
        while True:
            data, addr = self.socket.recvfrom(1024)
            command = data.decode()
            print(f"Received command: {command} from {addr}")
            
            # Process movement command: "move x y"
            if command.startswith("move"):
                parts = command.split()
                if len(parts) == 3:
                    try:
                        x = float(parts[1])
                        y = float(parts[2])
                        self.position = (x, y)
                        print(f"Moved to {self.position}")
                    except ValueError:
                        print("Invalid move command")
            
            # Process turn command: "turn angle"
            elif command.startswith("turn"):
                parts = command.split()
                if len(parts) == 2:
                    try:
                        angle = float(parts[1])
                        self.orientation = angle
                        print(f"Turned to {self.orientation} degrees")
                    except ValueError:
                        print("Invalid turn command")
            else:
                print("Unknown command")

if __name__ == "__main__":
    # Example usage: robot listens on 127.0.0.1:5000, sends to controller at 127.0.0.1:6000
    robot = ActualRobot("172.24.203.243", 5000, "172.24.203.214", 6000)      #first one is our own address second is of the base station
    robot.run()