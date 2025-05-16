import socket
import json
import threading
import time
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from std_msgs.msg import Float32MultiArray


class ActualRobot(Node):
    def __init__(self, robot_ip, robot_port, controller_ip, controller_port):
        # Initialize ROS2 node
        super().__init__('robot_controller')
        
        # Store IP and port details
        self.robot_ip = robot_ip
        self.robot_port = robot_port
        self.controller_addr = (controller_ip, controller_port)
        
        # Create and bind UDP socket
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.bind((self.robot_ip, self.robot_port))
        self.get_logger().info(f"Robot listening on {self.robot_ip}:{self.robot_port}")
        
        # Initialize robot state
        self.position = (0, 0)  # (x, y) in meters
        self.orientation = 0    # in degrees, 0 = east
        self.ball_position = None  # (x, y) if detected
        self.obstacles = []     # List of (x, y) positions
        
        # Set up ROS2 subscribers
        self.robot_pos_sub = self.create_subscription(
            Float32MultiArray,
            'robot_position',
            self.robot_pos_callback,
            10)
            
        self.ball_pos_sub = self.create_subscription(
            Float32MultiArray,
            'ball_position',
            self.ball_pos_callback,
            10)
            
        self.obstacles_sub = self.create_subscription(
            Float32MultiArray,
            'obstacles_position',
            self.obstacles_callback,
            10)
        
        # Start status sending thread
        self.send_status_thread = threading.Thread(target=self.send_status_periodically)
        self.send_status_thread.daemon = True
        self.send_status_thread.start()
    
    def robot_pos_callback(self, msg):
        """Callback for robot position updates."""
        pos = msg.data
        self.position = (pos[0], pos[1])
        self.orientation = pos[2]
        self.get_logger().debug(f"Updated robot position: ({pos[0]}, {pos[1]}), theta: {pos[2]}")
    
    def ball_pos_callback(self, msg):
        """Callback for ball position updates."""
        pos = msg.data
        self.ball_position = (pos[0], pos[1])
        self.get_logger().debug(f"Updated ball position: ({pos[0]}, {pos[1]})")
    
    def obstacles_callback(self, msg):
        """Callback for obstacles updates.
        Format: [num_obstacles, x1, y1, x2, y2, ...]
        """
        data = msg.data
        if len(data) > 0:
            num_obstacles = int(data[0])
            self.obstacles = []
            
            # Process obstacles data if format is valid
            if len(data) >= num_obstacles * 2 + 1:
                for i in range(num_obstacles):
                    x = data[1 + i*2]
                    y = data[2 + i*2]
                    self.obstacles.append((x, y))
                
                self.get_logger().debug(f"Updated obstacles: {self.obstacles}")
            else:
                self.get_logger().warning("Invalid obstacles data format")
    
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
        command_thread = threading.Thread(target=self.process_commands)
        command_thread.daemon = True
        command_thread.start()
        
        # Keep the ROS2 node spinning
        rclpy.spin(self)
    
    def process_commands(self):
        """Process commands received from the controller."""
        while True:
            data, addr = self.socket.recvfrom(1024)
            command = data.decode()
            self.get_logger().info(f"Received command: {command} from {addr}")
            
            # Process movement command: "move x y"
            if command.startswith("move"):
                parts = command.split()
                if len(parts) == 3:
                    try:
                        x = float(parts[1])
                        y = float(parts[2])
                        # Note: In a real implementation, you'd publish this to a movement topic
                        # Here we just update the local state for the status updates
                        self.get_logger().info(f"Move command to {x}, {y}")
                    except ValueError:
                        self.get_logger().error("Invalid move command")
                    
            # Process turn command: "turn angle"
            elif command.startswith("turn"):
                parts = command.split()
                if len(parts) == 2:
                    try:
                        angle = float(parts[1])
                        # Note: In a real implementation, you'd publish this to a movement topic
                        self.get_logger().info(f"Turn command to {angle} degrees")
                    except ValueError:
                        self.get_logger().error("Invalid turn command")
            else:
                self.get_logger().warning("Unknown command")


def main(args=None):
    # Initialize ROS2
    rclpy.init(args=args)
    
    # Create the robot instance
    # Example usage: robot listens on 172.24.203.243:5000, sends to controller at 172.24.203.214:6000
    robot = ActualRobot("172.24.203.243", 5000, "172.24.201.10", 6000)
    
    try:
        robot.run()
    except KeyboardInterrupt:
        pass
    finally:
        # Clean up
        robot.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()