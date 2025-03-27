import os
import cv2
import pyudev
import time

class CameraManager:
    def __init__(self):
        # Define your mapping from camera name to target ID path
        self.camera_mapping = {
            "front": "pci-0000:05:00.3-usb-0:2:1.0",
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
            
            # Find by-path symlinks corresponding to this device node
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
        devnode = device['by_path_links'][0] if device['by_path_links'] else device['devnode']

        # Extract the numeric index from a device node string like '/dev/video0'
        try:
            index = int(''.join(filter(str.isdigit, devnode)))
            return index
        except ValueError:
            raise RuntimeError(f"Could not extract camera index from device node: {devnode}")

# Example usage:
if __name__ == '__main__':
    cam_manager = CameraManager()
    try:
        front_index = cam_manager.get_camera_index("front")
        print("Front camera index:", front_index)
        
        # Optionally, you can open the camera with OpenCV:
        # cap = cv2.VideoCapture(front_index, cv2.CAP_V4L2)
        # if cap.isOpened():
        #     print("Successfully opened front camera")
        #     cap.release()
        # else:
        #     print("Failed to open front camera")
    except Exception as e:
        print(e)
