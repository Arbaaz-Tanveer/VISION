# localization.py
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import time

class Localizer:
    def __init__(self, gt_path, num_levels=6, threshold=127):
        self.gt_path = gt_path
        self.num_levels = num_levels
        self.ground_truth = self._load_ground_truth(gt_path, threshold)
        self.pyr_gt = self._build_pyramid(self.ground_truth, num_levels)
    
    @staticmethod
    def _build_pyramid(image, levels):
        """Build a Gaussian pyramid for the image."""
        pyramid = [image]
        for _ in range(1, levels):
            image = cv2.pyrDown(image)
            pyramid.append(image)
        return pyramid
    
    @staticmethod
    def _load_ground_truth(gt_path, threshold):
        """Load the ground truth map from file and convert to binary."""
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        if gt is None:
            raise ValueError("Error loading ground truth image. Check the file path.")
        _, gt_bin = cv2.threshold(gt, threshold, 255, cv2.THRESH_BINARY)
        return gt_bin
    
    @staticmethod
    def _create_overlay(img1, img2):
        """Create an RGB overlay: img1 in white, img2 in red."""
        composite = np.dstack([img1, img1, img1]).astype(np.uint8)
        mask = img2 > 0
        composite[mask] = [255, 0, 0]
        return composite

    def localize(self, sensor_map, approx_angle, approx_x_cartesian, approx_y_cartesian,
                 angle_range, trans_range, center, num_starts=10,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-4),
                 warp_mode=cv2.MOTION_EUCLIDEAN):
        
        start_time = time.time()
        num_levels = self.num_levels

        # Build pyramid for sensor map.
        pyr_sensor = self._build_pyramid(sensor_map, num_levels)
        
        # Compute approximate translation in sensor coordinate frame.
        approx_angle_rad = np.deg2rad(approx_angle)
        tx_approx = -approx_x_cartesian * math.cos(approx_angle_rad) + approx_y_cartesian * math.sin(approx_angle_rad)
        ty_approx = approx_x_cartesian * math.sin(approx_angle_rad) + approx_y_cartesian * math.cos(approx_angle_rad)
        
        # Adjust ranges at the coarsest level.
        scale_coarse = 1 / (2 ** (num_levels - 1))
        angle_range_coarse = (-approx_angle - angle_range, -approx_angle + angle_range)
        trans_range_coarse_x = ((tx_approx - trans_range) * scale_coarse, (tx_approx + trans_range) * scale_coarse)
        trans_range_coarse_y = ((ty_approx - trans_range) * scale_coarse, (ty_approx + trans_range) * scale_coarse)
        
        coarse_level = num_levels - 1
        template_coarse = self.pyr_gt[coarse_level]
        image_coarse = pyr_sensor[coarse_level]
        
        best_cc = -1e9
        best_warp = None
        
        # Multi-start search at coarsest level.
        for i in range(num_starts):
            angle_init = np.random.uniform(*angle_range_coarse)
            tx_init = np.random.uniform(*trans_range_coarse_x)
            ty_init = np.random.uniform(*trans_range_coarse_y)
            
            angle_rad = np.deg2rad(angle_init)
            cos_val = np.cos(angle_rad)
            sin_val = np.sin(angle_rad)
            cx_coarse = center[0] * scale_coarse
            cy_coarse = center[1] * scale_coarse
            
            # Build initial warp matrix (rotation about center plus translation)
            a = cos_val
            b = -sin_val
            c = -cx_coarse * cos_val + cy_coarse * sin_val + cx_coarse + tx_init
            d = sin_val
            e = cos_val
            f = -cx_coarse * sin_val - cy_coarse * cos_val + cy_coarse + ty_init
            init_warp = np.array([[a, b, c],
                                  [d, e, f]], dtype=np.float32)
            try:
                cc, warp_candidate = cv2.findTransformECC(template_coarse, image_coarse, init_warp,
                                                          warp_mode, criteria)
                if cc > best_cc:
                    best_cc = cc
                    best_warp = warp_candidate.copy()
            except cv2.error as e:
                print(f"Multi-start ECC failed at coarsest level for start {i}: {e}")
        
        if best_warp is None:
            best_warp = np.eye(2, 3, dtype=np.float32)
        
        # Refine warp from coarse to fine scales.
        current_warp = best_warp.copy()
        for level in reversed(range(num_levels - 1)):
            current_warp[0, 2] *= 2.0
            current_warp[1, 2] *= 2.0
            
            template = self.pyr_gt[level]
            image = pyr_sensor[level]
            try:
                cc, current_warp = cv2.findTransformECC(template, image, current_warp,
                                                        warp_mode, criteria)
            except cv2.error as e:
                print(f"ECC failed at pyramid level {level}: {e}")
                break
        warp_matrix = current_warp
        print(f"Time taken: {time.time() - start_time:.2f} seconds")
        
        # Extract rotation angle from the warp matrix.
        a = warp_matrix[0, 0]  # cos(θ)
        d = warp_matrix[1, 0]  # sin(θ)
        recovered_angle = np.arctan2(d, a) * 180 / np.pi
        
        # Recover translation (center based)
        cx, cy = center
        tx_center = warp_matrix[0, 2] - (-cx * a + cy * d + cx)
        ty_center = warp_matrix[1, 2] - (-cx * d - cy * a + cy)
        
        # Convert the translation to Cartesian coordinates.
        true_angle_rad = -np.deg2rad(recovered_angle)
        tx_cartesian = -tx_center * math.cos(true_angle_rad) + ty_center * math.sin(true_angle_rad)
        ty_cartesian = tx_center * math.sin(true_angle_rad) + ty_center * math.cos(true_angle_rad)
        
        # Compute the robot pose in the ground truth frame using the inverse transform.
        warp_3x3 = np.vstack([warp_matrix, [0, 0, 1]])
        T = np.linalg.inv(warp_3x3)
        robot_sensor = np.array([center[0], center[1], 1]).reshape(3, 1)
        robot_ground = T @ robot_sensor
        robot_ground = robot_ground.flatten()[:2]
        
        heading = -recovered_angle  # Heading in Cartesian coordinates.
        return (tx_cartesian, ty_cartesian, heading, best_cc, time.time()-start_time,
                warp_matrix, robot_ground)

    @staticmethod
    def plot_results(ground_truth, sensor_map, warp_matrix, robot_ground, recovered_angle, center,
                     arrow_length=50, true_angle=0):
        
        heading_angle = -recovered_angle
        heading_rad = np.deg2rad(heading_angle)
        dx = arrow_length * math.cos(heading_rad)
        dy = arrow_length * math.sin(heading_rad)
        
        # Warp sensor map to align with the ground truth.
        height, width = sensor_map.shape
        aligned_sensor = cv2.warpAffine(
            sensor_map, warp_matrix, (width, height),
            flags=cv2.WARP_INVERSE_MAP + cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0
        )
        overlay = Localizer._create_overlay(ground_truth, aligned_sensor)
        
        plt.figure(figsize=(18, 6))
        
        # Ground Truth Map with robot pose.
        plt.subplot(1, 3, 1)
        plt.imshow(ground_truth, cmap='gray')
        plt.title('Ground Truth Map')
        plt.axis('off')
        plt.plot(robot_ground[0], robot_ground[1], 'bo', markersize=10)
        plt.arrow(robot_ground[0], robot_ground[1], dx, dy, color='yellow', width=2, head_width=10)
        plt.text(robot_ground[0] + 10, robot_ground[1] + 10, 'Robot', color='blue', fontsize=12)
        
        # Sensor Map.
        plt.subplot(1, 3, 2)
        plt.imshow(sensor_map, cmap='gray')
        plt.title(f'Sensor Map\n(True Rotation = {true_angle}°)', fontsize=12)
        plt.axis('off')
        
        # Composite Overlay.
        plt.subplot(1, 3, 3)
        plt.imshow(overlay)
        plt.title('Composite Overlay', fontsize=12)
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()

# ---------------------------
# Example Usage (when running this file directly)
# ---------------------------
if __name__ == '__main__':
    # Path to sensor map file.
    sensor_path = 'src/vision_pkg/vision_pkg/maps/rotated_image.png'
    
    # Load sensor map in grayscale.
    sensor_map = cv2.imread(sensor_path, cv2.IMREAD_GRAYSCALE)
    if sensor_map is None:
        raise ValueError("Error loading sensor map. Check the file path.")
    
    # Convert sensor map to binary.
    _, sensor_map = cv2.threshold(sensor_map, 127, 255, cv2.THRESH_BINARY)
    
    height, width = sensor_map.shape
    center = (width // 2, height // 2)
    
    # Define approximate parameters.
    approx_angle = 15          # degrees
    approx_x_cartesian = -500  # pixels
    approx_y_cartesian = 300   # pixels
    angle_range = 10           # degrees error
    trans_range = 50           # pixels error
    
    # Create an instance of the Localizer.
    loc = Localizer(gt_path='src/vision_pkg/vision_pkg/maps/robocup_field.png', num_levels=5)
    
    # Perform localization.
    (tx_cartesian, ty_cartesian, heading, cc, time_taken,
     warp_matrix, robot_ground) = loc.localize(sensor_map, approx_angle,
                                               approx_x_cartesian, approx_y_cartesian,
                                               angle_range, trans_range, center,
                                               num_starts=50)
    
    print("Estimated robot position (Cartesian): ({:.2f}, {:.2f})".format(tx_cartesian, ty_cartesian))
    print("Robot heading (degrees): {:.2f}".format(heading))
    print("Correlation strength: {:.2f}".format(cc))
    print("Time taken: {:.2f} seconds".format(time_taken))
    print("Robot position on ground truth map:", robot_ground)
    
    # Plot results using the plot_results static method.
    loc.plot_results(loc.ground_truth, sensor_map, warp_matrix, robot_ground, -heading, center, true_angle=15)
