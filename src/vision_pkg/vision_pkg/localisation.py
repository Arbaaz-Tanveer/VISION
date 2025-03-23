import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import time

# ---------------------------
# Helper Functions
# ---------------------------

def build_pyramid(image, levels):
    """Build a Gaussian pyramid for the image."""
    pyramid = [image]
    for _ in range(1, levels):
        image = cv2.pyrDown(image)
        pyramid.append(image)
    return pyramid

def create_overlay(img1, img2):
    """Create an RGB overlay: img1 in white, img2 in red."""
    composite = np.dstack([img1, img1, img1]).astype(np.uint8)
    mask = img2 > 0
    composite[mask] = [255, 0, 0]
    return composite

# ---------------------------
# Localization Function
# ---------------------------
def localize(sensor_map, approx_angle, approx_x_cartesian, approx_y_cartesian, 
             angle_range, trans_range, pyr_gt, center, num_levels=6, num_starts=10, 
             criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-4),
             warp_mode=cv2.MOTION_EUCLIDEAN):
    
    start_time = time.time()
    
    # Build pyramid for sensor map
    pyr_sensor = build_pyramid(sensor_map, num_levels)
    # Compute approximate translation (convert approximate bot position from Cartesian to sensor coordinate frame)
    approx_angle_rad = np.deg2rad(approx_angle)
    tx_approx = -approx_x_cartesian * math.cos(approx_angle_rad) + approx_y_cartesian * math.sin(approx_angle_rad)
    ty_approx = approx_x_cartesian * math.sin(approx_angle_rad) + approx_y_cartesian * math.cos(approx_angle_rad)
    
    # Adjust ranges at the coarsest level.
    scale_coarse = 1 / (2 ** (num_levels - 1))
    angle_range_coarse = (-approx_angle - angle_range, -approx_angle + angle_range)
    trans_range_coarse_x = ((tx_approx - trans_range) * scale_coarse, (tx_approx + trans_range) * scale_coarse)
    trans_range_coarse_y = ((ty_approx - trans_range) * scale_coarse, (ty_approx + trans_range) * scale_coarse)
    
    coarse_level = num_levels - 1
    template_coarse = pyr_gt[coarse_level]
    image_coarse = pyr_sensor[coarse_level]
    
    best_cc = -1e9
    best_warp = None
    
    # Multi-start search at coarsest level
    for i in range(num_starts):
        angle_init = np.random.uniform(*angle_range_coarse)
        tx_init = np.random.uniform(*trans_range_coarse_x)
        ty_init = np.random.uniform(*trans_range_coarse_y)
        
        angle_rad = np.deg2rad(angle_init)
        cos_val = np.cos(angle_rad)
        sin_val = np.sin(angle_rad)
        cx_coarse = center[0] * scale_coarse
        cy_coarse = center[1] * scale_coarse
        
        # Build initial warp matrix (rotation about the center plus translation)
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
    
    # If no candidate was found, default to identity matrix..
    if best_warp is None:
        best_warp = np.eye(2, 3, dtype=np.float32)
    
    # Refine warp from coarse to fine scales.
    current_warp = best_warp.copy()
    for level in reversed(range(num_levels - 1)):
        # Scale translation for the next (finer) level.
        current_warp[0, 2] *= 2.0
        current_warp[1, 2] *= 2.0
        
        template = pyr_gt[level]
        image = pyr_sensor[level]
        try:
            cc, current_warp = cv2.findTransformECC(template, image, current_warp,
                                                    warp_mode, criteria)
        except cv2.error as e:
            print(f"ECC failed at pyramid level {level}: {e}")
            break
    warp_matrix = current_warp
    print(f"time taken till now = {time.time() - start_time}")
    
    # Extract rotation angle from the warp matrix.
    a = warp_matrix[0, 0]  # cos(θ)
    b = warp_matrix[0, 1]  # -sin(θ)
    d = warp_matrix[1, 0]  # sin(θ)
    recovered_angle = np.arctan2(d, a) * 180 / np.pi
    
    # Recover translation (center based)
    cx, cy = center
    tx_center = warp_matrix[0, 2] - (-cx * a + cy * d + cx)
    ty_center = warp_matrix[1, 2] - (-cx * d - cy * a + cy)
    
    # Convert the translation to Cartesian coordinates
    true_angle_rad = -np.deg2rad(recovered_angle)
    tx_cartesian = -tx_center * math.cos(true_angle_rad) + ty_center * math.sin(true_angle_rad)
    ty_cartesian = tx_center * math.sin(true_angle_rad) + ty_center * math.cos(true_angle_rad)
    
    # Compute the robot pose in the ground truth frame using the inverse transform.
    warp_3x3 = np.vstack([warp_matrix, [0, 0, 1]])
    T = np.linalg.inv(warp_3x3)
    robot_sensor = np.array([center[0], center[1], 1]).reshape(3, 1)
    robot_ground = T @ robot_sensor
    robot_ground = robot_ground.flatten()[:2]
    
    time_taken = time.time() - start_time
    
    # The heading (in Cartesian coordinates) is taken as the negative of the recovered angle.
    heading = -recovered_angle
    
    return (tx_cartesian, ty_cartesian, heading, best_cc, time_taken, warp_matrix, robot_ground)

# ---------------------------
# Plotting Function
# ---------------------------
def plot_results(ground_truth, sensor_map, warp_matrix, robot_ground, recovered_angle, center, arrow_length=50, true_angle=0):
    
    heading_angle = -recovered_angle
    heading_rad = np.deg2rad(heading_angle)
    dx = arrow_length * math.cos(heading_rad)
    dy = arrow_length * math.sin(heading_rad)
    
    # Create aligned sensor map using inverse warp.
    height, width = sensor_map.shape
    aligned_sensor = cv2.warpAffine(
        sensor_map, warp_matrix, (width, height),
        flags=cv2.WARP_INVERSE_MAP + cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0
    )
    overlay = create_overlay(ground_truth, aligned_sensor)
    
    plt.figure(figsize=(18, 6))
    
    # Ground Truth with robot pose overlay.
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
# Example Usage
# ---------------------------
# 1) Load binary images from files.
gt_path = 'robocup_field.png'
sensor_path = 'damaged_image.png'

# Read images in grayscale.
ground_truth = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
sensor_map = cv2.imread(sensor_path, cv2.IMREAD_GRAYSCALE)

if ground_truth is None or sensor_map is None:
    raise ValueError("Error loading one or both images. Check the file paths.")

# Convert images to binary using a threshold.
_, ground_truth = cv2.threshold(ground_truth, 127, 255, cv2.THRESH_BINARY)
_, sensor_map = cv2.threshold(sensor_map, 127, 255, cv2.THRESH_BINARY)

height, width = sensor_map.shape
center = (width // 2, height // 2)

# 2) Optional: Simulate a known rotation on the sensor map for testing.
# true_angle = 0  # degrees for testing
# M_true = cv2.getRotationMatrix2D(center, true_angle, 1.0)
# rotated_sensor = cv2.warpAffine(sensor_map, M_true, (width, height),
#                                 flags=cv2.INTER_NEAREST,
#                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)

# 3) Precompute pyramid for the ground truth map.
num_levels = 6
pyr_gt = build_pyramid(ground_truth, num_levels)

# 4) Define approximate parameters and expected errors.
approx_angle = 15          # in degrees
approx_x_cartesian = -500   # in pixels
approx_y_cartesian = 300     # in pixels
angle_range = 10            # expected error in angle (degrees)
trans_range = 50            # expected error in translation (pixels)

# 5) Call the localize function.
(tx_cartesian, ty_cartesian, heading, cc, time_taken, warp_matrix, robot_ground) = localize(
    sensor_map, approx_angle, approx_x_cartesian, approx_y_cartesian,
    angle_range, trans_range, pyr_gt, center, num_levels=num_levels, num_starts=50
)
print(robot_ground)
print(f"Estimated robot position (Cartesian): ({tx_cartesian:.2f}, {ty_cartesian:.2f})")
print(f"Robot heading (degrees): {heading:.2f}")
print(f"Correlation strength: {cc:.2f}")
print(f"Time taken: {time_taken:.2f} seconds")

# 6) Plot the results.
plot_results(ground_truth, sensor_map, warp_matrix, robot_ground, -heading, center, true_angle=15)
