# another_file.py
from localisation import Localizer
import cv2

# Load your sensor map (make sure the path is correct)
sensor_path = 'src/vision_pkg/vision_pkg/maps/rotated_image.png'
sensor_map = cv2.imread(sensor_path, cv2.IMREAD_GRAYSCALE)
_, sensor_map = cv2.threshold(sensor_map, 127, 255, cv2.THRESH_BINARY)

height, width = sensor_map.shape
center = (width // 2, height // 2)

# Create an instance of the Localizer.
loc = Localizer(gt_path='src/vision_pkg/vision_pkg/maps/robocup_field.png', num_levels=5)

# Call the localize method with the appropriate parameters.
(tx_cartesian, ty_cartesian, heading, cc, time_taken, warp_matrix, robot_ground) = loc.localize(
    sensor_map,
    approx_angle=15,
    approx_x_cartesian=-500,
    approx_y_cartesian=300,
    angle_range=10,
    trans_range=50,
    center=center,
    num_starts=50
)

# Optionally, display the results.
print(f"Robot position: ({tx_cartesian:.2f}, {ty_cartesian:.2f}), Heading: {heading:.2f}°")

# And use the built-in plotter.
# Localizer.plot_results(loc.ground_truth, sensor_map, warp_matrix, robot_ground, -heading, center, true_angle=15)




# #for vision and cameras
# import cv2
# import numpy as np
# from groundmapping import (
#     compute_calibration_params,
#     undistort_image,
#     white_threshold,
#     update_ground_map,
#     undistort_pixel,
#     pixel_to_ground,
#     CoordinateEstimator
# )

# def main():
#     # Load a test image (ensure 'sample.jpg' exists or update the path).
#     image_path = "camera_2_frame945.jpg"
#     image = cv2.imread(image_path)
#     if image is None:
#         print("Error: Could not load image. Please check the image path.")
#         return

#     # Compute calibration parameters and undistortion maps once.
#     map1, map2, K, D, new_K = compute_calibration_params(image, distortion_param=0.05, show=True)
    
#     # Undistort the image.
#     undistorted = undistort_image(image, map1, map2, show=True)
#     # undistorted = image
    
#     # Initialize the coordinate estimator using undistorted image dimensions.
#     h, w = undistorted.shape[:2]
#     print(f"height = {h} , width = {w}")
#     estimator = CoordinateEstimator(
#         image_width=w,
#         image_height=h,
#         fov_horizontal=96,
#         fov_vertical=85,
#         camera_height=0.8,  # meters
#         camera_tilt= 30  # degrees down from horizontal
#     )
    
#     # Create a common ground map.
#     # For example, a 40x40 meter map with 15 pixels per meter.
#     map_size_m = 40
#     scale = 15
#     map_size_px = int(map_size_m * scale)
#     common_ground_map = np.zeros((map_size_px, map_size_px), dtype=np.uint8)
    
#     # Update the common ground map using four simulated camera views.
#     for cam in ["front",]:
#         common_ground_map = update_ground_map(
#             common_ground_map,
#             undistorted,
#             estimator,
#             thresh_val=220,
#             scale=scale,
#             max_distance=15,
#             camera=cam,
#             show=False
#         )
    
#     # Display the merged ground map.
#     cv2.imshow("Merged Ground Map (All 4 Views)", common_ground_map)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
    
#     # Example: Convert a list of pixel coordinates (from the distorted image) to ground coordinates.
#     test_pixels = [(w//2, h//2), (100, 200), (300, 400)]
#     ground_coords = pixel_to_ground(test_pixels, estimator, K, D, new_K, show=True)
    
# if __name__ == "__main__":
#     main()