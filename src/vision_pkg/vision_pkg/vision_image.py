import cv2
import time
import os
import numpy as np
import matplotlib.pyplot as plt
from groundmapping import (
    compute_calibration_params,
    undistort_image,
    update_ground_map,
    CoordinateEstimator
)
from localisation2 import Localizer

def show_intermediate_results(blurred, edges, dilated_edges):
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

def process_ground_map(ground_map):
    """
    Process the ground map using Canny edge detection followed by dilation.
    Returns the processed map along with the intermediate images.
    """
    # Apply a slight Gaussian blur to reduce noise before edge detection.
    blurred = cv2.GaussianBlur(ground_map, (3, 3), 0)
    
    # Apply Canny edge detection.
    # Adjust the thresholds as needed.
    edges = cv2.Canny(blurred, threshold1=50, threshold2=300)
    
    # Define a kernel for dilation.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
    
    # Dilate the edges to thicken them.
    dilated_edges = cv2.dilate(edges, kernel, iterations=3)
    
    return blurred, edges, dilated_edges

def main():
    # Set image dimensions and calibration parameters.
    image_width, image_height = 1280, 960
    map1, map2, K, D, new_K = compute_calibration_params(
        image_height, image_width, distortion_param=0.05, show=False
    )
    
    # Set up the coordinate estimator with camera and ground parameters.
    estimator = CoordinateEstimator(
        image_width=image_width,
        image_height=image_height,
        fov_horizontal=95,  # degrees
        fov_vertical=82,    # degrees
        camera_height=0.75, # meters
        camera_tilt=30      # degrees
    )
    calibration = {
        "map1": map1,
        "map2": map2,
        "K": K,
        "D": D,
        "new_K": new_K,
        "estimator": estimator
    }

    # Load the image.
    image_path = "captured_frames/capture_20250328_180056.jpg"  # <-- Update with your image path
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not read the image.")
        return

    # Undistort the image.
    undistorted = undistort_image(image, calibration["map1"], calibration["map2"], show=True)

    # Create a blank common ground map.
    map_size_m = 28   # physical size in meters
    scale = 40        # pixels per meter
    map_size_px = int(map_size_m * scale)
    common_ground_map = np.zeros((map_size_px, map_size_px), dtype=np.uint8)

    # Update the ground map using the undistorted image.
    # Here we assume the camera role is "right" (you can change this as needed).
    common_ground_map = update_ground_map(
        common_ground_map,
        undistorted,
        calibration["estimator"],
        thresh_val=210,
        scale=scale,
        max_distance=5,
        camera="right",
        show=True
    )
    
    # Show the original common ground map using matplotlib.
    plt.figure(figsize=(5, 5))
    plt.imshow(common_ground_map, cmap='gray')
    plt.title("Original Common Ground Map")
    plt.axis('off')
    plt.show()
    
    # Process the ground map: apply Canny edge detection and then dilate the edges.
    time_start = time.time()
    blurred, edges, dilated_edges = process_ground_map(common_ground_map)
    print(f"time taken in processing = {time.time() - time_start}")
    show_intermediate_results(blurred, edges, dilated_edges)
    
    # Use the dilated edges for localization.
    processed_ground_map = dilated_edges

    # Initialize the Localizer.
    localizer = Localizer(gt_path='src/vision_pkg/vision_pkg/maps/test_field.png', threshold=127)
    h_map, w_map = processed_ground_map.shape
    center = (w_map // 2, h_map // 2)

    try:
        # Perform localization on the processed ground map.
        (tx_cartesian, ty_cartesian, heading, score, time_taken,
         warp_matrix, robot_ground) = localizer.localize(
            processed_ground_map, num_good_matches=100, center=center, plot_mode='best'
        )
        # Convert translation to meters.qq
        tx_cartesian_m = tx_cartesian / scale      
        ty_cartesian_m = ty_cartesian / scale

        print(f"Localization result: Position: ({tx_cartesian_m:.2f}, {ty_cartesian_m:.2f}), "
              f"Heading: {heading:.2f}°, Time taken: {time_taken:.2f}s")

        # Plot the localization results.
        Localizer.plot_results(
            localizer.ground_truth,
            processed_ground_map,
            warp_matrix,
            robot_ground,
            -heading,
            center,
            true_angle=15
        )

        # Optionally display the final processed ground map using cv2.imshow.
        cv2.imshow("Processed Ground Map", processed_ground_map)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"Error during localisation: {e}")

if __name__ == '__main__':
    main()
