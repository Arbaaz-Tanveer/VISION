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
