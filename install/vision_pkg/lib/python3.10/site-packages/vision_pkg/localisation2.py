import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import itertools
import time

#  case needs to be written when no valid candidates are found -----------------------------------------------------------------------------------------------
class Localizer:
    def __init__(self, gt_path, threshold=127):
        """
        Initialize the Localizer by loading the ground truth image and computing its ORB keypoints and descriptors.
        """
        self.gt_path = gt_path
        self.threshold = threshold
        self.gt_img = self._load_image(gt_path, threshold)
        # Precompute the binary mask of the ground truth image for faster overlap scoring.
        self.gt_mask = (self.gt_img == 255)
        # Compute ORB features for the ground truth image.
        self.orb = cv2.ORB_create(30)
        self.kp_gt, self.des_gt = self.orb.detectAndCompute(self.gt_img, None)
        print("orb detected succesfully")
    @staticmethod
    def _load_image(path, threshold):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError("Error loading image from path: {}".format(path))
        # Convert to binary image.
        _, binary = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
        return binary

    @staticmethod
    def compute_euclidean_transform(pt1_sensor, pt2_sensor, pt1_gt, pt2_gt):
        """
        Given two corresponding point pairs (from sensor and ground truth),
        compute the Euclidean transformation (rotation and translation).
        Returns:
          - warp_matrix: the 2x3 transformation matrix.
          - rotation: rotation angle in degrees.
          - translation: (tx, ty)
        """
        vec_sensor = np.array(pt2_sensor) - np.array(pt1_sensor)
        vec_gt = np.array(pt2_gt) - np.array(pt1_gt)
        
        angle_sensor = math.atan2(vec_sensor[1], vec_sensor[0])
        angle_gt = math.atan2(vec_gt[1], vec_gt[0])
        
        theta = angle_gt - angle_sensor
        cos_val = math.cos(theta)
        sin_val = math.sin(theta)
        
        R = np.array([[cos_val, -sin_val],
                      [sin_val,  cos_val]])
        
        pt1_sensor_arr = np.array(pt1_sensor).reshape(2, 1)
        pt1_gt_arr = np.array(pt1_gt).reshape(2, 1)
        T = pt1_gt_arr - R @ pt1_sensor_arr
        
        warp_matrix = np.hstack([R, T])
        return warp_matrix, np.degrees(theta), (T[0,0], T[1,0])
    
    def compute_overlap_score(self, sensor_warped):
        """
        Computes an overlap score using the precomputed ground truth image.
        This version uses OpenCV's optimized bitwise_and for speed.
        """
        # Compute the overlap using cv2.bitwise_and. Since both images are binary (0 or 255),
        # the result will be 255 where both images have white pixels.
        overlap = cv2.bitwise_and(sensor_warped, self.gt_img)
        score = cv2.countNonZero(overlap)
        return score

    
    @staticmethod
    def recover_transformations_from_matches(kp_gt, kp_sensor, good_matches, gt_img, sensor_img, compute_score_fn, center,bot_pos,bot_angle, pos_range,angle_range):
        """
        For every combination of two matches, compute the candidate Euclidean transformation, its 
        overlap score, and the robot position (by warping the sensor center).
        Returns a list of candidate dictionaries.
        """
        candidate_transforms = []
        # Define sensor center for transformation (3x1 vector).
        sensor_center = np.array([center[0], center[1], 1]).reshape(3, 1)
        
        for m1, m2 in itertools.combinations(good_matches, 2):
            pt1_gt = kp_gt[m1.queryIdx].pt
            pt2_gt = kp_gt[m2.queryIdx].pt
            
            pt1_sensor = kp_sensor[m1.trainIdx].pt
            pt2_sensor = kp_sensor[m2.trainIdx].pt
            
            # Skip degenerate cases.
            if np.linalg.norm(np.array(pt2_sensor) - np.array(pt1_sensor)) < 1e-6:
                continue
            
            warp_matrix, rotation, translation = Localizer.compute_euclidean_transform(
                pt1_sensor, pt2_sensor, pt1_gt, pt2_gt)
            
            
            
            # Compute robot ground position by warping the sensor center.
            warp_3x3 = np.vstack([warp_matrix, [0, 0, 1]])
            robot_ground = warp_3x3 @ sensor_center
            robot_ground = robot_ground.flatten()[:2]
            robot_ground[0] -= center[0]
            robot_ground[1] = center[1] - robot_ground[1]
            print(robot_ground)
            heading = -rotation
            pos_distance = math.sqrt((bot_pos[0] - robot_ground[0])**2 + (bot_pos[1] - robot_ground[1])**2)
            angle_distance = abs(bot_angle - heading)
            if((pos_distance < pos_range) and (angle_distance < angle_range)):
                sensor_warped = cv2.warpAffine(sensor_img, warp_matrix, (gt_img.shape[1], gt_img.shape[0]),
                                           flags=cv2.INTER_LINEAR)
                score = compute_score_fn(sensor_warped)          
                # score = 50 
                candidate_transforms.append({
                    'warp_matrix': warp_matrix,
                    'rotation': rotation,
                    'translation': translation,
                    'match_pair': (m1, m2),
                    'score': score,
                    'robot_ground': robot_ground,  # New field storing bot position.
                    'robot_heading': heading
                })
        return candidate_transforms

    def localize(self, sensor_img, num_good_matches=5, center=None, plot_mode='none',bot_pos = None,bot_angle = None, pos_range = None, angle_range = None):
        """
        Perform localization using ORB feature matching between the sensor image and the ground truth.
        
        Args:
            sensor_img: Grayscale (or binary) sensor image.
            num_good_matches: Number of top matches to use.
            center: The (x, y) center point in the sensor image (default: image center).
            plot_mode: 'multiple' to plot top10 candidate overlays, 'best' to show only the best candidate overlay,
                       or 'none' for no plotting.
        
        Returns:
            A tuple:
              (tx_cartesian, ty_cartesian, heading, score, time_taken, best_warp, robot_ground)
            where robot_ground is the computed robot position (obtained by warping the sensor center).
        """
        # Preprocess sensor image (if not already binary, threshold it)
        start_time = time.time()
        # _, sensor_bin = cv2.threshold(sensor_img, self.threshold, 255, cv2.THRESH_BINARY)
        candidate_transforms = []
        # Compute ORB features for the sensor image.
        kp_sensor, des_sensor = self.orb.detectAndCompute(sensor_img, None)
        if des_sensor is None or self.des_gt is None:
            raise ValueError("ORB failed to compute descriptors for one of the images.")
        
        # Brute-force matching.
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(self.des_gt, des_sensor)
        if not matches:
            raise ValueError("No matches found between the images.")
        matches = sorted(matches, key=lambda x: x.distance)
        good_matches = matches[:min(num_good_matches, len(matches))]
        
        # Define sensor center if not provided.
        h, w = sensor_img.shape
        if center is None:
            center = (w // 2, h // 2)
        
        # Recover candidate transformations along with robot position.
        candidate_transforms = Localizer.recover_transformations_from_matches(
            self.kp_gt, kp_sensor, good_matches, self.gt_img, sensor_img, self.compute_overlap_score, center, bot_pos,bot_angle, pos_range,angle_range)
        
        if len(candidate_transforms) == 0:
            raise ValueError("No valid candidate transformations could be computed.")
        
        # Choose the best candidate (highest overlap score).
        candidate_transforms.sort(key=lambda x: x['score'], reverse=True)
        best_candidate = candidate_transforms[0]
        best_warp = best_candidate['warp_matrix']
        best_score = best_candidate['score']
        best_rotation = best_candidate['rotation']
        best_position = best_candidate['robot_ground']
        heading = best_candidate['robot_heading']

        # Compute robot ground position using the chosen warp.
        sensor_center = np.array([center[0], center[1], 1]).reshape(3, 1)
        warp_3x3 = np.vstack([best_warp, [0, 0, 1]])
        robot_ground = warp_3x3 @ sensor_center
        robot_ground = robot_ground.flatten()[:2]
        
        # For consistency with the original interface, we define:
        # tx_cartesian, ty_cartesian: translation components (best_translation)
        # heading: negative of the rotation angle (as in the original code)
        
        time_taken = time.time() - start_time

        # Plotting options.
        if plot_mode == 'multiple':
            # Only plot the top 10 candidates.
            self.plot_candidates(sensor_img, candidate_transforms[:10])
        elif plot_mode == 'best':
            self.plot_best(sensor_img, best_warp, robot_ground, heading, center)
        
        return (best_position[0], best_position[1], heading, best_score, time_taken, best_warp)

    def plot_candidates(self, sensor_img, candidate_transforms):
        """
        Plot multiple candidate overlays in a grid.
        Only the top 10 candidates are plotted.
        """
        num_candidates = len(candidate_transforms)
        cols = min(5, num_candidates)
        rows = (num_candidates + cols - 1) // cols
        
        plt.figure(figsize=(15, 3 * rows))
        for idx, cand in enumerate(candidate_transforms):
            warp_matrix = cand['warp_matrix']
            sensor_warped = cv2.warpAffine(sensor_img, warp_matrix, (sensor_img.shape[1], sensor_img.shape[0]),
                                           flags=cv2.INTER_LINEAR)
            # Create composite overlay: sensor image in red on top of the ground truth image.
            composite = cv2.cvtColor(self.gt_img, cv2.COLOR_GRAY2BGR)
            mask = sensor_warped > 0
            composite[mask] = [0, 0, 255]
            plt.subplot(rows, cols, idx+1)
            plt.imshow(composite)
            plt.title("Cand {}:\nScore={}\nRot={:.1f}°\nRobot Pos=({:.1f}, {:.1f})".format(
                idx, cand['score'], cand['rotation'], cand['robot_ground'][0], cand['robot_ground'][1]))
            plt.axis('off')
        plt.tight_layout()
        plt.show()

    def plot_best(self, sensor_img, warp_matrix, robot_ground, heading, center, arrow_length=50, true_angle=0):
        """
        Plot the best candidate result using three subplots:
          1. Ground truth image with robot position and heading arrow.
          2. Sensor image.
          3. Composite overlay (ground truth in grayscale with warped sensor image in red).
        """
        # Warp sensor image using the best warp.
        height, width = sensor_img.shape
        sensor_warped = cv2.warpAffine(sensor_img, warp_matrix, (width, height),
                                       flags=cv2.INTER_LINEAR)
        # Create composite overlay: ground truth in grayscale and warped sensor image in red.
        overlay = cv2.cvtColor(self.gt_img, cv2.COLOR_GRAY2BGR)
        mask = sensor_warped > 0
        overlay[mask] = [0, 0, 255]
        
        # Compute arrow for heading.
        heading_rad = -np.deg2rad(heading)
        dx = arrow_length * math.cos(heading_rad)
        dy = arrow_length * math.sin(heading_rad)
        
        plt.figure(figsize=(18, 6))
        
        # Ground Truth with robot pose.
        plt.subplot(1, 3, 1)
        plt.imshow(self.gt_img, cmap='gray')
        plt.title('Ground Truth Map')
        plt.axis('off')
        plt.plot(robot_ground[0], robot_ground[1], 'bo', markersize=10)
        plt.arrow(robot_ground[0], robot_ground[1], dx, dy, color='yellow', width=2, head_width=10)
        plt.text(robot_ground[0] + 10, robot_ground[1] + 10, 'Robot', color='blue', fontsize=12)
        
        # Sensor Image.
        plt.subplot(1, 3, 2)
        plt.imshow(sensor_img, cmap='gray')
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
    # File paths.
    gt_path = 'src/vision_pkg/vision_pkg/maps/robocup_field.png'
    sensor_path = 'src/vision_pkg/vision_pkg/maps/rotated_image.png'
    
    # Load sensor image in grayscale.
    sensor_img = cv2.imread(sensor_path, cv2.IMREAD_GRAYSCALE)
    # _, sensor_img = cv2.threshold(sensor_img, 127, 255, cv2.THRESH_BINARY)
    if sensor_img is None:
        raise ValueError("Error loading sensor image. Check the file path.")
    
    height, width = sensor_img.shape
    center = (width // 2, height // 2)
    
    # Create an instance of Localizer.
    loc = Localizer(gt_path=gt_path, threshold=127)
    
    bot_pos = (-500,300)
    bot_angle = -15
    pos_range = 50
    angle_range = 20

    # Perform localization.
    # Choose plot_mode 'multiple' for candidate overlays or 'best' for the best candidate display.
    (tx_cartesian, ty_cartesian, heading, score, time_taken,
     warp_matrix) = loc.localize(sensor_img, num_good_matches=10, center=center, plot_mode='multiple',bot_pos = bot_pos,bot_angle=bot_angle,pos_range = pos_range,angle_range = angle_range)
    
    print("Estimated robot translation (Cartesian): ({:.2f}, {:.2f})".format(tx_cartesian, ty_cartesian))
    print("Robot heading (degrees): {:.2f}".format(heading))
    print("Overlap score: {:.2f}".format(score))
    print("Time taken: {:.2f} seconds".format(time_taken))
    # print("Robot position on ground truth map:", robot_ground)
