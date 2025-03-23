import cv2
import numpy as np

# Load the input image
img = cv2.imread(r'images\camera_2_frame945.jpg')
if img is None:
    print("Error: Could not load image. Check the path.")
    exit(1)
h, w = img.shape[:2]

# Define the approximate camera matrix assuming the optical center is at the image center.
K = np.array([[w / 2, 0, w / 2],
              [0, w / 2, h / 2],
              [0, 0, 1]], dtype=np.float32)

# Global variable to store the latest undistorted image.
latest_undistorted = None

# Utility function to map trackbar values in [0, 200] to [-1.0, 1.0]
def map_trackbar(val):
    return (val - 100) / 100.0

# Utility function to map scale trackbar values (e.g., 100 means 1.0, 200 means 2.0)
def map_scale(val):
    return val / 100.0

def update(_):
    global latest_undistorted
    # Get current trackbar positions for distortion parameters
    k1 = map_trackbar(cv2.getTrackbarPos('k1', 'Fisheye'))
    k2 = map_trackbar(cv2.getTrackbarPos('k2', 'Fisheye'))
    k3 = map_trackbar(cv2.getTrackbarPos('k3', 'Fisheye'))
    k4 = map_trackbar(cv2.getTrackbarPos('k4', 'Fisheye'))
    
    # Get the current scale factor for output image size
    scale = map_scale(cv2.getTrackbarPos('Scale', 'Fisheye'))
    
    # Build the distortion coefficient array for fisheye model
    D_new = np.array([[k1], [k2], [k3], [k4]], dtype=np.float32)
    
    # Calculate new output dimensions based on the scale factor
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Use balance=1 to include the full field of view (no cropping), and specify the new image size.
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                K, D_new, (w, h), np.eye(3), balance=1, new_size=(new_w, new_h))
    
    # Compute undistortion and rectification maps
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
                    K, D_new, np.eye(3), new_K, (new_w, new_h), cv2.CV_16SC2)
    # Apply remapping to obtain the undistorted image
    undistorted = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    
    # Store the current undistorted image for saving later.
    latest_undistorted = undistorted
    
    # Show the undistorted image with any black borders where data is missing.
    cv2.imshow('Fisheye', undistorted)

# Callback function for the save button.
def save_image(*args):
    if latest_undistorted is not None:
        save_path = 'undistorted_saved.jpg'
        cv2.imwrite(save_path, latest_undistorted)
        print(f"Image saved to {save_path}")
    else:
        print("No image to save yet.")

# Create a window for display (make sure your OpenCV has Qt support)
cv2.namedWindow('Fisheye', cv2.WINDOW_NORMAL)

# Create trackbars for the four distortion coefficients with range [0, 200] (100 corresponds to 0.0)
cv2.createTrackbar('k1', 'Fisheye', 100, 200, update)
cv2.createTrackbar('k2', 'Fisheye', 100, 200, update)
cv2.createTrackbar('k3', 'Fisheye', 100, 200, update)
cv2.createTrackbar('k4', 'Fisheye', 100, 200, update)
# Create a trackbar for the scale factor; starting at 100 (i.e. scale 1.0) up to 300 (scale 3.0)
cv2.createTrackbar('Scale', 'Fisheye', 100, 300, update)

# Create a save button (this requires OpenCV built with Qt support)
try:
    cv2.createButton('Save', save_image, None, cv2.QT_PUSH_BUTTON, 0)
except Exception as e:
    print("Save button could not be created. Ensure your OpenCV installation supports Qt.", e)

# Initial call to display the original undistorted image (with no extra scale)
update(0)

print("Adjust the sliders to correct fisheye distortion and increase the frame size. Press 'Esc' to exit.")
while True:
    key = cv2.waitKey(1) & 0xFF
    # Optional: allow saving with the 's' key if button is unavailable.
    if key == ord('s'):
        save_image()
    if key == 27:  # Exit on 'Esc' key
        break

cv2.destroyAllWindows()
