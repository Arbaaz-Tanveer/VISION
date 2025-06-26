import cv2
import numpy as np


def shift_image(binary_img, shift_x, shift_y):
    """
    Shift the binary image horizontally and vertically.

    Parameters:
        binary_img (numpy array): Input binary image.
        shift_x (int): Number of pixels to shift in x-direction (+ve right, -ve left).
        shift_y (int): Number of pixels to shift in y-direction (+ve down, -ve up).

    Returns:
        numpy array: Shifted binary image.
    """
    h, w = binary_img.shape
    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    shifted_img = cv2.warpAffine(binary_img, M, (w, h), borderValue=0)
    return shifted_img


def rotate_image(binary_img, angle):
    """
    Rotate the binary image around its center by a given angle.

    Parameters:
        binary_img (numpy array): Input binary image.
        angle (float): Angle in degrees (positive is counter-clockwise).

    Returns:
        numpy array: Rotated binary image with same dimensions.
    """
    h, w = binary_img.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_img = cv2.warpAffine(binary_img, M, (w, h), borderValue=0)
    return rotated_img


def crop_image(binary_img, new_width, new_height):
    """
    Crop the binary image with the center as reference to the desired dimensions.

    Parameters:
        binary_img (numpy array): Input binary image.
        new_width (int): Desired width after cropping.
        new_height (int): Desired height after cropping.

    Returns:
        numpy array: Cropped binary image.
    """
    h, w = binary_img.shape
    center_x, center_y = w // 2, h // 2
    x_start = max(center_x - new_width // 2, 0)
    y_start = max(center_y - new_height // 2, 0)
    x_end = min(center_x + new_width // 2, w)
    y_end = min(center_y + new_height // 2, h)

    cropped_img = binary_img[y_start:y_end, x_start:x_end]
    
    # Pad if needed to maintain desired size
    cropped_img = cv2.copyMakeBorder(
        cropped_img,
        (new_height - cropped_img.shape[0]) // 2,
        (new_height - cropped_img.shape[0] + 1) // 2,
        (new_width - cropped_img.shape[1]) // 2,
        (new_width - cropped_img.shape[1] + 1) // 2,
        cv2.BORDER_CONSTANT,
        value=0
    )
    
    return cropped_img


# ------------------------------
# Example Usage
# ------------------------------
if __name__ == "__main__":
    # Load binary image
    img_path = "robocup_field.png"
    binary_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
    # Ensure binary image
    _, binary_img = cv2.threshold(binary_img, 127, 255, cv2.THRESH_BINARY)

    # Shift image by 50 pixels right and 30 pixels down
    shifted_img = shift_image(binary_img, shift_x=50, shift_y=30)
    # cv2.imwrite("shifted_image.png", shifted_img)

    # Rotate image by 45 degrees
    rotated_img = rotate_image(binary_img, angle=340)
    cv2.imwrite("rotated_image.png", rotated_img)

    # Crop image to 200x200 pixels with the center remaining fixed
    cropped_img = crop_image(binary_img, new_width=200, new_height=200)
    # cv2.imwrite("cropped_image.png", cropped_img)

    print("All images processed and saved successfully!")
