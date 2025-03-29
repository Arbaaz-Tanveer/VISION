import cv2
import numpy as np
import random
import sys

def add_salt_noise(img, noise_prob=0.005):
    """
    Add salt noise (white pixels) to the image.
    
    Args:
        img (np.ndarray): Binary image.
        noise_prob (float): Probability that any pixel is set to white.
        
    Returns:
        np.ndarray: Image with added salt noise.
    """
    noisy = img.copy()
    # Generate random noise mask: each pixel gets white with probability noise_prob.
    noise_mask = np.random.rand(*img.shape) < noise_prob
    noisy[noise_mask] = 255
    return noisy

def add_white_clumps(img, num_clumps_range=(3, 10), clump_radius_range=(5, 20)):
    """
    Add random white clumps (circles) to the image.
    
    Args:
        img (np.ndarray): Binary image.
        num_clumps_range (tuple): Minimum and maximum number of clumps to add.
        clump_radius_range (tuple): Range for the radius of clumps.
    
    Returns:
        np.ndarray: Image with added white clumps.
    """
    damaged = img.copy()
    h, w = damaged.shape
    num_clumps = random.randint(*num_clumps_range)
    for _ in range(num_clumps):
        center = (random.randint(0, w - 1), random.randint(0, h - 1))
        radius = random.randint(*clump_radius_range)
        cv2.circle(damaged, center, radius, 255, -1)  # Filled white circle
    return damaged

def remove_clumps(img, num_remove_range=(3, 7), clump_radius_range=(5, 20), removal_type="full"):
    """
    Remove parts of the image by erasing random circular regions.
    
    Args:
        img (np.ndarray): Binary image.
        num_remove_range (tuple): Minimum and maximum number of clumps to remove.
        clump_radius_range (tuple): Range for the radius of the removed clumps.
        removal_type (str): "full" for completely blacking out the region,
                            "partial" for setting region to gray.
                            
    Returns:
        np.ndarray: Image with removed (damaged) clumps.
    """
    damaged = img.copy()
    h, w = damaged.shape
    num_remove = random.randint(*num_remove_range)
    for _ in range(num_remove):
        center = (random.randint(0, w - 1), random.randint(0, h - 1))
        radius = random.randint(*clump_radius_range)
        mask = np.zeros_like(damaged)
        cv2.circle(mask, center, radius, 255, -1)
        if removal_type == "full":
            damaged[mask == 255] = 0
        elif removal_type == "partial":
            damaged[mask == 255] = 128
    return damaged

def damage_binary_image(binary_img, noise_prob=0.005):
    """
    Apply a series of damage operations on a binary image.
    
    Args:
        binary_img (np.ndarray): Input binary image.
        noise_prob (float): Probability for salt noise.
    
    Returns:
        np.ndarray: Damaged image.
    """
    damaged = add_salt_noise(binary_img, noise_prob=noise_prob)
    damaged = remove_clumps(damaged, num_remove_range=(150, 300), clump_radius_range=(5, 20), removal_type="full")
    damaged = add_white_clumps(damaged, num_clumps_range=(20, 40), clump_radius_range=(5, 10))
    return damaged

def convert_to_binary(image, threshold=127):
    """
    Convert a grayscale image to binary using a threshold.
    
    Args:
        image (np.ndarray): Grayscale image.
        threshold (int): Threshold value.
        
    Returns:
        np.ndarray: Binary image.
    """
    _, binary = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    return binary

def main():
    # if len(sys.argv) < 2:
    #     print("Usage: python damage_image.py <path_to_image>")
    #     sys.exit(1)
        
    image_path = "src/vision_pkg/vision_pkg/maps/rotated_image.png"
    # Load the image in grayscale (if not already binary, we will threshold it)
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Unable to load image at {image_path}")
        sys.exit(1)
    
    # Convert to binary if necessary. If your image is already binary, this is fine.
    binary_img = convert_to_binary(img, threshold=127)
    
    # Damage the binary image.
    damaged_img = damage_binary_image(binary_img, noise_prob=0.01)
    
    # Display original and damaged images.
    cv2.imshow("Original Binary Image", binary_img)
    cv2.imshow("Damaged Image", damaged_img)
    print("Press any key to exit.")
    cv2.waitKey(0)
    
    # Save the damaged image.
    output_path = "damaged_image.png"
    cv2.imwrite(output_path, damaged_img)
    print(f"Damaged image saved to {output_path}")
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
