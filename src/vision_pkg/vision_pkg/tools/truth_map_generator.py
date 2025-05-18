import numpy as np
import cv2

def create_binary_map(width_m, height_m, resolution=5):
    """
    Create a binary (black and white) map image.
    
    Parameters:
        width_m (float): Width of the map in meters.
        height_m (float): Height of the map in meters.
        resolution (int): Number of pixels per meter.
    
    Returns:
        np.ndarray: Black image of size (height_px, width_px) with uint8 type.
    """
    width_px = int(round(width_m * resolution))
    height_px = int(round(height_m * resolution))
    return np.zeros((height_px, width_px), dtype=np.uint8)

def world_to_image(x, y, height_px, resolution=5):
    """
    Convert world (meters) coordinates to image (pixel) coordinates.
    
    World coordinate system:
      - x increases to the right.
      - y increases upward.
    Image coordinate system (OpenCV):
      - x increases to the right.
      - y increases downward.
    
    Parameters:
        x (float): x-coordinate in meters.
        y (float): y-coordinate in meters.
        height_px (int): Height of the image in pixels.
        resolution (int): Number of pixels per meter.
    
    Returns:
        (int, int): (x, y) pixel coordinate in the image.
    """
    x_px = int(round(x * resolution))
    y_px = height_px - int(round(y * resolution))
    return x_px, y_px

def draw_line(img, x1, y1, x2, y2, thickness_m, resolution=5):
    """
    Draw a white line on the binary map between two world points.
    
    Parameters:
        img (np.ndarray): The binary map image.
        x1, y1 (float): Starting point in meters.
        x2, y2 (float): Ending point in meters.
        thickness_m (float): Thickness of the line in meters.
        resolution (int): Pixels per meter.
    """
    height_px = img.shape[0]
    pt1 = world_to_image(x1, y1, height_px, resolution)
    pt2 = world_to_image(x2, y2, height_px, resolution)
    thickness_px = max(1, int(round(thickness_m * resolution)))
    cv2.line(img, pt1, pt2, color=255, thickness=thickness_px)

def draw_circle(img, x, y, radius_m, thickness_m=None, resolution=5, fill=False):
    height_px = img.shape[0]
    center = world_to_image(x, y, height_px, resolution)
    radius_px = int(round(radius_m * resolution))
    if fill:
        thickness = -1  # OpenCV convention for filled shape
    else:
        thickness = max(1, int(round(thickness_m * resolution))) if thickness_m is not None else 1
    cv2.circle(img, center, radius_px, color=255, thickness=thickness)

def draw_semi_circle(img, center_x, center_y, radius_m, thickness_m, resolution=5, start_angle=0, end_angle=180):
    height_px = img.shape[0]
    # Convert center from world to image coordinates.
    center = world_to_image(center_x, center_y, height_px, resolution)
    axes = (int(round(radius_m * resolution)), int(round(radius_m * resolution)))
    thickness_px = max(1, int(round(thickness_m * resolution)))
    # Draw the arc using cv2.ellipse.
    cv2.ellipse(img, center, axes, angle=0, startAngle=start_angle, endAngle=end_angle, color=255, thickness=thickness_px)


def save_map(img, filename):
    """
    Save the binary map image to a file.
    
    Parameters:
        img (np.ndarray): The binary map image.
        filename (str): Path where the image will be saved.
    """
    cv2.imwrite(filename, img)

def draw_robocup_msl_field(img, resolution=5, thickness_m = 0.1):
    """
    Draws the RoboCup MSL field markings on the provided binary map image.
    
    RoboCup MSL Field Specifications (default values, in meters):
      - Field dimensions: 9.0 m (width) x 6.0 m (height)
      - Field boundary: Outer rectangle
      - Center line: Vertical line splitting the field at x = 4.5 m
      - Center circle: Center at (4.5, 3.0) m with a radius of 0.75 m
    
    Parameters:
        img (np.ndarray): The binary map image.
        resolution (int): Pixels per meter.
    """
    field_width = 24.0
    field_height = 16.0

    # Draw field boundary (rectangle)
    draw_line(img, 1, 1, field_width-1, 1, thickness_m, resolution=resolution)       # Bottom edge
    draw_line(img, 1, field_height-1, field_width-1, field_height-1, thickness_m, resolution=resolution)  # Top edge
    draw_line(img, 1, 1, 1, field_height-1, thickness_m, resolution=resolution)       # Left edge
    draw_line(img, field_width-1, 1, field_width-1, field_height-1, thickness_m, resolution=resolution)   # Right edge

    # Draw center line (vertical center)
    draw_line(img, field_width/2, 1, field_width/2, field_height-1, thickness_m, resolution=resolution)

    # Draw center circle
    draw_circle(img, field_width/2, field_height/2, radius_m=1.5, thickness_m = 0.1, resolution=resolution, fill=False)

def draw_basketball_field_hall3(img, resolution=5, thickness_m = 0.35):
    """
    Draws the RoboCup MSL field markings on the provided binary map image.
    
    RoboCup MSL Field Specifications (default values, in meters):
      - Field dimensions: 9.0 m (width) x 6.0 m (height)
      - Field boundary: Outer rectangle
      - Center line: Vertical line splitting the field at x = 4.5 m
      - Center circle: Center at (4.5, 3.0) m with a radius of 0.75 m
    
    Parameters:
        img (np.ndarray): The binary map image.
        resolution (int): Pixels per meter.
    """
    field_width = 28.0
    field_height = 28.0
    x = 7.345   
    draw_circle(img, field_width/2, field_height/2, radius_m=3.75/2, thickness_m = 0.35, resolution=resolution, fill=False) #middle circle
    draw_circle(img, field_width/2 - x, field_height/2, radius_m=3.65/2, thickness_m = 0.35, resolution=resolution, fill=False) #left
    draw_circle(img, field_width/2 + x, field_height/2, radius_m=3.65/2, thickness_m = 0.35, resolution=resolution, fill=False) #right

    draw_semi_circle(img, field_width/2 - 11.185, field_height/2, radius_m=5.95, thickness_m=0.35, resolution=resolution, start_angle=-90, end_angle=90)  # bigger left semi circle
    draw_semi_circle(img, field_width/2 + 11.185, field_height/2, radius_m=5.95, thickness_m=0.35, resolution=resolution, start_angle=270, end_angle=90)  # bigger right semi circle


    draw_line(img, field_width/2 - 13.265, field_height/2 - 7.53, field_width/2 + 13.265, field_height/2 - 7.53, thickness_m, resolution=resolution)       # Bottom edge
    draw_line(img, field_width/2 - 13.265, field_height/2 + 7.53, field_width/2 + 13.265, field_height/2 + 7.53, thickness_m, resolution=resolution)       # upper edge
    draw_line(img, field_width/2 - 13.265, field_height/2 - 7.53, field_width/2 - 13.265, field_height/2 + 7.53, thickness_m, resolution=resolution)       # left edge
    draw_line(img, field_width/2 + 13.265, field_height/2 - 7.53, field_width/2 + 13.265, field_height/2 + 7.53, thickness_m, resolution=resolution)       # left edge

    draw_line(img, field_width/2 - x, field_height/2 - 1.825, field_width/2 - x, field_height/2 + 1.825, thickness_m, resolution=resolution)       # side circles line
    draw_line(img, field_width/2 + x, field_height/2 - 1.825, field_width/2 + x, field_height/2 + 1.825, thickness_m, resolution=resolution)       

    draw_line(img, field_width/2 - x, field_height/2 - 1.825, field_width/2 - 13.265, field_height/2 - 3.03, thickness_m, resolution=resolution)       # tilted lines left
    draw_line(img, field_width/2 - x, field_height/2 + 1.825, field_width/2 - 13.265, field_height/2 + 3.03, thickness_m, resolution=resolution)       

    draw_line(img, field_width/2 + x, field_height/2 - 1.825, field_width/2 + 13.265, field_height/2 - 3.03, thickness_m, resolution=resolution)       # tilted lines right
    draw_line(img, field_width/2 + x, field_height/2 + 1.825, field_width/2 + 13.265, field_height/2 + 3.03, thickness_m, resolution=resolution)       

    draw_line(img, field_width/2 - 13.265, field_height/2 + 5.95, field_width/2 + 2.08 - 13.265, field_height/2 + 5.95, thickness_m, resolution=resolution)       # semicircle left lines
    draw_line(img, field_width/2 - 13.265, field_height/2 - 5.95, field_width/2 + 2.08 - 13.265, field_height/2 - 5.95, thickness_m, resolution=resolution)       

    draw_line(img, field_width/2 + 13.265, field_height/2 + 5.95, field_width/2 + 13.265 - 2.08, field_height/2 + 5.95, thickness_m, resolution=resolution)       # semicircle right lines
    draw_line(img, field_width/2 + 13.265, field_height/2 - 5.95, field_width/2 + 13.265 - 2.08, field_height/2 - 5.95, thickness_m, resolution=resolution)      


    # Draw center circle

def draw_basketball_field_hall7(img, resolution=5, thickness_m = 0.35):
    """
    Draws the RoboCup MSL field markings on the provided binary map image.
    
    RoboCup MSL Field Specifications (default values, in meters):
      - Field dimensions: 9.0 m (width) x 6.0 m (height)
      - Field boundary: Outer rectangle
      - Center line: Vertical line splitting the field at x = 4.5 m
      - Center circle: Center at (4.5, 3.0) m with a radius of 0.75 m
    
    Parameters:
        img (np.ndarray): The binary map image.
        resolution (int): Pixels per meter.
    """
    field_width = 28.0
    field_height = 28.0
    x = 7.345   
    draw_circle(img, field_width/2, field_height/2, radius_m=3.65/2, thickness_m = 0.35, resolution=resolution, fill=False) #middle circle
    draw_circle(img, field_width/2 - x, field_height/2, radius_m=3.57/2, thickness_m = 0.35, resolution=resolution, fill=False) #left
    draw_circle(img, field_width/2 + x, field_height/2, radius_m=3.57/2, thickness_m = 0.35, resolution=resolution, fill=False) #right

    draw_semi_circle(img, field_width/2 - 11.185, field_height/2, radius_m=5.95, thickness_m=0.35, resolution=resolution, start_angle=-90, end_angle=90)  # bigger left semi circle
    draw_semi_circle(img, field_width/2 + 11.185, field_height/2, radius_m=5.95, thickness_m=0.35, resolution=resolution, start_angle=270, end_angle=90)  # bigger right semi circle


    draw_line(img, field_width/2 - 13.265, field_height/2 - 7.53, field_width/2 + 13.265, field_height/2 - 7.53, thickness_m, resolution=resolution)       # Bottom edge
    draw_line(img, field_width/2 - 13.265, field_height/2 + 7.53, field_width/2 + 13.265, field_height/2 + 7.53, thickness_m, resolution=resolution)       # upper edge
    draw_line(img, field_width/2 - 13.265, field_height/2 - 7.53, field_width/2 - 13.265, field_height/2 + 7.53, thickness_m, resolution=resolution)       # left edge
    draw_line(img, field_width/2 + 13.265, field_height/2 - 7.53, field_width/2 + 13.265, field_height/2 + 7.53, thickness_m, resolution=resolution)       # left edge

    draw_line(img, field_width/2 - x, field_height/2 - 1.785, field_width/2 - x, field_height/2 + 1.825, thickness_m, resolution=resolution)       # side circles line
    draw_line(img, field_width/2 + x, field_height/2 - 1.785, field_width/2 + x, field_height/2 + 1.825, thickness_m, resolution=resolution)       

    draw_line(img, field_width/2 - x, field_height/2 - 1.785, field_width/2 - 13.265, field_height/2 - 3.03, thickness_m, resolution=resolution)       # tilted lines left
    draw_line(img, field_width/2 - x, field_height/2 + 1.785, field_width/2 - 13.265, field_height/2 + 3.03, thickness_m, resolution=resolution)       

    draw_line(img, field_width/2 + x, field_height/2 - 1.785, field_width/2 + 13.265, field_height/2 - 3.03, thickness_m, resolution=resolution)       # tilted lines right
    draw_line(img, field_width/2 + x, field_height/2 + 1.785, field_width/2 + 13.265, field_height/2 + 3.03, thickness_m, resolution=resolution)       

    draw_line(img, field_width/2 - 13.265, field_height/2 + 5.95, field_width/2 + 2.08 - 13.265, field_height/2 + 5.95, thickness_m, resolution=resolution)       # semicircle left lines
    draw_line(img, field_width/2 - 13.265, field_height/2 - 5.95, field_width/2 + 2.08 - 13.265, field_height/2 - 5.95, thickness_m, resolution=resolution)       

    draw_line(img, field_width/2 + 13.265, field_height/2 + 5.95, field_width/2 + 13.265 - 2.08, field_height/2 + 5.95, thickness_m, resolution=resolution)       # semicircle right lines
    draw_line(img, field_width/2 + 13.265, field_height/2 - 5.95, field_width/2 + 13.265 - 2.08, field_height/2 - 5.95, thickness_m, resolution=resolution)      
    
    draw_line(img, field_width/2, field_height/2 - 7.595, field_width/2, field_height/2 + 7.595, thickness_m, resolution=resolution)       # Bottom edge


    # Draw center circle
def draw_test_map(img, resolution):
    """
    Draw a filled rectangle (box) on the binary map.
    
    The rectangle is 0.9 meters wide and 0.6 meters high with its lower-left corner at the origin (0, 0)
    and extending into the first quadrant.
    
    Parameters:
        img (np.ndarray): The binary map image.
        resolution (int): Pixels per meter.
    """
    height_px = img.shape[0]
    # Lower-left corner in world coordinates: (0,0)
    # Upper-right corner in world coordinates: (0.9, 0.6)
    pt_lower_left = world_to_image(0, 0, height_px, resolution)
    pt_upper_right = world_to_image(0.9, 0.6, height_px, resolution)
    
    # In the image coordinate system, the top-left corner is the one with a smaller y value.
    # Here, world_to_image converts (0,0) to (0, height_px) and (0.9,0.6) to (int(0.9*resolution), height_px - int(0.6*resolution)).
    # So, top-left in the image is (pt_lower_left[0], pt_upper_right[1])
    top_left = (pt_lower_left[0]+height_px//2, pt_upper_right[1]-height_px//2)
    bottom_right = (pt_upper_right[0]+ height_px//2, pt_lower_left[1]-height_px//2)
    
    cv2.rectangle(img, top_left, bottom_right, color=255, thickness=-1)
    
def draw_test_map2(img, resolution,thickness_m = 0.09):
    field_width = 10.0
    field_height = 10.0
    height_px = img.shape[0]
    # Lower-left corner in world coordinates: (0,0)
    # Upper-right corner in world coordinates: (0.9, 0.6)
    draw_line(img, field_height//2, field_height//2, field_height//2, field_height//2+1.2, thickness_m, resolution=resolution)       # Bottom edge
    draw_line(img, field_height//2, field_height//2, field_height//2+0.7, field_height//2, thickness_m, resolution=resolution)       # Bottom edge

    
    
    
    
  

# Example usage:
if __name__ == "__main__":
    # Set resolution: 5 pixels per meter
    resolution = 40

    # Create a binary map that exactly fits the RoboCup MSL field
    field_width_m, field_height_m = 28, 28.0
    field_img = create_binary_map(field_width_m, field_height_m, resolution)

    # # Draw additional shapes if desired (example line and circle)
    # draw_line(field_img, 1, 1, 8, 5, thickness_m=0.2, resolution=resolution)
    # draw_circle(field_img, 4.5, 3, radius_m=1.0, thickness_m=0.1, resolution=resolution, fill=False)

    # Draw the RoboCup MSL field on the same image (this will overlay the field markings)
    # draw_robocup_msl_field(field_img, resolution=resolution,thickness_m = 0.1)
    draw_test_map(field_img, resolution=resolution)
    # Save the resulting binary map image
    save_map(field_img, "src/vision_pkg/vision_pkg/maps/test_field.png")    