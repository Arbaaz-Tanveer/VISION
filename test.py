import cv2
def show_pyramid(image, levels):
    pyramid = [image]
    for level in range(1, levels):
        image = cv2.pyrDown(image)
        pyramid.append(image)
    for i, level_img in enumerate(pyramid):
        cv2.imshow(f'Pyramid Level {i}', level_img)
        cv2.waitKey(0)  # Wait for key press to proceed to next level
    cv2.destroyAllWindows()

# Example usage:
if __name__ == "__main__":
    sensor_path = 'src/vision_pkg/vision_pkg/maps/robocup_field.png'
    sensor_map = cv2.imread(sensor_path, cv2.IMREAD_GRAYSCALE)
    if sensor_map is None:
        raise ValueError("Error loading sensor map. Check the file path.")
    _, sensor_map = cv2.threshold(sensor_map, 127, 255, cv2.THRESH_BINARY)
    
    # Show pyramid levels
    show_pyramid(sensor_map, levels=3)