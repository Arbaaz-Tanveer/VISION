import cv2
import time
import os
from datetime import datetime

print(cv2.getBuildInformation())

def camera_preview(width=640, height=480, camera_index=0):
    # Create output directory if it doesn't exist
    output_dir = "captured_frames"
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)  # Use CAP_V4L2 for Linux

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, 60)  # Force FPS
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))  # Fast format
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  
    cap.set(cv2.CAP_PROP_BRIGHTNESS, 20)
    # Disable Auto Exposure (Windows)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  
    cap.set(cv2.CAP_PROP_EXPOSURE, 200)  
    print("Camera reported FPS:", cap.get(cv2.CAP_PROP_FPS))
    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    frame_count = 0
    start_time = time.time()
    fps = 0  
    last_capture_time = time.time()
    
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Actual resolution: {actual_width} x {actual_height}")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Can't receive frame")
                break

            frame_count += 1
            elapsed_time = time.time() - start_time
            if elapsed_time > 1:
                fps = frame_count / elapsed_time
                frame_count = 0
                start_time = time.time()
                print(f"FPS: {fps:.1f}")

            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Camera Preview (Press Q to quit)', frame)

            # Capture and save a picture every 1 second
            current_time = time.time()
            if current_time - last_capture_time >= 1:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = os.path.join(output_dir, f"capture_{timestamp}.jpg")
                cv2.imwrite(filename, frame)
                print(f"Saved {filename}")
                last_capture_time = current_time

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        cap.release()
        cv2.destroyAllWindows()

camera_preview(1280, 960)
