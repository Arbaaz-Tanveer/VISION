import cv2
import numpy as np
import time

def camera_preview(width=1644, height=1256, camera_index=2, scale=1.0):
    # Open the camera using V4L2 for Linux (adjust if necessary)
    cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, 60)  # Request 60 FPS
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimal buffering
    
    # Optional camera settings (adjust as needed)
    cap.set(cv2.CAP_PROP_BRIGHTNESS, 30)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # Disable auto exposure (if applicable)
    cap.set(cv2.CAP_PROP_EXPOSURE, 500)
    
    print("Camera reported FPS:", cap.get(cv2.CAP_PROP_FPS))
    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    # Read one frame to determine image dimensions
    ret, frame = cap.read()
    if not ret:
        print("Error: Couldn't read frame from camera")
        cap.release()
        return

    h, w = frame.shape[:2]

    # Define the approximate camera matrix assuming the optical center is at the image center.
    K = np.array([[w / 2, 0, w / 2],
                  [0, w / 2, h / 2],
                  [0, 0, 1]], dtype=np.float32)

    # Fixed distortion coefficients:
    # k1 is set to 105 (maps to (105 - 100)/100 = 0.05) and others to 100 (i.e., 0.0)
    k1 = (105 - 100) / 100.0  # 0.05
    k2 = (100 - 100) / 100.0  # 0.0
    k3 = (100 - 100) / 100.0  # 0.0
    k4 = (100 - 100) / 100.0  # 0.0
    D = np.array([[k1], [k2], [k3], [k4]], dtype=np.float32)

    # Compute new output dimensions based on the scale factor.
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Estimate the new camera matrix to avoid cropping (balance=1)
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                K, D, (w, h), np.eye(3), balance=1, new_size=(new_w, new_h))
    
    # Compute undistortion and rectification maps (computed once)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
                    K, D, np.eye(3), new_K, (new_w, new_h), cv2.CV_16SC2)
    
    # Variables for FPS calculation
    frame_count = 0
    start_time = time.time()
    fps = 0
    
    # Latest undistorted frame (for saving)
    latest_undistorted = None

    print("Press 'q' to exit or 's' to save the current frame.")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Couldn't retrieve frame")
                break

            # Remap to undistort the frame
            undistorted = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            latest_undistorted = undistorted  # update latest frame

            # FPS calculation
            frame_count += 1
            elapsed_time = time.time() - start_time
            if elapsed_time > 1:
                fps = frame_count / elapsed_time
                frame_count = 0
                start_time = time.time()

            # Overlay the FPS on the undistorted image
            cv2.putText(undistorted, f"FPS: {fps:.1f}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow('Undistorted Camera Feed', undistorted)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                save_path = 'undistorted_saved.jpg'
                cv2.imwrite(save_path, latest_undistorted)
                print(f"Image saved to {save_path}")
    finally:
        cap.release()
        cv2.destroyAllWindows()

# Call the camera preview function with desired dimensions.
camera_preview()
