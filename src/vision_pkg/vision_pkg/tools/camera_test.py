import cv2
import time
print(cv2.getBuildInformation())

def camera_preview(width=640, height=480, camera_index=4):
#     gst_pipeline = (
#     "v4l2src device=/dev/video2 ! "
#     "video/x-raw, width=640, height=480, framerate=30/1 ! "
#     "videoconvert ! appsink sync=false max-buffers=1 drop=true"
# )


#     cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
    cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)  # Use CAP_V4L2 for Linux

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, 60)  # Force FPS
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))  # Fast format
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffering
    cap.set(cv2.CAP_PROP_BRIGHTNESS, 30)
    # Disable Auto Exposure (Windows)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  
    cap.set(cv2.CAP_PROP_EXPOSURE, 700)  
    print("Camera reported FPS:", cap.get(cv2.CAP_PROP_FPS))
    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    frame_count = 0
    start_time = time.time()
    fps = 0  
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Actual resolution: {actual_width} x {actual_height}")

    try:
        while True:
            # for _ in range(3):  # Discard old frames
            #     cap.grab()
            
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
            # frame = cv2.resize(frame, (1280, 960))
            cv2.imshow('Camera Preview (Press Q to quit)', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        cap.release()
        cv2.destroyAllWindows()

# camera_preview(1920, 720)
# camera_preview(1920, 1467)

# camera_preview(3288, 2512)
# camera_preview(1644, 1256)

camera_preview(1280, 960)

# camera_preview(640, 489)
