import cv2
import numpy as np

# Step 1: Detect direction of object movement
def detect_object_direction(current_x, current_y, prev_x, prev_y, threshold=15):
    direction_x = ""
    direction_y = ""

    if abs(current_x - prev_x) > threshold:
        if current_x > prev_x:
            direction_x = "Right"
        else:
            direction_x = "Left"

    if abs(current_y - prev_y) > threshold:
        if current_y > prev_y:
            direction_y = "Down"
        else:
            direction_y = "Up"

    if direction_x and direction_y:
        return f"{direction_y}-{direction_x}"
    elif direction_x:
        return direction_x
    elif direction_y:
        return direction_y
    else:
        return "Stationary"

# Step 2: Real-time object tracking with direction for multiple objects
def object_tracking():
    cap = cv2.VideoCapture('car1.mp4')

    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=25, detectShadows=False)

    prev_positions = {}  # Dictionary to store previous positions of objects

    frame_counter = 0
    max_frames = 500  # Limit the number of frames to process

    while frame_counter < max_frames:
        ret, frame = cap.read()
        if not ret:
            break  # Exit when the video ends

        # Apply background subtraction
        fgmask = fgbg.apply(frame)

        # Smooth the mask to reduce noise
        fgmask = cv2.medianBlur(fgmask, 5)

        # Find contours to detect moving objects
        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Loop through all detected contours
        for i, contour in enumerate(contours):
            if cv2.contourArea(contour) > 500:  # Filter out small objects
                x, y, w, h = cv2.boundingRect(contour)
                centroid_x = x + w // 2
                centroid_y = y + h // 2

                # Get previous position of this object (using index as key)
                prev_x, prev_y = prev_positions.get(i, (centroid_x, centroid_y))

                # Detect direction of movement
                direction = detect_object_direction(centroid_x, centroid_y, prev_x, prev_y)

                # Update the object's previous position
                prev_positions[i] = (centroid_x, centroid_y)

                # Draw rectangle and movement info on the frame
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f'Object {i}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f'Direction: {direction}', (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        frame_counter += 1
        cv2.imshow('Real-Time Object Tracking', frame)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Step 3: Run the object tracking
if __name__ == '__main__':
    object_tracking()
