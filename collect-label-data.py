import cv2
import numpy as np

# Define the gestures and corresponding labels
gestures = ["gesture1", "gesture2", "gesture3"]

# Set up the video capture
cap = cv2.VideoCapture(0)

# Initialize variables
current_gesture = 0
gesture_frames = []
gesture_labels = []

# Recording loop
while True:
    # Read the video frame
    ret, frame = cap.read()

    # Display the frame for the user
    cv2.imshow("Collecting Gestures", frame)

    # Detect key presses
    key = cv2.waitKey(1)

    # Start collecting gesture frames when 'space' is pressed
    if key == ord(" "):
        gesture_frames = []
        current_gesture += 1 if current_gesture < len(gestures) - 1 else 0
        print("Collecting gesture:", gestures[current_gesture])

    # Stop collecting gesture frames and annotate them with the current gesture label when 's' is pressed
    elif key == ord("s"):
        if len(gesture_frames) > 0:
            for frame in gesture_frames:
                gesture_labels.append(current_gesture)
        print("Collected", len(gesture_frames), "frames for", gestures[current_gesture])

    # Quit the recording loop when 'q' is pressed
    elif key == ord("q"):
        break

    # Collect frames while recording
    if len(gesture_frames) > 0:
        gesture_frames.append(frame)

# Release the video capture and close the window
cap.release()
cv2.destroyAllWindows()

# Convert the collected gesture frames and labels to NumPy arrays
gesture_frames = np.array(gesture_frames)
gesture_labels = np.array(gesture_labels)

# Save the gesture data to files (e.g., CSV, NumPy arrays) for further processing and training
np.save("gesture_frames.npy", gesture_frames)
np.save("gesture_labels.npy", gesture_labels)