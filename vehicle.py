import cv2
import matplotlib.pyplot as plt
import os
import numpy as np

# Define the file path
file_path = r'vehicle counter/video.mp4'

# Verify if the file exists
if not os.path.exists(file_path):
    print(f"Error: File not found at {file_path}")
else:
    # Attempt to open the video
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        print("Error: Could not open video file. Check codecs or file integrity.")
    else:
        print("Press 'q' in the OpenCV window or any key in Matplotlib to stop the video.")

        # Initialize the plot
        plt.ion()  # Turn on interactive mode
        fig, ax = plt.subplots()  # Create a figure and axis
        frame_display = None  # Placeholder for the image plot

        min_width_react = 80  # Minimum width of rectangle
        min_height_react = 80  # Minimum height of rectangle

        count_line_position = 550

        # Initialize Background Subtractor
        algo = cv2.bgsegm.createBackgroundSubtractorMOG()

        def center_handle(x, y, w, h):
            """Calculate the center of a bounding box."""
            cx = x + w // 2
            cy = y + h // 2
            return cx, cy

        detect = {}  # Dictionary to store detected objects with unique IDs
        offset = 6  # Allowable error around the count line
        counter = 0
        next_vehicle_id = 1  # Unique ID for each detected vehicle

        while True:
            ret, frame1 = cap.read()
            if not ret:
                print("End of video or failed to read frame.")
                break

            # Preprocessing
            grey = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(grey, (3, 3), 5)

            # Apply Background Subtraction
            img_sub = algo.apply(frame1)
            dilat = cv2.dilate(img_sub, np.ones((5, 5)))

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            dilatada = cv2.morphologyEx(dilat, cv2.MORPH_CLOSE, kernel)
            dilatada = cv2.morphologyEx(dilatada, cv2.MORPH_CLOSE, kernel)

            # Find Contours
            contours, _ = cv2.findContours(dilatada, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # Draw a line on the frame
            cv2.line(frame1, (25, count_line_position), (1200, count_line_position), (225, 127, 0), 3)

            current_detect = {}  # Temporary dictionary for current frame detections

            # Draw rectangles for detected objects
            for (i, c) in enumerate(contours):
                (x, y, w, h) = cv2.boundingRect(c)
                validate_counter = (w >= min_width_react) and (h >= min_height_react)
                if not validate_counter:
                    continue
                cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 225, 0), 2)

                # Calculate the center and update detection
                center = center_handle(x, y, w, h)
                current_detect[center] = None  # Track centers for this frame

                cv2.circle(frame1, center, 4, (0, 0, 225), -1)

            # Compare detections across frames
            for center in current_detect.keys():
                x, y = center
                # Check if the object is crossing the line
                if (count_line_position - offset) <= y <= (count_line_position + offset):
                    if center not in detect.values():
                        counter += 1
                        detect[next_vehicle_id] = center
                        next_vehicle_id += 1
                        print(f"Vehicle Counter: {counter}")

            # Cleanup old detections not present in the current frame
            detect = {vid: pos for vid, pos in detect.items() if pos in current_detect}

            # Display the vehicle count
            cv2.putText(frame1, f"VEHICLE COUNTER: {counter}", (450, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 225), 5)

            # Convert the modified frame to RGB for Matplotlib
            frame_rgb = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)

            # Update the Matplotlib plot
            if frame_display is None:
                frame_display = ax.imshow(frame_rgb)
                ax.axis('off')
            else:
                frame_display.set_data(frame_rgb)
                

            plt.pause(0.001)

            # Display Results
            cv2.imshow('Frame with Detections', frame1)

            # Exit the loop if any key is pressed
            if plt.waitforbuttonpress(timeout=0.001) or cv2.waitKey(1) & 0xFF == ord('q'):
                print("Exiting video viewer.")
                break

        cap.release()
        cv2.destroyAllWindows()
        plt.close()
