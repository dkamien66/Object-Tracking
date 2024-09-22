# https://www.youtube.com/watch?v=GgGro5IV-cs by Pysourc
# dictionary .items() https://www.w3schools.com/python/ref_dictionary_items.asp
# openCV .putText() https://www.geeksforgeeks.org/python-opencv-cv2-puttext-method/

import cv2
import math
from object_detection import ObjectDetection

# Initialize object detection
od = ObjectDetection()

# Run video
cap = cv2.VideoCapture("people_walking.mp4")

# Create an array to store the center points of each box of the previous frame
center_points_prev_frame  = []

# Dictionary for object IDs and their center points
tracking_objects = {}

# Frame counter
frame_count = 0

# Loop for every frame of the video to run
while True:
    # Get one frame from the video
    # ret is true or false for frame existing
    ret, frame = cap.read()
    if not ret:
        break

    # Create an array to store the center points of each of the current frame
    center_points_curr_frame = []

    # Begin object IDs
    object_id = 1

    frame_count += 1

    # Detect objects on frame
    #   What object it is
    #   How confident the detection is
    #   Box border
    (class_ids, scores, boxes) = od.detect(frame)
    for box in boxes:
        # Get top left point x coordinate, y coordinate, width of box, height of box
        (x, y, w, h) = box

        # Make dot for center of box, create variables for center x and center y coordinates
        cx = int(x + 0.5*w)
        cy = int(y + 0.5*h)
        center_points_curr_frame.append((cx, cy))

        # Draw rectangle on the frame
        # With top left point and bottom right point
        # Green color
        # Thickness of 2
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    if frame_count <= 2:
        # Assign an ID to center dots that are close each other, meaning the dots are tracking the same object
        # I wish I didn't need to loop through every point in the previous frame array
        for point in center_points_curr_frame:
            for prev_point in center_points_prev_frame:
                # Calculate the distance between a dot of a current frame and previous frame
                distance = math.sqrt(abs(point[0] - prev_point[0])**2 + abs(point[1] - prev_point[1])**2)
                
                # When the distance is this small, they have to belong to the same object
                if distance < 20:
                    tracking_objects[object_id] = point
                    # Increase IDs for every new object
                    object_id += 1
    else:
        for id, original_point in tracking_objects.copy().items():
            # Create a flag to check if an object is no longer being updated = remove from tracking_objects dictionary
            object_tracked = False

            for point in center_points_curr_frame.copy():
                distance = math.sqrt(abs(point[0] - original_point[0])**2 + abs(point[1] - original_point[1])**2)

                # Update object ID position
                if distance < 20:
                    tracking_objects[id] = point
                    object_tracked = True
                    # Remove this point because it's being tracked so it will be shown -> this helps isolate the points not being tracked so we can create them later
                    if point in center_points_curr_frame:
                        center_points_curr_frame.remove(point)
            
            # Remove object ID as it's no longer being tracked
            if not object_tracked:
            #if distance == 0:
                tracking_objects.pop(id)

        # Add new object IDs for new center points being tracked
        for remaining_point in center_points_curr_frame:
            tracking_objects[object_id] = remaining_point
            object_id += 1


    # Draw ID and center points together
    for id, point in tracking_objects.items():
        cv2.circle(frame, point, 5, (0, 0, 255), -1)
        cv2.putText(frame, str(id), (point[0], point[1] - 5), 0, 1, (0, 0, 255), 2)

    # Update the previous frame array 
    center_points_prev_frame = center_points_curr_frame.copy()

    # Show the frame
    cv2.imshow("Frame", frame)
    # Frames change every 1 second
    key = cv2.waitKey(1)
    if key == 27: # escape key
        break

cap.release()
cv2.destroyAllWindows()