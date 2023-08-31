import cv2
import numpy as np
from object_detection import ObjectDetection
import math

# Initialize Object Detection
od = ObjectDetection()

# count line position
count_line_position = 350
offset = 2
counter = 0

# video location

cap = cv2.VideoCapture("C:/Users/91798/Documents/python/openCV/source_code/VID-20230418-WA0011.mp4")

# Initialize count
count = 0
center_points_prev_frame = []


tracking_objects = {}
track_id = 0
red_car_id = set([])

while True:
    ret, frame = cap.read()
    count += 1
    if not ret:
        break

    # Point current frame
    center_points_cur_frame = []
    center_points_red_frame = []
    #red_car_id = []
    #y = []
    # Detect objects on frame
    (class_ids, scores, boxes) = od.detect(frame)

    # Drawing  a line
    cv2.line(frame,(13,count_line_position),(445,count_line_position),(255,127,0),2)


    for box in boxes:
        (x, y, w, h) = box
        cx = int((x + x + w) / 2)
        cy = int((y + y + h) / 2)
        center_points_cur_frame.append((cx, cy))
        #print("FRAME N°", count, " ", x, y, w, h)
        # cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)


        # Get the HSV color of each pixel within the rectangle
        hsv_roi = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2HSV)
        # Set the range of color to detect (in this case, red)
        lower_red = np.array([0,70,50])
        upper_red = np.array([10,255,255])
        mask1 = cv2.inRange(hsv_roi, lower_red, upper_red)

        lower_red = np.array([170,70,50])
        upper_red = np.array([180,255,255])
        mask2 = cv2.inRange(hsv_roi, lower_red, upper_red)

        # Combine the masks to get the final mask for the red color
        mask = mask1 + mask2

        # Count the number of non-zero pixels in the mask within the rectangle
        masked_pixels = cv2.countNonZero(mask)
        total_pixels = w*h
        pixel_ratio = masked_pixels/float(total_pixels)

        if pixel_ratio > 0.05:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
            if ((cx,cy)) not in center_points_red_frame:
                if y<(count_line_position+offset) and y>(count_line_position-offset):
                    center_points_red_frame.append((cx,cy)) 
                    counter += 1
                    cv2.line(frame,(13, count_line_position), (445, count_line_position), (0,127,255), 3) 
                    #center_points_cur_frame.remove((x,y))
                    
                    #print('tracking id is :',track_id)
                    red_car_id.add(track_id)
                    
                    print(red_car_id)
                    print("Red Vehicle is detected : "+str(len(red_car_id)))
                    

    # Only at the beginning we compare previous and current frame
    if count <= 2:
        for pt in center_points_cur_frame:
            for pt2 in center_points_prev_frame:
                distance = math.hypot(pt2[0] - pt[0], pt2[1] - pt[1])

                if distance < 20:
                    tracking_objects[track_id] = pt
                    track_id += 1
    else:

        tracking_objects_copy = tracking_objects.copy()
        center_points_cur_frame_copy = center_points_cur_frame.copy()
        #red_car_id_copy = red_car_id.copy()

        for object_id, pt2 in tracking_objects_copy.items():
            object_exists = False
            for pt in center_points_cur_frame_copy:
                distance = math.hypot(pt2[0] - pt[0], pt2[1] - pt[1])

                # Update IDs position
                if distance < 20:
                    tracking_objects[object_id] = pt
                    object_exists = True
                    if pt in center_points_cur_frame:
                        center_points_cur_frame.remove(pt)
                    continue

                
            # Remove IDs lost
            if not object_exists:
                tracking_objects.pop(object_id)

        # Add new IDs found
        for pt in center_points_cur_frame:
            tracking_objects[track_id] = pt
            track_id += 1


    

    for object_id, pt in tracking_objects.items():
        cv2.circle(frame, pt, 5, (0, 0, 255), -1)
        cv2.putText(frame, str(object_id), (pt[0], pt[1] - 7), 0, 1, (0, 0, 255), 2)

    # print("Tracking objects")
    # print(tracking_objects)


    # print("CUR FRAME LEFT PTS")
    # print(center_points_cur_frame)

    cv2.putText(frame, " Red Vehicle Count : "+str(len(red_car_id)), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255),5)

    cv2.imshow("Original video", frame)

    # Make a copy of the points
    center_points_prev_frame = center_points_cur_frame.copy()
    

    key = cv2.waitKey(1)
    if key == 27:
        break

    

cap.release()
cv2.destroyAllWindows()
