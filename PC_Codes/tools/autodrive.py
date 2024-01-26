import scipy.ndimage as ndimage
import numpy as np
import cv2
import socket
import random
from time import sleep

class AutoDriveLaneDetection:
    
    def __init__(self, jetson_nano_ip, jetson_nano_port):
        self.jetson_ip = jetson_nano_ip
        self.jetson_port = jetson_nano_port
        self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send_to_jetson(self, char_value):
        message = char_value.encode()
        self.udp_socket.sendto(message, (self.jetson_ip, self.jetson_port))

    def calculate_direction(self, img_width, tracker_center_x, range_factor=3):
        # Calculate the range for mapping
        mapping_range = img_width // range_factor

        # Calculate the center of the image
        image_center_x = img_width // 2

        # Calculate the direction from the center of the image to the tracker
        direction = tracker_center_x - image_center_x

        # Map the direction onto 7 points
        if direction < -mapping_range:
            return 'a', "Far left"
        elif -mapping_range <= direction < -mapping_range / 2:
            return 'b', "Left"
        elif -mapping_range / 2 <= direction < 0:
            return 'c', "Slightly left"
        elif direction == 0:
            return 'd', "Center"
        elif 0 < direction <= mapping_range / 2:
            return 'e', "Slightly right"
        elif mapping_range / 2 < direction <= mapping_range:
            return 'f', "Right"
        elif direction > mapping_range:
            return 'g', "Far right"
    def plot_one_box(self, x, img, color=None, label=None, line_thickness=None, roi=(125, 275,355, 320)):
        tl = line_thickness or round(0.0001 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
        color = color or [random.randint(0, 255) for _ in range(3)]
        c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
        
        # Check if the bounding box intersects with the ROI
        roi_x1, roi_y1, roi_x2, roi_y2 = roi
        if c2[0] > roi_x1 and c2[1] > roi_y1 and c1[0] < roi_x2 and c1[1] < roi_y2:
            self.send_to_jetson('s')
            cv2.putText(img, f"Stop! Detected Object: {label}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)
        # Draw bounding box on the original image
        cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
        if label:
            tf = max(tl - 1, 1)  # font thickness
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    

    def auto_drive(self, img, result):

        
        color_area = np.zeros((result[0].shape[0], result[0].shape[1], 3), dtype=np.uint8)

        # Road segmentation (result[0])
        color_area[result[0] == 1] = [0, 0, 0]

        # Lane segmentation (result[1])
        lane_segments = result[1]

        # Find contours in the lane segment
        contours, _ = cv2.findContours(lane_segments, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw 9 lines like a ruler at the mid-bottom of the image
        ruler_line_count = 7
        ruler_line_height = img.shape[0] - 1
        ruler_line_spacing = 50  # Adjust the spacing between lines
        ruler_line_start_x = img.shape[1] // 2 - (ruler_line_count - 1) * ruler_line_spacing // 2

        for i in range(ruler_line_count):
            line_x = ruler_line_start_x + i * ruler_line_spacing
            line_length = 50 if i == 3 else 30  # Set the length to 50 for the third line, 10 for others
            cv2.line(color_area, (line_x, ruler_line_height), (line_x, ruler_line_height - line_length), [255, 0, 255], thickness=2)

        
        # If lanes are detected, draw them
        if len(contours) > 0:
            # Sort contours by x-coordinate of their centroid
            contours = sorted(contours, key=lambda x: cv2.minAreaRect(x)[0][0])
            direction_mapping = ''
            key = ''

            # Assign colors to lanes based on their index
            for i, contour in enumerate(contours):
                # Identify if it's the left or right lane
                color = [0, 0, 255] if i % 2 == 0 else [255, 0, 0]

                # Draw the lane on the color_area
                cv2.drawContours(color_area, [contour], -1, color, thickness=cv2.FILLED)

            # Calculate the center between the two lanes
            if len(contours) == 2:
                left_centroid = cv2.minAreaRect(contours[0])[0]
                right_centroid = cv2.minAreaRect(contours[1])[0]
                center_x = int((left_centroid[0] + right_centroid[0]) / 2)

                # Draw a line or marker at the center between the two lanes
                tracker_line_length = 100
                cv2.line(color_area, (center_x, color_area.shape[0] - 1), (center_x, color_area.shape[0] - 1 - tracker_line_length), [255, 255, 255], thickness=2)

                # Calculate the direction and map it onto 7 points
                key, direction_mapping = self.calculate_direction(img.shape[1], center_x)
                cv2.putText(img, f"Direction: {direction_mapping}", (img.shape[1] - 180, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            # If there's only one lane
            elif len(contours) == 1:
                # Calculate the center of the single lane
                single_lane_centroid = cv2.minAreaRect(contours[0])[0]
                center_x = int(single_lane_centroid[0])

                # Assign colors based on the position relative to the midpoint
                if center_x < img.shape[1] // 2:
                    color = [0, 0, 255]  # Blue for left lane
                    center_x = center_x + 160
                else:
                    color = [255, 0, 0]  # Red for right lane
                    center_x = center_x - 160

                # Draw the lane on the color_area
                cv2.drawContours(color_area, [contour], -1, color, thickness=cv2.FILLED)

                # Draw a line or marker at the adjusted center
                tracker_line_length = 100
                cv2.line(color_area, (center_x, color_area.shape[0] - 1), (center_x, color_area.shape[0] - 1 - tracker_line_length), [255, 255, 255], thickness=2)

                # Calculate the direction and map it onto 7 points
                key, direction_mapping = self.calculate_direction(img.shape[1], center_x)
                cv2.putText(img, f"Direction: {direction_mapping}", (img.shape[1] - 180, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                
            #Send character to jetson
            self.send_to_jetson(key)
            print(key)
        else:
            self.send_to_jetson('s')
        # Convert to BGR
        color_area = color_area[..., ::-1]

        # Resize color_area to match the input image size
        color_area = cv2.resize(color_area, (img.shape[1], img.shape[0]))

        # Blend the original image with the colored regions
        img = cv2.addWeighted(img, 1, color_area, 0.5, 0)

        # Resize the final image
        img = cv2.resize(img, (480, 320), interpolation=cv2.INTER_LINEAR)
        
        return img
    
    @staticmethod
    def process_lines(ll_seg_mask):
        image_height, image_width = ll_seg_mask.shape

        # Create a binary mask for the 1/3 bottom part crosswise
        roi_mask = np.zeros_like(ll_seg_mask)
        roi_mask[image_height * 2 // 3:, :] = 1

        # Apply the mask to the original image or ll_seg_mask
        ll_seg_mask = ll_seg_mask * roi_mask

        # Calculate the center of mass for each labeled region
        labels, num_labels = ndimage.label(ll_seg_mask)
        centers = ndimage.center_of_mass(roi_mask, labels, range(1, num_labels + 1))

        # Separate the labels into left and right based on the image's midpoint
        midpoint = image_width // 2
        left_labels = [label for label in range(1, num_labels + 1) if centers[label - 1][1] < midpoint]
        right_labels = [label for label in range(1, num_labels + 1) if centers[label - 1][1] >= midpoint]

        # Keep only one lane line on the left and one on the right
        selected_labels = []
        if left_labels:
            left_distances = [centers[label - 1][1] for label in left_labels]
            selected_labels.append(left_labels[np.argmax(left_distances)])
        if right_labels:
            right_distances = [centers[label - 1][1] for label in right_labels]
            selected_labels.append(right_labels[np.argmin(right_distances)])

        # Keep only the selected labels in the mask
        ll_seg_mask[~np.isin(labels, selected_labels)] = 0

        return ll_seg_mask
    
    