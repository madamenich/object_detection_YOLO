import cv2
import numpy as np
from ultralytics import YOLO
from argparse import ArgumentParser
from PIL import Image
from utils import *
import xml.etree.ElementTree as ET
# Load the image

img = cv2.imread("image.png")
ground_boxes = load_ground_truth(img,"ground_truth.xml")
h, w,rows,cols = apply_grid(img, num_grid=4)

print('grdound',ground_boxes)

# Define true positives, false positives, and false negatives
tp = 0
fp = 0
fn = 0


# -----------------------------------------------------------------

prediction_box =[]



# Define the zoom factor for the hovered grid
zoom_factor = 1
# Define a variable to store the zoomed-in image
zoomed_in_img = None
# Define a global variable to store the current cell
current_cell = None
last_predicted_cell = None
prediction_box = []

model = YOLO('yolov8n.pt')
blurred_img = apply_denoise_and_blur(img, blur=21)

import torch
device = '0' if torch.cuda.is_available() else 'cpu'

print('Running on %s' % device)

# Define a function to handle mouse events
def mouse_event(event, x, y, flags, param):
    e1 = cv2.getTickCount()
    # Access the global variable
    global current_cell, last_predicted_cell, prediction_box, tp, fp, fn

    if event == cv2.EVENT_MOUSEMOVE:
        # Calculate the grid index based on the mouse position
        row_index = y // (h // rows)
        col_index = x // (w // cols)

        # Check if the mouse is still in the same cell
        if (col_index, row_index) == current_cell:
            return
        # Update the last predicted cell
        last_predicted_cell = current_cell
        # Update the current cell
        current_cell = (col_index, row_index)


         
        # Blur the previously predicted cell (if any)
        if last_predicted_cell:
            prev_col, prev_row = last_predicted_cell
            prev_grid_x1 = prev_col * w // cols
            prev_grid_x2 = (prev_col + 1) * w // cols
            prev_grid_y1 = prev_row * h // rows
            prev_grid_y2 = (prev_row + 1) * h // rows
            blurred_img[prev_grid_y1:prev_grid_y2, prev_grid_x1:prev_grid_x2] = cv2.GaussianBlur(
                img[prev_grid_y1:prev_grid_y2, prev_grid_x1:prev_grid_x2], (21, 21), 0)
        # Zoom in on the hovered grid
        grid_x1 = col_index * w // cols
        grid_x2 = (col_index + 1) * w // cols
        grid_y1 = row_index * h // rows
        grid_y2 = (row_index + 1) * h // rows
        blurred_img[grid_y1:grid_y2, grid_x1:grid_x2] = img[grid_y1:grid_y2, grid_x1:grid_x2]

        if current_cell != last_predicted_cell:
            global zoomed_in_img
            zoomed_in_img = img[grid_y1:grid_y2, grid_x1:grid_x2]
            height, width= zoomed_in_img.shape[:2]
            #TODO: resize the image to the input size of the model and make a prediction
            zoomed_in_img = cv2.resize(zoomed_in_img, (width*zoom_factor, height*zoom_factor), interpolation=cv2.INTER_NEAREST)
            results = model.predict(zoomed_in_img, classes=0, imgsz=width, device=device, augment =True)
            #TODO: draw the bounding boxes on the zoomed-in image
            zoomed_in_img = results[0].plot()
            zoomed_in_img = cv2.resize(zoomed_in_img, (width, height), interpolation=cv2.INTER_NEAREST)
            blurred_img[grid_y1:grid_y2, grid_x1:grid_x2] = zoomed_in_img


            boxes = results[0].boxes.xyxy.cpu().numpy()
            print('boxes',boxes)
            # for box in boxes:
            #     prediction_box.append(box)
            cv2.imshow("Grid", blurred_img)
            cv2.imwrite(f"fvresult{cols}.png", blurred_img)
            #TODO: calculate the IoU between the predicted bounding boxes and the ground truth bounding boxes
            #Step1 scale the ground truth boxes
            scale_factorx = blurred_img.shape[1]/zoomed_in_img.shape[1]
            scale_factory = blurred_img.shape[0]/zoomed_in_img.shape[0]
            scaled_boxes = []
            for box in boxes:
                scaled_boxes.append([box[0]*scale_factorx,box[1]*scale_factory,box[2]*scale_factorx,box[3]*scale_factory])
            print('scaled_ground_boxes',scaled_boxes)
            for true_box in ground_boxes:
                for box in scaled_boxes:
                    print('true_box',true_box)
                    print('box',box)
                    calculate_iou(true_box,box)
                    
                  



# Set the mouse callback function for the window
cv2.namedWindow("Grid")
cv2.setMouseCallback("Grid", mouse_event)


# Show the image
cv2.imshow("Grid", cv2.GaussianBlur(img, (21, 21), 0))
cv2.waitKey(0)
cv2.destroyAllWindows()


