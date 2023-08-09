import cv2
import numpy as np
from ultralytics import YOLO
from argparse import ArgumentParser
from PIL import Image
# Load the image
img = cv2.imread("image.png")


#TODO: Applying Grid on IMAGE
# -----------------------------------------------------------------
# Get the height and width of the image
h, w = img.shape[:2]

# Define the number of rows and columns
rows = 4
cols = 4

# Draw the horizontal lines
for i in range(1, rows):
    y = i * h // rows
    cv2.line(img, (0, y), (w, y), (0, 0, 255), 1)

# Draw the vertical lines
for j in range(1, cols):
    x = j * w // cols
    cv2.line(img, (x, 0), (x, h), (0, 0, 255), 1)

# -----------------------------------------------------------------





# Define the zoom factor for the hovered grid
zoom_factor = 1
# Define a variable to store the zoomed-in image
zoomed_in_img = None
# Define a global variable to store the current cell
current_cell = None
last_predicted_cell = None

model = YOLO('yolov8n.pt')
img = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)

blurred_img = img.copy()

blurred_img = cv2.GaussianBlur(img, (21, 21), 0)  # Apply Gaussian blur to the whole image
# img  = cv2.GaussianBlur(img, (21, 21), 0)
import torch
device = '0' if torch.cuda.is_available() else 'cpu'

print('Running on %s' % device)

# Define a function to handle mouse events
def mouse_event(event, x, y, flags, param):
    # Access the global variable
    global current_cell, last_predicted_cell

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
        print("current cell: ", current_cell)
         # Blur the non-hovered grids with Gaussian blur
         
 # Blur the previously predicted cell (if any)
        if last_predicted_cell:
            prev_col, prev_row = last_predicted_cell
            prev_grid_x1 = prev_col * w // cols
            prev_grid_x2 = (prev_col + 1) * w // cols
            prev_grid_y1 = prev_row * h // rows
            prev_grid_y2 = (prev_row + 1) * h // rows
            blurred_img[prev_grid_y1:prev_grid_y2, prev_grid_x1:prev_grid_x2] = cv2.GaussianBlur(
                img[prev_grid_y1:prev_grid_y2, prev_grid_x1:prev_grid_x2], (21, 21), 0
            )
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
            zoomed_in_img = cv2.resize(zoomed_in_img, (width*zoom_factor, height*zoom_factor), interpolation=cv2.INTER_NEAREST)
            results = model.predict(zoomed_in_img, classes=0, imgsz=width, boxes=False, device=device)
            zoomed_in_img = results[0].plot()
            zoomed_in_img = cv2.resize(zoomed_in_img, (width, height), interpolation=cv2.INTER_NEAREST)
            blurred_img[grid_y1:grid_y2, grid_x1:grid_x2] = zoomed_in_img
            cv2.imshow("Grid", blurred_img)


# Set the mouse callback function for the window
cv2.namedWindow("Grid")
cv2.setMouseCallback("Grid", mouse_event)
# Show the image
cv2.imshow("Grid", cv2.GaussianBlur(img, (21, 21), 0))
cv2.waitKey(0)

cv2.destroyAllWindows()