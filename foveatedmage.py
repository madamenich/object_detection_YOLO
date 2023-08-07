import cv2
import numpy as np
import torch
from ultralytics import YOLO 
# Load the image
img = cv2.imread('s.png')
# Get the height and width of the image
h, w = img.shape[:2]

# Define the grid size
grid_size = 4

# Calculate the step size for each grid cell
step_h = h // grid_size
step_w = w // grid_size

# Define the maximum blur kernel size
max_kernel = 31


# Define the input size of Yolov8
input_size = 640


# Define the mouse callback function
def mouse_callback(event, x, y, flags, param):
    global img, grid_size, step_h, step_w, max_kernel

    # Check if the left mouse button is pressed
    if event == cv2.EVENT_MOUSEMOVE:
        # Copy the original image
        img_copy = img.copy()

        # Apply a Gaussian blur to the image copy
        img_blur = cv2.GaussianBlur(img_copy, (max_kernel, max_kernel), 0)

        # Find the grid cell that contains the mouse position
        i = y // step_h
        j = x // step_w
        # Get the coordinates of the cell
        x1 = j * step_w
        y1 = i * step_h
        x2 = (j + 1) * step_w
        y2 = (i + 1) * step_h

        # Create a mask with the cell as white and the rest as black
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

        # Extract the cell region from the original image using the mask
        img_cell = cv2.bitwise_and(img, img, mask=mask)
        # Calculate the new height and width after zooming
        new_h = img_cell.shape[0] * 2
        new_w = img_cell.shape[1] * 2

        img_zoom = cv2.resize(img_cell, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        model = YOLO('yolov8n.pt')
        results = model(img_zoom)
        for result in results:                                         # iterate results
                boxes = result.boxes.cpu().numpy()                         # get boxes on cpu in numpy
                for box in boxes:                                          # iterate boxes
                        r = box.xyxy[0].astype(int)                            # get corner points as int
                        print(r)                                               # print boxes
                        img_cell= cv2.rectangle(img_cell, r[:2], r[2:], (255, 0, 0), 2)   # draw boxes on img
    

        # Extract the non-cell region from the blurred image using the inverted mask
        mask_inv = cv2.bitwise_not(mask)
        img_non_cell = cv2.bitwise_and(img_blur, img_blur, mask=mask_inv)


        # Combine the cell and non-cell regions
        img_res = cv2.add(img_cell, img_non_cell)


        # Show the result
        cv2.imshow('Image', img_res)





# Create a window and set the mouse callback function
cv2.namedWindow('Image')
cv2.setMouseCallback('Image', mouse_callback)

# Show the original image with grid
for i in range(grid_size):
    for j in range(grid_size):
        # Get the coordinates of the current cell
        x1 = j * step_w
        y1 = i * step_h
        x2 = (j + 1) * step_w
        y2 = (i + 1) * step_h

        # Draw a rectangle around the cell
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
cv2.imshow('Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
