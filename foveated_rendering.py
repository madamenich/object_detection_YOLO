import cv2
import numpy as np
from ultralytics import YOLO

from PIL import Image
# Load the image
img = cv2.imread("s.png")

# Get the height and width of the image
h, w = img.shape[:2]

# Define the number of rows and columns
rows = 4
cols = 4

# Draw the horizontal lines
for i in range(1, rows):
    y = i * h // rows
    cv2.line(img, (0, y), (w, y), (0, 0, 255), 2)

# Draw the vertical lines
for j in range(1, cols):
    x = j * w // cols
    cv2.line(img, (x, 0), (x, h), (0, 0, 255), 2)



# Define the zoom factor for the hovered grid
zoom_factor = 2
# Define a variable to store the zoomed-in image
zoomed_in_img = None

# Define a function to handle mouse events
def mouse_event(event, x, y, flags, param):
    # If the left button is clicked
    if event == cv2.EVENT_MOUSEMOVE:
        # Calculate the grid index based on the mouse position
        row_index = y // (h // rows)
        col_index = x // (w // cols)
                # Create a copy of the original image
        blurred_img = img.copy()
        # Blur the non-hovered grids with Gaussian blur
        for i in range(rows):
            for j in range(cols):
                if i != row_index or j != col_index:
                    grid_x1 = j * w // cols
                    grid_x2 = (j + 1) * w // cols
                    grid_y1 = i * h // rows
                    grid_y2 = (i + 1) * h // rows
                    blurred_img[grid_y1:grid_y2, grid_x1:grid_x2] = cv2.GaussianBlur(
                        img[grid_y1:grid_y2, grid_x1:grid_x2], (11, 11), 0
                    )


        # Zoom in on the hovered grid
        grid_x1 = col_index * w // cols
        grid_x2 = (col_index + 1) * w // cols
        grid_y1 = row_index * h // rows
        grid_y2 = (row_index + 1) * h // rows
        blurred_img[grid_y1:grid_y2, grid_x1:grid_x2] = img[grid_y1:grid_y2, grid_x1:grid_x2]
        
        # Apply blending effect
        alpha = 0.7  # You can adjust this value to control the blending effect
        blended_img = cv2.addWeighted(img, alpha, blurred_img, 1 - alpha, 0)
        
        # Show the blended image
        # cv2.imshow("Grid", blended_img)
        global zoomed_in_img
        # Save the zoomed-in image for object detection
        zoomed_in_img = img[grid_y1:grid_y2, grid_x1:grid_x2]
        model = YOLO('yolov8n.pt')
        results = model(zoomed_in_img)
        zoomed_in_img = results[0].plot()
        blurred_img[grid_y1:grid_y2, grid_x1:grid_x2] = zoomed_in_img
        



        
        # cv2.imshow("Zoom", zoomed_in_img)

        # cv2.imshow("Grid", blended_img)
        cv2.imshow("Grid", blurred_img)
# Set the mouse callback function for the window
cv2.namedWindow("Grid")
cv2.setMouseCallback("Grid", mouse_event)
# Show the image
cv2.imshow("Grid", img)
cv2.waitKey(0)

cv2.destroyAllWindows()


#Save the zoomed-in image for object detection (if it exists)
# if zoomed_in_img is not None:
#     cv2.imwrite("zoom_in_part.png", zoomed_in_img)
#     model = YOLO("yolov8n.pt")
#     # Load the image
#     model.predict("zoom_in_part.png", save=True, conf = 0.5)
