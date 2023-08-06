import cv2
import numpy as np
import argparse
from dynamicgrid import grid_display
from zoomin import zoom_in_selected_region



# Define the callback function for mouse events
def foveated_rendering(event, x, y, flags, param):
    global img, fovea_size, blur_level
    # If left mouse button is clicked
    if event == cv2.EVENT_LBUTTONDOWN:
        # Copy the original image
        output = img.copy()
        # Get the height and width of the image
        h, w = img.shape[:2]
        # Create a mask with the same size as the image
        mask = np.zeros((h, w), dtype=np.uint8)
        # Draw a white circle on the mask centered at the mouse position with the fovea size as radius
        cv2.circle(mask, (x, y), fovea_size, 255, -1)
        # Blur the output image with the blur level
        output = cv2.GaussianBlur(output, (blur_level, blur_level), 0)
        # Copy the original image pixels inside the mask region to the output image
        output[mask > 0] = img[mask > 0]

        #Define the red color for the foveaated region
        cv2.circle(output, (x, y), fovea_size, (0, 0, 255), 3)
        print("x: ",x,"y: ",y)
        # Show the output image
        cv2.imshow("Foveated Image Rendering", output)
        cv2.imwrite("foveated.png",output)
        return 'foveated.png'
# Load the image
img = cv2.imread("funny_cats.jpg")
n_grid = 4
grid_img =cv2.imread(grid_display(n_grid,img))

# Define the fovea size (in pixels) and the blur level (odd number)
fovea_size = 100
blur_level = 21
# Create a window to display the image
cv2.namedWindow("Foveated Image Rendering")

# Grid Display working here
# n_grid = 4
# grid_display(n_grid,img)

# Set the mouse callback function for the window
cv2.setMouseCallback("Foveated Image Rendering", foveated_rendering)

#set Zoom Image goes here
zoom_in_selected_region(grid_img, 100, 100, 300, 300, 2.0)
# Show the original image
# cv2.imshow("Foveated Image Rendering", grid_img)


# Wait for a key press to exit
cv2.waitKey(0)
# Destroy all windows
cv2.destroyAllWindows()
