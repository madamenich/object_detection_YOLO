import cv2

def zoom_in_selected_region(image_path, top_left_x, top_left_y, bottom_right_x, bottom_right_y, zoom_factor):
    # Read the image
    # img = cv2.imread(image_path)
    img = image_path

    # Define the region of interest (ROI) coordinates
    x1, y1 = top_left_x, top_left_y
    x2, y2 = bottom_right_x, bottom_right_y

    # Extract the selected region from the image
    roi = img[y1:y2, x1:x2]

    # Calculate the new size for the zoomed region
    new_width = int((x2 - x1) * zoom_factor)
    new_height = int((y2 - y1) * zoom_factor)

    # Resize the selected region to the new size
    zoomed_roi = cv2.resize(roi, (new_width, new_height))

    # Create a canvas with the size of the selected region
    canvas = img.copy()

    # Place the zoomed region back on the canvas
    canvas[y1:y1 + new_height, x1:x1 + new_width] = zoomed_roi
    
  
    #Display the zoomed region
    cv2.imshow("Zoomed Region", canvas)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

# Example usage:
# image_path = "c.jpg"
# top_left_x, top_left_y = 100, 100  # Coordinates of the top-left corner of the ROI
# bottom_right_x, bottom_right_y = 300, 300  # Coordinates of the bottom-right corner of the ROI
# zoom_factor = 2.0  # Zoom factor (2.0 means 2x zoom)

# zoom_in_selected_region(image_path, top_left_x, top_left_y, bottom_right_x, bottom_right_y, zoom_factor)
