import cv2
import xml.etree.ElementTree as ET

def calculate_iou(box1, box2):
    # Box format: (x1, y1, x2, y2), where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner.
    
    # Calculate the coordinates of the intersection rectangle
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # Calculate the area of intersection
    intersection_area = max(0, (x2 - x1 + 1)) * max(0, (y2 - y1 + 1))
    
    print('intersection_area',intersection_area)
    
    # Calculate the area of both bounding boxes
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
    
    # Calculate the Union area
    union_area = float(box1_area + box2_area - intersection_area)
    # print('union_area',union_area)
    
    # Calculate IoU
    iou = intersection_area / union_area
    
    
    return iou



def apply_denoise_and_blur(img, blur=21):
    img = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)
    blurred_img = img.copy()

    blurred_img = cv2.GaussianBlur(img, (blur, blur), 0)  # Apply Gaussian blur to the whole image
    return blurred_img


def apply_grid(img,num_grid=4):
    h, w = img.shape[:2]

    # Define the number of rows and columns
    rows = cols = num_grid

    # Draw the horizontal lines
    for i in range(1, rows):
        y = i * h // rows
        cv2.line(img, (0, y), (w, y), (0, 0, 255), 1)

    # Draw the vertical lines
    for j in range(1, cols):
        x = j * w // cols
        cv2.line(img, (x, 0), (x, h), (0, 0, 255), 1)    
    return h,w,rows,cols
annotations = []
def load_ground_truth(img,xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    for obj in root.findall('object'):
        # Get the class label (assuming it's under the 'name' tag)
        label = obj.find('name').text

        # Get the bounding box coordinates
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)
        annotations.append((xmin, ymin, xmax, ymax))
    for xmin, ymin, xmax, ymax in annotations:
        # Draw the bounding box on the image
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.putText(img, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    return annotations

img_object ={}
def first_load(img,xml_file,num_grid=4,blur=21):
    annotations = load_ground_truth(img,xml_file)
    h, w,rows,cols = apply_grid(img, num_grid)
    blurred_img = apply_denoise_and_blur(img, blur)
    # img_object.popitem('annotations',annotations)
    # img_object['annotations'] = annotations
    # img_object['h'] = h
    # img_object['w'] = w
    # img_object['rows'] = rows
    # img_object['cols'] = cols
    # img_object['blurred_img'] = blurred_img
    return img_object
    # return annotations,h,w,rows,cols,blurred_img