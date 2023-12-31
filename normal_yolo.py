from PIL import Image
from ultralytics import YOLO
import xml.etree.ElementTree as ET
import cv2
# Load the XML annotation file generated by LabelImg
# xml_file = "image.xml"  # Replace with the path to your XML file
# tree = ET.parse(xml_file)
# root = tree.getroot()

# Load the image
image_path = "image.png"  # Replace with the path to your image
image = cv2.imread(image_path)
# # Iterate through the XML file to get bounding box coordinates and labels
# for obj in root.findall('object'):
#     # Get the class label (assuming it's under the 'name' tag)
#     label = obj.find('name').text

#     # Get the bounding box coordinates
#     bbox = obj.find('bndbox')
#     xmin = int(bbox.find('xmin').text)
#     ymin = int(bbox.find('ymin').text)
#     xmax = int(bbox.find('xmax').text)
#     ymax = int(bbox.find('ymax').text)

#     # Draw the bounding box on the image
#     image=cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
#     image=cv2.putText(image, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

# Load a pretrained YOLOv8n model
model = YOLO('yolov8n.pt')

# Run inference on 'bus.jpg'
results = model(image,classes =0)  # results list
# Show the results
for r in results:
    print(r)  # print raw predictions
    # r.save_txt('test.txt')  # save as results.txt
    im_array = r.plot()  # plot a BGR numpy array of predictions
    im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
    im.show()  # show image
    im.save('YOLO_Res.png')  # save image
