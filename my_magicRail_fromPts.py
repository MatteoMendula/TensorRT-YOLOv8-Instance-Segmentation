import numpy as np
import cv2
import onnxruntime as ort
from ultralytics import YOLO
import utils

model = YOLO('yolov8s-seg.pt')  # load an official model
# image = cv2.imread('bus3.jpg')
image = cv2.imread('original_bus_832x1088.jpg')
# get height and width of the image
desired_image_size = (832, 1088)
desired_image_size_resize = (1088, 832)

# Export the model
model.export(format='onnx', imgsz=desired_image_size)

# Load YOLOv8s-seg model
model_path = 'yolov8s-seg.onnx'
ort_session = ort.InferenceSession(model_path)

input_image, resized_image = utils.preprocess_image(image)

# Load the image
# resized_image = cv2.resize(image, desired_image_size_resize)
# input_image = resized_image.astype(np.float32) / 255.0
# input_image = np.transpose(input_image, [2, 0, 1])  # Change image layout to CHW (Channels First)
# input_image = np.expand_dims(input_image, axis=0)  # Add batch dimension

# Perform inference
outputs = ort_session.run(None, {'images': input_image})
output_0=np.squeeze(outputs[0])
_prototypes=np.squeeze(outputs[1])

l_mask,l_class,l_conf,l_boxes=utils.extract_masks(output_0,_prototypes,desired_image_size)

print("l_mask.len:", len(l_mask))

utils.show_masks(l_mask,l_class,l_conf,l_boxes, resized_image, should_save=True, classes=utils.CLASSES)
cv2.destroyAllWindows()