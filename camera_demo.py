import numpy as np
import cv2
import onnxruntime as ort
from ultralytics import YOLO
import utils

model = YOLO('yolov8s-seg.pt') 
# get height and width of the image
# desired_image_size = (640, 480)
desired_image_size = (480, 640)

# Export the model
model.export(format='onnx', imgsz=desired_image_size)

# Load YOLOv8s-seg model
model_path = 'yolov8s-seg.onnx'
ort_session = ort.InferenceSession(model_path)

capture = cv2.VideoCapture(0, cv2.CAP_V4L2)
capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
width = desired_image_size[0]
height = desired_image_size[1]
capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

# exit loop if 'q' is pressed

while True:
    # capture only one frame
    ret, image = capture.read()
    input_image, resized_image = utils.preprocess_image(image)

    start_time = cv2.getTickCount()

    outputs = ort_session.run(None, {'images': input_image})

    time = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()

    output_0=np.squeeze(outputs[0])
    _prototypes=np.squeeze(outputs[1])
    l_mask,l_class,l_conf,l_boxes=utils.extract_masks(output_0,_prototypes,desired_image_size)

    # print("l_mask.len:", len(l_mask))

    masked_image = utils.show_masks(l_mask,l_class,l_conf,l_boxes, resized_image, classes=utils.CLASSES, fps = 1/time)
    cv2.imshow('TRT Yolov8-seg', masked_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()


# Load the image
# resized_image = cv2.resize(image, desired_image_size_resize)
# input_image = resized_image.astype(np.float32) / 255.0
# input_image = np.transpose(input_image, [2, 0, 1])  # Change image layout to CHW (Channels First)
# input_image = np.expand_dims(input_image, axis=0)  # Add batch dimension

# Perform inference


