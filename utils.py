import numpy as np
from skimage.transform import resize

def nms(bounding_boxes, threshold):
    # If no bounding boxes, return empty list
    if len(bounding_boxes) == 0:
        return []

    # Bounding boxes
    boxes = np.array(bounding_boxes)
    # Extracting coordinates
    x_mid = boxes[:, 0]
    y_mid = boxes[:, 1]
    width = boxes[:, 2]
    height = boxes[:, 3]
    
    # Calculating start and end coordinates
    start_x = x_mid - width / 2
    start_y = y_mid - height / 2
    end_x = x_mid + width / 2
    end_y = y_mid + height / 2
    score=boxes[:, 4]


    # Picked bounding boxes
    picked_boxes = []
    picked_score = []

    # Compute areas of bounding boxes
    areas = (end_x - start_x + 1) * (end_y - start_y + 1)

    # Sort by confidence score of bounding boxes
    order = np.argsort(score)

    # Iterate bounding boxes
    while order.size > 0:
        # The index of largest confidence score
        index = order[-1]

        # Pick the bounding box with largest confidence score
        picked_boxes.append(bounding_boxes[index])

        # Compute ordinates of intersection-over-union(IOU)
        x1 = np.maximum(start_x[index], start_x[order[:-1]])
        x2 = np.minimum(end_x[index], end_x[order[:-1]])
        y1 = np.maximum(start_y[index], start_y[order[:-1]])
        y2 = np.minimum(end_y[index], end_y[order[:-1]])

        # Compute areas of intersection-over-union
        w = np.maximum(0.0, x2 - x1 + 1)
        h = np.maximum(0.0, y2 - y1 + 1)
        intersection = w * h

        # Compute the ratio between intersection and union
        ratio = intersection / (areas[index] + areas[order[:-1]] - intersection)

        left = np.where(ratio < threshold)
        order = order[left]

    return np.array(picked_boxes)
def crop_image(image, box):
    x, y, w, h = box.astype(int)
    
    # Ensure box coordinates are within image boundaries
    x1 = max(0, x - w // 2)
    y1 = max(0, y - h // 2)
    x2 = min(image.shape[1], x + w // 2)
    y2 = min(image.shape[0], y + h // 2)
    
    # Create a mask to zero out areas outside the box
    mask = np.zeros_like(image, dtype=np.float32)
    mask[y1:y2, x1:x2] = 1
    
    # Apply the mask to the original image
    cropped_image = image * mask
    
    return cropped_image
def threshold_image(image,threshold=0.1):
    thresholded_image = np.where(image > threshold, 255,0)
    return thresholded_image

def extract_masks(output_0,
                  prototypes,
                  input_size,
                  threshold_detection=0.2,
                  theshold_iou=0.5,
                  threshold_mask=0.1):
    nb_class=output_0.shape[0]-4-prototypes.shape[0]
    l_class=[[] for k in range(nb_class)]
    output_0_T=output_0.T
    for detection in output_0_T:
        conf=detection[4:nb_class+4]
        max_conv=np.max(conf)
        argmax_conv=np.argmax(conf)
        if(max_conv>threshold_detection):
            l_class[argmax_conv].append(np.concatenate((detection[:4], np.array([max_conv]),detection[4+nb_class:])))
    l_class_NMS=[]  
    for clas in l_class:
        l_class_NMS.append(nms(clas,theshold_iou)) 
    
    l_mask=[]
    l_class=[]
    l_conf=[]
    for k in range(len(l_class_NMS)):
        for detection in l_class_NMS[k]:
            coeff=detection[5:]
            mask=prototypes*coeff.reshape(prototypes.shape[0],1,1)
    
            # Initialize an empty array to store resized images
            resized_mask = np.empty((mask.shape[0], input_size[0], input_size[0]))
            
            # Resize each image in the array
            for i, image in enumerate(mask):
                resized_mask[i] = resize(image, input_size, anti_aliasing=True)
            l_mask.append(threshold_image(crop_image(np.mean(resized_mask, axis=0),detection[:4]),threshold_mask))
            l_class.append(k)
            l_conf.append(detection[4])
    return l_mask,l_class,l_conf


classes={0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus',
         6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant',
         11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat',
         16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear',
         22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag',
         27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard',
         32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove',
         36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle',
         40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon',
         45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange',
         50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut',
         55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed',
         60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse',
         65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven',
         70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock',
         75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}