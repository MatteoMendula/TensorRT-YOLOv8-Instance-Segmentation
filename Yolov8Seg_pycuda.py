import numpy as np
import os
import cv2
import warnings
from collections import namedtuple
from pathlib import Path
from typing import List, Optional, Tuple, Union
import argparse

import numpy as np
import pycuda.autoinit  # noqa F401
import pycuda.driver as cuda
import tensorrt as trt
from numpy import ndarray

os.environ['CUDA_MODULE_LOADING'] = 'LAZY'
warnings.filterwarnings(action='ignore', category=DeprecationWarning)

CLASSES = ['binary']

# run with: python3 Yolov8Seg_pycuda.py --engine_path yolov8s_rail_seg_best.engine --image_path resize1280x960_railway.jpg --save_output True

class TRTEngine:

    def __init__(self, weight: Union[str, Path]) -> None:
        self.weight = Path(weight) if isinstance(weight, str) else weight
        self.stream = cuda.Stream(0)
        self.__init_engine()
        self.__init_bindings()
        self.__warm_up()

    def __init_engine(self) -> None:
        logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(logger, namespace='')
        with trt.Runtime(logger) as runtime:
            model = runtime.deserialize_cuda_engine(self.weight.read_bytes())

        context = model.create_execution_context()

        names = [model.get_binding_name(i) for i in range(model.num_bindings)]
        self.num_bindings = model.num_bindings
        self.bindings: List[int] = [0] * self.num_bindings
        num_inputs, num_outputs = 0, 0

        for i in range(model.num_bindings):
            if model.binding_is_input(i):
                num_inputs += 1
            else:
                num_outputs += 1

        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.model = model
        self.context = context
        self.input_names = names[:num_inputs]
        self.output_names = names[num_inputs:]

    def __init_bindings(self) -> None:
        dynamic = False
        Tensor = namedtuple('Tensor', ('name', 'dtype', 'shape', 'cpu', 'gpu'))
        inp_info = []
        out_info = []
        out_ptrs = []
        for i, name in enumerate(self.input_names):
            assert self.model.get_binding_name(i) == name
            dtype = trt.nptype(self.model.get_binding_dtype(i))
            shape = tuple(self.model.get_binding_shape(i))
            if -1 in shape:
                dynamic |= True
            if not dynamic:
                cpu = np.empty(shape, dtype)
                gpu = cuda.mem_alloc(cpu.nbytes)
                cuda.memcpy_htod_async(gpu, cpu, self.stream)
            else:
                cpu, gpu = np.empty(0), 0
            inp_info.append(Tensor(name, dtype, shape, cpu, gpu))
        for i, name in enumerate(self.output_names):
            i += self.num_inputs
            assert self.model.get_binding_name(i) == name
            dtype = trt.nptype(self.model.get_binding_dtype(i))
            shape = tuple(self.model.get_binding_shape(i))
            if not dynamic:
                cpu = np.empty(shape, dtype=dtype)
                gpu = cuda.mem_alloc(cpu.nbytes)
                cuda.memcpy_htod_async(gpu, cpu, self.stream)
                out_ptrs.append(gpu)
            else:
                cpu, gpu = np.empty(0), 0
            out_info.append(Tensor(name, dtype, shape, cpu, gpu))

        self.is_dynamic = dynamic
        self.inp_info = inp_info
        self.out_info = out_info
        self.out_ptrs = out_ptrs

    def __warm_up(self) -> None:
        if self.is_dynamic:
            print('You engine has dynamic axes, please warm up by yourself !')
            return
        for _ in range(10):
            inputs = []
            for i in self.inp_info:
                inputs.append(i.cpu)
            self.__call__(inputs)

    def set_profiler(self, profiler: Optional[trt.IProfiler]) -> None:
        self.context.profiler = profiler \
            if profiler is not None else trt.Profiler()

    def __call__(self, *inputs) -> Tuple[ndarray, ndarray, ndarray, ndarray]:

        assert len(inputs) == self.num_inputs
        contiguous_inputs: List[ndarray] = [
            np.ascontiguousarray(i) for i in inputs
        ]

        for i in range(self.num_inputs):

            if self.is_dynamic:
                self.context.set_binding_shape(
                    i, tuple(contiguous_inputs[i].shape))
                self.inp_info[i].gpu = cuda.mem_alloc(
                    contiguous_inputs[i].nbytes)

            cuda.memcpy_htod_async(self.inp_info[i].gpu, contiguous_inputs[i],
                                   self.stream)
            self.bindings[i] = int(self.inp_info[i].gpu)

        output_gpu_ptrs: List[int] = []
        outputs: List[ndarray] = []

        for i in range(self.num_outputs):
            j = i + self.num_inputs
            if self.is_dynamic:
                shape = tuple(self.context.get_binding_shape(j))
                dtype = self.out_info[i].dtype
                cpu = np.empty(shape, dtype=dtype)
                gpu = cuda.mem_alloc(contiguous_inputs[i].nbytes)
                cuda.memcpy_htod_async(gpu, cpu, self.stream)
            else:
                cpu = self.out_info[i].cpu
                gpu = self.out_info[i].gpu
            outputs.append(cpu)
            output_gpu_ptrs.append(gpu)
            self.bindings[j] = int(gpu)

        self.context.execute_async_v2(self.bindings, self.stream.handle)
        self.stream.synchronize()

        for i, o in enumerate(output_gpu_ptrs):
            cuda.memcpy_dtoh_async(outputs[i], o, self.stream)

        return tuple(outputs) if len(outputs) > 1 else outputs[0]

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

    # x is the x-coordinate of the center of the box
    # y is the y-coordinate of the center of the box
    # w is the width of the box
    # h is the height of the box
    
    # Ensure box coordinates are within image boundaries
    x1 = max(0, x - w // 2)
    y1 = max(0, y - h // 2)
    x2 = min(image.shape[1], x + w // 2)
    y2 = min(image.shape[0], y + h // 2)
    h1 = min(h, image.shape[0] - y1)
    w1 = min(w, image.shape[1] - x1)
    
    # x1 is the starting x-coordinate of the image
    # x2 is the ending x-coordinate of the image
    # y1 is the starting y-coordinate of the image
    # y2 is the ending y-coordinate of the image 

    # Create a mask to zero out areas outside the box
    mask = np.zeros_like(image, dtype=np.float32)
    mask[y1:y2, x1:x2] = 1

    # Apply the mask to the original image
    cropped_image = image * mask
    
    return cropped_image, (x1, y1, w1, h1)

def threshold_image(image,threshold):
    thresholded_image = np.where(image > threshold, 255,0)
    return thresholded_image

def extract_masks(output_0,
                  prototypes,
                  input_size,
                  threshold_detection=0.5,
                  theshold_iou=0.45,
                  threshold_mask=0):
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
    l_boxes=[]
    for k in range(len(l_class_NMS)):
        for detection in l_class_NMS[k]:
            coeff=detection[5:]
            mask=prototypes*coeff.reshape(prototypes.shape[0],1,1)
    
            # Initialize an empty array to store resized images
            resized_mask = np.empty((mask.shape[0], input_size[0], input_size[1]))
            
            # Resize each image in the array
            for i, image in enumerate(mask):
                r = cv2.resize(image, (input_size[1], input_size[0]), interpolation=cv2.INTER_AREA)
                resized_mask[i] = r

            cropped_image, resized_box = crop_image(np.mean(resized_mask, axis=0),detection[:4])
            l_mask.append(threshold_image(cropped_image,threshold_mask))
            l_class.append(k)
            l_conf.append(detection[4])
            l_boxes.append(resized_box)

    return l_mask,l_class,l_conf,l_boxes


def show_masks(image, l_mask, l_class, l_conf, l_boxes, should_save=False):

    colors = [[0,255,0], [0,0,255], [255,0,0], [255,255,0], [255,0,255], [0,255,255], [128,0,0], [0,128,0], [0,0,128], [128,128,0], [128,0,128], [0,128,128], [128,128,128], [192,192,192], [128,0,0], [128,128,0], [0,128,0], [128,0,128], [0,128,128], [0,0,128], [255,255,255], [0,0,0]]

    masked_image = np.copy(image)
    
    for idx, mask in enumerate(l_mask):
        # Apply mask on the original image with opacity
        color = colors[idx]
        mask_rgb = np.zeros_like(image)
        mask_rgb[mask > 0] = color
        opacity = 0.3
        cv2.addWeighted(mask_rgb, opacity, masked_image, 1 - opacity, 0, masked_image)


        # Sort unpacked bounding 
        boxes = l_boxes[idx]

        x, y, w, h = boxes
        cv2.rectangle(masked_image, (x, y), (x + w, y + h),color, 2)
        # add highlight for the label
        cv2.rectangle(masked_image, (x, y - 40), (x + w, y),color, -1)
        # add label
        cv2.putText(masked_image, f'{CLASSES[l_class[idx]]} {l_conf[idx]:.2f}', (x + 2, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 5)

        # cv2.imshow(f'Masked Image {idx + 1}', masked_image)
        # cv2.waitKey(0)

    # save
    if should_save:
        cv2.imwrite(f'railway_masked_image_{idx + 1}.jpg', masked_image)


def preprocess_image(image: ndarray) -> ndarray:
    input_image = image.astype(np.float32) / 255.0
    input_image = np.transpose(input_image, [2, 0, 1])  # Change image layout to CHW (Channels First)
    input_image = np.expand_dims(input_image, axis=0) 
    return input_image

def get_input_image(args):
    if args.camera:
            capture = cv2.VideoCapture(args.camera_id, cv2.CAP_V4L2)
            capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
            width = args.width
            height = args.height
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

            # capture only one frame
            ret, image = capture.read()
            capture.release()
    else:
        image = cv2.imread(args.image_path)

    return image

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine_path', type=str, required=True, help='Path to the engine')
    parser.add_argument('--image_path', type=str, required=False, help='Path to the image')
    parser.add_argument('--camera', type=bool, default=False, help='Use camera')
    parser.add_argument('--camera_id', type=int, default=0, help='Use camera')
    parser.add_argument('--width', type=int, default=1280, help='Width of the camera frame')
    parser.add_argument('--height', type=int, default=960, help='Height of the camera frame')
    parser.add_argument('--threshold_detection', type=float, default=0.5, help='Threshold for detection')
    parser.add_argument('--theshold_iou', type=float, default=0.45, help='Threshold for IOU')
    parser.add_argument('--threshold_mask', type=float, default=0, help='Threshold for mask')
    parser.add_argument('--save_output', type=bool, default=False, help='Save output image')
    return parser.parse_args()

def main(args):

    # one of image_path or camera must be provided
    if not args.image_path and not args.camera:
        print('Please provide either image_path or camera_id')
        return
    
    # in case of camera, width and height must be provided
    if args.camera and not args.camera_id and not args.width and not args.height:
        print('Please provide camera_id, width and height for the camera')
        return

    # Load the image
    image = get_input_image(args)

    print("input image shape", image.shape)
    # Load the engine
    engine = TRTEngine(args.engine_path)

    # check if image size corresponds to the engine input size
    engine_h, engine_w = engine.inp_info[0].shape[-2:]
    image_h, image_w = image.shape[:2]

    if (engine_h, engine_w) != (image_h, image_w):
        print(f'Image size {image_h}x{image_w} does not match engine input size {engine_h}x{engine_w}')
        print('Please resize the image to match the engine input size')
        return
    
    input_size = (image_h, image_w)

    # Preprocess the image
    input_image = preprocess_image(image)
    # Run inference
    results = engine(input_image)
    prototypes = results[0][0]
    output_0 = results[1][0]
    # Extract masks
    l_mask,l_class,l_conf,l_boxes = extract_masks(output_0,
                                                  prototypes,
                                                  input_size,
                                                  threshold_detection=0.5,
                                                  theshold_iou=0.45,
                                                  threshold_mask=0)

    # n detection
    print("n masks: ", len(l_mask))

    # Show masks
    show_masks(image, l_mask,l_class,l_conf,l_boxes, should_save=args.save_output)

if __name__ == '__main__':
    args = parse_args()
    main(args)