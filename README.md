# TensorRT YOLOv8 Instance Segmentation

Python scripts performing instance segmentation using the **YOLOv8** model in Python3.
![! ONNX YOLOv8 Instance Segmentation](https://github.com/MatteoMendula/TensorRT-YOLOv8-Instance-Segmentation/blob/main/segmented.jpg?raw=true)
*Original image*: Ultralytics
This is the result of a finetuning on railway images, the model is able to detect the railway mask and the objects on it.

This represents my personal take on extensive online resources, although I couldn't replicate it verbatim. Hence, I developed my custom pipeline to execute Yolov8-seg on TRT.
I've exclusively tested this on Jetson TX2, so I cannot guarantee its compatibility with other platforms.
Though it was a challenging process and the outcome isn't flawless, I've shared it to contribute to the community's progress, combining my research efforts and imperfect programming towards a more refined solution.
The initial resource I used to start are linked below, many congratulation and thanks to the authors, I would not be able to make any significant progress without your directions.

## Requirements
 - Jetson TX2 module 
	- **nvidia@tegra-ubuntu:~/Documents/mybeatifulpath$** head -n 1 /etc/nv_tegra_release
	R32 (release), REVISION: 6.1, GCID: 27863751, BOARD: t186ref, EABI: aarch64, DATE: Mon Jul 26 19:36:31 UTC 2021
	
	 - **nvidia@tegra-ubuntu:~/Documents/mybeatifulpath$** python3 --version
	Python 3.6.9
 - Check the  **requirements.txt**  file.

## Installation

    git clone https://github.com/ibaiGorordo/ONNX-YOLOv8-Instance-Segmentation.git
    cd ONNX-YOLOv8-Instance-Segmentation
    pip install -r requirements.txt

## 1. Export ONNX model

Since ultralytics requires high-level resources and close-to-latest pip dependencies it is highly suggested to export the model on a laptop where you do not need to fight too much against CUDA and CUDNN versioning.
To check you nvidia drivers and CUDA version run:

    nvidia@tegra-ubuntu:~/Documents/mybeatifulpath$: nvidia-smi
    
This will return someting like:

![NVIDIA driver version is top left while CUDA version is top right](https://portal.databasemart.com/AvatarHandler.ashx?fid=484998&key=817012272)
*NVIDIA driver version is top left while CUDA version is top right*

Assuming GPU drivers are updated, this only requires to install torch following the direction provided at:
[PyTorch Start Locally](https://pytorch.org/get-started/locally/)

For other CUDA versions which are not listed in the GUI change:

    --index-url https://download.pytorch.org/whl/cu118
   
   into:
   

    --index-url https://download.pytorch.org/whl/cu"YOUR_VERSION_HERE_NO_DOTS"

pasting that url on your browser will give you the list of dependencies which will be installed with torch - torchvision and torchaudio.

Now can convert the Pytorch model to ONNX using the following Jupyter notebook: [Jupyter notebook](https://github.com/MatteoMendula/TensorRT-YOLOv8-Instance-Segmentation/blob/main/export_onnx.ipynb)

**N.B.** when you export the model remember to set the input size you used to run the inference with TRT 

## Build a TRT engine with trtexec
`trtexec`  is a command line wrapper that helps quickly utilize and protoype models with TensorRT, without requiring you to write your own inference application ([link here](https://docs.nvidia.com/tao/tao-toolkit/text/trtexec_integration/index.html)).

`trtexec` comes preinstalled on Jetson platforms together with TensorRT.
Its location should be: `/usr/src/tensorrt/bin/trtexec`

So, move the onnx file generated on your laptop at the previous step on the board using scp or a usb drive and run:

    /usr/src/tensorrt/bin/trtexec  --onnx="yolov8-seg${LETTER}.onnx"  --saveEngine="yolov8-seg${LETTER}.engine"  --explicitBatch  >  trtexec_log.txt

to build the engine and to save the log outputs to the *trtexec_log.txt* text file.
This will produce a lot of information about the building process and even if some warning pops out (depending on your TRT version) this should work flowless.

You can delete the current file by clicking the **Remove** button in the file explorer. The file will be moved into the **Trash** folder and automatically deleted after 7 days of inactivity.

## Run the inference

The engine parsing together with pre/post processing have been condensed in a single python file.
Here the inference is done with [pycuda](https://pypi.org/project/pycuda/), allocating memory from and to the GPU with high level APIs.
So simply run:

    python3 Yolov8Seg_pycuda.py --engine_path yolov8s-seg.engine --image_path resize1280x960_railway.jpg --save_output True

# Sources

Again a big thank you to the main authors:

-   [triple-Mu/YOLOv8-TensorRT](https://github.com/triple-Mu/YOLOv8-TensorRT) for the engine parsing
- [Pierre Meyrat: Overcoming Export Challenges for YOLOv8 segmentation Model to ONNX Model](https://medium.com/@jackpiroler?source=post_page-----b9507935d7e2--------------------------------) for the NMS python implementation and the mask postprocessing

### Other state of the art resources

 - [Model Export with Ultralytics YOLO](https://docs.ultralytics.com/modes/export/)
 - [Nvidia TensortRT](https://github.com/NVIDIA/TensorRT)
 - [Neutron.app](https://netron.app/) for ONNX model analysis
 - [ONNX runtime](https://onnxruntime.ai/docs/tutorials/mobile/pose-detection.html) for postprocessing debuggin on my laptop
