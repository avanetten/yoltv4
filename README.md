# YOLTv4 # 

![Alt text](/results/__examples/header.jpg?raw=true "")
 
 YOLTv4 builds upon [YOLT]( https://github.com/avanetten/yolt) and [SIMRDWN]( https://github.com/avanetten/simrdwn), and updates these frameworks to use the most performant version of [YOLO](https://pjreddie.com/darknet/yolo/), [YOLOv4](https://github.com/AlexeyAB/darknet). YOLTv4 is designed to detect objects in aerial or satellite imagery in arbitrarily large images that far exceed the ~600×600 pixel size typically ingested by deep learning object detection frameworks.  
 
 This repository is built upon the impressive work of AlexeyAB's [YOLOv4](https://github.com/AlexeyAB/darknet) implementation, which improves both speed and detection performance compared to YOLOv3 (which is implemented in SIMRDWN). We use YOLOv4 insead of "[YOLOv5](https://github.com/ultralytics/yolov5)", since YOLOv4 is endorsed by the original creators of YOLO, whereas "YOLOv5" is not; furthermore YOLOv4 appears to have superior performance. 
 
 Below, we provide examples of how to use this repository with the open-source [Rareplanes](https://registry.opendata.aws/rareplanes/) dataset. 
 
____
## Running YOLTv4

___

### 0. Installation

YOLTv4 is built to execute within a docker container on a GPU-enabled machine.  The docker command creates an Ubuntu 16.04 image with CUDA 9.2, python 3.6, and conda.

1. Clone this repository (e.g. to _/yoltv4/_).

2. Download model weights to _yoltv4/darknet/weights_).
	See:
	    https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.conv.137
    	https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights
    	https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights
        https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-csp.weights


2. Install [nvidia-docker](https://github.com/NVIDIA/nvidia-docker).
 
3. Build docker file.

		nvidia-docker build -t yoltv4_image /yoltv4/docker
	
4. Spin up the docker container (see the [docker docs](https://docs.docker.com/engine/reference/commandline/run/) for options).

        NV_GPU=0 nvidia-docker run -it -v /local_data:/local_data -v /yoltv4:/yoltv4 -ti --ipc=host --name yoltv4_gpu0 yoltv4_image
	
5. Compile the Darknet C program.

    First Set GPU=1 CUDNN=1, CUDNN_HALF=1, OPENCV=1 in /yoltv4/darknet/Makefile, then make:
	
	    cd /yoltv4/darknet
	    make
	

___

### 1. Train

#### A. Prepare Data

1. Make YOLO images and labels (see _yoltv4/notebooks/prep\_data.ipynb_ for further details).

2. Create a txt file listing the training images.

3. Create file obj.names file with each desired object name on its own line.

4. Create file obj.data in the directory _yoltv4/darknet/data_ containing necessary files.  For example:

	_/yoltv4/darknet/data/rareplanes\_train.data_
        
        classes = 30
        train =  /local_data/cosmiq/wdata/rareplanes/train/txt/train.txt
        valid =  /local_data/cosmiq/wdata/rareplanes/train/txt/valid.txt
        names =  /yoltv4/darknet/data/rareplanes.name
        backup = backup/

5. Prepare config files.

    See instructions [here](https://github.com/AlexeyAB/darknet#how-to-train-to-detect-your-custom-objects), or tweak _/yoltv4/darknet/cfg/yoltv4\_rareplanes.cfg_.

#### B. Execute Training
    
1. Execute.

        cd /yoltv4/darknet
	    time ./darknet detector train data/rareplanes_train.data  cfg/yoltv4_rareplanes.cfg weights/yolov4.conv.137  -dont_show -mjpeg_port 8090 -map

2. Review progress (plotted at: _/yoltv4/darknet/chart\_yoltv4\_rareplanes.png_).
  


___

### 2. Test

#### A. Prepare Data

1. Make sliced images (see _yoltv4/notebooks/prep\_data.ipynb_ for further details).

2. Create a txt file listing the training images.

3. Create file obj.data in the directory _yoltv4/darknet/data_ containing necessary files.  For example:

	_/yoltv4/darknet/data/rareplanes_test.data_
        classes = 30
        train = 
        valid =  /local_data/cosmiq/wdata/rareplanes/test/txt/test.txt
        names =  /yoltv4/darknet/data/rareplanes.name
        backup = backup/

#### B. Execute Testing


1. Execute (proceeds at >80 frames per second on a Tesla P100):

        cd /yoltv4/darknet
	    time ./darknet detector valid data/rareplanes_test.data cfg/yoltv4_rareplanes.cfg backup/ yoltv4_rareplanes_best.weights

2. Post-process detections:

	A. Move detections into results directory
    
		mkdir /yoltv4/darknet/results/rareplanes_preds_v0
		mkdir  /yoltv4/darknet/results/rareplanes_preds_v0/orig_txt
		mv /yoltv4/darknet/results/comp4_det_test_*  /yoltv4/darknet/results/rareplanes_preds_v0/orig_txt/
	B. Stitch detections back together and make plots
    
		time python /yoltv4/yoltv4/post_process.py \
		    --pred_dir=/yoltv4/darknet/results/rareplanes_preds_v0/orig_txt/ \
		    --raw_im_dir=/local_data/cosmiq/wdata/rareplanes/test/images/ \
		    --sliced_im_dir=/local_data/cosmiq/wdata/rareplanes/test/yoltv4/images_slice/ \
		    --out_dir= /yoltv4/darknet/results/rareplanes_preds_v0 \
		    --detection_thresh=0.25 \
		    --slice_size=416} \
		    --n_plots=8
		 

Outputs will look something like the figures below:

![Alt text](/results/__examples/rareplanes0.jpg?raw=true "")

![Alt text](/results/__examples/rareplanes1.jpg?raw=true "")

![Alt text](/results/__examples/rareplanes2.jpg?raw=true "")


