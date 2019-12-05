# SPADE_eval
This is the evaluation procedure for SPADE models. **Python3 preferred**

### 0. Preparation steps
* Prepare test dataset (semantic segmentation and images). We recommend using > 50000 segmentations to get accurate FID results.
    
* Use the model to be evaluated to generate images from test dataset segmentations.

### 1. mIoU and accu
* Estimate semantic segmentation from generated images in step 0 using [this](https://github.com/CSAILVision/semantic-segmentation-pytorch) repository. Follow the installation instructions. Download the upernet101 encoder and decoder from [here](http://sceneparsing.csail.mit.edu/model/pytorch/ade20k-resnet101-upernet/). and use the command below to estimate semantic segmentation:

````
python -u test.py \
--imgs <GENERATED-IMAGES-DIR> \
--cfg config/ade20k-resnet101-upernet.yaml \
TEST.result <OUTPUT-DIR> \
TEST.checkpoint epoch_50.pth
````

* Put the mIoU.py script in this repo in the ````semantic-segmentation-pytorch```` folder to calculate mIoU and accu scores.

````
python mIoU.py \
--gt_dir <TEST-DATASET-SEGMENTATION-DIR> \
--pred_dir <ESTIMATED-SEGMENTATION-DIR>
````

### 2. FID score

In the ````fid```` folder, the ````fid.py```` script computes the fid score. Specify the image dir or computed stats file in a config file (examples in ````fid/configs````) and run:

````
python fid.py configs/example.yaml
````
