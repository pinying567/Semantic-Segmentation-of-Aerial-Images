# Semantic Segmentation of Aerial Images

**Author: Pin-Ying Wu**

**Table of contents**
- Overview
- Code
- Result Analysis

## Overview
### Task
Perform semantic segmentation which predicts a label to each image with CNN models.

• Input : RGB image
• Output : Semantic Segmentation/Prediction

<!-- ![] (asset/task.png)-->
<!-- ![](https://i.imgur.com/kkLhKa8.png)-->
<img src=asset/task.png width=80%>  

* <font size=3>**Baseline model (VGG16 + FCN32s)**</font>
    - We use pre-trained VGG16 as the backbone of our CNN model, connected by FCN32s (Fully Convolutional Network).
    - Reference : [Long et al., “Fully Convolutional Networks for Semantic Segmentation”, CVPR 2015](https://arxiv.org/pdf/1411.4038.pdf)

* <font size=3>**An improved model (VGG16 + FCN8s)**</font>
    - We use pre-trained VGG16 as the backbone of our CNN model, connected by FCN8s.
    - Performance is better than the baseline model.

<!-- ![] (asset/fcn.png)-->
<!--![](https://i.imgur.com/iZUwJA0.png)-->
<img src=asset/fcn.png width=80%>  

### Dataset
Image size: `512x512`, Mask size: `512x512`. There are 7 possible classes for each pixel.
1. Urban `(0, 255 255)`
2. Agriculture`(255, 255, 0)`
3. Rangeland `(255, 0, 255)`
4. Forest`(0, 255, 0)`
5. Water`(0, 0, 255)`
6. Barren`(255, 255, 255)`
7. Unknown`(0, 0, 0)`

* train/
    - Contains 2000 image-mask (ground truth) pairs
    - Satellite images are named 'xxxx_sat.jpg’
    - Mask images (ground truth) are named 'xxxx_mask.png'
* validation/
    - Contains 257 image-mask pairs

## Code
### Prerequisites
```
pip install -r requirements.txt
```
### Data Preparation
```
bash ./get_dataset.sh
```
The shell script will automatically download the dataset and store the data in a folder called `data/p2_data`.

### Training
1. Baseline model (VGG16 + FCN32s)
```
bash ./train_baseline.sh
```
2. An improved model (VGG16 + FCN8s)
```
bash ./train_improved.sh
```

### Checkpoints
| baseline | improved |
|:---:|:---:|
| [baseline_model](https://www.dropbox.com/s/w29d0huil1tlhe3/baseline_model.pkl?dl=1)  |  [improved_model](https://www.dropbox.com/s/7tpzysqb96860xe/improved_model.pkl?dl=1)  |

### Evaluation
1. Baseline model (VGG16 + FCN32s)
```
python3 baseline.py --phase test --checkpoint <checkpoint> --test_dir <path_to_test_img_dir> --save_dir <path_to_output_img_dir>
```
2. An improved model (VGG16 + FCN8s)
```
python3 improved.py --phase test --checkpoint <checkpoint> --test_dir path_to_test_img_dir> --save_dir <path_to_output_img_dir>
```

### Tools
-  mIoU:
```
python3 mean_iou_evaluate.py <-g ground_truth_directory> <-p prediction_directory>
```

- Visualization (draw semantic segmentation map on RGB image):
```
python3 viz_mask.py < --img_path path_to_the_rgb_image> < --seg_path path_to_the_segmentation>
```

## Result Analysis
### Evaluation metric
mean Intersection over Union (mIoU)

For each class, IoU is defined as: $\frac{True Positive}{True Positive + False Positive + False Negative}$

mean IoU is calculated by averaging over all classes except Unknown(0,0,0).
mIoU is calculated over **all test images**.

![](https://i.imgur.com/Lb7l097.png)

- Validation mIOU = **0.661148**
- Test mIOU = **0.698591**

    
### Visualization
**1. Baseline model (VGG16 + FCN32s)**
<!-- ![] (asset/baseline.png)-->
<!-- ![](https://i.imgur.com/Ez889Ti.png)-->
<img src=asset/baseline.png width=80%>  


- We can find that as epoch increases, the edge of segmentation becomes smoother and closer to the mask picture. During the early stage, most of the edges look like squares, which don’t in line with reality.
- During the middle stage, the rough outline appears, but still lose some details. In final stage, the edge becomes closer to the mask picture. However, it still can’t recognize the small areas such as a tiny river.

**2. An improved model (VGG16 + FCN8s)**
<!-- ![] (asset/improved.png)-->
<!-- ![](https://i.imgur.com/oIux0eo.png)-->
<img src=asset/improved.png width=80%>  


- As we observe from the improved model, the edge of segmentation becomes smoother and closer to the mask picture. In addition, the edges are smoother than that in the baseline model even in the early stage.
- In final stage, the shapes of the segmentations are closer to the mask pictures than that from the baseline model. Taking the 0107_sat.jpg for example, the pink areas at the bottom left corner are mainly split into two pieces (left and right) as in the ground truth mask; however, the baseline model can’t do this even in the final stage. The improved model captures more details of the images, but it still can’t recognize the small areas like a tiny river.