# Computer Vision models
Implementations of common computer vision deep learning models.
**Update:** as expected, there are far better (and way more updated) versions out there, consider this side project archived.
Check out the beautiful [timm](https://github.com/rwightman/pytorch-image-models) for bleeding edge encoders.

**There are better versions already out there with pretrained weights, why bother?**

First and foremost, it's a bit for fun and a bit for learning purposes. \

## Classification
CV models aimed at classification tasks, with a standard classifier head.

### Current models:
- ResNet variations (18, 34, 50, 101, 152)
- Xception (8 mid-flow blocks)

| Model                | pretrained  | Score (TODO)      |
|----------------------|-------------|-------------------|
| ResNet18             | ✔           |                   |
| ResNet34             | ✔           |                   |
| ResNet50             | ✔           |                   |
| ResNet101            | ✔           |                   |
| ResNet152            | ✔           |                   |
| Xception (8 mid flow)| ✔           |                   |

## Segmentation
Models for semantic and/or instance segmentation, including backbones when necessary.

### Current backbones:
| Model                 | pretrained  | Score (TODO)      |
|-----------------------|-------------|-------------------|
| ResNet18              | ✔           |                   |
| ResNet34              | ✔           |                   |
| ResNet50              | ✔           |                   |
| ResNet101             | ✔           |                   |
| ResNet152             | ✔           |                   |
| Xception (8 mid flow) | ✔           |                   |
| Xception (16 mid flow)| ✔           |                   |


### Current models
| Model                 |
|-----------------------|
| UNet                  |
| DeepLabV3             |
| DeepLabV3+            |
