# Grape Detection and Tracking Using Deep Neural Networks
This Repository allows modular grape detection and tracking using detection models like Faster R-CNN, Mask R-CNN, etc. and tracking algorithms like SORT, DeepSORT, ByteTrack, etc. 

By clicking on the video, you will be referred to YouTube:
<div align="center">
  <a href="https://www.youtube.com/watch?v=QXpD1_R7mbo" target="_blank">
    <img src="https://img.youtube.com/vi/QXpD1_R7mbo/0.jpg" alt="Grape Detection and Tracking" width="500" height="300">
  </a>
  
  Grape Detection and Tracking Using Mask R-CNN and ByteTrack
</div>


This Repository contains two submodules. The submodule "lib1" contains an altered version of MMCV from MMLab to make it work for Windows. The second submodule "lib2" contains an altered Version of MMLab's MMTracking.

---

## 🍇 Full Configuration Details

This repository supports the paper _"Enhanced Grape Tracking Using Deep Neural Networks"_ and documents every detail required to reproduce the results.

---

### 📦 Mask R-CNN Detector Configuration

The detector used in all tracking algorithms is a customized **Mask R-CNN** network trained on the UAV RGB Video Dataset. The goal is reliable detection and segmentation of grape clusters.

#### 🧠 Architecture Summary

| Component | Setting |
|----------|---------|
| **Backbone** | ResNet-50, 4 stages (Stage 1 frozen), pretrained on COCO |
| **Neck** | Feature Pyramid Network (FPN) |
| **Input Resolution** | 4096×2160 px |
| **RoI Alignment** | Used for both bbox and mask branches |

#### 🔧 Region Proposal Network (RPN)

| Parameter | Value |
|----------|-------|
| Anchors | (8×8), (8×16), (16×8) |
| Strides | 4, 8, 16, 32, 64 |
| IoU Threshold (Positive) | > 0.7 |
| IoU Threshold (Negative) | < 0.3 |
| Number of Proposals (pre/post NMS) | 2000 → 1000 |
| NMS Threshold | 0.6 |
| Bbox Coder | Mean: [0,0,0,0]; Std: [1,1,1,1] |
| Losses | L1 for bbox, Cross-Entropy for classification |

#### 🎯 RoI Head & Output

| Parameter | Value |
|----------|-------|
| BBox Head | 2 FC layers, 1024 units each |
| BBox Output Coder | Mean: [0,0,0,0]; Std: [0.1,0.1,0.2,0.2] |
| Mask Head | 4 Conv layers → (28×28) binary mask |
| Losses | L1 for bbox, Cross-Entropy for classification & masks |
| Output Classes | 1 (grape clusters) |

#### 🧪 Training Settings

| Parameter | Value |
|----------|-------|
| Training Set Size | 528 annotated images |
| Test Set Size | 136 annotated images |
| Epochs | 25 |
| Optimizer | SGD (momentum 0.9) or Adam |
| Weight Decay | 0.0001 |
| Learning Rate Scheduler | Warm-up → decay after 3 epochs |
| Batch Size | 2 images per GPU |
| Mask Threshold | 0.5 |

#### 🔄 Data Augmentation (Train)

| Transformation | Value |
|----------------|-------|
| Brightness | ±50% |
| Contrast | ±50% |
| Saturation | ±50% |
| Hue | ±18° shift in HSV |
| Flip | 50% probability (horizontal) |
| Normalization Mean | [123.675, 116.28, 103.53] |
| Normalization Std | [58.395, 57.12, 57.375] |
| Padding | Image padded to dimensions divisible by 32 |

---

### 📍 SORT Configuration

| Parameter | Value |
|----------|-------|
| Detector | Mask R-CNN |
| Detection Confidence Threshold | 0.6 |
| Track Termination | After 50 unmatched frames |
| Track Confirmation | After 5 matched frames |
| Kalman Filter Weights | Position: 1/10, Velocity: 1/80 |
| Matching Method | Hungarian Algorithm on (1 - IoU) |
| IoU Threshold | > 0.035 (3.5%) |

---

### 📍 SORT+ Configuration

Enhanced version using multi-step matching:

| Step | Similarity | Threshold |
|------|------------|-----------|
| 1 | IoU overlap | > 0.2 |
| 2 | Mahalanobis Distance (cx, cy) | < 4.605 (χ², 0.9) |
| 3 | IoU overlap | > 0.035 |
| 4 | Euclidean Distance | < bbox_width + bbox_height / 2 |

Other settings match SORT. Matching steps are executed sequentially via Hungarian Algorithm.

---

### 📍 DeepSORT Configuration

| Parameter | Value |
|----------|-------|
| Detector | Mask R-CNN |
| Confidence Threshold | 0.6 |
| Kalman Filter | Same as SORT |
| Appearance Feature Extractor | ResNet-50 → Global Avg Pooling → FC Layer |
| Feature Output Dim | 128 |
| Feature Matching | Euclidean distance < 1.5 |
| Mahalanobis Distance | < 10.597 (χ², 0.995) |
| Fallback Matching | IoU overlap > 0.035 |

---

### 📍 DeepSORT+ Configuration

DeepSORT+ adds three additional matching stages on top of DeepSORT:

| Step | Similarity | Threshold |
|------|------------|-----------|
| 1 | Mahalanobis < 10.597 (χ² 0.995) + Feature Distance < 1.5 |
| 2 | IoU overlap > 0.2 |
| 3 | Mahalanobis < 4.605 (χ² 0.9) |
| 4 | IoU overlap > 0.035 |
| 5 | Euclidean Distance < bbox_width + bbox_height / 2 |

---

### 📍 ByteTrack Configuration

ByteTrack uses confidence-based splitting of detections:

| Type | Confidence |
|------|------------|
| High Confidence | > 0.6 |
| Low Confidence | > 0.05 |
| Track Initialization | Requires high-confidence detection |

| Stage | Matching Target | IoU Threshold |
|-------|------------------|----------------|
| 1 | Confirmed ↔ High Conf | 0.965 |
| 2 | Tentative ↔ High Conf | 0.915 |
| 3 | Confirmed ↔ Low Conf | 0.8 |

---

### 🧪 Classification Network for DeepSORT(+)

| Component | Details |
|----------|---------|
| Backbone | ResNet-50 (ImageNet pretrained) |
| Neck | Global Average Pooling |
| Head | FC: 2048 → 1024 → 128 |
| Classes | 586 |
| Loss | Cross-Entropy + Triplet Loss (margin: 0.5) |
| Optimizer | Adam or SGD |
| Input Size | 128 × 256 |
| Normalization | Same as Mask R-CNN |
| Flip Augmentation | 50% |


