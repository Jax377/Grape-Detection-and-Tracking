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

## 📦 Mask R-CNN Configuration

This section outlines all components of the **Mask R-CNN** detector used across all tracking algorithms.

### 🔍 Detector Configuration Table

| Component | Setting |
|----------|---------|
| **Backbone** | ResNet-50 (4 stages, Stage 1 frozen) |
|            | Pretrained on the COCO dataset |
| **Neck** | Feature Pyramid Network (FPN) |
| In Channels | (1088×2048×256), (544×1024×512), (272×512×1024), (136×256×2048) |
| Out Channels | (1088×2048×256), (544×1024×256), (272×512×256), (136×256×256), (68×128×256) |
| **Region Proposal Network Head** | |
| In Channels | same as FPN output |
| Anchor Generator | (8×8), (8×16), (16×8) with strides 4, 8, 16, 32, 64 |
| Bbox `[x, y, h, w]` Coder | mean = [0, 0, 0, 0], std = [1, 1, 1, 1] |
| Bbox Loss | L1 loss |
| Classification Loss | Cross-entropy |
| **NMS (Non-Maximum Suppression)** | |
| Proposals kept before NMS | 2000 |
| Proposals kept after NMS | 1000 |
| IoU Threshold | 0.6 |
| **Region of Interest (RoI) Head** | RoI Align |
| Bbox Branch Output | 7×7×256 |
| Mask Branch Output | 14×14×256 |
| Strides | 4, 8, 16, 32 |
| **BBox Head** | 2 Fully Connected Layers |
| In Channels | 7×7×256 |
| Out Channels (per FC) | 1024 |
| Classes | 1 |
| Bbox Coder | mean = [0, 0, 0, 0], std = [0.1, 0.1, 0.2, 0.2] |
| Bbox Loss | L1 loss |
| Classification Loss | Cross-entropy |
| **Mask Head** | 4 Convolutions |
| In Channels | 14×14×256 |
| Out Channels | 4×(14×14×256), then 28×28×256, then 28×28 |
| Classes | 1 |
| Mask Loss | Cross-entropy |

---

## 🛠️ Tracker Configurations

Below are the exact configurations of all tracking algorithms used.

### 📍 SORT+ Configuration

| Parameter | Value |
|----------|-------|
| **Detector** | Mask R-CNN |
| Confidence Score Threshold | 0.6 |
| Frames until removing unmatched tracks | 50 |
| Frames until confirming tentative tracks | 5 |
| **Kalman Filter** | |
| Position Weight | `w_p = 1/10` |
| Velocity Weight | `w_v = 1/80` |
| **Matching Steps (Hungarian Algorithm)** | |
| 1. `1 - IoU-overlap` | Threshold: 0.8 |
| 2. Mahalanobis Distance | Threshold: χ²₀.₉,₂ = 4.605 |
|   Variables Used | (cₓ, cᵧ) |
| 3. `1 - IoU-overlap` | Threshold: 0.965 |
| 4. Euclidean Distance | Threshold: `w_pred + h_pred / 2` |

---

### 📍 DeepSORT+ Configuration

| Parameter | Value |
|----------|-------|
| **Detector** | Mask R-CNN |
| Confidence Score Threshold | 0.6 |
| Frames until removing unmatched tracks | 50 |
| Frames until confirming tentative tracks | 5 |
| **Kalman Filter** | |
| Position Weight | `w_p = 1/10` |
| Velocity Weight | `w_v = 1/80` |
| **Classification Network** | |
| Backbone | ResNet-50 |
| Neck | Global Average Pooling |
| Head | 1 Fully Connected Layer |
| **Matching Steps (Hungarian Algorithm)** | |
| 1. Appearance Features + Mahalanobis Distance | Mahalanobis < 10.597, Feature Distance < 1.5 |
|   Variables Used | (cₓ, cᵧ) |
| 2. `1 - IoU-overlap` | Threshold: 0.8 |
| 3. Mahalanobis Distance | Threshold: χ²₀.₉₉₅,₂ = 4.605 |
| 4. `1 - IoU-overlap` | Threshold: 0.965 |
| 5. Euclidean Distance | Threshold: `w_pred + h_pred / 2` |

---

### 📍 Mask R-CNN Training Configuration

| Component | Setting |
|----------|---------|
| **Anchor Assignment** | |
| Positive Anchors | IoU > 0.7 |
| Negative Anchors | IoU < 0.3 |
| Low Quality Matches | Anchors with IoU > 0.3 |
| Random Samples | 256 (50% positive) |
| **Mask/BBox Assignment** | |
| Positive/Negative Threshold | IoU = 0.5 |
| Random Samples | 512 (25% positive), ground truth included |
| Mask Binary Threshold | 0.5 |
| **Maximum Epochs** | 25 |

---

### 📷 Data Transformations (Mask R-CNN)

| Transformation | Value |
|----------------|-------|
| Brightness Adjustment | Scalar ∈ [0.5, 1.5] |
| Contrast Adjustment | Scalar ∈ [0.5, 1.5] |
| Saturation Adjustment | Scalar ∈ [0.5, 1.5] |
| Hue Adjustment | Degrees ∈ [–18, 18] using HSV rotation |
| Horizontal Flip | 50% chance |
| Normalization Mean | [123.675, 116.28, 103.53] |
| Normalization Std | [58.395, 57.12, 57.375] |
| Padding | Pad to dimensions divisible by 32 |

---

### 📷 Data Transformations (Classification Network)

| Transformation | Value |
|----------------|-------|
| Resize | (128, 256) |
| Horizontal Flip | 50% chance |
| Normalization Mean | [123.675, 116.28, 103.53] |
| Normalization Std | [58.395, 57.12, 57.375] |

---

If you use this configuration in your own work, please cite:

> Jacob Piazolo & Benedikt Fischer (2025).  
> _Enhanced Grape Tracking Using Deep Neural Networks with an Extended Matching Algorithm for SORT and DeepSORT_.  
> Computers and Electronics in Agriculture.



