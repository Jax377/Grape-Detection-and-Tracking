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

## ⚙️ Full Configuration Overview

This repository evaluates grape cluster tracking using SORT, DeepSORT, ByteTrack, and the extended versions SORT+ and DeepSORT+. All trackers use a uniform **Mask R-CNN** detector trained on the UAV RGB Video dataset.

---

## 🧠 Mask R-CNN Detector

### 📐 Architecture Configuration

| Component | Setting |
|----------|---------|
| **Backbone** | ResNet-50 (4 stages, Stage 1 frozen), pretrained on COCO |
| **Neck** | Feature Pyramid Network (FPN) |
| In Channels | (1088×2048×256), (544×1024×512), (272×512×1024), (136×256×2048) |
| Out Channels | (1088×2048×256), (544×1024×256), (272×512×256), (136×256×256), (68×128×256) |
| **RPN Head** | |
| Anchor Generator | (8×8), (8×16), (16×8) with strides 4, 8, 16, 32, 64 |
| Bbox `[x, y, h, w]` coder | mean = [0, 0, 0, 0]; std = [1, 1, 1, 1] |
| Bbox Loss | L1 |
| Classification Loss | Cross-entropy |
| **NMS** | |
| Proposals Before NMS | 2000 |
| Proposals After NMS | 1000 |
| IoU Threshold | 0.6 |
| **RoI Head** | RoI Align |
| Bbox Branch Output | 7×7×256 |
| Mask Branch Output | 14×14×256 |
| Strides | 4, 8, 16, 32 |
| **BBox Head** | 2 Fully Connected Layers |
| In Channels | 7×7×256 |
| Out Channels (FC Layers) | 1024 |
| Classes | 1 |
| Bbox Coder | mean = [0, 0, 0, 0]; std = [0.1, 0.1, 0.2, 0.2] |
| Bbox Loss | L1 |
| Classification Loss | Cross-entropy |
| **Mask Head** | 4 Convolutions |
| In Channels | 14×14×256 |
| Out Channels | 4×(14×14×256), then 28×28×256, then 28×28 |
| Classes | 1 |
| Mask Loss | Cross-entropy |

---

### 🧪 Training Configuration

| Category | Setting |
|----------|---------|
| **Anchor Assignment** | |
| Positive: IoU > | 0.7 |
| Negative: IoU < | 0.3 |
| Positive (low quality): IoU > | 0.3 |
| Random Samples | 256 (50% positive) |
| **Mask/Bbox Assignment** | |
| Positive: IoU > | 0.5 |
| Negative: IoU < | 0.5 |
| Random Samples | 512 (25% positive); GT included |
| Mask Binary Threshold | 0.5 |
| **Epochs** | 25 |

---

### 🌱 Data Transformations

#### For Mask R-CNN

| Transformation | Value |
|----------------|-------|
| Brightness | [0.5, 1.5] |
| Contrast | [0.5, 1.5] |
| Saturation | [0.5, 1.5] |
| Hue | [–18°, +18°] using HSV |
| Horizontal Flip | 50% chance |
| Normalization Mean | [123.675, 116.28, 103.53] |
| Normalization Std | [58.395, 57.12, 57.375] |
| Padding | Pad to size divisible by 32 |

#### For DeepSORT Classification Network

| Transformation | Value |
|----------------|-------|
| Resize | (128, 256) |
| Horizontal Flip | 50% chance |
| Normalization Mean | [123.675, 116.28, 103.53] |
| Normalization Std | [58.395, 57.12, 57.375] |

---

## 🎯 Tracker Configurations

All trackers are configured to use the same detector and Kalman Filter tuning.

### 1️⃣ SORT

| Parameter | Value |
|----------|-------|
| Detector | Mask R-CNN |
| Confidence Threshold | 0.6 |
| Frames to Remove Unmatched Tracks | 50 |
| Frames to Confirm Tentative Tracks | 5 |
| **Kalman Filter** | `w_p = 1/10`, `w_v = 1/80` |
| **Matching** | Hungarian Algorithm on `1 - IoU overlap` |
| IoU Threshold | 0.965 |

---

### 2️⃣ SORT+

| Parameter | Value |
|----------|-------|
| Detector | Mask R-CNN |
| Confidence Threshold | 0.6 |
| Frames to Remove Unmatched Tracks | 50 |
| Frames to Confirm Tentative Tracks | 5 |
| **Kalman Filter** | `w_p = 1/10`, `w_v = 1/80` |
| **Matching Steps (Hungarian Algorithm)** | |
| 1. `1 - IoU overlap` | Threshold: 0.8 |
| 2. Mahalanobis Distance | Threshold: χ²₀.₉,₂ = 4.605 |
|   Variables Used | (cₓ, cᵧ) |
| 3. `1 - IoU overlap` | Threshold: 0.965 |
| 4. Euclidean Distance | Threshold: `w_pred + h_pred / 2` |

---

### 3️⃣ DeepSORT

| Parameter | Value |
|----------|-------|
| Detector | Mask R-CNN |
| Confidence Threshold | 0.6 |
| Frames to Remove Unmatched Tracks | 50 |
| Frames to Confirm Tentative Tracks | 5 |
| **Kalman Filter** | `w_p = 1/10`, `w_v = 1/80` |
| **Classification Network** | ResNet-50 + GAP + 1 FC Layer |
| **Matching Steps (Hungarian Algorithm)** | |
| 1. Mahalanobis Distance | < 10.597 (χ²₀.₉₉₅,₂) |
| 1. Feature Distance | < 1.5 |
|   Variables Used | (cₓ, cᵧ) |
| 2. `1 - IoU overlap` | Threshold: 0.965 |

---

### 4️⃣ DeepSORT+

| Parameter | Value |
|----------|-------|
| Detector | Mask R-CNN |
| Confidence Threshold | 0.6 |
| Frames to Remove Unmatched Tracks | 50 |
| Frames to Confirm Tentative Tracks | 5 |
| **Kalman Filter** | `w_p = 1/10`, `w_v = 1/80` |
| **Classification Network** | ResNet-50 + GAP + 1 FC Layer |
| **Matching Steps (Hungarian Algorithm)** | |
| 1. Mahalanobis Distance | < 10.597 (χ²₀.₉₉₅,₂) |
| 1. Feature Distance | < 1.5 |
|   Variables Used | (cₓ, cᵧ) |
| 2. `1 - IoU overlap` | Threshold: 0.8 |
| 3. Mahalanobis Distance | Threshold: 4.605 (χ²₀.₉,₂) |
| 4. `1 - IoU overlap` | Threshold: 0.965 |
| 5. Euclidean Distance | Threshold: `w_pred + h_pred / 2` |

---

### 5️⃣ ByteTrack

| Parameter | Value |
|----------|-------|
| Detector | Mask R-CNN |
| Frames to Remove Unmatched Tracks | 50 |
| Frames to Confirm Tentative Tracks | 5 |
| **Kalman Filter** | `w_p = 1/10`, `w_v = 1/80` |
| **Detection Thresholds** | |
| High Confidence | > 0.6 |
| Low Confidence | > 0.05 |
| Required for Track Init | > 0.6 |
| **Matching Steps (Hungarian Algorithm)** | |
| 1. Confirmed ↔ High Conf Detections | Threshold: 0.965 |
| 2. Tentative ↔ High Conf Detections | Threshold: 0.915 |
| 3. Confirmed ↔ Low Conf Detections | Threshold: 0.8 |


