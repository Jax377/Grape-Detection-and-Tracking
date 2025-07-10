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

## Configuration Details

This repository implements and evaluates the enhanced tracking algorithms **SORT+** and **DeepSORT+** for grape cluster tracking in UAV vineyard videos. These trackers extend the original SORT and DeepSORT by incorporating a multi-step matching cascade with complementary similarity measures.

All algorithms use a **Mask R-CNN** detector with consistent parameters for a fair comparison.

---

### Mask R-CNN Configuration

| Component | Setting |
|----------|---------|
| **Backbone** | ResNet-50, 4 stages (Stage 1 frozen), pretrained on COCO |
| **Neck** | Feature Pyramid Network (FPN) |
| **Anchor Generator** | (8×8), (8×16), (16×8) with strides 4, 8, 16, 32, 64 |
| **RPN Head** | L1 bbox loss, Cross-Entropy classification loss |
| **NMS Threshold** | IoU = 0.6 |
| **Max Proposals After NMS** | 1000 |
| **RoI Head** | RoI Align |
| **BBox Head** | 2 Fully Connected Layers (1024 units) |
| **Mask Head** | 4 Conv layers → upsample to 28×28 |
| **Loss Functions** | L1 loss (bbox), Cross-Entropy (class & mask) |
| **Input Image Size** | 4096×2160 px |
| **Detector Confidence Threshold** | 0.6 |

---

### SORT+ Configuration

SORT+ augments the original SORT tracker by chaining four matching steps using diverse similarity metrics to enhance robustness and recovery of partial detections.

#### General Parameters

| Parameter | Value |
|----------|-------|
| Detector | Mask R-CNN |
| Confidence Threshold | 0.6 |
| Remove Unmatched Tracks After | 50 frames |
| Confirm Tentative Track After | 5 frames |

#### Kalman Filter

| Parameter | Value |
|----------|-------|
| Position Weight (`w_p`) | 1 / 10 |
| Velocity Weight (`w_v`) | 1 / 80 |

#### Matching Cascade

| Step | Similarity Measure | Threshold |
|------|--------------------|-----------|
| 1 | IoU overlap | > 0.2 |
| 2 | Mahalanobis Distance (center coords) | < 4.605 (Chi² 0.9) |
| 3 | IoU overlap | > 0.035 |
| 4 | Euclidean Distance | < `w_pred + h_pred / 2` |

---

### DeepSORT+ Configuration

DeepSORT+ enhances DeepSORT by adding more matching steps and extending the use of appearance features. This results in better identity preservation under occlusion.

#### General Parameters

| Parameter | Value |
|----------|-------|
| Detector | Mask R-CNN |
| Confidence Threshold | 0.6 |
| Remove Unmatched Tracks After | 50 frames |
| Confirm Tentative Track After | 5 frames |

#### Kalman Filter

| Parameter | Value |
|----------|-------|
| Position Weight (`w_p`) | 1 / 10 |
| Velocity Weight (`w_v`) | 1 / 80 |

#### Classification Network

| Component | Architecture |
|----------|--------------|
| Base | ResNet-50 |
| Pooling | Global Average Pooling |
| Head | 1 Fully Connected Layer |
| Feature Distance | Euclidean |

#### Matching Cascade

| Step | Similarity Measure | Threshold |
|------|--------------------|-----------|
| 1 | Mahalanobis Distance (center coords) | < 10.597 (Chi² 0.995) |
|   | + Appearance Feature Distance | < 1.5 |
| 2 | IoU overlap | > 0.2 |
| 3 | Mahalanobis Distance | < 4.605 (Chi² 0.9) |
| 4 | IoU overlap | > 0.035 |
| 5 | Euclidean Distance | < `w_pred + h_pred / 2` |

---

### Notes

- The matching cascade uses the **Hungarian Algorithm** at each step for optimal one-to-one assignment.
- All trackers are implemented using [MMTracking](https://github.com/open-mmlab/mmtracking) with custom extensions.
- This setup is designed for **grape cluster tracking** using the **UAV RGB Video dataset**.

---
