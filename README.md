# 火Detector
Deep Learning 2024 Challenge: Does the image contain 🔥?

# Fire Detection Project

**Authors:** Lingfeng Jin, Damir Tassybayev, Abduazizkhon Shomansurov  
**Affiliation:** Sapienza Università di Roma  

---

## Abstract

This project explores fire detection in images using deep learning. Starting with a custom CNN achieving **97.5% accuracy**, we experimented with advanced architectures, including Vision Transformers (ViT) and ResNet variants. The best single-model performance was achieved using a **pretrained Vision Transformer** with selective layer freezing, reaching **98.8% accuracy**. An ensemble model further improved validation accuracy to **99.04%**. Domain-specific normalization played a crucial role in enhancing performance.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Methodology](#methodology)
   - [Dataset Challenges](#dataset-challenges)
   - [Base CNN Architecture](#base-cnn-architecture)
   - [Training Progression](#training-progression)
   - [Advanced Architectures](#advanced-architectures)
   - [Optimization and Preprocessing](#optimization-and-preprocessing)
   - [Domain-Specific Normalization](#domain-specific-normalization)
3. [Results](#results)
   - [Pretrained Vision Transformer](#pretrained-vision-transformer)
   - [Ensemble Model](#ensemble-model)
4. [Conclusion](#conclusion)
5. [References](#references)

---

## Introduction

Fire detection in images is a critical computer vision application with potential life-saving implications. This project compares various architectures and introduces novel methods like domain-specific normalization and ensemble modeling to enhance detection accuracy.

---

## Methodology

### Dataset Challenges

The dataset contained non-obvious labeling issues, such as:
- Boats and landscapes labeled as "Fire."
- Firefighters labeled as "Fire."
- Fireworks labeled as "No Fire."

### Base CNN Architecture

A custom CNN model with:
- Four convolutional blocks with increasing depth.
- Batch normalization and ReLU activation.
- Global average pooling before the classification head.

Achieved **97.5% accuracy** with only **4.2M parameters**.

### Training Progression

Tracked training metrics with TensorBoard, highlighting accuracy and loss convergence.

### Advanced Architectures

Explored:
- **Deep CNN with Attention:** Achieved 86% accuracy.
- **Vision Transformers (ViT):** Slow convergence but promising results.
- **ResNet Variants:** Similar performance to the base CNN.
- **Custom DeepResNet:** Achieved **97.7% accuracy**.

### Optimization and Preprocessing

- **Optimizers:** AdamW (faster convergence) and SGD with momentum (consistent improvement).
- **Data Augmentation:** Applied color jitter, horizontal flips, and rotations.

### Domain-Specific Normalization

Calculated dataset-specific statistics:
- Mean: `[0.449, 0.398, 0.345]`
- Standard Deviation: `[0.258, 0.244, 0.262]`

Improved performance by emphasizing the red channel's importance.

---

## Results

### Pretrained Vision Transformer

- Achieved **98.8% accuracy** on the validation set with selective layer freezing.

### Ensemble Model

Combined:
- Pretrained Vision Transformer.
- Base CNN.
- Custom DeepResNet.

Resulted in **99.04% accuracy** on the validation set.

---

## Conclusion

While Vision Transformers achieved the best individual performance, simpler CNN models demonstrated competitive results. The ensemble approach provided marginal improvements, and domain-specific normalization consistently enhanced accuracy.

---

## References

1. Dosovitskiy et al., "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale," 2020.
2. He et al., "Deep Residual Learning for Image Recognition," 2016.
3. Loshchilov and Hutter, "Decoupled Weight Decay Regularization," 2018.
4. Vaswani et al., "Attention is All You Need," 2017.
5. Xie et al., "Aggregated Residual Transformations for Deep Neural Networks," 2017.
