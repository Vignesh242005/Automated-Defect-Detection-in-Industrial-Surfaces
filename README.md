# Defect Segmentation on Kolektor SDD2 Dataset

## Overview
This project implements defect segmentation using computer vision techniques on the **Kolektor SDD2 dataset**. It includes image preprocessing, segmentation using K-Means clustering, edge detection, morphological operations, and evaluation with IoU and Dice scores. The results are visualized and analyzed.

## Features
- **Image Preprocessing**: Histogram equalization, filtering (Gaussian, Median, Bilateral)
- **Segmentation**: K-Means clustering, region growing, edge detection
- **Evaluation Metrics**: Intersection over Union (IoU) and Dice Coefficient
- **Visualization**: Segmentation comparison plots, histogram, boxplot, and trend analysis
- **Parallel Processing**: Speeds up dataset processing using threading

## Dataset
- **Train & Test Data**: Stored in respective folders within `KolektorSDD2`
- **Ground Truth Masks**: Used for evaluating segmentation performance

## Installation
Ensure you have the required dependencies:
```bash
pip install opencv-python numpy matplotlib scikit-image scikit-learn pandas seaborn
```

## Usage
### 1. Run Image Processing & Segmentation
```bash
python cv.py
```
This script processes images, applies segmentation, evaluates results, and saves output images & metrics.

### 2. Run Analysis & Visualization
```bash
python virtual_comparison.py
```
This script reads segmentation metrics, generates statistical plots, and saves visualizations.

## Output
- **Segmented Images**: Stored in `KolektorSDD2/`
- **Metrics CSV**: `segmentation_metrics.csv`
- **Visualization Plots**: Stored in `processed_results/`

## Example Workflow
1. Place dataset images and ground truth masks in `KolektorSDD2/train`, `KolektorSDD2/train_gt`, `KolektorSDD2/test`, `KolektorSDD2/test_gt`
2. Run `cv.py` to process images
3. Run `virtual_comparison.py` to analyze segmentation performance

## Author
Developed by **Vignesh**.

---
This project demonstrates defect detection and segmentation techniques using the Kolektor SDD2 dataset. Feel free to contribute or raise issues!

