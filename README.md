# Clustering Upper Limb Impairment in Stroke Patients Using Data from Inertial Measurement Units

## Overview
This repository contains the code parts of the implementation of my bachelor thesis research on clustering upper limb impairment in stroke patients using wearable IMU sensor data. The project develops and compares two novel approaches for categorizing real-life upper limb performance, addressing the gap between clinical assessments and everyday functional ability.

## Bachelor Thesis Information
- **Title:** Clustering Upper Limb Impairment in Stroke Patients Using Data from Inertial Measurement Units
- **Author:** Janic Moser
- **University:** ETH Zurich
- **Department:** Department of Computer Science
- **Supervisor:** Prof. Dr. Julia E. Vogt
- **Advisors:** A. Ryser, J. Pohl, C. A. Easthope
- **Date:** January 19, 2025

## Abstract
Current clinical assessments of upper limb function fail to capture the complexities of daily life, often not accurately reflecting upper limb performance in everyday activities. This work aims to derive a simplified categorization of real-life upper limb performance in stroke patients based on wrist sensor data.

Two approaches are proposed:
1. **Feature-based approach**: Uses five established upper limb performance features with dimensionality reduction (PCA) and clustering (K-Means)
2. **Deep learning approach**: Learns features directly from raw time series using contrastive learning (SimCLR framework)

**Key Results:**
- Feature-based approach provides valid categorization into low, medium, and high performance groups
- Deep learning approach successfully captures complex patterns but requires further refinement for clinical applicability
- Both approaches offer practical frameworks for clinicians to evaluate therapy outcomes

## Problem Statement & Motivation
- **Challenge**: Clinical assessments performed in standardized settings don't reflect real-world upper limb performance
- **Opportunity**: Wearable IMU sensors provide precise daily activity measurements
- **Barrier**: Complexity in accessing and interpreting sensor information
- **Solution**: Simplified categorization system for clinical decision-making

## Methodology

### Approach 1: Feature-Based Clustering
Based on established methods by Barth et al.:

1. Preprocessing:

- Feature extraction from IMU signals
- Data cleaning and normalization

2. Feature Selection: Five established upper limb performance features:

- Arm use patterns for affected and unaffected limbs
- Bilateral coordination measures
- Activity intensity metrics
- Movement variability indicators
- Functional use ratios

3. Dimensionality Reduction:

- Applied PCA to feature set for performance representations

4. Clustering:

- K-means clustering on reduced representations
- Validation against clinical scores (Fugl-Meyer Assessment)

### Approach 2: Deep Learning Clustering (SimCLR)
Novel adaptation of SimCLR framework for IMU time series:

1. Preprocessing:

- Time series segmentation into windows
- Data augmentation for contrastive learning
- Normalization and filtering

2. Contrastive Learning:

- Architecture: Custom neural network encoder
- Framework: SimCLR adapted for IMU data
- Training: Contrastive loss on time series slices
- Augmentations: Time-based transformations suitable for IMU signals

3. Training Pipeline:

- Optimizer: AdamW with decoupled weight decay
- Hyperparameter Optimization: Optuna framework
- Batch Design: Specialized batching for contrastive learning

4. Embedding Extraction & Clustering:

- Learn embeddings from raw sensor data
- Cluster embeddings for performance categorization
- Correlation analysis with Fugl-Meyer Assessment scores

## Key Tools & Technologies:

- Machine Learning Frameworks
  - PyTorch: Deep learning implementation
  - Scikit-learn: Traditional ML algorithms and evaluation metrics
  - PCA: Dimensionality reduction and visualization
  - Optuna: Hyperparameter optimization

- Data Processing
  - HDF5 (h5py): Efficient storage and processing of large IMU datasets
  - NumPy/Pandas: Data manipulation and analysis
  - YAML: Configuration management

- Clinical Validation
  - Fugl-Meyer Assessment: Gold standard for upper limb function evaluation
  - Correlation Analysis: Statistical validation of clustering results
  - Clinical Interpretability: Alignment with rehabilitation practices

## Results Summary
### Feature-Based Clustering
- Successfully categorizes patients into three distinct groups (low, medium, high performance)
- Clinical validation against Fugl-Meyer Assessment scores
- Interpretable features aligned with clinical understanding
- Stable clustering across different validation approaches

### Deep Learning Clustering
- Complex pattern capture from raw IMU time series
- Learned representations show clinical relevance
- Clustering quality requires further refinement for clinical application
- Future potential for more sophisticated categorization

## Clinical Significance
This work addresses a critical gap in stroke rehabilitation by:

- Bridging clinical assessment and real-world performance
- Providing objective, continuous monitoring capabilities
- Enabling personalized rehabilitation strategies
- Supporting evidence-based clinical decision making

## Future Directions
- Enhanced deep learning architectures for better clustering quality
- Multi-modal data integration (IMU + clinical + imaging)
- Longitudinal analysis for recovery trajectory prediction
- Clinical trial integration for therapy optimization
- Real-time monitoring systems for immediate feedback

Contact:
jamoser@ethz.ch

Acknowledgments:  
Supervisor: Prof. Dr. Julia E. Vogt  
Advisors: A. Ryser, J. Pohl, C. A. Easthope  
ETH Zurich: Department of Computer Science  
Clinical Partners and data providers: Lake Lucerne Institute (LLUI)  

For detailed methodology, clinical validation, and comprehensive results, please refer to the full thesis document.

Keywords: Stroke Rehabilitation, Upper Limb Assessment, IMU Sensors, Contrastive Learning, SimCLR, Clinical Machine Learning, Wearable Technology, Clustering, Digital Health