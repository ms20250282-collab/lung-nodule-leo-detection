# Enhanced Deep Feature Selection Using LEO for Lung Nodule Detection

A comprehensive deep learning system for pulmonary nodule detection and classification in CT (Computed Tomography) images using Lagrange Elementary Optimization (LEO) for intelligent feature selection.

## 📋 Project Overview

This project implements an advanced framework for:
- **Nodule Detection**: Locating candidate nodules in CT scans
- **Segmentation**: Precise boundary delineation of detected nodules
- **Classification**: Benign vs Malignant classification
- **Feature Selection**: Using LEO optimization for optimal feature subset selection
- **Radiomics Analysis**: Extracting and selecting radiomics features
- **Deep Feature Fusion**: Combining CNN and Transformer representations

## 🏗️ Project Architecture

```
lung-nodule-leo-detection/
├── data/
│   ├── raw/                 # Raw MHD CT images
│   ├── processed/           # Preprocessed data
│   └── datasets/            # LUNA16, LIDC-IDRI splits
│
├── models/
│   ├── detection/           # Detection models (YOLO, Faster R-CNN)
│   ├── segmentation/        # Segmentation models (U-Net, SegNet)
│   ├── classification/      # Classification models (ResNet, DenseNet)
│   └── transformers/        # Vision Transformer architectures
│
├── leo/
│   ├── optimizer.py         # LEO optimization algorithm
│   ├── feature_selector.py  # Enhanced feature selection wrapper
│   └── constraints.py       # Constraint handling for LEO
│
├── feature_selection/
│   ├── radiomics.py         # Radiomics feature extraction
│   ├── deep_features.py     # Deep feature extraction from CNNs
│   ├── fusion.py            # Feature fusion strategies
│   ├── lasso.py             # LASSO selector
│   └── mrmr.py              # Minimum Redundancy Maximum Relevance
│
├── preprocessing/
│   ├── mhd_loader.py        # MHD file loading and parsing
│   ├── normalization.py     # CT intensity normalization
│   ├── augmentation.py      # Data augmentation
│   └── windowing.py         # CT window/level adjustment
│
├── evaluation/
│   ├── metrics.py           # Detection & classification metrics
│   ├── validation.py        # Cross-validation strategies
│   ├── calibration.py       # Probability calibration
│   └── stability.py         # Feature stability analysis
│
├── config/
│   ├── config.yaml          # Main configuration
│   ├── leo_config.yaml      # LEO-specific parameters
│   └── model_config.yaml    # Model architectures
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_model_training.ipynb
│   ├── 04_feature_selection.ipynb
│   └── 05_evaluation.ipynb
│
├── tests/
│   ├── test_mhd_loader.py
│   ├── test_leo_optimizer.py
│   ├── test_models.py
│   └── test_preprocessing.py
│
├── scripts/
│   ├── train_detector.py
│   ├── train_classifier.py
│   ├── apply_leo_selection.py
│   └── evaluate_model.py
│
├── utils/
│   ├── logger.py            # Logging utilities
│   ├── helpers.py           # Helper functions
│   └── visualization.py     # Visualization tools
│
├── requirements.txt
├── setup.py
└── README.md
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/ms20250282-collab/lung-nodule-leo-detection.git
cd lung-nodule-leo-detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Data

```bash
# Place MHD files in data/raw/
# MHD files should be paired with RAW files
# Example structure:
# data/raw/
#   ├── scan_001.mhd
#   ├── scan_001.raw
#   ├── scan_002.mhd
#   └── scan_002.raw
```

### 3. Preprocessing

```python
from preprocessing.mhd_loader import MHDLoader
from preprocessing.normalization import CTNormalizer

# Load MHD file
loader = MHDLoader('data/raw/scan_001.mhd')
ct_image, metadata = loader.load()

# Normalize CT intensity
normalizer = CTNormalizer()
ct_normalized = normalizer.normalize(ct_image)
```

### 4. Train Detection Model

```bash
python scripts/train_detector.py --config config/config.yaml
```

### 5. Apply LEO Feature Selection

```bash
python scripts/apply_leo_selection.py --model-path checkpoints/detector.pth
```

## 📊 Supported Tasks

### 1. **Nodule Detection**
- YOLO v8 for real-time detection
- Faster R-CNN for high accuracy
- RetinaNet for balanced performance
- Sensitivity, Specificity, F1-Score metrics

### 2. **Nodule Segmentation**
- U-Net architecture
- SegNet with encoder-decoder
- Dice loss and Cross-entropy hybrid loss
- Boundary-aware segmentation

### 3. **Malignancy Classification**
- ResNet-50/101 pre-trained on ImageNet
- DenseNet for feature reuse
- Inception-v3 for multi-scale features
- Transformer-based classifiers

### 4. **Feature Selection**
- **LASSO**: L1-regularized linear regression
- **mRMR**: Minimum Redundancy Maximum Relevance
- **LEO**: Lagrange Elementary Optimization (population-based)
- **Hybrid LEO**: Enhanced with stability and constraint handling

## 🧬 LEO Algorithm Features

### Core Optimization
```
LEO = Lagrange Elementary Optimization
├── Population-based Exploration
├── Genetic Crossover & Mutation
├── Lagrange Multiplier Guidance
├── Stationary Point Navigation
└── Adaptive Parameter Tuning
```

### Enhanced Features
- **Nested Cross-Validation**: Prevents data leakage
- **Stability Analysis**: Feature consistency across folds
- **Constraint Handling**: Domain-specific constraints
- **Multi-objective**: Balance accuracy vs feature count
- **External Validation**: Generalization assessment

## 📈 Evaluation Metrics

### Detection Metrics
- Sensitivity (True Positive Rate)
- Specificity (True Negative Rate)
- F1-Score
- FROC (Free Response Operating Characteristic)
- Average Precision (AP@IoU=0.5)

### Classification Metrics
- Accuracy, Precision, Recall, F1
- ROC-AUC
- PR-AUC
- Calibration Error
- Brier Score

### Feature Selection Metrics
- Stability Index
- Redundancy Measure
- Feature Importance Scores
- Cross-validation Performance

## 🔧 Configuration

### Main Config (config/config.yaml)
```yaml
data:
  data_dir: "data/"
  train_split: 0.7
  val_split: 0.15
  test_split: 0.15
  num_workers: 4

preprocessing:
  target_spacing: [1.0, 1.0, 1.0]  # mm
  window_level: 40
  window_width: 400
  normalize: true

model:
  detector: "yolov8"
  classifier: "resnet50"
  device: "cuda"
  batch_size: 16

training:
  epochs: 100
  learning_rate: 0.001
  optimizer: "adam"
  early_stopping: true
  patience: 10

leo:
  population_size: 50
  generations: 200
  mutation_rate: 0.1
  crossover_rate: 0.8
  feature_selection_mode: "hybrid"
```

## 📚 Datasets

### Supported Datasets
- **LUNA16**: Pulmonary Nodule Analysis Challenge (888 CTs)
- **LIDC-IDRI**: Lung Image Database Consortium (1,018 CTs)
- **LUNGx**: Challenge dataset (73 CTs)
- **Custom MHD datasets**: Any MHD/RAW format CT data

## 🎯 Research Priorities (from Review)

✅ **Implemented:**
1. Patient-level leakage control (nested CV)
2. External validation framework
3. Stability analysis across folds
4. Calibrated risk estimates
5. Ablation study utilities

⏳ **To Implement:**
1. Multi-center testing protocol
2. Radiologist interaction interface
3. Failure mode analysis
4. Subgroup performance analysis

## 📖 Usage Examples

### Example 1: Load and Preprocess MHD File

```python
from preprocessing.mhd_loader import MHDLoader
from preprocessing.normalization import CTNormalizer

# Load MHD file
loader = MHDLoader('path/to/scan.mhd')
image, metadata = loader.load()

# Normalize
normalizer = CTNormalizer()
image_norm = normalizer.normalize(image)

print(f"Image shape: {image_norm.shape}")
print(f"Metadata: {metadata}")
```

### Example 2: Train Detection Model

```python
import torch
from models.detection.yolo_detector import YOLODetector
from torch.utils.data import DataLoader

# Initialize model
model = YOLODetector(pretrained=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Load data
train_loader = DataLoader(train_dataset, batch_size=16)

# Train
for epoch in range(100):
    for batch in train_loader:
        outputs = model(batch['images'])
        loss = model.compute_loss(outputs, batch['targets'])
        loss.backward()
```

### Example 3: Apply LEO Feature Selection

```python
from leo.feature_selector import LEOFeatureSelector
from feature_selection.radiomics import RadiomicsExtractor

# Extract features
extractor = RadiomicsExtractor()
features = extractor.extract(ct_image, nodule_mask)

# Apply LEO selection
selector = LEOFeatureSelector(
    population_size=50,
    generations=200,
    objective='balanced'  # balance accuracy vs feature count
)

selected_indices = selector.fit_transform(
    X=features,
    y=labels,
    cv=5  # 5-fold cross-validation
)

selected_features = features[:, selected_indices]
```

### Example 4: Evaluate with Stability Analysis

```python
from evaluation.stability import StabilityAnalyzer

stability_analyzer = StabilityAnalyzer()
stability_scores = stability_analyzer.analyze(
    feature_selector=selector,
    X=features,
    y=labels,
    num_iterations=10,
    test_size=0.2
)

print(f"Stability Index: {stability_scores['stability_index']:.3f}")
print(f"Jaccard Index: {stability_scores['jaccard_index']:.3f}")
```

## 🧪 Testing

```bash
# Run all tests
python -m pytest tests/

# Run specific test
python -m pytest tests/test_mhd_loader.py -v

# With coverage
python -m pytest tests/ --cov=. --cov-report=html
```

## 📊 Visualization

```python
from utils.visualization import CTVisualizer

visualizer = CTVisualizer()

# Visualize CT slice with nodule
visualizer.plot_ct_with_nodule(
    ct_image=ct_image,
    nodule_mask=nodule_mask,
    title='Lung Nodule Detection'
)

# Visualize feature importance
visualizer.plot_feature_importance(
    features=feature_names,
    importance=importance_scores
)
```

## 🔍 Ablation Studies

The framework includes utilities for conducting ablations:

```python
from evaluation.ablation import AblationStudy

ablation = AblationStudy()
results = ablation.run(
    model=model,
    feature_groups={
        'preprocessing': ['normalization', 'windowing'],
        'features': ['radiomics', 'deep'],
        'selection': ['lasso', 'leo']
    },
    dataset=test_dataset
)
```

## 📄 Citation

If you use this project in your research, please cite:

```bibtex
@software{leo_lung_nodule_2024,
  title={Enhanced Deep Feature Selection Using LEO for Lung Nodule Detection},
  author={Your Name},
  year={2024},
  url={https://github.com/ms20250282-collab/lung-nodule-leo-detection}
}
```

## 📝 License

MIT License - See LICENSE file for details

## 👥 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 🤝 Support

For issues, questions, or suggestions, please open an GitHub issue.

## 📚 References

1. Li et al. - LUNA16: Pulmonary Nodule Analysis Challenge
2. Xu et al. - Nodule Growth Prediction using Radiomics and Deep Features
3. Aladdin & Rashid - LEO: Lagrange Elementary Optimization
4. 2025 Review - LEO Exploitation Stage Analysis

## 🎓 Further Reading

- [CT Image Analysis Guide](https://www.ncbi.nlm.nih.gov/pubmed/)
- [Deep Learning for Medical Imaging](https://arxiv.org/)
- [Feature Selection in Machine Learning](https://scikit-learn.org/)

---

**Last Updated:** 2024
**Status:** Active Development 🚀
