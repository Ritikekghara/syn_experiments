This repository implements a binary medical image classification pipeline using a pretrained ResNet-18 backbone. The framework supports curriculum learning with synthetic data, class imbalance handling, and comprehensive evaluation metrics.

## Dataset Structure

```
SYN_EXPERIMENTS/
│
├── data/
│   ├── train/                       # Real training data
│   │   ├── BNV/
│   │   └── MEL/
│   │
│   ├── train_synthetic/             # Synthetic data (training only)
│   │   └── MEL/
│   │
│   ├── val/                         # Validation set (real only)
│   │   ├── BNV/
│   │   └── MEL/
│   │
│   └── test/                        # Test set (real only)
│       ├── BNV/
│       └── MEL/
```

## Repository Structure

```
syn_experiments/
│
├── evaluate_thresholds.py           # Threshold evaluation entry point
├── generate_dataset_with_jsd.py     # Dataset synthesis/analysis helper
├── threshold_train_main.py          # Training entry point for thresholding
├── requirements.txt                 # Python dependencies
├── README.md                        # Project documentation
│
├── models/
│   ├── __init__.py
│   ├── resnet50.py                  # ResNet-50 backbone
│   ├── swin.py                      # Swin Transformer backbone
│   └── vit.py                       # Vision Transformer backbone
│
└── threshold/
	├── training/
	│   └── train.py                 # Threshold training loop
	│
	└── utils/
		├── compute_recall_threshold.py
		├── custom_dataset.py
		├── dataloader.py
		└── threshold_batch_sampler.py
```
## Install the requirements
```
pip install -r requirements.txt
```
## Run the Script

```
python main.py
```
