from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import pandas as pd
import sys
import os
import torch

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from threshold.utils.custom_dataset import CytologyCombinedDataset
from threshold.utils.threshold_batch_sampler import ThresholdBasedBatchSampler


def collate_fn_with_metadata(batch):
    """
    Custom collate function to handle batches with metadata (is_syn, jsd_scores).
    
    Args:
        batch: List of (img, label, is_syn, jsd_score) tuples
    
    Returns:
        Batched tensors: (imgs, labels, is_syn, jsd_scores)
    """
    imgs = torch.stack([item[0] for item in batch])
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    is_syn = torch.tensor([item[2] for item in batch], dtype=torch.bool)
    jsd_scores = torch.tensor([item[3] for item in batch], dtype=torch.float32)
    
    return imgs, labels, is_syn, jsd_scores


def get_train_loader_threshold(
    data_dir,
    batch_size,
    threshold=None,
    dataset_csv='dataset.csv',
    num_workers=0
):
    """
    Get train loader with threshold-based filtering.
    
    Args:
        data_dir: Base data directory
        batch_size: Batch size
        threshold: JSD threshold (None = no synthetic)
        dataset_csv: Path to dataset CSV with JSD scores
        num_workers: Number of DataLoader workers
    
    Returns:
        DataLoader with threshold-filtered samples
    """
    
    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(10),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2
        ),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.05, 0.05),
            scale=(0.9, 1.1)
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Load JSD scores from CSV BEFORE creating dataset
    df = pd.read_csv(dataset_csv)
    
    # Filter for synthetic images and only those in train_synthetic folder
    # by checking if 'train_synthetic' is in the filename path
    synthetic_df = df[
        (df['is_syn'] == 1) & 
        (df['filename'].str.contains('train_synthetic', case=False, na=False))
    ].reset_index(drop=True)
    
    # Extract JSD scores
    synthetic_jsd_scores = synthetic_df['jsd_score'].tolist()
    
    # Create dataset first to get counts
    train_ds_temp = datasets.ImageFolder(
        f"{data_dir}/train_synthetic",
        transform=train_tf
    )
    synth_count = len(train_ds_temp)
    
    # Verify and pad if needed
    if len(synthetic_jsd_scores) != synth_count:
        print(f"\n⚠️  WARNING: Synthetic count mismatch!")
        print(f"   CSV has {len(synthetic_jsd_scores)} but dataset has {synth_count}")
        print(f"   Padding with zeros (will treat as high-quality)\n")
        
        # Pad with zeros if needed
        while len(synthetic_jsd_scores) < synth_count:
            synthetic_jsd_scores.append(0.0)
        synthetic_jsd_scores = synthetic_jsd_scores[:synth_count]
    
    # Now create the combined dataset with JSD scores
    train_ds = CytologyCombinedDataset(
        real_root=f"{data_dir}/train",
        synth_root=f"{data_dir}/train_synthetic",
        transform=train_tf,
        jsd_scores=synthetic_jsd_scores
    )
    
    real_indices = list(range(train_ds.real_len))
    synth_indices = list(range(train_ds.real_len, len(train_ds)))

    # Count real class distribution from dataset
    num_benign = sum(1 for t in train_ds.real_ds.targets if t == 0)
    num_malignant = train_ds.real_len - num_benign
    
    print(f"Loading training data:")
    print(f"  Real images: {train_ds.real_len} (BNV: {num_benign}, MEL: {num_malignant})")
    print(f"  Synthetic in folder: {train_ds.synth_len}")
    print(f"  Synthetic in CSV (train_synthetic): {len(synthetic_df)}")
    
    sampler = ThresholdBasedBatchSampler(
        real_indices=real_indices,
        synthetic_indices=synth_indices,
        synthetic_jsd_scores=synthetic_jsd_scores,
        batch_size=batch_size,
        threshold=threshold,
        num_benign=num_benign
    )
    
    return DataLoader(
        train_ds,
        batch_sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_fn_with_metadata,
    )


def get_val_loader(data_dir, batch_size, num_workers=0):
    """
    Get validation loader (only real images, no synthetic).
    """
    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Use ImageFolder directly for validation (no synthetic data)
    val_ds = datasets.ImageFolder(
        root=f"{data_dir}/val",
        transform=val_tf
    )
    
    return DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )


def get_test_loader(data_dir, batch_size, num_workers=0):
    """
    Get test loader (only real images, no synthetic).
    """
    test_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Use ImageFolder directly for test (no synthetic data)
    test_ds = datasets.ImageFolder(
        root=f"{data_dir}/test",
        transform=test_tf
    )
    
    return DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
