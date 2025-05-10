import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import os
import pickle
import numpy as np

# Import the individual dataset loaders
from dataset_ck import CKPlusPairedDataset
from dataset_kdef import KDEFPairedDataset

def create_combined_dataloaders(
    ckplus_data_path='../processed_data/aligned_ck_data.pkl',
    kdef_data_path='../processed_data/aligned_kdef_data.pkl',
    image_size=224,
    batch_size=16,
    num_workers=2,
    target_emotion=None,
    augmentation_factor=0,
    train_ratio=0.85
):
    """
    Create combined DataLoaders for both CK+ and KDEF aligned datasets with on-the-fly 90-10 split
    
    Args:
        ckplus_data_path: Path to CK+ aligned data file
        kdef_data_path: Path to KDEF aligned data file
        image_size: Size to resize images to
        batch_size: Batch size for DataLoader
        num_workers: Number of workers for DataLoader
        target_emotion: Optional emotion(s) to filter for (e.g., 'happiness' or ['happiness', 'anger'])
        augmentation_factor: How many augmented versions to add for each original pair
        train_ratio: Ratio of training data (default: 0.9)
        
    Returns:
        Tuple of (train_loader, test_loader)
    """
    print("Creating combined CK+ and KDEF dataloaders...")
    
    # Load CK+ data
    with open(ckplus_data_path, 'rb') as f:
        ckplus_data = pickle.load(f)
    
    ckplus_df = ckplus_data['dataset_df']
    ckplus_df = ckplus_df[ckplus_df['dataset'] == 'CK+'].reset_index(drop=True)
    
    # Get unique CK+ subjects
    ckplus_subjects = ckplus_df['subject_id'].unique()
    
    # Shuffle subjects for random splits
    np.random.seed(42)  # For reproducibility
    np.random.shuffle(ckplus_subjects)
    
    # Split CK+ subjects into train and test
    n_train_ck = int(len(ckplus_subjects) * train_ratio)
    ckplus_train_subjects = ckplus_subjects[:n_train_ck]
    ckplus_test_subjects = ckplus_subjects[n_train_ck:]
    
    # Create dataframes for CK+ train and test splits
    ckplus_train_df = ckplus_df[ckplus_df['subject_id'].isin(ckplus_train_subjects)].reset_index(drop=True)
    ckplus_test_df = ckplus_df[ckplus_df['subject_id'].isin(ckplus_test_subjects)].reset_index(drop=True)
    
    # Load KDEF data
    with open(kdef_data_path, 'rb') as f:
        kdef_data = pickle.load(f)
    
    kdef_df = kdef_data['dataset_df']
    kdef_df = kdef_df[kdef_df['dataset'] == 'KDEF'].reset_index(drop=True)
    
    # Get unique KDEF subjects
    kdef_subjects = kdef_df['subject_id'].unique()
    
    # Shuffle subjects for random splits
    np.random.seed(42)  # For reproducibility
    np.random.shuffle(kdef_subjects)
    
    # Split KDEF subjects into train and test
    n_train_kdef = int(len(kdef_subjects) * train_ratio)
    kdef_train_subjects = kdef_subjects[:n_train_kdef]
    kdef_test_subjects = kdef_subjects[n_train_kdef:]
    
    # Create dataframes for KDEF train and test splits
    kdef_train_df = kdef_df[kdef_df['subject_id'].isin(kdef_train_subjects)].reset_index(drop=True)
    kdef_test_df = kdef_df[kdef_df['subject_id'].isin(kdef_test_subjects)].reset_index(drop=True)
    
    # Create CK+ datasets
    ckplus_train_dataset = CKPlusPairedDataset(
        ckplus_data_path, 
        split_df=ckplus_train_df,
        image_size=image_size, 
        augmentation_factor=augmentation_factor,
        target_emotion=target_emotion
    )
    
    ckplus_test_dataset = CKPlusPairedDataset(
        ckplus_data_path, 
        split_df=ckplus_test_df,
        image_size=image_size, 
        augmentation_factor=0,  # No augmentation for testing
        target_emotion=target_emotion
    )
    
    # Create KDEF datasets
    kdef_train_dataset = KDEFPairedDataset(
        kdef_data_path, 
        split_df=kdef_train_df,
        image_size=image_size, 
        augmentation_factor=augmentation_factor,
        target_emotion=target_emotion
    )
    
    kdef_test_dataset = KDEFPairedDataset(
        kdef_data_path, 
        split_df=kdef_test_df,
        image_size=image_size, 
        augmentation_factor=0,  # No augmentation for testing
        target_emotion=target_emotion
    )
    
    # Create combined datasets
    combined_train_dataset = ConcatDataset([ckplus_train_dataset, kdef_train_dataset])
    combined_test_dataset = ConcatDataset([ckplus_test_dataset, kdef_test_dataset])
    
    # Create DataLoaders
    train_loader = DataLoader(
        combined_train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        combined_test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Print dataset statistics
    print(f"Combined train dataset: {len(combined_train_dataset)} pairs")
    print(f"  - CK+: {len(ckplus_train_dataset)} pairs from {len(ckplus_train_subjects)} subjects")
    print(f"  - KDEF: {len(kdef_train_dataset)} pairs from {len(kdef_train_subjects)} subjects")
    print(f"Combined test dataset: {len(combined_test_dataset)} pairs")
    print(f"  - CK+: {len(ckplus_test_dataset)} pairs from {len(ckplus_test_subjects)} subjects")
    print(f"  - KDEF: {len(kdef_test_dataset)} pairs from {len(kdef_test_subjects)} subjects")
    
    return train_loader, test_loader

# For testing
if __name__ == "__main__":
    train_loader, test_loader = create_combined_dataloaders()
    print(f"Combined train loader: {len(train_loader.dataset)} samples")
    print(f"Combined test loader: {len(test_loader.dataset)} samples")
    
    # Test by loading a batch
    for batch in train_loader:
        print(f"Batch size: {len(batch['source_image'])}")
        print(f"Source image shape: {batch['source_image'].shape}")
        print(f"Target image shape: {batch['target_image'].shape}")
        break
