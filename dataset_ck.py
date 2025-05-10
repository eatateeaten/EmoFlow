import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from PIL import Image
import pickle
from collections import defaultdict
import os
import random
import numpy as np

class PairedTransform:
    """
    Apply the same random transformations to a pair of images
    to ensure alignment between source and target faces.
    """
    def __init__(self, image_size=224, augment=True):
        self.image_size = image_size
        self.augment = augment
        
    def __call__(self, source_img, target_img):
        # Resize both images
        source_img = F.resize(source_img, (self.image_size, self.image_size))
        target_img = F.resize(target_img, (self.image_size, self.image_size))
        
        if self.augment:
            # Apply the SAME random horizontal flip to both images
            if random.random() > 0.5:
                source_img = F.hflip(source_img)
                target_img = F.hflip(target_img)
            
            # Apply the SAME random brightness/contrast adjustment
            brightness_factor = 1.0 + random.uniform(-0.3, 0.3)
            contrast_factor = 1.0 + random.uniform(-0.3, 0.3)
            
            source_img = F.adjust_brightness(source_img, brightness_factor)
            source_img = F.adjust_contrast(source_img, contrast_factor)
            
            target_img = F.adjust_brightness(target_img, brightness_factor)
            target_img = F.adjust_contrast(target_img, contrast_factor)
        
        # Convert to tensor and normalize
        source_tensor = F.to_tensor(source_img)
        target_tensor = F.to_tensor(target_img)
        
        source_tensor = F.normalize(source_tensor, mean=[0.5], std=[0.5])
        target_tensor = F.normalize(target_tensor, mean=[0.5], std=[0.5])
        
        return source_tensor, target_tensor

class CKPlusPairedDataset(Dataset):
    """
    Dataset for CK+ paired facial expressions
    
    Produces pairs of [source, target] images where:
    - Both source and target are from the same subject
    - Source emotion is always 'neutral'
    - Target emotion is any emotion except 'neutral', or a specific emotion if provided
    
    Args:
        data_path: Path to processed aligned data pickle file
        split_df: DataFrame containing the train or test split
        image_size: Size to resize images to
        augmentation_factor: How many augmented versions to add for each original pair
        target_emotion: Optional specific target emotion(s) to filter for (e.g., 'happiness' or ['happiness', 'anger'])
                        If None, includes all non-neutral emotions
    """
    def __init__(self, data_path='../processed_data/aligned_ck_data.pkl', 
                 split_df=None, image_size=224, augmentation_factor=0,
                 target_emotion=None):
        # Load the processed data
        if split_df is None:
            with open(data_path, 'rb') as f:
                data = pickle.load(f)
                self.df = data['dataset_df']
                self.emotion_embeddings = data['emotion_embeddings']
        else:
            self.df = split_df
            # Load emotion embeddings separately if needed
            with open(data_path, 'rb') as f:
                data = pickle.load(f)
                self.emotion_embeddings = data['emotion_embeddings']
        
        # Filter only CK+ entries
        self.df = self.df[self.df['dataset'] == 'CK+'].reset_index(drop=True)
        
        # Store target_emotion if provided
        self.target_emotion = target_emotion
        
        # Convert single target emotion to list for consistent handling
        if self.target_emotion is not None and not isinstance(self.target_emotion, list):
            self.target_emotion = [self.target_emotion]
        
        # Group images by subject
        self.subject_images = defaultdict(lambda: defaultdict(list))
        for idx, row in self.df.iterrows():
            subject_id = row['subject_id']
            emotion = row['emotion']
            self.subject_images[subject_id][emotion].append(idx)
        
        # Create pairs of [neutral, non-neutral] images
        self.pairs = []
        for subject_id, emotions in self.subject_images.items():
            # Check if this subject has neutral images
            if 'neutral' in emotions:
                neutral_indices = emotions['neutral']  # List of neutral image indices
                
                # For each neutral image
                for neutral_idx in neutral_indices:
                    # Pair with target emotion(s) or all non-neutral emotions for this subject
                    if self.target_emotion:
                        # Only include the specific target emotion(s)
                        for target_emotion_name in self.target_emotion:
                            if target_emotion_name in emotions:
                                for target_idx in emotions[target_emotion_name]:
                                    # Verify indices are valid
                                    if neutral_idx < len(self.df) and target_idx < len(self.df):
                                        self.pairs.append((neutral_idx, target_idx))
                    else:
                        # Include all non-neutral emotions
                        for emotion, indices in emotions.items():
                            if emotion != 'neutral':
                                for target_idx in indices:
                                    # Verify indices are valid
                                    if neutral_idx < len(self.df) and target_idx < len(self.df):
                                        self.pairs.append((neutral_idx, target_idx))
        
        # Print warning if we discarded any pairs
        if self.target_emotion:
            description = f"neutral-to-{'/'.join(self.target_emotion)}"
        else:
            description = "neutral-to-emotion"
        
        # Map emotions to indices
        self.emotion_to_idx = {
            'neutral': 0, 
            'anger': 1, 
            'contempt': 2, 
            'disgust': 3, 
            'fear': 4, 
            'happiness': 5, 
            'sadness': 6, 
            'surprise': 7
        }
        
        # Store original pairs and metadata before augmentation
        self.original_pairs = self.pairs.copy()
        self.num_original_pairs = len(self.original_pairs)
        
        # Add augmented pairs if requested
        self.augmentation_factor = augmentation_factor
        self.image_size = image_size
        
        if augmentation_factor > 0:
            # The original pairs will be accessed with their original indices
            # Augmented pairs will have indices after the original ones
            original_pairs_count = len(self.original_pairs)
            
            for i in range(augmentation_factor):
                # Add copies of original pairs that will get different augmentations
                # These are identified by a tuple (original_idx, augmentation_id)
                for j in range(original_pairs_count):
                    self.pairs.append(self.original_pairs[j])
                    
            print(f"Added {augmentation_factor * original_pairs_count} augmented pairs")
        
        # Create transforms
        self.regular_transform = PairedTransform(image_size=image_size, augment=False)
        self.augment_transform = PairedTransform(image_size=image_size, augment=True)
        
        print(f"Created {len(self.pairs)} {description} pairs for {len(self.subject_images)} subjects")
    
    def __len__(self):
        """Get the number of pairs in the dataset"""
        return len(self.pairs)
    
    def __getitem__(self, idx):
        """
        Get a pair of source (neutral) and target (emotional) images
        
        Args:
            idx: Index of the pair to fetch
            
        Returns:
            dict containing source and target image tensors and metadata
        """
        # Handle index out of range by wrapping around
        if idx >= len(self.pairs):
            idx = idx % len(self.pairs)

        source_idx, target_idx = self.pairs[idx]

        # Get source image data (neutral)
        source_entry = self.df.iloc[source_idx]
        source_path = source_entry['image_path']
        source_emotion = source_entry['emotion']
        subject_id = source_entry['subject_id']

        # Get target image data (non-neutral emotion)
        target_entry = self.df.iloc[target_idx]
        target_path = target_entry['image_path']
        target_emotion = target_entry['emotion']

        # Load images as grayscale
        source_image = Image.open(source_path).convert('L')
        target_image = Image.open(target_path).convert('L')

        # Determine if this is an original or augmented sample
        is_augmented = idx >= self.num_original_pairs
        
        # Apply appropriate transform (augmented or regular)
        if is_augmented:
            source_tensor, target_tensor = self.augment_transform(source_image, target_image)
        else:
            source_tensor, target_tensor = self.regular_transform(source_image, target_image)

        # Get emotion indices
        source_emotion_idx = self.emotion_to_idx.get(source_emotion, 0)  # Should always be 0 (neutral)
        target_emotion_idx = self.emotion_to_idx.get(target_emotion, 0)

        # Create one-hot encoding
        source_one_hot = torch.zeros(8)
        source_one_hot[source_emotion_idx] = 1

        target_one_hot = torch.zeros(8)
        target_one_hot[target_emotion_idx] = 1
        
        # Get CLIP embeddings
        source_embedding = self.emotion_embeddings[source_emotion]
        target_embedding = self.emotion_embeddings[target_emotion]
        
        return {
            'source_image': source_tensor,
            'target_image': target_tensor,
            'subject_id': subject_id,
            'source_emotion': source_emotion,
            'target_emotion': target_emotion,
            'source_emotion_idx': source_emotion_idx,
            'target_emotion_idx': target_emotion_idx,
            'source_one_hot': source_one_hot,
            'target_one_hot': target_one_hot,
            'source_embedding': source_embedding,
            'target_embedding': target_embedding,
            'source_path': source_path,
            'target_path': target_path,
            'is_augmented': is_augmented,
        }

class CKPlusNeutralHappyDataset(CKPlusPairedDataset):
    """
    Dataset specifically for neutral-to-happy facial expression pairs
    
    This is a convenience class that filters for only 'happiness' emotion
    """
    def __init__(self, data_path='../processed_data/aligned_ck_data.pkl', 
                 split_df=None, image_size=224, augmentation_factor=0):
        super().__init__(data_path, split_df, image_size, augmentation_factor, 
                         target_emotion='happiness')

def create_dataloaders(data_path='../processed_data/aligned_ck_data.pkl',
                       image_size=224,
                       batch_size=16, 
                       num_workers=2,
                       target_emotion=None,
                       augmentation_factor=0,
                       train_ratio=0.9):
    """
    Create DataLoaders for CK+ dataset with on-the-fly 90-10 split
    
    Args:
        data_path: Path to aligned_ck_data.pkl file
        image_size: Size to resize images to
        batch_size: Batch size for DataLoader
        num_workers: Number of workers for DataLoader
        target_emotion: Optional emotion(s) to filter for (e.g., 'happiness' or ['happiness', 'anger'])
        augmentation_factor: How many augmented versions to add for each original pair
        train_ratio: Ratio of training data (default: 0.9)
        
    Returns:
        Tuple of (train_loader, test_loader)
    """
    # Load the processed data
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    df = data['dataset_df']
    
    # Filter only CK+ entries
    df = df[df['dataset'] == 'CK+'].reset_index(drop=True)
    
    # Get unique subjects
    subjects = df['subject_id'].unique()
    
    # Shuffle subjects for random splits
    np.random.seed(42)  # For reproducibility
    np.random.shuffle(subjects)
    
    # Split subjects into train and test
    n_train = int(len(subjects) * train_ratio)
    train_subjects = subjects[:n_train]
    test_subjects = subjects[n_train:]
    
    # Create dataframes for train and test splits
    train_df = df[df['subject_id'].isin(train_subjects)].reset_index(drop=True)
    test_df = df[df['subject_id'].isin(test_subjects)].reset_index(drop=True)
    
    # Create datasets
    train_dataset = CKPlusPairedDataset(
        data_path, 
        split_df=train_df,
        image_size=image_size, 
        augmentation_factor=augmentation_factor,
        target_emotion=target_emotion
    )
    
    test_dataset = CKPlusPairedDataset(
        data_path, 
        split_df=test_df,
        image_size=image_size, 
        augmentation_factor=0,  # No augmentation for test set
        target_emotion=target_emotion
    )
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"Created train split with {len(train_df)} images from {len(train_subjects)} subjects")
    print(f"Created test split with {len(test_df)} images from {len(test_subjects)} subjects")
    
    return (train_loader, test_loader)
