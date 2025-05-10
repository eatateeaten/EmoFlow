# Facial Expression Editing System

A deep learning system for editing facial expressions while preserving identity, using a UNet-based Conditional Flow Matching approach.

## Features

- **Identity preservation**: Edit expressions without affecting identity
- **Smooth transitions**: Create natural transitions between emotions
- **Perceptual quality**: Uses LPIPS and VGG-based perceptual losses

## Installation

1. Clone the repository:
   ```bash
   git clone [repository-url]
   cd [repository-directory]
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training

Train the model using the following command:

```bash
python train.py
```

### Evaluation

The system evaluates generated expressions across different emotions:
- Transforms neutral faces to target emotions
- Supports evaluation on specific emotion subsets
- Visualizes results with WandB integration when enabled

## Dataset

The system is trained on two datasets:
- **Extended Cohn-Kanade (CK+)**: Contains sequences of facial expressions from neutral to peak emotion
- **KDEF**: Karolinska Directed Emotional Faces dataset with multiple expressions
