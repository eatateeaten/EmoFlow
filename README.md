![image](https://github.com/user-attachments/assets/561281d9-bc6a-48c2-a56f-c2556a06a024)


# Facial Expression Editing System

A fast-as-f(>80fps) for editing facial expressions while preserving identity, using a UNet-based Conditional Flow Matching approach.

## Features

- **Identity preservation**: Edit expressions without affecting identity
- **Smooth transitions**: Create natural transitions between emotions
- **Perceptual quality**: Uses LPIPS and VGG-based perceptual losses

## Installation

1. Clone the repository:
```bash
git clone [repo]
cd emoflow
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Apply patch to torchcfm:
```bash
python patch.py  # fix the path in the file if it's not correct
```

4. Unzip data:
```bash
cd data
unzip CK+_aligned.zip
unzip KDEF_aligned.zip
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
