import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import random
from tqdm import tqdm
import argparse
from datetime import datetime

# Import model and datasets
from torchcfm.models.unet import UNetModel
from dataset_combined import create_combined_dataloaders
import config

# For adaptive solver
from torchdyn.core import NeuralODE

# Define specific subject-emotion pairs to evaluate
# Format: [(subject_id, emotion), ...]
# Leave empty to use random selection
SELECTED_PAIRS = [
    ('S129', 'surprise'),   # CK+ subject
    ('S076', 'happiness'),  # CK+ subject
    ('K3', 'surprise'),       # KDEF subject
    ('K68', 'happiness'),      # KDEF subject
    ('K35', 'anger')     # KDEF subject
]

def set_seed(seed):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def euler_integrate(model, source_image, target_emotion, device, steps=5):
    """
    Generate a transformed image using Euler integration
    
    Args:
        model: Trained UNet model
        source_image: Source image tensor (batch_size, channels, height, width)
        target_emotion: Target emotion indices tensor (batch_size,)
        device: Computation device
        steps: Number of steps for Euler integration
        
    Returns:
        Transformed image tensor
    """
    model.eval()

    # Ensure target_emotion has the correct shape (batch_size,)
    if target_emotion.dim() > 1:  # One-hot encoding
        target_emotion = torch.argmax(target_emotion, dim=1)
    elif target_emotion.dim() == 0:  # Single scalar value
        target_emotion = target_emotion.unsqueeze(0)

    # Ensure target_emotion is the same batch size as source_image
    if target_emotion.shape[0] != source_image.shape[0]:
        target_emotion = target_emotion.expand(source_image.shape[0])

    source_image = source_image.to(device)
    target_emotion = target_emotion.to(device)
    
    dt = 1.0 / steps
    t_steps = torch.linspace(0, 1.0 - dt, steps).to(device)
    x = source_image
    
    with torch.no_grad():
        for t in t_steps:
            t_batch = torch.ones(x.shape[0], device=device) * t
            v = model(t_batch, x, y=target_emotion)
            x = x + dt * v
            # Clamp values to prevent extreme values
            x = torch.clamp(x, -1.0, 1.0)
            
    return x

def generate_trajectory(model, source_image, target_emotion, device, steps=10, method="euler", 
                        adaptive_solver_type="dopri5", sample_points=6):
    """
    Generate a transformation trajectory using the specified integration method
    
    Args:
        model: Trained UNet model
        source_image: Source image tensor (batch_size, channels, height, width)
        target_emotion: Target emotion indices tensor (batch_size,)
        device: Computation device
        steps: Number of steps for integration
        method: "euler" or "adaptive"
        adaptive_solver_type: Type of adaptive solver ("dopri5", "rk4", etc.)
        sample_points: Number of points to sample from the trajectory
        
    Returns:
        List of sampled frames from the trajectory
    """
    model.eval()
    
    # Ensure target_emotion has the correct shape
    if target_emotion.dim() > 1:  # One-hot encoding
        target_emotion = torch.argmax(target_emotion, dim=1)
    elif target_emotion.dim() == 0:  # Single scalar value
        target_emotion = target_emotion.unsqueeze(0)
    
    # Ensure shapes match
    if target_emotion.shape[0] != source_image.shape[0]:
        target_emotion = target_emotion.expand(source_image.shape[0])
    
    # Move to device
    source_image = source_image.to(device)
    target_emotion = target_emotion.to(device)
    
    if method == "euler":
        # Generate trajectory with Euler method
        frames = []
        frames.append(source_image.clone())  # Start with source image
        
        dt = 1.0 / steps
        t_steps = torch.linspace(0, 1.0, steps+1).to(device)  # Include endpoint
        x = source_image.clone()
        
        with torch.no_grad():
            for t in t_steps[1:]:  # Skip first point (t=0) as we already added it
                t_batch = torch.ones(x.shape[0], device=device) * (t - dt)
                v = model(t_batch, x, y=target_emotion)
                x = x + dt * v
                x = torch.clamp(x, -1.0, 1.0)
                frames.append(x.clone())
        
        # Sample specific points from trajectory
        if len(frames) > sample_points:
            # Calculate indices for evenly spaced samples
            indices = np.linspace(0, len(frames)-1, sample_points, dtype=int)
            sampled_frames = [frames[i] for i in indices]
        else:
            sampled_frames = frames
        
        return sampled_frames
        
    elif method == "adaptive":
        # Define vector field for torchdyn
        class UNetVectorField(nn.Module):
            def __init__(self, unet_model, target_emotion):
                super().__init__()
                self.unet_model = unet_model
                self.target_emotion = target_emotion
            
            def forward(self, t, x, *args, **kwargs):
                # Create a batch of time values
                t_batch = torch.ones(x.shape[0], device=x.device) * t.to(x.device)
                # UNetModel expects (t, x, y) parameter order
                return self.unet_model(t_batch, x, y=self.target_emotion)
        
        # Create vector field and Neural ODE
        vector_field = UNetVectorField(model, target_emotion)
        neural_ode = NeuralODE(
            vector_field,
            sensitivity='adjoint',
            solver=adaptive_solver_type,
            atol=1e-4,
            rtol=1e-4
        )
        
        # Set up time points
        t_span = torch.linspace(0., 1., sample_points).to(device)
        
        # Generate trajectory
        with torch.no_grad():
            trajectory = neural_ode.trajectory(
                source_image,
                t_span
            )
        
        return trajectory
    
    else:
        raise ValueError(f"Unknown integration method: {method}")

def tensor_to_image(tensor):
    """Convert a PyTorch tensor to a numpy array for plotting"""
    # Move to CPU if needed
    if tensor.is_cuda:
        tensor = tensor.cpu()
    
    # Handle different channel dimensions
    if tensor.dim() == 4:  # Batch dimension
        tensor = tensor[0]  # Take first image in batch
        
    # Convert from CHW to HWC for matplotlib
    if tensor.shape[0] == 3 or tensor.shape[0] == 1:  # Channel dimension first
        img = tensor.permute(1, 2, 0).numpy()
    else:
        img = tensor.numpy()
        
    # Normalize from [-1, 1] to [0, 1]
    img = (img + 1) / 2
    
    # If single channel, squeeze
    if img.shape[-1] == 1:
        img = img.squeeze(-1)
        
    return img

def create_trajectory_comparison_grid(model, sample, device, output_path=None, sample_points=6, 
                                      euler_steps=20, adaptive_solver_type="dopri5"):
    """
    Create a visualization grid comparing Euler and adaptive integration
    
    Args:
        model: Trained UNet model
        sample: Dictionary with sample information
        device: Computation device
        output_path: Path to save the output image
        sample_points: Number of frames to show in the trajectory
        euler_steps: Number of steps for Euler integration
        adaptive_solver_type: Type of adaptive solver to use
        
    Returns:
        Path to saved image file
    """
    subject_id = sample['subject_id']
    target_emotion = sample['target_emotion']
    source_image = sample['source_image'].to(device)
    target_emotion_idx = torch.tensor([sample['target_emotion_idx']], device=device)
    
    print(f"Generating trajectory comparison for {subject_id} to {target_emotion}...")
    
    # Generate trajectories
    euler_frames = generate_trajectory(
        model, 
        source_image, 
        target_emotion_idx, 
        device, 
        steps=euler_steps,
        method="euler",
        sample_points=sample_points
    )
    
    adaptive_frames = generate_trajectory(
        model,
        source_image,
        target_emotion_idx,
        device,
        method="adaptive",
        adaptive_solver_type=adaptive_solver_type,
        sample_points=sample_points
    )
    
    # Create visualization grid
    fig, axes = plt.subplots(2, sample_points, figsize=(3 * sample_points, 6))
    
    # Set title
    fig.suptitle(f"Subject {subject_id}: {target_emotion} - Euler vs Adaptive", fontsize=16)
    
    # Add row labels
    axes[0, 0].set_ylabel("Euler", rotation=90, size='large', labelpad=10)
    axes[1, 0].set_ylabel("Adaptive", rotation=90, size='large', labelpad=10)
    
    # Add time labels
    for i in range(sample_points):
        t = i / (sample_points - 1)
        axes[0, i].set_title(f"t={t:.2f}")
    
    # Convert frames to images and plot
    for i, (euler_frame, adaptive_frame) in enumerate(zip(euler_frames, adaptive_frames)):
        euler_img = tensor_to_image(euler_frame)
        adaptive_img = tensor_to_image(adaptive_frame)
        
        # Plot images
        if len(euler_img.shape) == 2:  # Grayscale
            axes[0, i].imshow(euler_img, cmap='gray')
            axes[1, i].imshow(adaptive_img, cmap='gray')
        else:  # RGB
            axes[0, i].imshow(euler_img)
            axes[1, i].imshow(adaptive_img)
        
        # Remove axis ticks
        axes[0, i].set_xticks([])
        axes[0, i].set_yticks([])
        axes[1, i].set_xticks([])
        axes[1, i].set_yticks([])
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  # Make room for the title
    
    # Save figure
    if output_path is None:
        # Create default output path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"trajectory_{subject_id}_{target_emotion}_{timestamp}.png"
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved trajectory comparison to {output_path}")
    
    plt.close(fig)
    
    return output_path

def evaluate_model(model_path, output_path=None, selected_pairs=None,
                   generate_trajectories=True, sample_points=6,
                   euler_steps=20, adaptive_solver="dopri5"):
    """
    Evaluate the model by creating a grid visualization of transformations
    
    Args:
        model_path: Path to the trained model
        output_path: Path to save the output image
        selected_pairs: List of (subject_id, emotion) pairs to evaluate specifically
        generate_trajectories: If True, also generate trajectory comparison grids
        sample_points: Number of sample points for trajectory visualization
        euler_steps: Number of steps for Euler integration
        adaptive_solver: Type of adaptive solver to use
    """
    # Use the global selected pairs if none provided
    if selected_pairs is None:
        selected_pairs = SELECTED_PAIRS
    
    print(f"Using {len(selected_pairs)} specific subject-emotion pairs")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize model
    model = UNetModel(
        dim=(config.IN_CHANNELS, config.IMAGE_SIZE, config.IMAGE_SIZE),
        num_channels=config.MODEL_CHANNELS,
        num_res_blocks=config.NUM_RES_BLOCKS,
        attention_resolutions=config.ATTENTION_RESOLUTIONS,
        dropout=config.DROPOUT,
        channel_mult=config.CHANNEL_MULT,
        class_cond=True,
        num_classes=config.CONTEXT_DIM,
        use_checkpoint=False,
        use_scale_shift_norm=config.USE_SCALE_SHIFT_NORM,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_fp16=False,
        use_new_attention_order=False
    ).to(device)
    
    # Load model weights
    if not Path(model_path).exists():
        print(f"Error: Model file {model_path} not found!")
        return
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print(f"Model loaded successfully from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Load validation data
    print("Loading validation data...")
    _, val_loader = create_combined_dataloaders(
        ckplus_data_path=config.CK_PATH,
        kdef_data_path=config.KDEF_PATH,
        image_size=config.IMAGE_SIZE,
        batch_size=1,  # Process one sample at a time
        num_workers=config.NUM_WORKERS,
        target_emotion=None,  # Include all emotions
        augmentation_factor=0  # No augmentation for validation
    )
    
    # For tracking the specific pairs we need to find
    found_pairs = {}
    for pair in selected_pairs:
        found_pairs[pair] = False
    
    print("Searching for specified subject-emotion pairs...")
    for batch in tqdm(val_loader):
        # Get subject and emotion
        subject_id = batch['subject_id'][0]
        target_emotion = batch['target_emotion'][0]
        #print(subject_id, target_emotion)
        
        # Check if this matches one of our requested pairs
        for pair in selected_pairs:
            pair_subject, pair_emotion = pair
            if subject_id == pair_subject and target_emotion == pair_emotion:
                # Store the sample
                sample = {
                    'subject_id': subject_id,
                    'source_image': batch['source_image'],
                    'target_image': batch['target_image'],
                    'source_emotion': batch['source_emotion'][0],
                    'target_emotion': target_emotion,
                    'target_emotion_idx': batch['target_emotion_idx'][0].item()
                }
                found_pairs[pair] = sample
    
    # Check if all pairs were found
    missing_pairs = [pair for pair in selected_pairs if found_pairs.get(pair) is False]
    
    # Assert that all specified pairs were found
    assert not missing_pairs, f"Could not find the following subject-emotion pairs in the validation set: {missing_pairs}"
    
    # Get the samples in the order they were specified
    selected_samples = [found_pairs[pair] for pair in selected_pairs]
    print(f"All {len(selected_samples)} requested subject-emotion pairs found")
    
    # Generate transformed images
    print("Generating transformed images...")
    results = []
    
    for sample in tqdm(selected_samples):
        source_image = sample['source_image'].to(device)
        target_emotion_idx = torch.tensor([sample['target_emotion_idx']], device=device)
        
        # Generate transformed image using Euler integration
        generated_image = euler_integrate(
            model,
            source_image,
            target_emotion_idx,
            device,
            steps=10
        )
        
        # Store result
        result = {
            'subject_id': sample['subject_id'],
            'source_image': source_image,
            'target_image': sample['target_image'],
            'generated_image': generated_image,
            'source_emotion': sample['source_emotion'],
            'target_emotion': sample['target_emotion']
        }
        
        results.append(result)
    
    # Create comparison visualization grid
    print("Creating main comparison grid...")
    
    # Calculate actual number of subjects
    actual_num_subjects = len(results)
    
    # Create appropriate figure size
    fig, axes = plt.subplots(actual_num_subjects, 3, figsize=(12, 4 * actual_num_subjects))
    
    # Handle special case when we have only one subject (axes won't be 2D)
    if actual_num_subjects == 1:
        axes = axes.reshape(1, -1)
        
    # Set titles for columns
    axes[0, 0].set_title("Source (Neutral)")
    axes[0, 1].set_title("Target (Ground Truth)")
    axes[0, 2].set_title("Generated")
    
    for i, result in enumerate(results):
        # Convert tensors to numpy arrays for plotting
        source_img = tensor_to_image(result['source_image'])
        target_img = tensor_to_image(result['target_image'])
        generated_img = tensor_to_image(result['generated_image'])
        
        # Extract subject ID and emotions
        subject_id = result['subject_id']
        target_emotion = result['target_emotion']
        
        # Set row title
        axes[i, 0].set_ylabel(f"{subject_id}\n({target_emotion})", rotation=0, labelpad=40, va='center', ha='right')
        
        # Plot images
        if len(source_img.shape) == 2:  # Grayscale
            axes[i, 0].imshow(source_img, cmap='gray')
            axes[i, 1].imshow(target_img, cmap='gray')
            axes[i, 2].imshow(generated_img, cmap='gray')
        else:  # RGB
            axes[i, 0].imshow(source_img)
            axes[i, 1].imshow(target_img)
            axes[i, 2].imshow(generated_img)
        
        # Remove axis ticks
        for j in range(3):
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
    
    # Adjust layout
    plt.tight_layout()
    
    # Save main comparison grid
    if output_path is None:
        # Create a default output path with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"evaluation_grid_{timestamp}.png"
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Main comparison grid saved to {output_path}")
    plt.close(fig)
    
    # Generate trajectory grids if requested
    if generate_trajectories:
        print("\nGenerating trajectory comparison grids...")
        trajectory_paths = []
        
        for sample in tqdm(selected_samples):
            # Create a unique filename for this subject-emotion pair
            subject_id = sample['subject_id']
            target_emotion = sample['target_emotion']
            trajectory_path = f"trajectory_{subject_id}_{target_emotion}.png"
            
            # Create and save the trajectory comparison grid
            path = create_trajectory_comparison_grid(
                model,
                sample,
                device,
                output_path=trajectory_path,
                sample_points=sample_points,
                euler_steps=euler_steps,
                adaptive_solver_type=adaptive_solver
            )
            
            trajectory_paths.append(path)
        
        print(f"Generated {len(trajectory_paths)} trajectory comparison grids")
        
    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate facial emotion transformation model")
    parser.add_argument("--model", type=str, default="checkpoints/unet_final.pt", 
                        help="Path to the trained model")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save the main comparison grid")
    parser.add_argument("--custom_pairs", type=str, default=None,
                        help="Path to a text file with custom subject-emotion pairs, one per line in format 'subject_id,emotion'")
    parser.add_argument("--no_trajectories", action="store_true",
                        help="Skip generating trajectory comparison grids")
    parser.add_argument("--sample_points", type=int, default=6,
                        help="Number of sample points to show in trajectory grids")
    parser.add_argument("--euler_steps", type=int, default=20,
                        help="Number of steps for Euler integration")
    parser.add_argument("--adaptive_solver", type=str, default="dopri5",
                        choices=["dopri5", "rk4", "euler", "midpoint"],
                        help="Adaptive solver type to use")
    args = parser.parse_args()
    
    # Check if we should load custom pairs from a file
    selected_pairs = SELECTED_PAIRS
    if args.custom_pairs:
        try:
            custom_pairs = []
            with open(args.custom_pairs, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        subject_id, emotion = line.split(',')
                        custom_pairs.append((subject_id.strip(), emotion.strip()))
            
            if custom_pairs:
                print(f"Loaded {len(custom_pairs)} custom pairs from {args.custom_pairs}")
                selected_pairs = custom_pairs
            else:
                print(f"No valid pairs found in {args.custom_pairs}, using default pairs")
        except Exception as e:
            print(f"Error loading custom pairs: {e}")
            print("Using default pairs from script")
    
    # Evaluate model
    evaluate_model(
        model_path=args.model,
        output_path=args.output,
        selected_pairs=selected_pairs,
        generate_trajectories=not args.no_trajectories,
        sample_points=args.sample_points,
        euler_steps=args.euler_steps,
        adaptive_solver=args.adaptive_solver
    )
