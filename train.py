import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import wandb
import numpy as np
import random
import math

# Import LPIPS loss for perceptual similarity
from lpips_loss import LPIPSLoss
from torchcfm.models.unet import UNetModel
from torchcfm.conditional_flow_matching import ExactOptimalTransportConditionalFlowMatcher
from torchdyn.core import NeuralODE
from torchvision.models import vgg16, VGG16_Weights
from torchvision.transforms import Resize

# Import dataset and config
from dataset_combined import create_combined_dataloaders
import config

def set_seed(seed):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class PerceptualLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        # Load pretrained VGG16 model
        vgg_model = vgg16(weights=VGG16_Weights.DEFAULT).to(device).eval()

        # Select multiple layers for feature extraction at different scales
        self.layers = {
            '0': 'conv1_1',   # Very early features (edges, colors)
            '5': 'conv2_2',   # Early features (textures)
            '10': 'conv3_3',  # Mid-level features
            '17': 'conv4_3',  # Higher level features
        }

        self.model_blocks = nn.ModuleDict()
        # Extract layers from vgg_model.features
        for layer_idx, name in self.layers.items():
            seq_layers = list(vgg_model.features)[:int(layer_idx) + 1]
            self.model_blocks[layer_idx] = nn.Sequential(*seq_layers).to(device)

        # Freeze parameters
        for param in self.parameters():
            param.requires_grad = False

        self.resize = Resize(224)
        self.mse = nn.MSELoss()  # Use default reduction='mean'

    def forward(self, x, y):
        # Handle grayscale (1 channel) to RGB (3 channels)
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        if y.shape[1] == 1:
            y = y.repeat(1, 3, 1, 1)

        # Resize to 224x224 if needed
        if x.shape[-1] != 224 or x.shape[-2] != 224:
            x = self.resize(x)
        if y.shape[-1] != 224 or y.shape[-2] != 224:
            y = self.resize(y)

        # Denormalize from [-1, 1] to [0, 1]
        x = (x + 1) / 2
        y = (y + 1) / 2

        # Apply ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x.device)
        x = (x - mean) / std
        y = (y - mean) / std

        # Compute feature losses with layer weights
        total_loss = 0.0
        layer_weights = {
            '0': 1.0/32,    # Lower weight for low-level features
            '5': 1.0/16,
            '10': 1.0/8,
            '17': 1.0/4     # Higher weight for high-level features
        }

        for layer_idx, _ in self.layers.items():
            with torch.no_grad():  # We don't need gradients for the feature extraction
                features_y = self.model_blocks[layer_idx](y)

            # But we do need gradients for x
            features_x = self.model_blocks[layer_idx](x)

            # L2 loss between features, weighted by importance
            layer_loss = self.mse(features_x, features_y) * layer_weights[layer_idx]
            total_loss += layer_loss

        return total_loss

def euler_integrate(model, source_image, target_emotion, device, steps=5, requires_grad=False):
    if requires_grad:
        model.train()
    else:
        model.eval()

    # Add batch dimension if needed
    if source_image.dim() == 3:
        source_image = source_image.unsqueeze(0)

    # Ensure target_emotion has the correct shape (batch_size,)
    # The UNetModel expects a 1D tensor of class indices
    if target_emotion.dim() > 1:  # One-hot encoding
        target_emotion = torch.argmax(target_emotion, dim=1)
    elif target_emotion.dim() == 0:  # Single scalar value
        target_emotion = target_emotion.unsqueeze(0)

    # Ensure target_emotion is the same batch size as source_image
    if target_emotion.shape[0] != source_image.shape[0]:
        target_emotion = target_emotion.expand(source_image.shape[0])

    # Explicitly verify the shape for debugging
    batch_size = source_image.shape[0]
    assert target_emotion.shape == (batch_size,), f"Target emotion shape {target_emotion.shape} != expected {(batch_size,)}"

    source_image = source_image.to(device)
    target_emotion = target_emotion.to(device)
    dt = 1.0 / steps
    t_steps = torch.linspace(0, 1.0 - dt, steps).to(device)
    x = source_image
    with torch.set_grad_enabled(requires_grad):
        for t in t_steps:
            t_batch = torch.ones(x.shape[0], device=device) * t
            v = model(t_batch, x, y=target_emotion)
            x = x + dt * v
    return x

def generate_sample(model, source_image, target_emotion, device, steps=None):
    """
    Generate a transformed image using the trained model

    Args:
        model: Trained UNet model
        source_image: Source image tensor
        target_emotion: Target emotion one-hot tensor or class index
        device: Computation device
        steps: Number of steps for ODE solver (optional)

    Returns:
        Generated image tensor
    """
    model.eval()

    # Add batch dimension if needed
    if source_image.dim() == 3:
        source_image = source_image.unsqueeze(0)

    # Convert target_emotion to class index if it's one-hot
    if target_emotion.dim() > 1 or (target_emotion.dim() == 1 and len(target_emotion) > 1):
        target_emotion = torch.argmax(target_emotion, dim=-1)
    elif target_emotion.dim() == 0:  # Single scalar value
        target_emotion = target_emotion.unsqueeze(0)

    # Move to device
    source_image = source_image.to(device)
    target_emotion = target_emotion.to(device)

    # Define vector field as a class for torchdyn
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
        solver='dopri5',
        atol=1e-4,
        rtol=1e-4
    )

    # Set up time points
    t_span = torch.tensor([0., 1.], device=device)

    # Generate trajectory
    with torch.no_grad():
        trajectory = neural_ode.trajectory(
            source_image,
            t_span
        )

    # Return final result (the transformed image)
    return trajectory[-1]

def train_unet(target_emotion=None):
    """
    Train a UNet model using flow matching on the CK+ dataset
    
    Args:
        target_emotions: Optional list of specific target emotions to filter for (e.g., ['happiness', 'anger'])
                        If None, includes all non-neutral emotions
    """
    # Set random seed
    set_seed(config.SEED)

    # Create checkpoint directory
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

    # Set device
    device = torch.device("cuda" if config.USE_CUDA and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    
    print(f"Loading paired data...")
    train_loader, val_loader = create_combined_dataloaders(
        ckplus_data_path='../processed_data/aligned_ck_data.pkl',
        kdef_data_path='../processed_data/aligned_kdef_data.pkl',
        image_size=config.IMAGE_SIZE,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        target_emotion=None,  # Now accepts a list of emotions or a single emotion
        augmentation_factor=0
    )

    print(f"Training with {len(train_loader.dataset)} pairs, validating with {len(val_loader.dataset)} pairs")

    # Initialize the model with aggressive downsampling
    model = UNetModel(
        dim=(config.IN_CHANNELS, config.IMAGE_SIZE, config.IMAGE_SIZE),
        num_channels=config.MODEL_CHANNELS,
        num_res_blocks=config.NUM_RES_BLOCKS,
        attention_resolutions=config.ATTENTION_RESOLUTIONS,
        dropout=config.DROPOUT,
        channel_mult=config.CHANNEL_MULT,  # Use custom channel multipliers for aggressive downsampling
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

    # Set up the flow matcher
    if config.FLOW_MATCHER_TYPE == 'exact_optimal_transport':
        flow_matcher = ExactOptimalTransportConditionalFlowMatcher(sigma=config.SIGMA)
        print(f"Using ExactOptimalTransportConditionalFlowMatcher with sigma: {config.SIGMA}")
    else:
        from torchcfm.conditional_flow_matching import ConditionalFlowMatcher
        flow_matcher = ConditionalFlowMatcher(sigma=config.SIGMA)
        print(f"Using ConditionalFlowMatcher with sigma: {config.SIGMA}")

    # Initialize LPIPS loss if enabled
    lpips_loss_fn = None
    if config.USE_LPIPS_LOSS:
        print(f"Initializing LPIPS loss with {config.LPIPS_NET} backbone...")
        lpips_loss_fn = LPIPSLoss(device=device, net=config.LPIPS_NET)
    
    # Initialize generator (UNet) optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    
    # Initialize learning rate scheduler if enabled
    scheduler = None
    if config.USE_LR_SCHEDULER:
        # Define a warmup scheduler that linearly increases LR from 0 to base LR
        def warmup_lambda(epoch):
            if epoch < config.WARMUP_EPOCHS:
                return float(epoch) / float(max(1, config.WARMUP_EPOCHS))
            else:
                # Cosine annealing after warmup
                progress = float(epoch - config.WARMUP_EPOCHS) / float(max(1, config.EPOCHS - config.WARMUP_EPOCHS))
                return max(config.MIN_LR / config.LEARNING_RATE, 0.5 * (1.0 + math.cos(math.pi * progress)))
        
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lambda)
        print(f"Using cosine annealing LR scheduler with {config.WARMUP_EPOCHS} epochs warmup")
        print(f"Initial LR: {config.LEARNING_RATE}, Min LR: {config.MIN_LR}")

    # Loss functions
    mse_loss = nn.MSELoss()

    # Initialize perceptual loss if enabled
    if config.USE_PERCEPTUAL_LOSS:
        perceptual_loss = PerceptualLoss(device)
        print("Using perceptual loss with VGG16")

    # Initialize wandb if enabled
    if config.LOG_TO_WANDB:
        # Create a clean dictionary for wandb config
        wandb_config = {
            # Data settings
            'data_path': config.DATA_PATH,
            'splits_path': config.SPLITS_PATH,
            'image_size': config.IMAGE_SIZE,
            'batch_size': config.BATCH_SIZE,
            'num_workers': config.NUM_WORKERS,

            # Model settings
            'in_channels': config.IN_CHANNELS,
            'model_channels': config.MODEL_CHANNELS,
            'num_res_blocks': config.NUM_RES_BLOCKS,
            'attention_resolutions': config.ATTENTION_RESOLUTIONS,
            'dropout': config.DROPOUT,
            'use_scale_shift_norm': config.USE_SCALE_SHIFT_NORM,

            # Flow matching settings
            'sigma': config.SIGMA,
            'flow_matcher_type': config.FLOW_MATCHER_TYPE,

            # Perceptual loss settings
            'use_perceptual_loss': config.USE_PERCEPTUAL_LOSS,
            'perceptual_loss_weight': config.PERCEPTUAL_LOSS_WEIGHT,
            'perceptual_loss_delay_epochs': config.PERCEPTUAL_LOSS_DELAY_EPOCHS,
            'euler_steps': config.EULER_STEPS,

            # Training settings
            'seed': config.SEED,
            'learning_rate': config.LEARNING_RATE,
            'weight_decay': config.WEIGHT_DECAY,
            'epochs': config.EPOCHS,
            'grad_clip_norm': config.GRAD_CLIP_NORM,
        }

        wandb.init(
            project=config.WANDB_PROJECT,
            name=config.EXPERIMENT_NAME,
            config=wandb_config
        )
        # Log model architecture
        wandb.watch(model, log_freq=100)

    # Training loop
    best_val_loss = float('inf')

    for epoch in range(config.EPOCHS):
        # Training phase
        model.train()
        total_train_loss = 0.0
        total_flow_matching_loss = 0.0
        total_perceptual_loss = 0.0
        num_batches = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.EPOCHS} [Train]")
        for batch_idx, batch_data in enumerate(progress_bar):
            # Move data to device
            source_image = batch_data['source_image'].to(device)
            target_image = batch_data['target_image'].to(device)
            target_one_hot = batch_data['target_one_hot'].to(device)

            # Convert one-hot to class indices for the UNet model
            target_class_indices = torch.argmax(target_one_hot, dim=1)

            # Zero gradients
            optimizer.zero_grad()

            # Sample time points
            batch_size = source_image.shape[0]
            t = torch.rand(batch_size, device=device)

            # Get conditional flow target from flow matcher
            t, xt, ut = flow_matcher.sample_location_and_conditional_flow(
                source_image, target_image, t
            )

            # Compute predicted vector field using UNet model
            predicted_ut = model(t, xt, y=target_class_indices)

            # Flow matching loss
            flow_matching_loss = mse_loss(predicted_ut, ut)

            # Initialize total loss with flow matching loss
            total_loss = flow_matching_loss

            # Generate images for perceptual and LPIPS losses if needed
            if (config.USE_PERCEPTUAL_LOSS and epoch >= config.PERCEPTUAL_LOSS_DELAY_EPOCHS) or \
               (config.USE_LPIPS_LOSS and epoch >= config.LPIPS_LOSS_DELAY_EPOCHS):
                # Generate sample with Euler integration and track gradients
                generated_images = euler_integrate(
                    model, 
                    source_image, 
                    target_class_indices, 
                    device,
                    steps=config.EULER_STEPS, 
                    requires_grad=True
                )
            
            # Apply perceptual loss if enabled and after delay epochs
            perceptual_loss_val = torch.tensor(0.0, device=device)
            if config.USE_PERCEPTUAL_LOSS and epoch >= config.PERCEPTUAL_LOSS_DELAY_EPOCHS and generated_images is not None:
                perceptual_loss_val = perceptual_loss(generated_images, target_image)
                
                # Add weighted perceptual loss to total loss
                total_loss = total_loss + config.PERCEPTUAL_LOSS_WEIGHT * perceptual_loss_val
                
            # Apply LPIPS loss if enabled and after delay epochs
            lpips_loss_val = torch.tensor(0.0, device=device)
            if config.USE_LPIPS_LOSS and epoch >= config.LPIPS_LOSS_DELAY_EPOCHS and generated_images is not None:
                # Normalize images to [0, 1] for LPIPS
                generated_norm = (generated_images + 1) / 2
                target_norm = (target_image + 1) / 2
                
                # Compute LPIPS loss
                lpips_loss_val = lpips_loss_fn(generated_norm, target_norm)
                
                # Add weighted LPIPS loss to total loss
                total_loss = total_loss + config.LPIPS_LOSS_WEIGHT * lpips_loss_val

            # Backward pass
            total_loss.backward()

            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP_NORM)
            if grad_norm > config.GRAD_CLIP_NORM * 0.9:
                print(f"Warning: Gradient norm {grad_norm:.4f} near clipping threshold")

            # Update weights
            optimizer.step()

            # Update metrics
            total_train_loss += total_loss.item()
            total_flow_matching_loss += flow_matching_loss.item()
            total_perceptual_loss += perceptual_loss_val.item() if config.USE_PERCEPTUAL_LOSS else 0
            total_lpips_loss = 0.0 if 'total_lpips_loss' not in locals() else total_lpips_loss
            total_lpips_loss += lpips_loss_val.item() if config.USE_LPIPS_LOSS else 0
            num_batches += 1

            # Update progress bar
            progress_bar.set_postfix({
                'loss': total_loss.item(),
                'flow_match': flow_matching_loss.item(),
                'perc_loss': perceptual_loss_val.item() if config.USE_PERCEPTUAL_LOSS else 0,
                'lpips_loss': lpips_loss_val.item() if config.USE_LPIPS_LOSS else 0,
            })

            # Log to wandb at intervals
            if config.LOG_TO_WANDB and batch_idx % config.LOG_INTERVAL == 0:
                # Log loss metrics
                wandb.log({
                    'train_batch_loss': total_loss.item(),
                    'train_batch_flow_matching_loss': flow_matching_loss.item(),
                    'train_batch_perceptual_loss': perceptual_loss_val.item() if config.USE_PERCEPTUAL_LOSS else 0,
                    'train_batch_lpips_loss': lpips_loss_val.item() if config.USE_LPIPS_LOSS else 0,
                    'train_step': epoch * len(train_loader) + batch_idx,
                    'gradient_norm': grad_norm
                })

                # Generate and log sample images
                if batch_idx % (config.LOG_INTERVAL * 5) == 0:  # Less frequent to save resources
                    with torch.no_grad():
                        # Generate a transformed image from the first source image in batch
                        sample_img = euler_integrate(
                            model, 
                            source_image[:1], 
                            target_class_indices[:1], 
                            device,
                            steps=config.EULER_STEPS, 
                            requires_grad=False
                        )
        
                        # Convert tensors to format suitable for visualization
                        sample_vis = (sample_img[0].cpu().permute(1, 2, 0).clamp(-1, 1) + 1) / 2
                        source_vis = (source_image[0].cpu().permute(1, 2, 0).clamp(-1, 1) + 1) / 2
                        target_vis = (target_image[0].cpu().permute(1, 2, 0).clamp(-1, 1) + 1) / 2
        
                        # Log images to wandb
                        wandb.log({
                            'train/generated_image': wandb.Image(sample_vis.numpy(), caption='Generated Image'),
                            'train/source_image': wandb.Image(source_vis.numpy(), caption='Source Image'),
                            'train/target_image': wandb.Image(target_vis.numpy(), caption='Target Image')
                        })

        # Calculate average training losses
        avg_train_loss = total_train_loss / num_batches
        avg_flow_matching_loss = total_flow_matching_loss / num_batches
        avg_perceptual_loss = total_perceptual_loss / num_batches if num_batches > 0 else 0
        avg_lpips_loss = total_lpips_loss / num_batches if num_batches > 0 else 0

        print(f"Epoch {epoch+1}/{config.EPOCHS} - Train Loss: {avg_train_loss:.6f}, "
              f"Flow Matching: {avg_flow_matching_loss:.6f}, "
              f"Perceptual: {avg_perceptual_loss:.6f}, "
              f"LPIPS: {avg_lpips_loss:.6f}")

        # Evaluation phase
        if (epoch + 1) % config.EVALUATION_INTERVAL == 0 or epoch == config.EPOCHS - 1:
            model.eval()
            total_val_loss = 0.0
            total_val_flow_matching_loss = 0.0
            total_val_perceptual_loss = 0.0
            total_val_lpips_loss = 0.0
            val_batches = 0

            print(f"Validating model at epoch {epoch+1}...")
            with torch.no_grad():
                val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{config.EPOCHS} [Val]")
                for batch_data in val_bar:
                    # Move data to device
                    source_image = batch_data['source_image'].to(device)
                    target_image = batch_data['target_image'].to(device)
                    target_one_hot = batch_data['target_one_hot'].to(device)
    
                    # Convert one-hot to class indices for the UNet model
                    target_class_indices = torch.argmax(target_one_hot, dim=1)
    
                    # Sample time points
                    batch_size = source_image.shape[0]
                    t = torch.rand(batch_size, device=device)
    
                    # Get conditional flow target from flow matcher
                    t, xt, ut = flow_matcher.sample_location_and_conditional_flow(
                        source_image, target_image, t
                    )
    
                    # Compute predicted vector field using UNet model
                    predicted_ut = model(t, xt, y=target_class_indices)
    
                    # Flow matching loss
                    flow_matching_loss = mse_loss(predicted_ut, ut)
    
                    # Generate images for evaluation (used for both perceptual and LPIPS loss)
                    generated_images = None
                    if config.USE_PERCEPTUAL_LOSS or config.USE_LPIPS_LOSS:
                        generated_images = euler_integrate(
                            model, 
                            source_image, 
                            target_class_indices, 
                            device,
                            steps=config.EULER_STEPS, 
                            requires_grad=False
                        )
                    
                    # Compute perceptual loss for validation (without affecting gradients)
                    perceptual_loss_val = torch.tensor(0.0, device=device)
                    if config.USE_PERCEPTUAL_LOSS and generated_images is not None:
                        perceptual_loss_val = perceptual_loss(generated_images, target_image)
                    
                    # Compute LPIPS loss for validation
                    lpips_loss_val = torch.tensor(0.0, device=device)
                    if config.USE_LPIPS_LOSS and generated_images is not None:
                        # Normalize images to [0, 1] for LPIPS
                        generated_norm = (generated_images + 1) / 2
                        target_norm = (target_image + 1) / 2
                        lpips_loss_val = lpips_loss_fn(generated_norm, target_norm)
    
                    # Compute total validation loss (same formula as training)
                    total_loss = flow_matching_loss
                    if config.USE_PERCEPTUAL_LOSS and epoch >= config.PERCEPTUAL_LOSS_DELAY_EPOCHS:
                        total_loss = total_loss + config.PERCEPTUAL_LOSS_WEIGHT * perceptual_loss_val
                    if config.USE_LPIPS_LOSS and epoch >= config.LPIPS_LOSS_DELAY_EPOCHS:
                        total_loss = total_loss + config.LPIPS_LOSS_WEIGHT * lpips_loss_val
    
                    # Log raw loss values for the first batch
                    if val_batches == 0:
                        print(f"Validation raw loss - Flow: {flow_matching_loss.item():.6f}, "
                              f"Perceptual: {perceptual_loss_val.item() if config.USE_PERCEPTUAL_LOSS else 0:.6f}, "
                              f"LPIPS: {lpips_loss_val.item() if config.USE_LPIPS_LOSS else 0:.6f}")
    
                    # Update metrics
                    total_val_loss += total_loss.item()
                    total_val_flow_matching_loss += flow_matching_loss.item()
                    total_val_perceptual_loss += perceptual_loss_val.item() if config.USE_PERCEPTUAL_LOSS else 0
                    total_val_lpips_loss += lpips_loss_val.item() if config.USE_LPIPS_LOSS else 0
                    val_batches += 1
    
                # Generate and log sample validation images
                if config.LOG_TO_WANDB:
                    # Use a small sample of validation images
                    num_samples = min(4, len(source_image))
                    for i in range(num_samples):
                        # Generate transformed image
                        sample_img = euler_integrate(
                            model, 
                            source_image[i:i+1], 
                            target_class_indices[i:i+1], 
                            device,
                            steps=config.EULER_STEPS, 
                            requires_grad=False
                        )
        
                        # Convert tensors to format suitable for visualization
                        sample_vis = (sample_img[0].cpu().permute(1, 2, 0).clamp(-1, 1) + 1) / 2
                        source_vis = (source_image[i].cpu().permute(1, 2, 0).clamp(-1, 1) + 1) / 2
                        target_vis = (target_image[i].cpu().permute(1, 2, 0).clamp(-1, 1) + 1) / 2
        
                        # Log images to wandb
                        wandb.log({
                            f'val/generated_image_{i}': wandb.Image(sample_vis.numpy(), caption=f'Generated Image {i}'),
                            f'val/source_image_{i}': wandb.Image(source_vis.numpy(), caption=f'Source Image {i}'),
                            f'val/target_image_{i}': wandb.Image(target_vis.numpy(), caption=f'Target Image {i}')
                        })

            # Calculate average validation loss
            avg_val_loss = total_val_loss / val_batches
            avg_val_flow_matching_loss = total_val_flow_matching_loss / val_batches
            avg_val_perceptual_loss = total_val_perceptual_loss / val_batches if val_batches > 0 else 0
            avg_val_lpips_loss = total_val_lpips_loss / val_batches if val_batches > 0 else 0

            print(f"Validation - Loss: {avg_val_loss:.6f}, "
                  f"Flow Matching: {avg_val_flow_matching_loss:.6f}, "
                  f"Perceptual: {avg_val_perceptual_loss:.6f}, "
                  f"LPIPS: {avg_val_lpips_loss:.6f}")

            # Step the learning rate scheduler if enabled
            if config.USE_LR_SCHEDULER and scheduler is not None:
                scheduler.step()
                current_lr = scheduler.get_last_lr()[0]
                print(f"Learning rate updated to: {current_lr:.7f}")
                
                # Log learning rate to wandb
                if config.LOG_TO_WANDB:
                    wandb.log({'learning_rate': current_lr, 'epoch': epoch})
            
            # Save best model
            if avg_val_loss < best_val_loss or not config.SAVE_BEST_ONLY:
                best_val_loss = avg_val_loss
                model_path = os.path.join(config.CHECKPOINT_DIR, "unet_best.pt")
                torch.save(model.state_dict(), model_path)
                print(f"New best model saved with validation loss: {avg_val_loss:.6f}")

        # Save checkpoint at intervals
        if (epoch + 1) % config.SAVE_INTERVAL == 0:
            checkpoint_path = os.path.join(config.CHECKPOINT_DIR, f"unet_epoch_{epoch+1}.pt")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved at epoch {epoch+1}")

        # Log to wandb if enabled
        if config.LOG_TO_WANDB:
            wandb.log({
                'epoch': epoch,
                'train_loss': avg_train_loss,
                'train_flow_matching_loss': avg_flow_matching_loss,
                'train_perceptual_loss': avg_perceptual_loss,
                'train_lpips_loss': avg_lpips_loss,
                'validation_loss': avg_val_loss if (epoch + 1) % config.EVALUATION_INTERVAL == 0 or epoch == config.EPOCHS - 1 else None,
                'validation_flow_matching_loss': avg_val_flow_matching_loss if (epoch + 1) % config.EVALUATION_INTERVAL == 0 or epoch == config.EPOCHS - 1 else None,
                'validation_perceptual_loss': avg_val_perceptual_loss if (epoch + 1) % config.EVALUATION_INTERVAL == 0 or epoch == config.EPOCHS - 1 else None,
                'validation_lpips_loss': avg_val_lpips_loss if (epoch + 1) % config.EVALUATION_INTERVAL == 0 or epoch == config.EPOCHS - 1 else None,
                'best_val_loss': best_val_loss,
                'using_perceptual_loss': epoch >= config.PERCEPTUAL_LOSS_DELAY_EPOCHS and config.USE_PERCEPTUAL_LOSS,
                'using_lpips_loss': epoch >= config.LPIPS_LOSS_DELAY_EPOCHS and config.USE_LPIPS_LOSS
            })

    # Save final model
    final_model_path = os.path.join(config.CHECKPOINT_DIR, "unet_final.pt")
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to {final_model_path}")

    if config.LOG_TO_WANDB:
        wandb.save(final_model_path)
        wandb.finish()

    print("UNet training completed!")
    return model

def evaluate_emotions(model, device, emotions=None, num_samples=4):
    """
    Evaluate the model by generating samples for multiple emotions
    
    Args:
        model: Trained UNet model
        device: Computation device
        emotions: List of emotions to evaluate. If None, evaluates all emotions except neutral
        num_samples: Number of samples to generate for each emotion
        
    Returns:
        Dictionary mapping emotion names to lists of generated samples
    """
    # Define all possible emotions (based on the CKPlusPairedDataset mapping)
    all_emotions = {
        0: 'neutral', 
        1: 'anger', 
        2: 'contempt', 
        3: 'disgust', 
        4: 'fear', 
        5: 'happiness', 
        6: 'sadness', 
        7: 'surprise'
    }
    
    # Define which emotions to evaluate
    if emotions is None:
        # Use all non-neutral emotions
        emotions_to_test = [emotion for idx, emotion in all_emotions.items() if emotion != 'neutral']
    elif isinstance(emotions, list):
        emotions_to_test = emotions
    else:
        # Single emotion case
        emotions_to_test = [emotions]
    
    print(f"Evaluating model with {len(emotions_to_test)} target emotions: {', '.join(emotions_to_test)}")
    
    # Load validation data - asking specifically for neutral source images paired with any target emotion
    _, val_loader = create_combined_dataloaders(
        ckplus_data_path='../processed_data/aligned_ck_data.pkl',
        kdef_data_path='../processed_data/aligned_kdef_data.pkl',
        image_size=config.IMAGE_SIZE,
        batch_size=num_samples,  # Just need a small batch
        num_workers=1,
        # Don't filter by target emotion here, just get any valid pairs
        target_emotion=None  
    )
    
    # Get a batch of source images (neutral faces)
    model.eval()
    source_images = None
    source_subjects = []
    
    for batch_data in val_loader:
        # Extract only neutral source images
        source_images = batch_data['source_image'][:num_samples].to(device)
        
        # Store subject IDs for reference
        source_subjects = batch_data['subject_id'][:num_samples]
        
        # Only need one batch
        break
    
    if source_images is None or len(source_images) == 0:
        print("No validation data available.")
        return {}
    
    print(f"Using {len(source_images)} neutral face images from subjects: {source_subjects}")
    
    # Map emotion names to indices
    emotion_to_idx = {
        'neutral': 0, 
        'anger': 1, 
        'contempt': 2, 
        'disgust': 3, 
        'fear': 4, 
        'happiness': 5, 
        'sadness': 6, 
        'surprise': 7
    }
    
    # Generate samples for each emotion
    results = {}
    
    with torch.no_grad():
        for emotion in emotions_to_test:
            print(f"Generating {emotion} expressions...")
            
            # Create target class indices for this emotion
            emotion_idx = emotion_to_idx.get(emotion, 0)
            target_class_indices = torch.full((num_samples,), emotion_idx, dtype=torch.long, device=device)
            
            # Generate samples using Euler integration
            samples = euler_integrate(
                model,
                source_images,
                target_class_indices,
                device,
                steps=config.EULER_STEPS,
                requires_grad=False
            )
            
            # Store results
            results[emotion] = samples
            
            # Convert for visualization
            if config.LOG_TO_WANDB:
                for i in range(num_samples):
                    sample_vis = (samples[i].cpu().permute(1, 2, 0).clamp(-1, 1) + 1) / 2
                    source_vis = (source_images[i].cpu().permute(1, 2, 0).clamp(-1, 1) + 1) / 2
                    
                    # Log images to wandb
                    wandb.log({
                        f'eval/{emotion}/generated_{i}': wandb.Image(sample_vis.numpy(), caption=f'{emotion.capitalize()} {i}'),
                        f'eval/{emotion}/source_{i}': wandb.Image(source_vis.numpy(), caption=f'Source {i}')
                    })
    
    print(f"Generated samples for {len(emotions_to_test)} emotions")
    return results

if __name__ == "__main__":
    # Train with all emotion types (neutral → any emotion)
    model = train_unet()  # No target emotion filter means all emotions will be used
    
    # Evaluate on each emotion type
    evaluate_emotions(model, torch.device("cuda" if config.USE_CUDA and torch.cuda.is_available() else "cpu"))
    
    # Alternative: train with specific emotions only
    # model = train_unet(target_emotions=['happiness', 'anger', 'surprise'])
    
    # Alternative: train with single emotion (original behavior)
    # model = train_unet(target_emotions='happiness')
