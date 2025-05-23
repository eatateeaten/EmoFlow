import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import gradio as gr
from pathlib import Path
from PIL import Image
import io
import time  # Add time module for timing measurements
import torch.nn.functional as F

# Import LPIPS loss for perceptual similarity
from torchcfm.models.unet import UNetModel

# Import dataset and config
from dataset_combined import create_combined_dataloaders
import config as config

device = "cuda"

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
).cuda()

# Load the model
model_path = "checkpoints/unet_epoch_100.pt"
if not Path(model_path).exists():
    print(f"Error: Model file {model_path} not found!")
    print("Please ensure you have the pretrained model file in the correct location.")
    exit(1)

# Load model weights
try:
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

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
            # Clamp values to prevent underflow/overflow of black and white pixels
            x = torch.clamp(x, -1.0, 1.0)
    return x

def generate_trajectory(model, source_image, target_emotion, device, steps=10, solver_type="euler", adaptive_solver="dopri5"):
    """
    Generate a transformed image using the trained model

    Args:
        model: Trained UNet model
        source_image: Source image tensor
        target_emotion: Target emotion one-hot tensor or class index
        device: Computation device
        steps: Number of steps for trajectory generation
        solver_type: Type of ODE solver to use ("euler" or "adaptive")
        adaptive_solver: Specific adaptive solver to use if solver_type is "adaptive"

    Returns:
        List of intermediate trajectory frames
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
    
    # Generate trajectory frames
    frames = []
    
    # Store the initial source image
    frames.append(source_image.clone())
    
    # Timing the inference process
    start_time = time.time()
    
    # Choose solver based on solver_type parameter
    if solver_type == "euler":
        # Euler integration
        dt = 1.0 / steps
        t_steps = torch.linspace(0, 1.0 - dt, steps).to(device)
        x = source_image.clone()
        
        with torch.no_grad():
            for t in t_steps:
                # Create a batch of time values
                t_batch = torch.ones(x.shape[0], device=device) * t
                
                # Predict the flow field
                v = model(t_batch, x, y=target_emotion)
                
                # Update the current state
                x = x + dt * v
                
                # Clamp values to prevent underflow/overflow
                x = torch.clamp(x, -1.0, 1.0)
                
                # Store the updated state
                frames.append(x.clone())
    
    elif solver_type == "adaptive":
        # Use torchdyn for adaptive solvers
        from torchdyn.core import NeuralODE
        
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
            solver=adaptive_solver,
            atol=1e-4,
            rtol=1e-4
        )
        
        # Create evenly spaced time points for visualization
        t_eval = torch.linspace(0, 1.0, steps + 1, device=device)
        
        with torch.no_grad():
            # Generate full trajectory
            trajectory = neural_ode.trajectory(
                source_image,
                t_span=torch.tensor([0., 1.], device=device),
                t_eval=t_eval
            )
            
            # Store all frames
            frames = [x.clone() for x in trajectory]
    
    else:
        raise ValueError(f"Unknown solver type: {solver_type}. Use 'euler' or 'adaptive'.")
    
    # Calculate total inference time
    inference_time = time.time() - start_time
    inference_time_per_step = inference_time / steps
    inference_time_per_image = inference_time / (steps * source_image.shape[0])
    
    print(f"Inference time: {inference_time:.4f} seconds")
    print(f"Time per step: {inference_time_per_step:.4f} seconds")
    print(f"Time per image per step: {inference_time_per_image:.4f} seconds")
    print(f"Solver type: {solver_type}" + (f", {adaptive_solver}" if solver_type == "adaptive" else ""))
    
    return frames

# Add Gradio app functionality
if __name__ == "__main__":
    
    print("Initializing Gradio app for facial expression editing...")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # List of available emotions (excluding neutral which is the source)
    emotions = ['anger', 'contempt', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']
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
    
    # Load validation dataset
    print("Loading validation dataset...")
    _, val_loader = create_combined_dataloaders(
        ckplus_data_path='data/aligned_ck_data.pkl',
        kdef_data_path='data/aligned_kdef_data.pkl',
        image_size=config.IMAGE_SIZE,
        batch_size=8,  # Small batch for display
        num_workers=config.NUM_WORKERS,
        target_emotion=None  # Load all emotions
    )
    
    # Extract all unique subjects with their neutral faces
    print("Extracting all unique subjects...")
    unique_subjects = {}
    subject_ids = set()
    
    # Track valid subject-emotion combinations
    valid_subject_emotion_pairs = {}
    
    # Iterate through the entire validation set
    for batch in tqdm(val_loader):
        for i in range(len(batch['source_image'])):
            # Get subject ID if available, or create a unique identifier
            if 'subject_id' in batch:
                subject_id = batch['subject_id'][i]
            else:
                # Create a unique ID from the source image itself (simple hash)
                img = batch['source_image'][i]
                subject_id = f"subj_{hash(img.cpu().numpy().tobytes()) % 10000}"
            
            # Only add each subject once (using their neutral face)
            if subject_id not in subject_ids:
                subject_ids.add(subject_id)
                unique_subjects[subject_id] = batch['source_image'][i]
            
            # Record the emotion for this subject (assuming target_one_hot is available)
            if 'target_one_hot' in batch:
                emotion_idx = torch.argmax(batch['target_one_hot'][i]).item()
                emotion_name = list(emotion_to_idx.keys())[list(emotion_to_idx.values()).index(emotion_idx)]
                
                # Initialize an empty set if this subject isn't in the dict yet
                if subject_id not in valid_subject_emotion_pairs:
                    valid_subject_emotion_pairs[subject_id] = set()
                
                # Add this emotion to the subject's valid emotions
                valid_subject_emotion_pairs[subject_id].add(emotion_name)
    
    print(f"Found {len(unique_subjects)} unique subjects in validation set")
    
    # Log available emotions for each subject
    for subject_id, emotions_set in valid_subject_emotion_pairs.items():
        print(f"Subject {subject_id} has {len(emotions_set)} emotions: {', '.join(emotions_set)}")
    
    # Convert to list for easier indexing
    all_subjects = list(unique_subjects.items())
    
    # Cache for the current data
    cache = {
        'subjects': all_subjects,
        'custom_image': None  # Will store the uploaded image tensor
    }
    
    # Convert images for display
    def tensor_to_display(img_tensor):
        """Convert normalized tensor to displayable numpy array"""
        # First move tensor to CPU before converting to numpy
        img_tensor = img_tensor.cpu()
        
        # Check if the image is single-channel (grayscale) or multi-channel (RGB)
        if img_tensor.dim() == 3 and img_tensor.shape[0] == 1:  # Grayscale [1, H, W]
            return ((img_tensor.squeeze().numpy() + 1) / 2 * 255).astype(np.uint8)
        elif img_tensor.dim() == 3 and img_tensor.shape[0] == 3:  # RGB [3, H, W]
            # Convert from CHW to HWC format for display
            img_array = img_tensor.permute(1, 2, 0).numpy()
            return ((img_array + 1) / 2 * 255).astype(np.uint8)
        elif img_tensor.dim() == 4:  # Batched images [B, C, H, W]
            if img_tensor.shape[1] == 1:  # Grayscale
                return ((img_tensor[0, 0].numpy() + 1) / 2 * 255).astype(np.uint8)
            else:  # RGB
                img_array = img_tensor[0].permute(1, 2, 0).numpy()
                return ((img_array + 1) / 2 * 255).astype(np.uint8)
        else:
            raise ValueError(f"Unexpected tensor shape: {img_tensor.shape}")
    
    # Function to process uploaded images
    def process_uploaded_image(uploaded_image):
        """
        Process an uploaded image to match the format expected by the model
        
        Args:
            uploaded_image: PIL image or numpy array from Gradio upload
        
        Returns:
            Processed tensor in the correct format
        """
        if uploaded_image is None:
            return None
        
        # Convert to PIL Image if needed
        if isinstance(uploaded_image, np.ndarray):
            image = Image.fromarray(uploaded_image)
        else:
            image = uploaded_image
        
        # Resize to 224x224
        image = image.resize((224, 224), Image.Resampling.LANCZOS)
        
        # Convert to RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to numpy array
        img_array = np.array(image)
        
        # Normalize to [-1, 1] range
        img_array = img_array.astype(np.float32) / 127.5 - 1.0
        
        # Convert to tensor with proper channel dimensions
        tensor = torch.from_numpy(img_array).permute(2, 0, 1)  # Change from HWC to CHW format
        
        return tensor
    
    def load_gallery_images():
        """Load all unique subjects for the gallery"""
        gallery_imgs = []
        
        # Display all unique subjects or up to 24 if there are too many
        max_display = min(len(cache['subjects']), 24)
        
        for i in range(max_display):
            subject_id, source_img = cache['subjects'][i]
            display_img = tensor_to_display(source_img)
            gallery_imgs.append((display_img, f"Subject {i} ({subject_id})"))
        
        return gallery_imgs
    
    def update_slider(image_idx, target_emotion_name):
        """Update the slider when a new image or emotion is selected"""
        if not 0 <= image_idx < len(cache['subjects']) or target_emotion_name not in emotion_to_idx:
            return gr.update(value=0, maximum=10), [], gr.update(value=None), "Invalid selection"
        
        # Get the source image for the selected subject
        subject_id, source_image = cache['subjects'][image_idx]
        source_image = source_image.unsqueeze(0).to(device)  # Add batch dimension and move to device
        
        # Get target emotion index
        target_emotion_idx = emotion_to_idx[target_emotion_name]
        # Create target emotion tensor (model expects class indices, not one-hot)
        target_emotion = torch.tensor([target_emotion_idx], device=device)
        
        # Call generate_trajectory with all required parameters
        trajectory = generate_trajectory(model, source_image, target_emotion, device, steps=10)
        
        # Convert frames for display
        frames = [tensor_to_display(frame[0]) for frame in trajectory]
        
        caption = f"Subject {image_idx} ({subject_id}) → {target_emotion_name}"
        
        return gr.update(value=0, maximum=len(frames)-1), frames, gr.update(value=frames[0]), caption
    
    def show_frame(frames, slider_value):
        """Show the frame at the slider position"""
        if not frames or slider_value >= len(frames):
            return None
        return frames[slider_value]
    
    # Create the Gradio interface
    with gr.Blocks(title="Facial Expression Editing") as app:
        gr.Markdown("# Facial Expression Editing with Flow Matching")
        gr.Markdown("1. Select a subject from the dropdown or upload your own image\n2. Choose a target emotion\n3. Select solver type\n4. Click 'Generate' to create the transformation\n5. Use the slider to view the progression")
        
        with gr.Row():
            with gr.Column(scale=2):
                # Create dropdown options with subject IDs
                subject_options = [f"Subject {i} ({subject_id})" for i, (subject_id, _) in enumerate(cache['subjects'][:30])]
                # Add option for custom uploaded image
                subject_options.insert(0, "Custom Uploaded Image")
                
                subject_dropdown = gr.Dropdown(
                    choices=subject_options,
                    value=subject_options[1] if len(subject_options) > 1 else subject_options[0],
                    label="Select Subject",
                    elem_id="subject_select"
                )
            
            with gr.Column(scale=2):
                target_emotion = gr.Dropdown(
                    choices=emotions,
                    value="happiness",
                    label="Target Emotion"
                )
            
            with gr.Column(scale=1):
                solver_type = gr.Radio(
                    choices=["euler", "adaptive"],
                    value="euler",
                    label="ODE Solver Type"
                )
                
                adaptive_solver = gr.Dropdown(
                    choices=["dopri5", "rk4", "euler", "midpoint"],
                    value="dopri5",
                    label="Adaptive Solver (if selected)",
                    visible=False
                )
            
        with gr.Row():
            with gr.Column(scale=1):
                generate_btn = gr.Button("Generate Transformation")
        
        with gr.Row():
            # Add image upload component
            image_upload = gr.Image(
                label="Upload Custom Image (will be resized to 224x224 and converted to grayscale)",
                type="pil",
                elem_id="image_upload"
            )
        
        # Hidden storage for frames and subject index
        frames_store = gr.State([])
        
        with gr.Row():
            with gr.Column(scale=1):
                # Add a small preview of the selected subject
                source_preview = gr.Image(
                    label="Source Face (Neutral)",
                    height=200,
                    width=200,
                    type="numpy"
                )
            
            with gr.Column(scale=1):
                output_image = gr.Image(
                    label="Transformed Image",
                    height=200,
                    width=200,
                    type="numpy"
                )
        
        with gr.Row():
            slider = gr.Slider(
                minimum=0,
                maximum=10,
                step=1,
                value=0,
                label="Transformation Progress"
            )

        output_label = gr.Label(label="Transformation Details")
        
        # Function to extract subject index from dropdown selection
        def get_subject_idx(subject_selection):
            if not subject_selection:
                return 0
            # Handle custom uploaded image case
            if subject_selection == "Custom Uploaded Image":
                return -1  # Special value for custom image
            # Extract the index part from "Subject X (id)"
            try:
                return int(subject_selection.split()[1].split('(')[0])
            except:
                return 0
        
        # Function to update the source preview when subject is selected
        def update_source_preview(subject_selection):
            idx = get_subject_idx(subject_selection)
            if idx == -1:  # Custom uploaded image
                if cache['custom_image'] is not None:
                    return tensor_to_display(cache['custom_image'])
                return None
            elif 0 <= idx < len(cache['subjects']):
                _, source_image = cache['subjects'][idx]
                return tensor_to_display(source_image)
            return None
        
        # Function to process uploaded image and update cache
        def handle_image_upload(image):
            if image is not None:
                tensor = process_uploaded_image(image)
                cache['custom_image'] = tensor
                # Update dropdown to select custom image
                return gr.update(value="Custom Uploaded Image"), tensor_to_display(tensor)
            return gr.update(), None
        
        # Add visibility toggle for adaptive solver dropdown
        def toggle_adaptive_solver_visibility(solver_choice):
            return gr.update(visible=solver_choice == "adaptive")
        
        solver_type.change(
            toggle_adaptive_solver_visibility,
            inputs=[solver_type],
            outputs=[adaptive_solver]
        )
        
        # Update the generate_transformation function to check for valid subject-emotion pairs
        def generate_transformation(subject_selection, emotion_name, solver_choice, adaptive_solver_choice):
            idx = get_subject_idx(subject_selection)
            
            # Get the appropriate source image
            if idx == -1:  # Custom uploaded image
                if cache['custom_image'] is None:
                    return gr.update(), [], gr.update(), "No custom image uploaded", None
                source_image = cache['custom_image'].unsqueeze(0).to(device)  # Add batch dimension
                subject_label = "Custom Image"
                subject_id = "custom"
            else:
                # Regular subject from dataset
                if not 0 <= idx < len(cache['subjects']):
                    return gr.update(), [], gr.update(), "Invalid subject selection", None
                subject_id, source_image = cache['subjects'][idx]
                source_image = source_image.unsqueeze(0).to(device)  # Add batch dimension
                subject_label = f"Subject {idx} ({subject_id})"
            
            # Check if this subject-emotion pair exists in the validation set
            valid_pair = True
            warning_message = ""
            
            if subject_id != "custom" and subject_id in valid_subject_emotion_pairs:
                if emotion_name not in valid_subject_emotion_pairs[subject_id]:
                    valid_pair = False
                    available_emotions = ", ".join(valid_subject_emotion_pairs[subject_id])
                    warning_message = f"Warning: No '{emotion_name}' expression available for {subject_label}.\nAvailable emotions: {available_emotions}.\nGenerated result may be unreliable."
            
            # Get target emotion index
            target_emotion_idx = emotion_to_idx[emotion_name]
            # Create target emotion tensor (model expects class indices, not one-hot)
            target_emotion = torch.tensor([target_emotion_idx], device=device)
            
            # Call generate_trajectory with all required parameters including solver choice
            trajectory = generate_trajectory(
                model, 
                source_image, 
                target_emotion, 
                device, 
                steps=10, 
                solver_type=solver_choice,
                adaptive_solver=adaptive_solver_choice
            )
            
            # Convert frames for display
            frames = [tensor_to_display(frame[0]) for frame in trajectory]
            
            # Prepare caption with warning if needed
            base_caption = f"{subject_label} → {emotion_name} (Solver: {solver_choice}" + (f", {adaptive_solver_choice})" if solver_choice == "adaptive" else ")")
            caption = warning_message + "\n" + base_caption if warning_message else base_caption
            
            # Also update the source preview
            source_preview = tensor_to_display(source_image[0])
            
            return gr.update(value=0, maximum=len(frames)-1), frames, gr.update(value=frames[0]), caption, source_preview
        
        # Set up event handlers
        subject_dropdown.change(
            update_source_preview,
            inputs=[subject_dropdown],
            outputs=[source_preview]
        )
        
        # Add handler for image upload
        image_upload.change(
            handle_image_upload,
            inputs=[image_upload],
            outputs=[subject_dropdown, source_preview]
        )
        
        generate_btn.click(
            generate_transformation,
            inputs=[subject_dropdown, target_emotion, solver_type, adaptive_solver],
            outputs=[slider, frames_store, output_image, output_label, source_preview]
        )
        
        slider.change(
            show_frame,
            inputs=[frames_store, slider],
            outputs=[output_image]
        )
    
    # Launch the app
    app.launch(share=True)

class DetailReconstructionLoss(nn.Module):
    def __init__(self, device, weight=1.0):
        """
        Detail reconstruction loss to preserve high-frequency facial details
        
        Args:
            device: Computation device
            weight: Weight for the loss
        """
        super().__init__()
        self.device = device
        self.weight = weight
        
        # Create Laplacian kernel for edge detection
        laplacian_kernel = torch.tensor([
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]
        ], dtype=torch.float32).view(1, 1, 3, 3).to(device)
        
        self.laplacian_kernel = laplacian_kernel.repeat(3, 1, 1, 1)
        
        # MSE loss for comparing details
        self.mse = nn.MSELoss()
        
    def extract_details(self, image):
        """
        Extract high-frequency details using Laplacian filtering
        
        Args:
            image: Input image tensor (B, C, H, W)
            
        Returns:
            Details tensor
        """
        # Apply padding to maintain original dimensions
        padded = F.pad(image, (1, 1, 1, 1), mode='reflect')
        
        # Apply the Laplacian filter to extract high-frequency details
        details = F.conv2d(padded, self.laplacian_kernel, groups=3)
        
        return details
    
    def forward(self, generated_images, source_images):
        """
        Compute detail preservation loss between generated and source images
        
        Args:
            generated_images: Generated face images with new expressions (B, C, H, W)
            source_images: Original source face images (B, C, H, W)
            
        Returns:
            loss: Detail preservation loss (scalar)
        """
        # Handle grayscale to RGB conversion if needed
        if generated_images.shape[1] == 1:
            generated_images = generated_images.repeat(1, 3, 1, 1)
        if source_images.shape[1] == 1:
            source_images = source_images.repeat(1, 3, 1, 1)
        
        # Extract high-frequency details
        source_details = self.extract_details(source_images)
        generated_details = self.extract_details(generated_images)
        
        # Calculate the MSE between details
        detail_loss = self.mse(generated_details, source_details)
        
        return self.weight * detail_loss
