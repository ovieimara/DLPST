import os
import numpy as np
import torch
import time
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import SimpleITK as sitk
from collections import OrderedDict
import argparse
import nibabel as nib


import os
import sys

# Add current directory to Python path to ensure sam2 can be imported
sys.path.insert(0, os.getcwd())

try:
    from sam2.build_sam import build_sam2_video_predictor_npz
    MODEL_AVAILABLE = True
    print("Successfully imported build_sam2_video_predictor_npz")
except ImportError as e:
    print(f"Import error: {e}")
    print("Warning: MedSAM2 model not found. Will run in evaluation-only mode.")
    MODEL_AVAILABLE = False
except Exception as e:
    print(f"Other error during import: {e}")
    MODEL_AVAILABLE = False

torch.set_float32_matmul_precision('high')
torch.manual_seed(2024)
torch.cuda.manual_seed(2024)
np.random.seed(2024)

def getLargestCC(segmentation):
    """Get the largest connected component from a binary segmentation"""
    from skimage import measure
    labels = measure.label(segmentation)
    if labels.max() == 0:
        return segmentation
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
    return largestCC

def resize_grayscale_to_rgb_and_resize(array, image_size):
    """
    Resize a 3D grayscale NumPy array to an RGB image and then resize it.
    
    Parameters:
        array (np.ndarray): Input array of shape (d, h, w).
        image_size (int): Desired size for the width and height.
    
    Returns:
        np.ndarray: Resized array of shape (d, 3, image_size, image_size).
    """
    from PIL import Image
    d, h, w = array.shape
    resized_array = np.zeros((d, 3, image_size, image_size))
    
    for i in range(d):
        img_pil = Image.fromarray(array[i].astype(np.uint8))
        img_rgb = img_pil.convert("RGB")
        img_resized = img_rgb.resize((image_size, image_size))
        img_array = np.array(img_resized).transpose(2, 0, 1)  # (3, image_size, image_size)
        resized_array[i] = img_array
    
    return resized_array

def calculate_dice_score(pred, gt):
    """Calculate Dice score between prediction and ground truth"""
    # Ensure binary masks
    pred_binary = (pred > 0).astype(np.float32)
    gt_binary = (gt > 0).astype(np.float32)
    
    # Calculate Dice coefficient
    intersection = np.sum(pred_binary * gt_binary)
    dice = 2 * intersection / (np.sum(pred_binary) + np.sum(gt_binary) + 1e-8)
    
    return dice

def load_reference_nifti(reference_path):
    """Load a reference NIfTI file to get header and affine information"""
    ref_nii = nib.load(reference_path)
    return ref_nii.affine, ref_nii.header

def create_dummy_box_prompt(img_shape):
    """Create a dummy bounding box prompt in the center of the resized image"""
    d, h, w = img_shape  # This will be (depth, 512, 512) for resized images
    # Create a box in the center of the image
    x_min, x_max = w//4, 3*w//4
    y_min, y_max = h//4, 3*h//4
    z_min, z_max = d//4, 3*d//4
    
    # For a 2D box prompt (on a middle slice), we'll use the middle slice
    middle_slice = d // 2
    return np.array([x_min, y_min, x_max, y_max]), middle_slice

def run_inference_on_npz(predictor, npz_data, case_id, use_box_prompt=True):
    """Run MedSAM2 inference on NPZ data"""
    imgs = npz_data['imgs']  # 3D volume (D, W, H)
    print(f"  Input shape: {imgs.shape}")
    
    # Preprocess images
    img_3D_ori = imgs.astype(np.float32)
    assert np.max(img_3D_ori) < 256, f'input data should be in range [0, 255], but got {np.max(img_3D_ori)}'
    
    original_depth, original_height, original_width = img_3D_ori.shape
    
    # Resize images to 512x512 for model input
    img_resized = resize_grayscale_to_rgb_and_resize(img_3D_ori, 512)
    img_resized = img_resized / 255.0
    img_resized = torch.from_numpy(img_resized).cuda()
    
    # Normalize with ImageNet statistics
    img_mean = (0.485, 0.456, 0.406)
    img_std = (0.229, 0.224, 0.225)
    img_mean = torch.tensor(img_mean, dtype=torch.float32)[:, None, None].cuda()
    img_std = torch.tensor(img_std, dtype=torch.float32)[:, None, None].cuda()
    img_resized = (img_resized - img_mean) / img_std
    
    # Get video dimensions
    video_depth, _, video_height, video_width = img_resized.shape
    
    # Initialize segmentation mask with original dimensions
    segs_3D = np.zeros((original_depth, original_height, original_width), dtype=np.uint8)
    
    # Create prompt on resized images
    if use_box_prompt:
        # Create box prompt on the resized 512x512 image
        bbox, key_slice_idx = create_dummy_box_prompt((video_depth, video_height, video_width))
        key_slice_idx_offset = key_slice_idx
    else:
        # If not using box prompt, use center slice
        key_slice_idx = video_depth // 2
        key_slice_idx_offset = key_slice_idx
        bbox = None
    
    # Run inference
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        inference_state = predictor.init_state(img_resized, video_height, video_width)
        
        if use_box_prompt and bbox is not None:
            try:
                _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=key_slice_idx_offset,
                    obj_id=1,
                    box=bbox,
                )
            except Exception as e:
                print(f"    Warning: Failed to add box prompt: {e}")
                pass
        
        # Forward propagation
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
            # Get the mask prediction (512x512)
            mask_512 = (out_mask_logits[0] > 0.0).cpu().numpy()[0]
            
            # Resize mask back to original dimensions
            from PIL import Image
            mask_pil = Image.fromarray(mask_512.astype(np.uint8))
            mask_resized = mask_pil.resize((original_width, original_height), Image.NEAREST)
            mask_original = np.array(mask_resized)
            
            # Apply to the segmentation mask
            segs_3D[out_frame_idx, :, :] = mask_original
        
        predictor.reset_state(inference_state)
        
        # Backward propagation
        if use_box_prompt and bbox is not None:
            try:
                _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=key_slice_idx_offset,
                    obj_id=1,
                    box=bbox,
                )
            except Exception as e:
                print(f"    Warning: Failed to add box prompt for reverse: {e}")
                pass
        
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state, reverse=True):
            # Get the mask prediction (512x512)
            mask_512 = (out_mask_logits[0] > 0.0).cpu().numpy()[0]
            
            # Resize mask back to original dimensions
            from PIL import Image
            mask_pil = Image.fromarray(mask_512.astype(np.uint8))
            mask_resized = mask_pil.resize((original_width, original_height), Image.NEAREST)
            mask_original = np.array(mask_resized)
            
            # Apply to the segmentation mask
            segs_3D[out_frame_idx, :, :] = mask_original
        
        predictor.reset_state(inference_state)
    
    # Post-process segmentation
    if np.max(segs_3D) > 0:
        segs_3D = getLargestCC(segs_3D)
        segs_3D = np.uint8(segs_3D)
    
    return segs_3D

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        default="checkpoints/checkpoint.pt",
        help='checkpoint path',
    )
    parser.add_argument(
        '--cfg',
        type=str,
        default="configs/sam2.1_hiera_t512.yaml",
        help='model config',
    )
    
    parser.add_argument(
        '-i',
        '--imgs_path',
        type=str,
        default="/space/fast/oim/medsam_raw/Dataset109_ProstrateCTV/conversions2/test",
        help='imgs path',
    )
    parser.add_argument(
        '--gts_path',
        type=str,
        default="/space/fast/oim/nnUNet_raw/Dataset109_ProstrateCTV/labelsTs",
        help='ground truth path',
    )
    parser.add_argument(
        '--original_imgs_path',
        type=str,
        default="/space/fast/oim/nnUNet_raw/Dataset109_ProstrateCTV/imagesTs",
        help='original images path for NIfTI header',
    )
    parser.add_argument(
        '-o',
        '--pred_save_dir',
        type=str,
        default="/space/fast/oim/medsam_raw/inference_results_nii",
        help='path to save segmentation results',
    )
    parser.add_argument(
        '--propagate_with_box',
        default=True,
        action='store_true',
        help='whether to propagate with box'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.pred_save_dir, exist_ok=True)
    
    # Get test files
    test_path = Path(args.imgs_path)
    npz_files = sorted(list(test_path.glob("*.npz")))
    
    print(f'Found {len(npz_files)} test files')
    
    # Initialize results tracking
    results_data = {
        'case_id': [],
        'dice_score': [],
        'inference_time_seconds': [],
        'input_shape': [],
        'prediction_shape': []
    }
    
    # Initialize predictor if model is available
    if MODEL_AVAILABLE:
        try:
            print("Loading MedSAM2 model...")
            predictor = build_sam2_video_predictor_npz(
                "/configs/sam2.1_hiera_t512.yaml",  # Use a standard config
                "/app/MedSAM2/checkpoints/checkpoint.pt"
            )
            predictor.eval()
            if torch.cuda.is_available():
                predictor = predictor.cuda()
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Failed to load model: {e}")
            import traceback
            traceback.print_exc()
            return
    
    # Process each test case
    for npz_file in tqdm(npz_files, desc="Processing test cases"):
        try:
            case_id = npz_file.stem  # e.g., "case_001"
            print(f"\nProcessing {case_id}...")
            
            # Start timing
            start_time = time.time()
            
            if MODEL_AVAILABLE and predictor is not None:
                # Load test data
                test_data = np.load(npz_file)
                
                # Run inference
                pred_mask = run_inference_on_npz(
                    predictor, 
                    test_data, 
                    case_id, 
                    use_box_prompt=args.propagate_with_box
                )
                
                # End timing
                end_time = time.time()
                inference_time = end_time - start_time
                
                # Convert prediction from (D, W, H) back to (W, H, D) for NIfTI
                pred_mask_nifti = np.transpose(pred_mask, (1, 2, 0))  # (D, W, H) -> (W, H, D)
                
                # Save prediction as .nii.gz using reference header/affine
                original_img_path = os.path.join(args.original_imgs_path, f"{case_id}_0000.nii.gz")
                pred_filename = f"{case_id}_pred.nii.gz"
                pred_path = os.path.join(args.pred_save_dir, pred_filename)
                
                if os.path.exists(original_img_path):
                    # Use original image header and affine for proper orientation
                    try:
                        affine, header = load_reference_nifti(original_img_path)
                        pred_nii = nib.Nifti1Image(pred_mask_nifti, affine, header)
                    except Exception as e:
                        print(f"    Warning: Failed to load reference NIfTI, using default: {e}")
                        pred_nii = nib.Nifti1Image(pred_mask_nifti, np.eye(4))
                else:
                    # Create without reference (may affect orientation)
                    print(f"    Warning: Reference image not found at {original_img_path}")
                    pred_nii = nib.Nifti1Image(pred_mask_nifti, np.eye(4))
                
                nib.save(pred_nii, pred_path)
                print(f"  Prediction saved to: {pred_path}")
                
            else:
                # If model not available, check if prediction already exists
                pred_path = os.path.join(args.pred_save_dir, f"{case_id}_pred.nii.gz")
                if os.path.exists(pred_path):
                    print(f"  Using existing prediction: {pred_path}")
                    # Load existing prediction
                    pred_nii = nib.load(pred_path)
                    pred_mask_nifti = pred_nii.get_fdata()
                    # Convert back to (D, W, H) format for evaluation
                    pred_mask = np.transpose(pred_mask_nifti, (2, 0, 1))  # (W, H, D) -> (D, W, H)
                    inference_time = 0  # No inference time for existing predictions
                else:
                    print(f"  Warning: No model available and no existing prediction found for {case_id}")
                    continue
            
            # Load ground truth for evaluation
            gt_file_path = os.path.join(args.gts_path, f"{case_id}.nii.gz")
            if os.path.exists(gt_file_path):
                try:
                    gt_nii = nib.load(gt_file_path)
                    gt_data = gt_nii.get_fdata()
                    print(f"  Ground truth shape: {gt_data.shape}")
                    
                    # Calculate Dice score
                    dice = calculate_dice_score(pred_mask_nifti, gt_data)
                    
                    # Store results
                    results_data['case_id'].append(case_id)
                    results_data['dice_score'].append(dice)
                    results_data['inference_time_seconds'].append(inference_time)
                    results_data['input_shape'].append(str(pred_mask.shape))
                    results_data['prediction_shape'].append(str(pred_mask.shape))
                    
                    print(f"  Dice score: {dice:.4f}")
                    if inference_time > 0:
                        print(f"  Inference time: {inference_time:.2f} seconds")
                except Exception as e:
                    print(f"  Error evaluating {case_id}: {e}")
            else:
                print(f"  Warning: Ground truth not found for {case_id}")
                
        except Exception as e:
            print(f"Error processing {npz_file.name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save results to CSV
    if results_data['case_id']:
        results_df = pd.DataFrame(results_data)
        
        # Calculate summary statistics
        avg_dice = np.mean(results_data['dice_score'])
        std_dice = np.std(results_data['dice_score'])
        avg_time = np.mean([t for t in results_data['inference_time_seconds'] if t > 0]) if any(t > 0 for t in results_data['inference_time_seconds']) else 0
        
        # Add summary row
        summary_data = {
            'case_id': ['SUMMARY'],
            'dice_score': [avg_dice],
            'inference_time_seconds': [avg_time],
            'input_shape': [f"Mean ± Std: {avg_dice:.4f} ± {std_dice:.4f}"],
            'prediction_shape': [f"Average inference time: {avg_time:.2f}s"]
        }
        summary_df = pd.DataFrame(summary_data)
        final_results_df = pd.concat([results_df, summary_df], ignore_index=True)
        
        results_csv_path = os.path.join(args.pred_save_dir, "evaluation_results.csv")
        final_results_df.to_csv(results_csv_path, index=False)
        
        print(f"\n=== EVALUATION SUMMARY ===")
        print(f"Total cases processed: {len(results_data['case_id'])}")
        print(f"Average Dice score: {avg_dice:.4f} ± {std_dice:.4f}")
        if avg_time > 0:
            print(f"Average inference time: {avg_time:.2f} seconds")
        print(f"Results saved to: {results_csv_path}")
        print(f"Predictions saved to: {args.pred_save_dir} (as .nii.gz files)")
    else:
        print("No results to save")

if __name__ == "__main__":
    main()