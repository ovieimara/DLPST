import os
import numpy as np
import nibabel as nib
from pathlib import Path

def prepare_medsam_standard_format(nnunet_dir, output_dir, train_split=0.8):
    """
    Convert nnUNet dataset to standard MedSAM format with train/val split from imagesTr
    and separate test set from imagesTs for later evaluation
    
    Args:
        nnunet_dir: Path to nnUNet dataset directory
        output_dir: Path to output directory
        train_split: Fraction of imagesTr to use for training (rest goes to validation)
    """
    nnunet_path = Path(nnunet_dir)
    output_path = Path(output_dir)
    
    # Create standard MedSAM structure
    train_dir = output_path / "train"
    val_dir = output_path / "val"
    test_dir = output_path / "test"
    
    for dir_path in [train_dir, val_dir, test_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Process training data with train/val split
    images_tr = nnunet_path / "imagesTr"
    labels_tr = nnunet_path / "labelsTr"
    
    if images_tr.exists() and labels_tr.exists():
        print("Processing imagesTr and labelsTr with train/validation split...")
        
        # Get all image files and sort for reproducible splits
        image_files = sorted(list(images_tr.glob("*.nii.gz")))
        
        # Calculate split index
        split_idx = int(len(image_files) * train_split)
        
        # Split files
        train_files = image_files[:split_idx]
        val_files = image_files[split_idx:]
        
        print(f"Total files: {len(image_files)}")
        print(f"Training files: {len(train_files)} ({len(train_files)/len(image_files)*100:.1f}%)")
        print(f"Validation files: {len(val_files)} ({len(val_files)/len(image_files)*100:.1f}%)")
        
        # Process training split
        print("\n--- Processing Training Split ---")
        process_split_with_file_list(train_files, labels_tr, train_dir)
        
        # Process validation split  
        print("\n--- Processing Validation Split ---")
        process_split_with_file_list(val_files, labels_tr, val_dir)
    else:
        print("Warning: imagesTr or labelsTr directory not found!")
    
    # Process test data (imagesTs/labelsTs) for later evaluation
    images_ts = nnunet_path / "imagesTs"
    labels_ts = nnunet_path / "labelsTs" 
    if images_ts.exists():
        print("\n--- Processing Test Set (imagesTs/labelsTs) ---")
        if labels_ts.exists():
            process_split_standard(images_ts, labels_ts, test_dir)
        else:
            print("Warning: labelsTs not found, processing imagesTs without labels")
            process_split_standard(images_ts, None, test_dir)

def process_split_with_file_list(image_files, labels_dir, output_dir):
    """Process a specific list of image files for train/val split"""
    for img_file in image_files:
        try:
            # Load image
            img_nii = nib.load(img_file)
            img_data = img_nii.get_fdata().astype(np.float32)
            
            print(f"Original image shape: {img_data.shape}")
            
            # Check and potentially reorder dimensions to (D, W, H)
            # Your data is (W, H, D) = (1024, 1024, 88), need to convert to (D, W, H) = (88, 1024, 1024)
            if len(img_data.shape) == 3:
                img_data = np.transpose(img_data, (2, 0, 1))  # (W, H, D) -> (D, W, H)
            
            # Normalize to [0, 255] with better handling
            img_min, img_max = np.percentile(img_data, [1, 99])  # More robust than min/max
            if img_max > img_min:
                img_normalized = np.clip((img_data - img_min) / (img_max - img_min) * 255, 0, 255)
            else:
                img_normalized = np.zeros_like(img_data)
            img_normalized = img_normalized.astype(np.uint8)
            
            # Handle nnUNet naming
            img_stem = img_file.stem.replace('.nii', '')
            if '_' in img_stem and img_stem.endswith('_0000'):
                label_name = img_stem[:-5] + '.nii.gz'
                output_name = img_stem[:-5]
            else:
                label_name = img_stem + '.nii.gz'
                output_name = img_stem
            
            label_file = labels_dir / label_name if labels_dir else None
            
            if labels_dir is None:
                # Test set without labels - save image only
                output_filename = f"{output_name}.npz"
                output_path = output_dir / output_filename
                
                np.savez_compressed(
                    output_path,
                    imgs=img_normalized  # Only images for test set
                )
                
                print(f"✓ Processed {output_name} (image only): {img_normalized.shape} -> {output_filename}")
                
            elif label_file.exists():
                # Load mask
                mask_nii = nib.load(label_file)
                mask_data = mask_nii.get_fdata().astype(np.int32)
                
                print(f"Original mask shape: {mask_data.shape}")
                
                # Apply same dimension reordering as image
                # Your mask data is also (W, H, D), convert to (D, W, H)
                if len(mask_data.shape) == 3:
                    mask_data = np.transpose(mask_data, (2, 0, 1))  # (W, H, D) -> (D, W, H)
                
                # Verify shapes match
                if img_normalized.shape != mask_data.shape:
                    print(f"⚠ Warning: Shape mismatch - Image: {img_normalized.shape}, Mask: {mask_data.shape}")
                
                # Create single .npz file with both imgs and gts (MedSAM2 format)
                output_filename = f"{output_name}.npz"
                output_path = output_dir / output_filename
                
                # Save in MedSAM2 expected format
                np.savez_compressed(
                    output_path,
                    imgs=img_normalized,  # (D, W, H), [0, 255]
                    gts=mask_data        # (D, W, H), integer labels
                )
                
                print(f"✓ Processed {output_name}: {img_normalized.shape} -> {output_filename}")
                
            else:
                if labels_dir:
                    print(f"⚠ Warning: No label found for {img_file.name} (looking for {label_name})")
                else:
                    print(f"⚠ Warning: No labels directory provided")
                
        except Exception as e:
            print(f"✗ Error processing {img_file.name}: {e}")
            import traceback
            traceback.print_exc()

def process_split_standard(images_dir, labels_dir, output_dir):
    """Process data split with proper nnUNet naming convention for MedSAM2"""
    if images_dir is None:
        return
        
    image_files = list(images_dir.glob("*.nii.gz"))
    
    for img_file in image_files:
        try:
            # Load image
            img_nii = nib.load(img_file)
            img_data = img_nii.get_fdata().astype(np.float32)
            
            print(f"Original image shape: {img_data.shape}")
            
            # Check and potentially reorder dimensions to (D, W, H)
            # Your data is (W, H, D) = (1024, 1024, 88), need to convert to (D, W, H) = (88, 1024, 1024)
            if len(img_data.shape) == 3:
                img_data = np.transpose(img_data, (2, 0, 1))  # (W, H, D) -> (D, W, H)
            
            # Normalize to [0, 255] with better handling
            img_min, img_max = np.percentile(img_data, [1, 99])  # More robust than min/max
            if img_max > img_min:
                img_normalized = np.clip((img_data - img_min) / (img_max - img_min) * 255, 0, 255)
            else:
                img_normalized = np.zeros_like(img_data)
            img_normalized = img_normalized.astype(np.uint8)
            
            # Handle nnUNet naming
            img_stem = img_file.stem.replace('.nii', '')
            if '_' in img_stem and img_stem.endswith('_0000'):
                label_name = img_stem[:-5] + '.nii.gz'
                output_name = img_stem[:-5]
            else:
                label_name = img_stem + '.nii.gz'
                output_name = img_stem
            
            label_file = labels_dir / label_name
            
            if label_file.exists():
                # Load mask
                mask_nii = nib.load(label_file)
                mask_data = mask_nii.get_fdata().astype(np.int32)
                
                print(f"Original mask shape: {mask_data.shape}")
                
                # Apply same dimension reordering as image
                # Your mask data is also (W, H, D), convert to (D, W, H)
                if len(mask_data.shape) == 3:
                    mask_data = np.transpose(mask_data, (2, 0, 1))  # (W, H, D) -> (D, W, H)
                
                # Verify shapes match
                if img_normalized.shape != mask_data.shape:
                    print(f"⚠ Warning: Shape mismatch - Image: {img_normalized.shape}, Mask: {mask_data.shape}")
                
                # Create single .npz file with both imgs and gts (MedSAM2 format)
                output_filename = f"{output_name}.npz"
                output_path = output_dir / output_filename
                
                # Save in MedSAM2 expected format
                # np.savez_compressed(
                #     output_path,
                #     imgs=img_normalized,  # (D, W, H), [0, 255]
                #     gts=mask_data        # (D, W, H), integer labels
                # )
                
                
                # Optional: Add RECIST markers if needed for training
                recist = create_recist_marker(mask_data)
                np.savez_compressed(output_path, imgs=img_normalized, gts=mask_data, recist=recist)
                
                print(f"✓ Processed {output_name}: {img_normalized.shape} -> {output_filename}")

                
            else:
                print(f"⚠ Warning: No label found for {img_file.name} (looking for {label_name})")
                
        except Exception as e:
            print(f"✗ Error processing {img_file.name}: {e}")
            import traceback
            traceback.print_exc()
import numpy as np
from scipy import ndimage
from scipy.ndimage import center_of_mass, binary_erosion, binary_dilation

def create_recist_marker(mask_data, marker_type='center_slice', erosion_iterations=2):
    """
    Create RECIST markers for MedSAM2 training from ground truth masks.
    
    Args:
        mask_data: 3D numpy array (D, W, H) with integer labels
        marker_type: str, method for creating markers
            - 'center_slice': Mark on slice with largest lesion area
            - 'largest_component': Mark largest connected component on center slice
            - 'centroid': Mark centroid of lesion on center slice
        erosion_iterations: int, number of erosion iterations to create smaller markers
    
    Returns:
        recist: 3D numpy array (D, W, H) with binary RECIST markers {0, 1}
    """
    # Initialize RECIST marker array
    recist = np.zeros_like(mask_data, dtype=np.uint8)
    
    # Handle case where mask is empty
    if np.sum(mask_data > 0) == 0:
        return recist
    
    # Find the slice with the largest lesion area
    slice_areas = []
    for z in range(mask_data.shape[0]):
        area = np.sum(mask_data[z] > 0)
        slice_areas.append(area)
    
    # Get the middle slice with largest area
    max_area_slice = np.argmax(slice_areas)
    
    if slice_areas[max_area_slice] == 0:
        return recist
    
    # Get the 2D mask for the selected slice
    slice_mask = (mask_data[max_area_slice] > 0).astype(np.uint8)
    
    if marker_type == 'center_slice':
        # Simple approach: erode the mask to create a marker
        marker_2d = binary_erosion(slice_mask, iterations=erosion_iterations)
        
        # If erosion removes everything, use original mask
        if np.sum(marker_2d) == 0:
            marker_2d = slice_mask
            
    elif marker_type == 'largest_component':
        # Find largest connected component and erode it
        labeled_mask, num_labels = ndimage.label(slice_mask)
        if num_labels > 0:
            # Find largest component
            component_sizes = [np.sum(labeled_mask == i) for i in range(1, num_labels + 1)]
            largest_component_label = np.argmax(component_sizes) + 1
            
            # Create marker from largest component
            largest_component = (labeled_mask == largest_component_label).astype(np.uint8)
            marker_2d = binary_erosion(largest_component, iterations=erosion_iterations)
            
            # If erosion removes everything, use original component
            if np.sum(marker_2d) == 0:
                marker_2d = largest_component
        else:
            marker_2d = slice_mask
            
    elif marker_type == 'centroid':
        # Create marker at centroid of lesion
        if np.sum(slice_mask) > 0:
            # Find centroid
            cy, cx = center_of_mass(slice_mask)
            cy, cx = int(round(cy)), int(round(cx))
            
            # Create small marker around centroid
            marker_2d = np.zeros_like(slice_mask)
            
            # Create a small square marker (adjust size as needed)
            marker_size = max(3, min(slice_mask.shape) // 20)  # Adaptive marker size
            y_start = max(0, cy - marker_size // 2)
            y_end = min(slice_mask.shape[0], cy + marker_size // 2 + 1)
            x_start = max(0, cx - marker_size // 2)
            x_end = min(slice_mask.shape[1], cx + marker_size // 2 + 1)
            
            marker_2d[y_start:y_end, x_start:x_end] = 1
            
            # Make sure marker is within the original lesion
            marker_2d = marker_2d * slice_mask
        else:
            marker_2d = slice_mask
    
    else:
        raise ValueError(f"Unknown marker_type: {marker_type}")
    
    # Place the 2D marker into the 3D RECIST array
    recist[max_area_slice] = marker_2d.astype(np.uint8)
    
    return recist

def create_recist_marker_multi_slice(mask_data, num_slices=3, erosion_iterations=2):
    """
    Create RECIST markers on multiple slices around the center.
    
    Args:
        mask_data: 3D numpy array (D, W, H) with integer labels
        num_slices: int, number of slices to mark (should be odd)
        erosion_iterations: int, erosion iterations for marker creation
    
    Returns:
        recist: 3D numpy array (D, W, H) with binary RECIST markers
    """
    recist = np.zeros_like(mask_data, dtype=np.uint8)
    
    if np.sum(mask_data > 0) == 0:
        return recist
    
    # Find slices with lesions
    slice_areas = [np.sum(mask_data[z] > 0) for z in range(mask_data.shape[0])]
    
    # Find center of mass in z-direction
    z_indices = np.arange(len(slice_areas))
    z_weights = np.array(slice_areas)
    
    if np.sum(z_weights) == 0:
        return recist
    
    center_z = int(np.average(z_indices, weights=z_weights))
    
    # Select slices around center
    half_slices = num_slices // 2
    start_z = max(0, center_z - half_slices)
    end_z = min(mask_data.shape[0], center_z + half_slices + 1)
    
    for z in range(start_z, end_z):
        if slice_areas[z] > 0:
            slice_mask = (mask_data[z] > 0).astype(np.uint8)
            marker_2d = binary_erosion(slice_mask, iterations=erosion_iterations)
            
            if np.sum(marker_2d) == 0:
                marker_2d = slice_mask
            
            recist[z] = marker_2d.astype(np.uint8)
    
    return recist

def create_box_prompt_from_recist(recist_marker):
    """
    Create a bounding box prompt from RECIST marker for inference.
    
    Args:
        recist_marker: 3D numpy array (D, W, H) with binary RECIST markers
    
    Returns:
        box_coords: dict with 'slice_idx' and 'bbox' [x_min, y_min, x_max, y_max]
    """
    # Find slice with marker
    slice_with_marker = None
    for z in range(recist_marker.shape[0]):
        if np.sum(recist_marker[z]) > 0:
            slice_with_marker = z
            break
    
    if slice_with_marker is None:
        return None
    
    # Get 2D marker
    marker_2d = recist_marker[slice_with_marker]
    
    # Find bounding box of marker
    rows, cols = np.where(marker_2d > 0)
    
    if len(rows) == 0:
        return None
    
    x_min, x_max = cols.min(), cols.max()
    y_min, y_max = rows.min(), rows.max()
    
    # Add some padding to the box
    padding = 5
    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(marker_2d.shape[1] - 1, x_max + padding)
    y_max = min(marker_2d.shape[0] - 1, y_max + padding)
    
    return {
        'slice_idx': slice_with_marker,
        'bbox': [x_min, y_min, x_max, y_max]
    }


# Example usage
if __name__ == "__main__":
    # Convert your dataset with 80% train, 20% validation split
    prepare_medsam_standard_format(
        "/space/fast/oim/nnUNet_raw/Dataset109_ProstrateCTV",
        "/space/fast/oim/medsam_raw/Dataset109_ProstrateCTV/conversions2",
        train_split=0.8  # 80% train, 20% validation
    )    
 
