#!/bin/bash

export nnUNet_raw=/space/fast/oim/nnUNet_raw
export nnUNet_preprocessed=/space/fast/oim/nnUNet_preprocessed
export nnUNet_results=/space/fast/oim/nnUNet_results

#nnunets folder paths
INPUT_FOLDER="/space/fast/oim/nnUNet_raw/Dataset109_ProstrateCTV/imagesTs"
OUTPUT_BASE="/space/fast/oim/nnUNet_results/Dataset109_ProstrateCTV/pred"
LABELS_FOLDER="/space/fast/oim/nnUNet_raw/Dataset109_ProstrateCTV/imagesTs/ground_truth"
DATASET_JSON="/space/fast/oim/nnUNet_preprocessed/Dataset109_ProstrateCTV/dataset.json"
PLANS_JSON="/space/fast/oim/nnUNet_preprocessed/Dataset109_ProstrateCTV/nnUNetPlans.json"




# Run 2D inference
nnUNetv2_predict -d Dataset109_ProstrateCTV \
                 -i $INPUT_FOLDER \
                 -o $OUTPUT_FOLDER_2D \
                 -f 0 1 2 3 4 \
                 -tr nnUNetTrainer \
                 -c 2d \
                 -p nnUNetPlans \
                 --save_probabilities
# Run 3D inference
nnUNetv2_predict -d Dataset109_ProstrateCTV \
                 -i $INPUT_FOLDER \
                 -o $OUTPUT_FOLDER_3D_FULLRES \
                 -f 0 1 2 3 4 \
                 -tr nnUNetTrainer \
                 -c 3d_fullres \
                 -p nnUNetPlans \
                 --save_probabilities

# Run ensemble inference
nnUNetv2_ensemble \
    -i $OUTPUT_FOLDER_2D $OUTPUT_FOLDER_3D_FULLRES \
    -o $ENSEMBLE_FINAL_OUTPUT \
    -np 8  # Use 8 processes for faster ensembling

nnUNetv2_evaluate_folder \
    -djfile $DATASET_JSON \
    -pfile $PLANS_JSON \
    $LABELS_FOLDER \
    $ENSEMBLE_FINAL_OUTPUT \
    -o $DICE_OUTPUT \
    -np 8