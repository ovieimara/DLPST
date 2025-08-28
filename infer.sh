#!/bin/bash

export nnUNet_raw=/space/fast/oim/nnUNet_raw
export nnUNet_preprocessed=/space/fast/oim/nnUNet_preprocessed
export nnUNet_results=/space/fast/oim/nnUNet_results

# Your provided paths
# INPUT_FOLDER="/space/fast/oim/nnUNet_raw/Dataset109_ProstrateCTV/imagesTs"
# OUTPUT_BASE="/space/fast/oim/nnUNet_results/Dataset109_ProstrateCTV/pred"
# LABELS_FOLDER="/space/fast/oim/nnUNet_raw/Dataset109_ProstrateCTV/imagesTs/ground_truth"
# DATASET_JSON="/space/fast/oim/nnUNet_preprocessed/Dataset109_ProstrateCTV/dataset.json"
# PLANS_JSON="/space/fast/oim/nnUNet_preprocessed/Dataset109_ProstrateCTV/nnUNetPlans.json"




# Run inference
# nnUNetv2_predict -d Dataset109_ProstrateCTV \
#                  -i $INPUT_FOLDER \
#                  -o $OUTPUT_FOLDER_2D \
#                  -f 0 1 2 3 4 \
#                  -tr nnUNetTrainer \
#                  -c 2d \
#                  -p nnUNetPlans \
#                  --save_probabilities

# nnUNetv2_predict -d Dataset109_ProstrateCTV \
#                  -i $INPUT_FOLDER \
#                  -o $OUTPUT_FOLDER_3D_FULLRES \
#                  -f 0 1 2 3 4 \
#                  -tr nnUNetTrainer \
#                  -c 3d_fullres \
#                  -p nnUNetPlans \
#                  --save_probabilities

# nnUNetv2_ensemble \
#     -i $OUTPUT_FOLDER_2D $OUTPUT_FOLDER_3D_FULLRES \
#     -o $ENSEMBLE_FINAL_OUTPUT \
#     -np 8  # Use 8 processes for faster ensembling

# nnUNetv2_evaluate_folder \
#     -djfile $DATASET_JSON \
#     -pfile $PLANS_JSON \
#     $LABELS_FOLDER \
#     $ENSEMBLE_FINAL_OUTPUT \
#     -o $DICE_OUTPUT \
#     -np 8

INPUT_FOLDER="/space/fast/oim/nnUNet_raw/Dataset109_ProstrateCTV/imagesTs"
LABELS_FOLDER="/space/fast/oim/nnUNet_raw/Dataset109_ProstrateCTV/labelsTs"
OUTPUT_BASE="/space/fast/oim/nnUNet_results/Dataset109_ProstrateCTV/fresh_run"
OUTPUT_2D="${OUTPUT_BASE}/2D"
OUTPUT_3D="${OUTPUT_BASE}/3D_fullres"
ENSEMBLE_OUTPUT="${OUTPUT_BASE}/ensemble"
TIMING_CSV="${OUTPUT_BASE}/timing_results.csv"
LOG_DIR="${OUTPUT_BASE}/logs"

# nnUNet dataset paths
DATASET_JSON="/space/fast/oim/nnUNet_preprocessed/Dataset109_ProstrateCTV/dataset.json"
PLANS_JSON="/space/fast/oim/nnUNet_preprocessed/Dataset109_ProstrateCTV/nnUNetPlans.json"

# Create fresh output directories
mkdir -p "$OUTPUT_2D" "$OUTPUT_3D" "$ENSEMBLE_OUTPUT" "$LOG_DIR"

# Initialize timing CSV only
echo "step,time_seconds,timestamp,exit_code" > "$TIMING_CSV"

echo "================================================"
echo "FRESH NNUNET PIPELINE WITH DEFAULT EVALUATION"
echo "================================================"
echo "Input: $INPUT_FOLDER"
echo "Labels: $LABELS_FOLDER"
echo "Output Base: $OUTPUT_BASE"
echo "Timing CSV: $TIMING_CSV"
echo "================================================"

# Function to run command with timing and logging
run_step() {
    local step_name=$1
    shift
    local cmd=("$@")
    
    echo "Starting: $step_name"
    echo "Command: ${cmd[*]}"
    
    start_time=$(date +%s.%N)
    
    # Run the command and capture output
    "${cmd[@]}" 2>&1 | tee "${LOG_DIR}/${step_name}.log"
    
    exit_code=$?
    end_time=$(date +%s.%N)
    
    elapsed=$(echo "$end_time - $start_time" | bc)
    timestamp=$(date +"%Y-%m-%d %H:%M:%S")
    
    # Save to timing CSV
    echo "$step_name,$elapsed,$timestamp,$exit_code" >> "$TIMING_CSV"
    
    echo "Completed: $step_name in $elapsed seconds (Exit code: $exit_code)"
    echo "----------------------------------------"
    
    return $exit_code
}

# Function to run evaluation with correct nnUNet syntax
run_evaluation() {
    local model_name=$1
    local predictions_dir=$2
    
    echo "================================================"
    echo "EVALUATING: $model_name"
    echo "================================================"
    
    eval_output_dir="${predictions_dir}_evaluation"
    mkdir -p "$eval_output_dir"
    
    # Create .json output file path
    json_output="${eval_output_dir}/summary.json"
    
    # Run nnUNet evaluation with correct syntax
    # -o should be a .json file, not a directory
    nnUNetv2_evaluate_folder \
        -djfile "$DATASET_JSON" \
        -pfile "$PLANS_JSON" \
        "$LABELS_FOLDER" \
        "$predictions_dir" \
        -o "$json_output" \
        -np 8
    
    echo "Evaluation completed for $model_name"
    echo "Results saved in: $json_output"
    echo "================================================"
}

# Function to run inference with detailed timing
run_inference_with_detailed_timing() {
    local step_name=$1
    local model_type=$2
    local output_dir=$3
    
    echo "Starting: $step_name"
    
    start_time=$(date +%s.%N)
    
    # Run with timer flag
    output=$(nnUNetv2_predict -d Dataset109_ProstrateCTV \
        -i "$INPUT_FOLDER" \
        -o "$output_dir" \
        -f 0 1 2 3 4 \
        -tr nnUNetTrainer \
        -c "$model_type" \
        -p nnUNetPlans \
        --save_probabilities \
        --timer 2>&1)
    
    exit_code=$?
    end_time=$(date +%s.%N)
    
    # Save full output
    echo "$output" > "${LOG_DIR}/${step_name}.log"
    
    # Extract timing information
    total_time=$(echo "$output" | grep "Total time" | awk '{print $4}')
    time_per_case=$(echo "$output" | grep "Time per case" | awk '{print $5}')
    preprocessing_time=$(echo "$output" | grep "Time for preprocessing" | awk '{print $5}')
    prediction_time=$(echo "$output" | grep "Time for prediction" | awk '{print $5}')
    postprocessing_time=$(echo "$output" | grep "Time for postprocessing" | awk '{print $5}')
    
    elapsed=$(echo "$end_time - $start_time" | bc)
    timestamp=$(date +"%Y-%m-%d %H:%M:%S")
    
    # Save to CSV - both total and per-case
    echo "$step_name,$elapsed,$total_time,$time_per_case,$preprocessing_time,$prediction_time,$postprocessing_time,$timestamp,$exit_code" >> "$TIMING_CSV"
    
    echo "Completed: $step_name"
    echo "Total time: $total_time seconds"
    echo "Time per case: $time_per_case seconds"
    echo "----------------------------------------"
    
    return $exit_code
}

# Step 1: 2D Inference
run_step "2d_inference" \
    nnUNetv2_predict -d Dataset109_ProstrateCTV \
    -i "$INPUT_FOLDER" \
    -o "$OUTPUT_2D" \
    -f 0 1 2 3 4 \
    -tr nnUNetTrainer \
    -c 2d \
    -p nnUNetPlans \
    --save_probabilities    

# Evaluate 2D model with default nnUNet output
run_evaluation "2D Model" "$OUTPUT_2D"

# Step 2: 3D FullRes Inference  
run_step "3d_fullres_inference" \
    nnUNetv2_predict -d Dataset109_ProstrateCTV \
    -i "$INPUT_FOLDER" \
    -o "$OUTPUT_3D" \
    -f 0 1 2 3 4 \
    -tr nnUNetTrainer \
    -c 3d_fullres \
    -p nnUNetPlans \
    --save_probabilities

# Evaluate 3D model with default nnUNet output
run_evaluation "3D FullRes Model" "$OUTPUT_3D"

# Step 3: Ensemble
run_step "ensemble" \
    nnUNetv2_ensemble \
    -i "$OUTPUT_2D" "$OUTPUT_3D" \
    -o "$ENSEMBLE_OUTPUT" \
    -np 8

# Evaluate ensemble with default nnUNet output
run_evaluation "Ensemble" "$ENSEMBLE_OUTPUT"

echo "================================================"
echo "PIPELINE COMPLETED"
echo "================================================"
echo "2D Output: $OUTPUT_2D"
echo "3D Output: $OUTPUT_3D"
echo "Ensemble Output: $ENSEMBLE_OUTPUT"
echo "Timing CSV: $TIMING_CSV"
echo "Logs: $LOG_DIR/"
echo ""
echo "EVALUATION RESULTS ARE IN:"
echo "  - ${OUTPUT_2D}_evaluation/summary.json"
echo "  - ${OUTPUT_3D}_evaluation/summary.json"
echo "  - ${ENSEMBLE_OUTPUT}_evaluation/summary.json"
echo "================================================"

# Show final timing only
echo "FINAL TIMING RESULTS:"
echo "======================"
cat "$TIMING_CSV"

# Show how to view results
echo ""
echo "TO VIEW RESULTS:"
echo "================="
echo "jq . ${OUTPUT_2D}_evaluation/summary.json"
echo "jq . ${OUTPUT_3D}_evaluation/summary.json"
echo "jq . ${ENSEMBLE_OUTPUT}_evaluation/summary.json"
echo ""
echo "# If jq is not installed:"
echo "python -m json.tool ${OUTPUT_2D}_evaluation/summary.json"

                 