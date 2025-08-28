#!/bin/bash

# Configuration
INPUT_FOLDER="/space/fast/oim/nnUNet_raw/Dataset109_ProstrateCTV/imagesTs"
OUTPUT_BASE="/space/fast/oim/nnUNet_results/Dataset109_ProstrateCTV/per_case_timing"
PER_CASE_TIMING_CSV="${OUTPUT_BASE}/per_case_timing.csv"
LOG_DIR="${OUTPUT_BASE}/logs"

# Create fresh output directories
mkdir -p "$OUTPUT_BASE" "$LOG_DIR"

# Initialize per-case timing CSV
echo "case_id,2d_time_seconds,3d_time_seconds,ensemble_time_seconds,total_time_seconds" > "$PER_CASE_TIMING_CSV"

echo "================================================"
echo "PER-CASE INFERENCE TIMING"
echo "================================================"
echo "Input: $INPUT_FOLDER"
echo "Output Base: $OUTPUT_BASE"
echo "Per-Case Timing CSV: $PER_CASE_TIMING_CSV"
echo "================================================"

# Use the working function to get case files
get_case_files() {
    local input_dir=$1
    # Count files that look like case files (not labels)
    ls "$input_dir"/*.nii.gz 2>/dev/null | grep -v "_seg" | grep -v "_label" | head -1
}

get_num_cases() {
    local input_dir=$1
    # Count files that look like case files (not labels)
    local num_cases=$(ls "$input_dir"/*.nii.gz 2>/dev/null | grep -v "_seg" | grep -v "_label" | wc -l)
    echo $num_cases
}

# Get number of cases and list of files
num_cases=$(get_num_cases "$INPUT_FOLDER")
echo "Found $num_cases cases to process"

# Get the first file to understand the naming pattern
first_file=$(get_case_files "$INPUT_FOLDER")
echo "First file: $(basename $first_file)"

# Process each case individually
for case_file in "$INPUT_FOLDER"/*.nii.gz; do
    # Skip segmentation/label files
    if echo "$case_file" | grep -q "_seg\|_label"; then
        continue
    fi
    
    case_id=$(basename "$case_file" .nii.gz)
    echo "Processing: $case_id"
    
    # Create temporary directory for this case
    TEMP_DIR="${OUTPUT_BASE}/temp_${case_id}"
    mkdir -p "$TEMP_DIR"
    
    # Copy only this case to temporary directory
    cp "$case_file" "$TEMP_DIR/"
    
    # Initialize timing variables
    time_2d=0
    time_3d=0
    time_ensemble=0
    
    # Time 2D inference for this single case
    echo "  Running 2D inference..."
    start_time=$(date +%s.%N)
    nnUNetv2_predict -d Dataset109_ProstrateCTV \
        -i "$TEMP_DIR" \
        -o "${OUTPUT_BASE}/2D_temp" \
        -f 0 \
        -tr nnUNetTrainer \
        -c 2d \
        -p nnUNetPlans \
        --save_probabilities > "${LOG_DIR}/${case_id}_2d.log" 2>&1
    time_2d=$(echo "$(date +%s.%N) - $start_time" | bc)
    
    # Check if 2D inference succeeded
    if [ $? -ne 0 ]; then
        echo "  2D inference failed for $case_id"
        time_2d=0
    fi
    
    # Time 3D inference for this single case
    echo "  Running 3D inference..."
    start_time=$(date +%s.%N)
    nnUNetv2_predict -d Dataset109_ProstrateCTV \
        -i "$TEMP_DIR" \
        -o "${OUTPUT_BASE}/3D_temp" \
        -f 0 \
        -tr nnUNetTrainer \
        -c 3d_fullres \
        -p nnUNetPlans \
        --save_probabilities > "${LOG_DIR}/${case_id}_3d.log" 2>&1
    time_3d=$(echo "$(date +%s.%N) - $start_time" | bc)
    
    # Check if 3D inference succeeded
    if [ $? -ne 0 ]; then
        echo "  3D inference failed for $case_id"
        time_3d=0
    fi
    
    # Time ensemble for this single case (only if both inferences succeeded)
    if [ -d "${OUTPUT_BASE}/2D_temp" ] && [ -d "${OUTPUT_BASE}/3D_temp" ] && [ $time_2d != 0 ] && [ $time_3d != 0 ]; then
        echo "  Running ensemble..."
        start_time=$(date +%s.%N)
        nnUNetv2_ensemble \
            -i "${OUTPUT_BASE}/2D_temp" "${OUTPUT_BASE}/3D_temp" \
            -o "${OUTPUT_BASE}/ensemble_temp" \
            -np 1 > "${LOG_DIR}/${case_id}_ensemble.log" 2>&1
        time_ensemble=$(echo "$(date +%s.%N) - $start_time" | bc)
        
        # Check if ensemble succeeded
        if [ $? -ne 0 ]; then
            echo "  Ensemble failed for $case_id"
            time_ensemble=0
        fi
    else
        echo "  Skipping ensemble - inference failed"
        time_ensemble=0
    fi
    
    # Calculate total time
    total_time=$(echo "$time_2d + $time_3d + $time_ensemble" | bc)
    
    # Save to CSV
    echo "$case_id,$time_2d,$time_3d,$time_ensemble,$total_time" >> "$PER_CASE_TIMING_CSV"
    
    echo "  Completed: 2D=${time_2d}s, 3D=${time_3d}s, Ensemble=${time_ensemble}s, Total=${total_time}s"
    
    # Clean up temporary directories
    rm -rf "$TEMP_DIR" "${OUTPUT_BASE}/2D_temp" "${OUTPUT_BASE}/3D_temp" "${OUTPUT_BASE}/ensemble_temp" 2>/dev/null
    
    echo "----------------------------------------"
done

echo "================================================"
echo "PER-CASE TIMING COMPLETED"
echo "================================================"
echo "CSV Output: $PER_CASE_TIMING_CSV"
echo "Logs: $LOG_DIR/"
echo "================================================"

# Show summary
echo "TIMING SUMMARY:"
echo "==============="
awk -F',' 'NR>1 {sum_2d+=$2; sum_3d+=$3; sum_ens+=$4; sum_total+=$5; count++} 
END {
    if (count > 0) {
        printf "Average 2D time: %.2f seconds\n", sum_2d/count;
        printf "Average 3D time: %.2f seconds\n", sum_3d/count;
        printf "Average ensemble time: %.2f seconds\n", sum_ens/count;
        printf "Average total time: %.2f seconds\n", sum_total/count;
        printf "Total cases processed: %d\n", count;
    } else {
        printf "No cases processed successfully\n";
    }
}' "$PER_CASE_TIMING_CSV"

# Show first few lines of CSV
echo ""
echo "FIRST 10 CASES:"
echo "==============="
head -11 "$PER_CASE_TIMING_CSV" | column -t -s','