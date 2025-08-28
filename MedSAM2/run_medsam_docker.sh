# docker build -t medsam2-gpu-docker:latest .
# docker run --name medsam2_lund_probe_train_109  --gpus all --shm-size=8G -it --rm -v /home/oim/DLPST/MedSAM2/sam2/configs:/app/MedSAM2/sam2/configs   -v /home/oim/DLPST/MedSAM2/checkpoints:/app/MedSAM2/checkpoints -v /home/oim/DLPST/MedSAM2:/app/MedSAM2 -v /space/fast/oim/medsam_raw/Dataset109_ProstrateCTV:/app/MedSAM2/data -v /space/fast/oim/tmp:/tmp -e TMPDIR=/tmp -e PYTHONPATH=/app/MedSAM2  medsam2-gpu-docker:latest
# Run the container in detached mode (-d) and execute the script

docker run -it --gpus all \
    -v /space/fast/oim/medsam_raw/Dataset109_ProstrateCTV/conversions2:/app/MedSAM2/conversions2 \
    -v /space/fast/oim/nnUNet_raw/Dataset109_ProstrateCTV/imagesTs:/app/MedSAM2/imagesTs \
    -v /space/fast/oim/nnUNet_raw/Dataset109_ProstrateCTV/labelsTs:/app/MedSAM2/labelsTs \
    -v /home/oim/DLPST/MedSAM2/checkpoints:/app/MedSAM2/checkpoints \
    -v /space/fast/oim/medsam_raw/Dataset109_ProstrateCTV/conversions2/seg_results:/app/output \
    -v /home/oim/DLPST/MedSAM2:/app/MedSAM2 \
    -e PYTHONPATH=/app/MedSAM2/sam2/config \
    medsam2-gpu-docker:latest  
    
    
    # python /app/MedSAM2/pred_eval_npz.py \
    # --imgs_path /app/MedSAM2/conversions2/test \
    # --gts_path /app/MedSAM2/labelsTs \
    # --original_imgs_path /app/MedSAM2/imagesTs \
    # --pred_save_dir /app/output \
    # --checkpoint app/MedSAM2/checkpoints/checkpoint.pt \
    # --cfg /app/MedSAM2/sam2/configs/sam2.1_hiera_tiny512_FLARE_RECIST.yaml


# sh single_node_train_medsam2.sh
# Optional: Immediately check if the container started
# echo "Attempting to start container..."
# docker ps -q -f name=medsam2_lund_probe_train_109 > /dev/null && echo "Container started successfully in detached mode." || echo "Failed to start container."

# # Optional: To see the logs as it runs (like 'tail -f')
# # docker logs -f medsam2_lund_probe_train_109

# # Optional: Check if the container started successfully
# echo "Container started with ID: $(docker ps -q -f name=medsam2_lund_probe_train_109)"

# docker logs -f medsam2_lund_probe_train_109

#sh single_node_train_medsam2.sh
# docker build -t monai_docker:latest .

# docker run --name swinUNETr_lund_probe_train_109 -v /space/fast/oim/nnUNet_raw:/app/data -v /space/fast/oim/tmp:/tmp -e TMPDIR=/tmp  --gpus all -d --rm --shm-size=8G  -it  monai_docker:latest  sh single_node_train_medsam2.sh

