# #!/bin/bash

# # Set dataset path and other common parameters
# DATASET_PATH="/home/eddy/Desktop/cropped_image"
# WORKERS=20
# # SAVE_DIR="swin_center_lr_0.5_loss_3e-4_smmothing_0.1"
# SMOOTHING=0.1
# # MODEL_WEIGHT_PATH='/home/eddy/Desktop/MasterThesis/mainProgram/Veri776_datasets_train/swin_center_lr_0.5_loss_3e-4_smmothing_0.1/swin_centerloss_best.pth'
# # Define the list of backbones to iterate over
# # BACKBONES=("resnet" "resnext" "seresnet" "densenet" "resnet34" "swin")
# # BACKBONES=swin
# # python train.py --dataset "$DATASET_PATH" --workers $WORKERS --save_dir "$SAVE_DIR" --backbone "$BACKBONE" --smoothing $SMOOTHING --custom_weights "$MODEL_WEIGHT_PATH" --center_loss --smoothing $SMOOTHING

# SAVE_DIR="swin_loss_3e-4_smmothing_0.1"
# MODEL_WEIGHT_PATH='/home/eddy/Desktop/MasterThesis/mainProgram/Veri776_datasets_train/swin_loss_3e-4_smmothing_0.1/swin_best.pth'
# # python train.py --dataset "$DATASET_PATH" --workers $WORKERS --save_dir "$SAVE_DIR" --backbone "$BACKBONE" --smoothing $SMOOTHING --custom_weights "$MODEL_WEIGHT_PATH" --smoothing $SMOOTHING

# BACKBONES=("swin")
# # Loop over each backbone
# # python train.cnn.py --dataset "$DATASET_PATH" --workers $WORKERS --save_dir "$SAVE_DIR" --backbone "resnext" --center_loss
# for BACKBONE in "${BACKBONES[@]}"; do
#     echo "Running training with backbone: $BACKBONE (without center loss)"
    
#     # Run the training without center loss
#     python train.py --dataset "$DATASET_PATH" --workers $WORKERS --save_dir "$SAVE_DIR" --backbone "$BACKBONE" --smoothing $SMOOTHING --custom_weights "$MODEL_WEIGHT_PATH" 
    
#     # echo "Running training with backbone: $BACKBONE (with center loss)"
    
#     # # Run the training with center loss
#     # python train.py --dataset "$DATASET_PATH" --workers $WORKERS --save_dir "$SAVE_DIR" --backbone "$BACKBONE" --center_loss --smoothing $SMOOTHING
# done


# 
#!/bin/bash

# Set dataset path and other common parameters
# DATASET_PATH="../Veri776_datasets"
DATASET_PATH="/home/eddy/Desktop/cropped_image"
WORKERS=20
SMOOTHING=0.1
LR=1e-4

# Define the list of backbones to iterate over
BACKBONES=("resnet_a" "resnet_b")

# Declare an associative array to map backbones to their respective save directories
declare -A SAVE_DIRS=(
    ["resnet_a"]="resnet_a_smoothing_0.1"
    ["resnet_a_center"]="resnet_a_center_lr_0.5_loss_3e-4_smoothing_0.1"
    ["resnet_b"]="resnet_b_smoothing_0.1"
    ["resnet_b_center"]="resnet_b_center_lr_0.5_loss_3e-4_smoothing_0.1"
)
declare -A MODEL_WEIGHT_PATHS=(
    ["resnet_a"]="/home/eddy/Desktop/MasterThesis/mainProgram/Veri776_datasets_train/resnet_a_smoothing_0.1/resnet_a_best.pth"
    ["resnet_a_center"]="/home/eddy/Desktop/MasterThesis/mainProgram/Veri776_datasets_train/resnet_a_center_lr_0.5_loss_3e-4_smoothing_0.1/resnet_a_centerloss_best.pth"
    ["resnet_b"]="/home/eddy/Desktop/MasterThesis/mainProgram/Veri776_datasets_train/resnet_b_smoothing_0.1/resnet_b_best.pth"
    ["resnet_b_center"]="/home/eddy/Desktop/MasterThesis/mainProgram/Veri776_datasets_train/resnet_b_center_lr_0.5_loss_3e-4_smoothing_0.1/resnet_b_centerloss_best.pth"
)

# Loop over each backbone
for BACKBONE in "${BACKBONES[@]}"; do
    echo "Running training with backbone: $BACKBONE (without center loss)"
    
    # Set the save directory for the current backbone without center loss
    SAVE_DIR="${SAVE_DIRS[$BACKBONE]}"
    MODEL_WEIGHT_PATH="${MODEL_WEIGHT_PATHS[$BACKBONE]}"
    # Run the training without center loss
    python train.py --dataset "$DATASET_PATH" --workers $WORKERS --save_dir "$SAVE_DIR" --backbone "$BACKBONE" --smoothing $SMOOTHING --custom_weights "$MODEL_WEIGHT_PATH"
    
    echo "Running training with backbone: $BACKBONE (with center loss)"
    
    # Set the save directory for the current backbone with center loss
    SAVE_DIR="${SAVE_DIRS[${BACKBONE}_center]}"
    MODEL_WEIGHT_PATH="${MODEL_WEIGHT_PATHS[${BACKBONE}_center]}"
    # Run the training with center loss
    python train.py --dataset "$DATASET_PATH" --workers $WORKERS --save_dir "$SAVE_DIR" --backbone "$BACKBONE" --center_loss --smoothing $SMOOTHING --custom_weights "$MODEL_WEIGHT_PATH"
done
