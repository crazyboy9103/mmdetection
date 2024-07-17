data=$1
model=$2

# Check if data is provided
if [ -z "$data" ]; then
    echo "Need data name"
    exit 1
fi

# Define models and their configuration prefixes
declare -A models_config=(
    [deformable_detr]="deformable_detr_r50_1xb2-15e_"
    [efficientdet]="efficientdet_effb3_bifpn_1xb16-crop512-15e_"
    [nas_fcos]="nas-fcos_r50-caffe_fpn_fcoshead-gn-head_1xb4-1x_"
    [retinanet]="retinanet_r50_fpn_1x_"
)

# Function to train a model
train_model() {
    local model_name=$1
    local cfg_prefix=$2
    python tools/train.py configs/neurocle/$model_name/$cfg_prefix$data.py --work-dir "./work_dirs/$data/$model_name"
}

# Check if specific model is given and is in the list of models
if [[ -n "$model" && -n "${models_config[$model]}" ]]; then
    train_model "$model" "${models_config[$model]}"
else
    # If no specific model or invalid model is given, iterate over all models
    for model in "${!models_config[@]}"; do
        train_model "$model" "${models_config[$model]}"
    done
fi
