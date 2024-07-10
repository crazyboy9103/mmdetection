data=$1

if [ -z "$data" ]; then
    echo "need data name"
    exit 1
fi

models=("detr" "efficientdet" "nas_fcos" "retinanet")
cfg_prefixs=("detr_r50_1xb2-15e_" "efficientdet_effb0_bifpn_1xb16-crop512-15e_" "nas-fcos_r50-caffe_fpn_fcoshead-gn-head_1xb4-1x_" "retinanet_r50_fpn_1x_")

for i in "${!models[@]}"; do
  model=${models[$i]}
  cfg_prefix=${cfg_prefixs[$i]}
  python tools/train.py configs/neurocle/$model/$cfg_prefix$data.py --work-dir "./work_dirs/$data/$model"
done