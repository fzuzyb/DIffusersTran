# 实验1
#accelerate launch --config_file accelerate_config.yaml train_flux_instantid.py \
#  --pretrained_model_name_or_path="/home/iv/Algo_new/Zhouyuanbo/IV_WORKING/Model/Flux" \
#  --output_dir="/home/iv/Data/Zhouyuanbo/AI写真数据集/AI写真训练结果/exp1_flux_instantid" \
#  --image_root="/home/iv/Data/Zhouyuanbo/AI写真数据集/Train/v2_face_embedding_half_body" \
#  --resume_from_checkpoint="/home/iv/Data/Zhouyuanbo/AI写真数据集/AI写真训练结果/exp1_flux_instantid/checkpoint-8000" \
#  --mixed_precision="bf16" \
#  --resolution=512 \
#  --train_batch_size=4 \
#  --dataloader_num_workers=8 \
#  --guidance_scale=3.5 \
#  --rank=64 \
#  --gradient_accumulation_steps=4 \
#  --gradient_checkpointing \
#  --use_8bit_adam \
#  --learning_rate=1e-4 \
#  --report_to="tensorboard" \
#  --lr_scheduler="constant" \
#  --lr_warmup_steps=0   \
#  --max_train_steps=100000 \
#  --validation_steps=500 \
#  --max_val_samples=10 \
#  --checkpointing_steps=500 \
#  --checkpoints_total_limit=30 \
#  --proportion_empty_prompts=0.5 \
#  --seed="0"
#accelerate launch --config_file accelerate_config.yaml train_flux_instantid_addfrontface.py \
#  --pretrained_model_name_or_path="/run/user/0/Flux" \
#  --output_dir="/run/user/0/experiments" \
#  --image_root="/run/user/0/v2_face_embedding_half_body" \
#  --resume_from_checkpoint="/run/user/0/experiments/full/checkpoint-5000" \
#  --mixed_precision="bf16" \
#  --resolution=512 \
#  --train_batch_size=4 \
#  --dataloader_num_workers=32 \
#  --guidance_scale=3.5 \
#  --rank=64 \
#  --gradient_accumulation_steps=4 \
#  --gradient_checkpointing \
#  --learning_rate=1e-4 \
#  --report_to="tensorboard" \
#  --lr_scheduler="constant" \
#  --lr_warmup_steps=0   \
#  --max_train_steps=1000000 \
#  --validation_steps=500 \
#  --max_val_samples=10 \
#  --checkpointing_steps=500 \
#  --checkpoints_total_limit=20 \
#  --proportion_empty_prompts=0.5 \
#  --seed="0"


#accelerate launch --config_file accelerate_config.yaml train_flux_instantid_addfrontface.py \
#  --pretrained_model_name_or_path="/run/user/0/Flux" \
#  --output_dir="/run/user/0/experiments" \
#  --image_root="/run/user/0/v2_face_embedding_half_body" \
#  --resume_from_checkpoint="/run/user/0/experiments/full/checkpoint-5000" \
#  --mixed_precision="bf16" \
#  --resolution=512 \
#  --train_batch_size=4 \
#  --dataloader_num_workers=32 \
#  --guidance_scale=3.5 \
#  --rank=64 \
#  --gradient_accumulation_steps=4 \
#  --gradient_checkpointing \
#  --learning_rate=1e-4 \
#  --report_to="tensorboard" \
#  --lr_scheduler="constant" \
#  --lr_warmup_steps=0   \
#  --max_train_steps=1000000 \
#  --validation_steps=500 \
#  --max_val_samples=10 \
#  --checkpointing_steps=500 \
#  --checkpoints_total_limit=20 \
#  --proportion_empty_prompts=0.5 \
#  --seed="0"

export IV_FACE_ROOT="/home/iv/Algo_new/DengWei/Project/StableDiffusion/ComfyUI/models/ivface"
export IMAGE2TEXT_MODEL="/home/iv/Algo_new/Zhouyuanbo/IV_WORKING/Model/gitbase"
accelerate launch --config_file default_config.yaml train_flux_instantid_webdataset.py \
  --pretrained_model_name_or_path="/home/iv/Algo_new/Zhouyuanbo/IV_WORKING/Model/Flux" \
  --resume_from_checkpoint="/home/iv/Data/Zhouyuanbo/AI写真数据集/AI写真训练结果/exp1_flux_instantid/checkpoint-63000" \
  --output_dir="/home/iv/Data/Zhouyuanbo/AI写真数据集/AI写真训练结果/exp2_flux_instantid_webdataset" \
  --train_image_root="/home/iv/images/new_aws_tar_v2" \
  --val_image_root="/home/iv/Data/Zhouyuanbo/AI写真数据集/Train/v2_face_embedding_half_body" \
  --mixed_precision="bf16" \
  --resolution=512 \
  --train_batch_size=4 \
  --dataloader_num_workers=1 \
  --guidance_scale=3.5 \
  --use_8bit_adam \
  --rank=64 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --learning_rate=1e-4 \
  --report_to="tensorboard" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0   \
  --max_train_steps=100000 \
  --validation_steps=500 \
  --max_val_samples=10 \
  --checkpointing_steps=500 \
  --checkpoints_total_limit=500 \
  --proportion_empty_prompts=0.5 \
  --seed="0"


#export CUDA_LAUNCH_BLOCKING=1
#export IV_FACE_ROOT="/home/iv/Algo_new/DengWei/Project/StableDiffusion/ComfyUI/models/ivface"
#export IMAGE2TEXT_MODEL="/home/iv/Algo_new/Zhouyuanbo/IV_WORKING/Model/gitbase"
#accelerate launch --config_file default_config.yaml train_flux_instantid_addfrontface.py \
#  --pretrained_model_name_or_path="/home/iv/Algo_new/Zhouyuanbo/IV_WORKING/Model/Flux" \
#  --output_dir="/home/iv/Data/Zhouyuanbo/AI写真数据集/AI写真训练结果/exp2_flux_instantid_webdataset" \
#  --image_root="/home/iv/images/new_aws_tar_v2" \
#  --mixed_precision="bf16" \
#  --resolution=512 \
#  --train_batch_size=4 \
#  --dataloader_num_workers=1 \
#  --guidance_scale=3.5 \
#  --use_8bit_adam \
#  --rank=64 \
#  --gradient_accumulation_steps=4 \
#  --gradient_checkpointing \
#  --learning_rate=1e-4 \
#  --report_to="tensorboard" \
#  --lr_scheduler="constant" \
#  --lr_warmup_steps=0   \
#  --max_train_steps=100000 \
#  --validation_steps=500 \
#  --max_val_samples=10 \
#  --checkpointing_steps=500 \
#  --checkpoints_total_limit=500 \
#  --proportion_empty_prompts=0.5 \
#  --seed="0"
#
#

