#!/bin/bash
#######################################################################################################################
#
# Run demo-training-prepare.sh with the same MODEL_TYPE & N_LAYER & N_EMBD first
# Or, rename your base model to rwkv-init.pth and put it in the output folder
#
# The trainer will load the last rwkv-*.pth in the folder, such that it can continue from a stopped run
# Therefore check the log (### Loading rwkv-xxx.pth... ###), and make sure you don't have extra rwkv-*.pth there
#
#######################################################################################################################
#
MODEL_TYPE="x070" #
#
N_LAYER="12"
N_EMBD="768"
N_MOBA_LAYER="4"
MOBA_CHUNK_SIZE="2048"
MOBA_TOPK="3"
#
CTX_LEN="512" # !!! change magic_prime if you change ctx_len !!!
WANDB="rwkv-moba-hybrid-prolong64k"
export WANDB_MODE=offline
RUN_NAME="minipile_L"$N_LAYER"-D"$N_EMBD"-C"$CTX_LEN"_ML"$N_MOBA_LAYER"_CS"$MOBA_CHUNK_SIZE"_TK"$MOBA_TOPK
PROJ_DIR="out/$RUN_NAME"
#
MODEL_FILE="/root/autodl-tmp/RWKV-x070-World-0.1B-v2.8-20241210-ctx4096.pth"
DATA_FILE="data/minipile"
EXIT_TOKENS="1498226207"
MAGIC_PRIME="2926181"
#######################################################################################################################
#
# Note bsz & lr affects model & training performance
# Small data => use smaller bsz & slightly smaller LR
# Large data => use larger bsz & slightly larger LR
# Larger model => use smaller LR
# Finetuning => use very small LR, such as 1e-5
#
M_BSZ="16" # takes ~9G VRAM here => reduce this to save VRAM, increase this for faster speed
LR_INIT="1e-5"
LR_FINAL="1e-5"
GRAD_CP=0 # 1 => not working when freezing rwkv
EPOCH_SAVE=50 # save every 10 "miniepochs" (1 miniepoch = 40320 * ctx_len tokens) => decrease if your GPU is weak
#
#######################################################################################################################
#
# magic_prime = the largest 3n+2 prime smaller than datalen/ctxlen-1 (= 1498226207/512-1 = 2926222.06 in this case) = 2926181 in this case
# use https://www.dcode.fr/prime-numbers-search
#
N_NODE=1 # number of nodes
GPU_PER_NODE=1 # number of GPUs per node
#
DS_BUCKET_MB=2 # set to 2 for consumer GPUs, set to 200 for A100 / H100 (affects speed & vram usage)
#
python train.py --load_pretrain $MODEL_FILE --wandb $WANDB --run_name $RUN_NAME --proj_dir $PROJ_DIR \
 --ctx_len $CTX_LEN --my_pile_stage 3 --epoch_count 999999 --epoch_begin 0 \
 --data_file $DATA_FILE --my_exit_tokens $EXIT_TOKENS --magic_prime $MAGIC_PRIME \
 --num_nodes $N_NODE --micro_bsz $M_BSZ --n_layer $N_LAYER --n_embd $N_EMBD --pre_ffn 0 --head_qk 0 \
 --lr_init $LR_INIT --lr_final $LR_FINAL --warmup_steps 10 --beta1 0.9 --beta2 0.99 --adam_eps 1e-18 \
 --my_pile_edecay 0 --data_type "binidx" --vocab_size 65536 --my_testing $MODEL_TYPE \
 --weight_decay 0.001 --epoch_save $EPOCH_SAVE --head_size_a 64 \
 --accelerator gpu --devices $GPU_PER_NODE --precision bf16 --strategy deepspeed_stage_1 --grad_cp $GRAD_CP \
 --enable_progress_bar True --ds_bucket_mb $DS_BUCKET_MB \
 --n_moba_layer $N_MOBA_LAYER --moba_chunk_size $MOBA_CHUNK_SIZE --moba_topk $MOBA_TOPK --only_train_moba