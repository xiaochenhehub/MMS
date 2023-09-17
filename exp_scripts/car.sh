# GIN and IPA for prostate images
SCRIPT=train.py
GPUID1=0
NUM_WORKER=8
MODEL='efficient_b2_unet'
CPT='prostate_ginipa_example'

# visualization
PRINT_FREQ=50000
VAL_FREQ=50000
TEST_EPOCH=50
EXP_TYPE='ginipa'

BSIZE=20
NL_GIN=4
N_INTERM=2

LAMBDA_WCE=1.0 # not using weights, actually standard multi-class ce
LAMBDA_DICE=1.0
LAMBDA_CONSIST=10.0 # Xu et al.

SAVE_PRED=False # save predictions or not

DATASET='CARDIAC'
CHECKPOINTS_DIR="./my_exps/$DATASET"
NITER=50
NITER_DECAY=950
IMG_SIZE=192

OPTM_TYPE='adam'
LR=3e-4
ADAM_L2=0.00003
TE_DOMAIN="NA" # will be override by exclu_domain

# blender config
BLEND_GRID_SIZE=24

ALL_TRS="bSSFP"
NCLASS=4

# KL term
CONSIST_TYPE='kld'

# save
SAVE_EPOCH=500
SAVE_DIR="/root/autodl-tmp/output/CAR/IDsAE"

# train type
SHUFFLE=True
TRAIN_TYPE="seg"
ALPHA=2
REFER_SHIFT=2

# continue train
CONTINUE_TRAIN=True
RELOAD_MODEL_DIR="/root/autodl-tmp/output/CAR/IAE/bSSFP/ENC/net_1000.pth"

# fiter_all_0
FILTER_ALL_0=False

for TR_DOMAIN in "${ALL_TRS[@]}"
do
    set -ex
    export CUDA_VISIBLE_DEVICES=$GPUID1

    NAME=${CPT}_tr${TR_DOMAIN}_exclude${TR_DOMAIN}_${MODEL}
    LOAD_DIR=$NAME

    python3 $SCRIPT with exp_type=$EXP_TYPE\
        name=$NAME\
        model=$MODEL\
        nThreads=$NUM_WORKER\
        print_freq=$PRINT_FREQ\
        validation_freq=$VAL_FREQ\
        batchsize=$BSIZE\
        lambda_wce=$LAMBDA_WCE\
        lambda_dice=$LAMBDA_DICE\
        save_epoch_freq=$SAVE_EPOCH\
        load_dir=$LOAD_DIR\
        infer_epoch_freq=$TEST_EPOCH\
        niter=$NITER\
        niter_decay=$NITER_DECAY\
        fineSize=$IMG_SIZE\
        lr=$LR\
        adam_weight_decay=$ADAM_L2\
        data_name=$DATASET\
        nclass=$NCLASS\
        tr_domain=$TR_DOMAIN\
        te_domain=$TE_DOMAIN\
        optimizer=$OPTM_TYPE\
        save_prediction=$SAVE_PRED\
        lambda_consist=$LAMBDA_CONSIST\
        blend_grid_size=$BLEND_GRID_SIZE\
        exclu_domain=$TR_DOMAIN\
        consist_type=$CONSIST_TYPE\
        display_freq=$PRINT_FREQ\
        gin_nlayer=$NL_GIN\
        gin_n_interm_ch=$N_INTERM\
        save_dir=$SAVE_DIR\
        shuffle=$SHUFFLE\
        train_type=$TRAIN_TYPE\
        continue_train=$CONTINUE_TRAIN\
        reload_model_dir=$RELOAD_MODEL_DIR\
        alpha=$ALPHA\
        refer_shift=$REFER_SHIFT\
        filter_all_0=$FILTER_ALL_0

done
