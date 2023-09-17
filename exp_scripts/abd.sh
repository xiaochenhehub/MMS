SCRIPT=train.py
GPUID1=0
NUM_WORKER=8
MODEL='efficient_b2_unet'
CPT='abdominal_gin_ipa_example'

# visualizations
PRINT_FREQ=50000
VAL_FREQ=50000
TEST_EPOCH=50
EXP_TYPE='ginipa'

BSIZE=20
NL_GIN=4
N_INTERM=2

LAMBDA_WCE=1.0 # actually we are not using weights so it is in effect plain ce
LAMBDA_DICE=1.0
LAMBDA_CONSIST=10.0 # Xu et al.

SAVE_PRED=False # save prediction results or not

DATASET='ABDOMINAL'
CHECKPOINTS_DIR="./my_exps/${DATASET}"
NITER=50
NITER_DECAY=950
# rec/seg: 50,1950/950,3e-4
# inter: 10,490,3e-5
IMG_SIZE=192
NCLASS=5

OPTM_TYPE='exp'
LR=3e-4  # inter_train: 3e-5
ADAM_L2=0.00003
TE_DOMAIN="CHAOST2"

# blender config
BLEND_GRID_SIZE=24 # 24 * 2 = 48, 1/4 of 192

# validation fold
ALL_TRS=( "SABSCT")

# KL term
CONSIST_TYPE='kld'

# save
SAVE_EPOCH=500
SAVE_DIR="/root/autodl-tmp/output/abdominal/SABSCT/IDsAE"

# reload
CONTINUE_TRAIN=False
RELOAD_MODEL_DIR="/root/autodl-tmp/output/abdominal/SABSCT/IDsAE/ENC/net_1000.pth"

TRAIN_STAGE=3
ALPHA=2
REFER_SHIFT=2
SHUFFLE=True

LR_POLICY="exp"

TRAIN_TYPE="rec"

for TR_DOMAIN in "${ALL_TRS[@]}"
do
    set -ex
    export CUDA_VISIBLE_DEVICES=$GPUID1

    NAME=${CPT}_${TR_DOMAIN}_${MODEL}
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
        consist_type=$CONSIST_TYPE\
        display_freq=$PRINT_FREQ\
        gin_nlayer=$NL_GIN\
        gin_n_interm_ch=$N_INTERM\
        save_dir=$SAVE_DIR\
        continue_train=$CONTINUE_TRAIN\
        reload_model_dir=$RELOAD_MODEL_DIR\
        reload_optim_dir=$RELOAD_OPTIM_DIR\
        train_stage=$TRAIN_STAGE\
        alpha=$ALPHA\
        refer_shift=$REFER_SHIFT\
        shuffle=$SHUFFLE\
        lr_policy=$LR_POLICY\
        train_type=$TRAIN_TYPE
done
