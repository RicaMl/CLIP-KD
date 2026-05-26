#!/bin/bash
cd src


export USE_AUG=1

METHODS=("CrossKD" "ICL" "GD" "FD" "CKD" "AFD")
LR=1e-5
WD=0.2

for METHOD in "${METHODS[@]}"; do
    echo "========================================="
    echo "Running Distillation Method: $METHOD"
    echo "========================================="
    
    CKD=0; ICL=0; CROSS_KD=0; FD=0; GD=0; AFD=0
    
    case $METHOD in
        "CrossKD") CROSS_KD=1.0 ;;
        "ICL")     ICL=1.0 ;;
        "GD")      GD=1.0 ;;
        "FD")      FD=1.0 ;;
        "CKD")     CKD=1.0 ;;
        "AFD")     AFD=1.0 ;;
    esac

    TAG="fathomnet-distill-${METHOD}-student-vit-b"
    
    /../venv310/bin/torchrun --nproc_per_node 1 -m \
        training.main_kd \
        --save-frequency 0 \
        --zeroshot-frequency 0 \
        --report-to tensorboard \
        --train-data "csvfiles/train/captions_train_new-2.csv" \
        --val-data "csvfiles/val/captions_val_new-2.csv" \
        --data-root /Users/ricamouele/Documents/TER/CLIP-KD/ \
        --val-data-root /Users/ricamouele/Documents/TER/CLIP-KD/ \
        --csv-img-key filepath \
        --csv-caption-key caption \
        --csv-separator , \
        --warmup 200 \
        --batch-size=64 \
        --lr=${LR} \
        --wd=${WD} \
        --epochs 20 \
        --workers=4 \
        --model ViT-T-16 \
        --pretrained openai \
        --t-model ViT-B-16 \
        --t-model-checkpoint pretrained_models/ViT_B_16-laion400m_teacher-marine_e15.pt \
        --logs ../logs/ \
        --alpha_ckd_loss ${CKD} \
        --alpha_icl_loss ${ICL} \
        --alpha_cross_kd_loss ${CROSS_KD} \
        --alpha_fd_loss ${FD} \
        --alpha_gd_loss ${GD} \
        --alpha_afd_loss ${AFD} \
        --tag ${TAG}

    # Clean up checkpoints after each method to save disk space
    echo "Cleaning up checkpoints for $METHOD..."
    rm -f ../logs/*-tag_${TAG}/checkpoints/epoch_*.pt
done
