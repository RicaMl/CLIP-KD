#!/bin/bash

#   FD   →  --alpha_fd_loss 2000.
#   MFD  →  --alpha_fd_loss 2000. --mask_ratio 0.75
#   CRD  →  --alpha_ckd_loss 1.
#   GD   →  --alpha_gd_loss 1e8
#   ICL  →  --alpha_icl_loss 1.
#   AFD  →  --alpha_afd_loss 1.

cd src
python -m training.main_kd \
    --train-data "csvfiles/train/captions_train_new-2.csv" \
    --val-data "csvfiles/val/captions_val_new-2.csv" \
    --csv-img-key filepath \
    --csv-caption-key caption \
    --csv-separator , \
    --data-root /Users/ricamouele/Documents/TER/CLIP-KD/ \
    --val-data-root /Users/ricamouele/Documents/TER/CLIP-KD/ \
    --save-frequency 18 \
    --zeroshot-frequency 0 \
    --report-to tensorboard \
    --warmup 30 \
    --batch-size 8 \
    --lr 1e-04 \
    --wd 0.1 \
    --epochs 30 \
    --workers 1 \
    --seed 42 \
    --logs "../logs" \
    --name "RN50-CLIP-KD-CC3M-Distilled-FineTuned-Marine-2" \
    --model RN50 \
    --model-checkpoint pretrained_models/ViT-B-16_laion400m_kd_RN50_cc3m_12m_ep32.pt \
    --t-model ViT-B-16 \
    --t-model-checkpoint pretrained_models/ViT_B_16-laion400m_teacher-marine_e15.pt \
    --alpha_ckd_loss 1. \
    --alpha_icl_loss 1. \
    --alpha_fd_loss 2000. \
    --tag distill-new