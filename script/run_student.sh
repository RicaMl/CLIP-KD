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
    --save-frequency 20 \
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
    --name "OPENAI-RN50-CC3M-Distilled-FineTuned-Marine-3" \
    --model RN50 \
    --pretrained openai \
    --t-model ViT-B-16 \
    --t-model-checkpoint pretrained_models/ViT_B_16-laion400m_teacher-marine_e15.pt \
    --alpha_ckd_loss 1. \
    --alpha_icl_loss 1. \
    --alpha_fd_loss 2000. \
    --tag distill-new