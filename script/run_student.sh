#!/bin/bash

#   FD   →  --alpha_fd_loss 2000.
#   MFD  →  --alpha_fd_loss 2000. --mask_ratio 0.75
#   CRD  →  --alpha_ckd_loss 1.
#   GD   →  --alpha_gd_loss 1e8
#   ICL  →  --alpha_icl_loss 1.
#   AFD  →  --alpha_afd_loss 1.

cd src
python -m training.main_kd \
    --train-data "csvfiles/train/captions_morphology_train_clip_original.csv" \
    --val-data="csvfiles/val/captions_morphology_val_clip_original.csv" \
    --csv-img-key filepath \
    --csv-caption-key caption \
    --csv-separator , \
    --data-root /Users/ricamouele/Documents/TER/CLIP-KD/ \
    --val-data-root /Users/ricamouele/Documents/TER/CLIP-KD/ \
    --model ViT-T-16 \
    --t-model ViT-B-16 \
    --t-model-checkpoint pretrained_models/ViT_B_16-laion400m_teacher-marine.pt \
    --save-frequency 7 \
    --zeroshot-frequency 0 \
    --report-to tensorboard \
    --warmup 50 \
    --batch-size 8 \
    --lr 1e-5 \
    --wd 0.1 \
    --epochs 15 \
    --workers=1 \
    --seed 42 \
    --logs "../logs" \
    --alpha_fd_loss 2000. \
    --name "ViT_B_16-laion400m_teacher-marine-FD-e15"