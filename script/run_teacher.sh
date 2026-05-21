
#!/bin/bash
cd src
python -m training.main \
    --train-data "csvfiles/train/captions_morphology_train_clip_original.csv" \
    --val-data="csvfiles/val/captions_morphology_val_clip_original.csv" \
    --csv-img-key filepath \
    --csv-caption-key caption \
    --csv-separator , \
    --data-root /Users/ricamouele/Documents/TER/CLIP-KD/ \
    --val-data-root /Users/ricamouele/Documents/TER/CLIP-KD/ \
    --model ViT-B-16 \
    --model-checkpoint pretrained_models/ViT_B_16-laion400m_e32.pt \
    --save-frequency 7 \
    --zeroshot-frequency 0 \
    --report-to tensorboard \
    --warmup 50 \
    --batch-size 8 \
    --lr 1e-6 \
    --wd 0.1 \
    --epochs 15 \
    --workers=1 \
    --seed 42 \
    --logs "../logs" \
    --name "ViT_B_16-laion400m_teacher-marine"
