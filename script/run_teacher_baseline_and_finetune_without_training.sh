cd src
python -m training.main \
    --val-data="csvfiles/val/captions_val_new-2.csv" \
    --csv-img-key filepath \
    --csv-caption-key caption \
    --csv-separator , \
    --val-data-root /Users/ricamouele/Documents/TER/CLIP-KD/ \
    --model ViT-B-16 \
    --model-checkpoint pretrained_models/ViT_B_16-laion400m_teacher-marine_e15.pt \
    --eval \
    --workers=1 \
    --seed 42 \
    --logs "../logs" \
    --zeroshot-frequency 0 \
    --name "eval-pretrained-vitb16-2"


cd src
python -m training.main \
    --imagenet-val "csvfiles/val/captions_val_new-2.csv" \
    --model ViT-B-16 \
    --resume pretrained_models/ViT_B_16-laion400m_e32.pt \
    --eval \
    --zeroshot-frequency 1 \
    --batch-size 256 \
    --workers 4 \
    --seed 42 \
    --logs "../logs" \
    --name "zeroshot-vitb16"