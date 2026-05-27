#!/bin/bash

# teacher ViT_B_16-laion400m pas entrainé sur notre dataset
cd src
python -m training.zero_shot_standard \
    --model ViT-B-16 \
    --checkpoint pretrained_models/ViT_B_16-laion400m_e32.pt \
    --csv csvfiles/zero-shot/captions_zero_shot2.csv \
    --logs ../logs/zero_shot_classif_baseline3

# teacher fine-tuné sur FathomNet notre dataset
python -m training.zero_shot_standard \
    --model ViT-B-16 \
    --checkpoint pretrained_models/ViT_B_16-laion400m_teacher-marine_e15.pt \
    --csv csvfiles/zero-shot/captions_zero_shot2.csv \
    --logs ../logs/zero_shot_classif_finetuned3

#student RN50 DISTILLÉ avec notre teacher fine tuné
python -m training.zero_shot_standard \
    --model RN50 \
    --checkpoint pretrained_models/OPENAI-RN50_Distilled_student.pt \
    --csv csvfiles/zero-shot/captions_zero_shot2.csv \
    --logs ../logs/zero_shot_classif_student3

#student distillé avec un ViT_B_16-laion400m
python -m training.zero_shot_standard \
    --model RN50 \
    --checkpoint pretrained_models/RN50_cc3m_12m_ep32.pt \
    --csv csvfiles/zero-shot/captions_zero_shot2.csv \
    --logs ../logs/zero_shot_classif_student3