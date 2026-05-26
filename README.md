# CLIP-KD — Distillation de connaissances sur domaine marin (FathomNet)

> Travail d'Étude et de Recherche (TER) — Master 1 Vision et Machine Intelligente  
> Université Paris Cité — 2025/2026  
> **Auteurs :** Rica Mouele Yandza Itotoba, Azouaou Zouaoui

---

## Présentation

Ce dépôt contient le code et les expériences réalisées dans le cadre d'une étude empirique de la distillation de connaissances appliquée au modèle CLIP sur un dataset d'images marines issues de la base [FathomNet](https://fathomnet.org). Il s'appuie sur l'implémentation de référence [CLIP-KD](https://github.com/winycg/CLIP-KD).

L'objectif est triple :
1. Tester les différentes stratégies de distillation proposées dans l'article CLIP-KD (FD, MFD, CRD, GD, ICL, AFD)
2. Investiguer des adaptations spécifiques au domaine marin (hyperparamètres, captions, nettoyage de données)
3. Évaluer les performances en classification zero-shot sur les données marines

---

## Structure du projet

```
CLIP-KD/
├── src/
│   └── training/
│       ├── main.py            # Entraînement standard (teacher)
│       ├── main_kd.py         # Entraînement avec distillation (student)
│       └── ...
├── csvfiles/
│   ├── train/
│   │   └── captions_train_new-2.csv
│   └── val/
│       └── captions_val_new-2.csv
├── pretrained_models/
│   ├── ViT_B_16-laion400m_e32.pt
│   ├── ViT_B_16-laion400m_teacher-marine_e15.pt
│   ├── RN50_cc3m_12m_ep32.pt
│   ├── ViT_B_16_laion400m_kd_RN50_cc3m_12m_ep32.pt
│   └── ViT_B_16_laion400m_kd_ViT_T_16_cc3m_12m_ep32.pt
├── logs/
├── scripts/
│   ├── train_teacher.sh
│   └── train_student.sh
└── README.md
```

---

## Dataset

Le dataset est composé d'images sous-marines annotées issues de l'API FathomNet :

- **Train :** 350 images natives (après nettoyage des duplicatas et des augmentations figées)
- **Val :** 39 images
- **Format :** CSV avec colonnes `filepath` et `caption`

### Récupération des images

Les images originales ont été récupérées directement via l'API FathomNet pour éliminer le bruit gaussien et les augmentations pré-calculées présentes dans le dataset initial. Les augmentations sont appliquées **exclusivement à la volée** pendant l'entraînement :

```python
RandomResizedCrop(image_size, scale=(0.9, 1.0), interpolation=InterpolationMode.BICUBIC),
RandomHorizontalFlip(p=0.5),
RandomVerticalFlip(p=0.5),
RandomApply([GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0))], p=0.3),
ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2)
```

### Captions

Deux versions de captions ont été générées via un VLM :

| Version | Description | Résultat |
|---|---|---|
| V1 | Few-shot avec exemple fixe, structure syntaxique rigide | Shortcut learning, mauvaise généralisation |
| **V2** | **Few-shot libre, descriptions variées et riches** | **Meilleure généralisation, retenu pour les expériences** |

---

## Modèles

### Modèle enseignant (Teacher)

| Paramètre | Valeur |
|---|---|
| Architecture | ViT-B/16 |
| Checkpoint de départ | `ViT_B_16-laion400m_e32.pt` |
| Checkpoint final | `ViT_B_16-laion400m_teacher-marine_e15.pt` (epoch 13) |
| R@1 I→T (epoch 13) | 51.28% |
| R@10 T→I (epoch 13) | 100.00% |
| Loss val (epoch 13) | 0.508 |

### Modèles étudiants (Students) testés

| Checkpoint | Architecture | R@1 max | Loss val min |
|---|---|---|---|
| `RN50_cc3m_12m_ep32` | RN50 | 25.6% | 1.381 |
| **`ViT_B_16_laion400m_kd_RN50_cc3m_12m_ep32`** | **RN50** | **28.2%** | **1.462** |
| `ViT_B_16_laion400m_kd_ViT_T_16_cc3m_12m_ep32` | ViT-T/16 | 20.5% | 1.554 |
| `ViT_B_16_laion400m_e32` | ViT-B/16 | 15.4% | 1.658 |

Le **RN50 pré-distillé KD** est retenu comme meilleur étudiant sur l'ensemble des métriques.

---

## Installation

```bash
git clone https://github.com/winycg/CLIP-KD.git
cd CLIP-KD
pip install -r requirements.txt
```

Placer les checkpoints pré-entraînés dans `pretrained_models/`. Les checkpoints OpenCLIP sont disponibles sur [Hugging Face](https://huggingface.co/laion).

---

## Entraînement

### 1. Fine-tuning du modèle enseignant

```bash
cd src
python -m training.main \
    --train-data "csvfiles/train/captions_train_new-2.csv" \
    --val-data "csvfiles/val/captions_val_new-2.csv" \
    --csv-img-key filepath \
    --csv-caption-key caption \
    --csv-separator , \
    --data-root /path/to/CLIP-KD/ \
    --val-data-root /path/to/CLIP-KD/ \
    --model ViT-B-16 \
    --model-checkpoint pretrained_models/ViT_B_16-laion400m_e32.pt \
    --save-frequency 1 \
    --zeroshot-frequency 0 \
    --report-to tensorboard \
    --warmup 150 \
    --batch-size 8 \
    --lr 1e-05 \
    --wd 0.1 \
    --epochs 15 \
    --workers 1 \
    --seed 42 \
    --logs "../logs" \
    --name "ViT_B_16-laion400m_teacher-marine"
```

### 2. Distillation du modèle étudiant

Configuration optimale identifiée empiriquement (FD + ICL + CKD) :

```bash
cd src
python -m training.main_kd \
    --train-data "csvfiles/train/captions_train_new-2.csv" \
    --val-data "csvfiles/val/captions_val_new-2.csv" \
    --csv-img-key filepath \
    --csv-caption-key caption \
    --csv-separator , \
    --data-root /path/to/CLIP-KD/ \
    --val-data-root /path/to/CLIP-KD/ \
    --save-frequency 0 \
    --zeroshot-frequency 0 \
    --report-to tensorboard \
    --warmup 30 \
    --batch-size 8 \
    --lr 1e-4 \
    --wd 0.1 \
    --epochs 30 \
    --workers 1 \
    --seed 42 \
    --logs "../logs" \
    --name "RN50-KD-Marine" \
    --model RN50 \
    --model-checkpoint pretrained_models/ViT_B_16_laion400m_kd_RN50_cc3m_12m_ep32.pt \
    --t-model ViT-B-16 \
    --t-model-checkpoint pretrained_models/ViT_B_16-laion400m_teacher-marine_e15.pt \
    --alpha_fd_loss 2000. \
    --alpha_icl_loss 1. \
    --alpha_ckd_loss 1. \
    --tag distill-marine
```

---

## Stratégies de distillation disponibles

| Stratégie | Flag | Valeur recommandée | Description |
|---|---|---|---|
| FD | `--alpha_fd_loss` | `2000.` | MSE sur les features image et texte |
| MFD | `--alpha_fd_loss --mask_ratio` | `2000. 0.75` | FD avec masquage de patches (75%) |
| CRD | `--alpha_ckd_loss` | `1.` | KL sur les distributions de similarité |
| GD | `--alpha_gd_loss` | `1e8` | MSE sur les gradients (coûteux) |
| ICL | `--alpha_icl_loss` | `1.` | Contrastif croisé student/teacher |
| AFD | `--alpha_afd_loss` | `1.` | Fusion + projection des features |

Les stratégies sont combinables. La combinaison **FD + ICL + CRD** est la plus performante sur notre dataset.

---

## Hyperparamètres clés — adaptations pour le domaine marin

| Paramètre | Valeur standard | Valeur optimale (marin) | Raison |
|---|---|---|---|
| `--batch-size` | 128 | **8** | Homogénéité visuelle → faux négatifs |
| `--lr` (student) | 1e-3 | **1e-4** | Dataset petit, convergence plus douce |
| `--logit-scale` | 100 | **10** | Images similaires → distribution plus lisse |
| `--alpha_fd_loss` | 2000 | **2000** | Valeur de référence maintenue |
| `--warmup` | 10000 | **30–150** | Adapté à 350 images |

---

## Visualisation avec TensorBoard

```bash
tensorboard --logdir logs/
```

---

## Résultats principaux

### Enseignant vs meilleur étudiant

| Modèle | R@1 I→T | R@10 I→T | R@1 T→I | R@10 T→I | Loss val |
|---|---|---|---|---|---|
| Teacher ViT-B/16 (ep. 13) | 51.3% | 97.4% | 43.6% | 100.0% | 0.508 |
| **Student RN50-KD (ep. 18)** | **25.6%** | **87.2%** | **28.2%** | **82.1%** | **1.462** |
| Baseline sans distillation | 15.4% | 66.7% | 23.1% | 69.2% | 1.658 |

> **Note importante :** Ces résultats sont à interpréter avec précaution. Le jeu de validation ne compte que 39 images, ce qui implique une variance élevée des métriques (1 exemple = 2.56% de R@1). Les tendances qualitatives sont fiables, mais les valeurs absolues ne doivent pas être surinterprétées.

---

## Limitations connues

- Dataset de très petite taille (350 train / 39 val) — variance élevée des métriques
- Évaluation sur le même split pour la validation et la sélection du checkpoint (absence de test set indépendant)
- Gradient Distillation (GD) non pleinement évaluée faute de ressources GPU
- Entraînement mono-GPU sur CPU local (MacBook) — temps d'exécution long

---

## Perspectives

- Enrichir le dataset via l'API FathomNet (plusieurs milliers d'images disponibles) ou la génération synthétique par modèles de diffusion
- Utiliser un teacher plus puissant : ViT-L/14 ou ViT-H/14 sur LAION-2B (OpenCLIP)
- Adopter la perte SigLIP pour le teacher (+5% R@1 observé sur notre domaine)
- Explorer MobileCLIP ou TinyCLIP comme architectures étudiantes
- Implémenter un loss scheduling pour \(\alpha_{\text{FD}}\) dynamique

---

## Références

- **CLIP** : Radford et al., *Learning Transferable Visual Models From Natural Language Supervision*, ICML 2021
- **CLIP-KD** : Yang et al., *CLIP-KD: An Empirical Study of CLIP Model Distillation*, CVPR 2024
- **SigLIP** : Zhai et al., *Sigmoid Loss for Language Image Pre-Training*, ICCV 2023
- **Knowledge Distillation** : Hinton et al., *Distilling the Knowledge in a Neural Network*, NeurIPS 2015
- **FathomNet** : [fathomnet.org](https://fathomnet.org)
- **OpenCLIP** : [github.com/mlfoundations/open_clip](https://github.com/mlfoundations/open_clip)