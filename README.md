# CLIP-KD : Distillation de connaissances pour CLIP sur le domaine marin

Ce dépôt contient l’implémentation et les expérimentations réalisées dans le cadre d’un TER (Master 1 VMI, Université Paris Cité). L’objectif est d’étudier la distillation de connaissances (Knowledge Distillation) du modèle CLIP sur un jeu de données d’images sous‑marines (FathomNet), afin d’obtenir un modèle étudiant plus compact tout en conservant de bonnes performances en retrieval image‑texte.

## Contexte

CLIP (Contrastive Language-Image Pre-training) est un modèle de vision‑langage qui aligne images et descriptions textuelles dans un espace commun. La distillation de connaissances permet de transférer les connaissances d’un grand modèle enseignant (teacher) vers un modèle plus petit (student), réduisant ainsi le coût en mémoire et en calcul.

Dans ce projet, nous utilisons :
- **Teacher** : ViT-B/16 fine‑tuné sur le dataset marin FathomNet.
- **Student** : RN50 (initialisé par OpenAI ou pré‑distillé) et d’autres architectures (ViT-T-16, etc.).
- **Stratégies de distillation** : FD (Feature Distillation), ICL (Interactive Contrastive Learning), CRD, MFD, AFD, GD – avec combinaison optimale `FD + ICL + CKD`.

## Données

Le jeu de données est issu de l’API [FathomNet](https://fathomnet.org/) : images sous‑marines d’espèces variées, nettoyées des doublons et des artefacts (bruit, padding figé).  
- **Train** : 350 images uniques  
- **Validation** : 39 images  
- **Test zero‑shot** : 14 espèces non vues à l’entraînement.

Deux versions de captions (descriptions textuelles) ont été générées par un VLM :
- **Version 1** : structure rigide (few‑shot fixe) → encourage le *shortcut learning*.
- **Version 2** : descriptions libres et diversifiées → meilleure généralisation.

## Installation

1. **Cloner le dépôt**
   ```bash
   git clone https://github.com/RicaMl/CLIP-KD.git
   cd CLIP-KD
   ```

2. **Créer un environnement virtuel (Python 3.10 recommandé)**
   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux/Mac
   # ou venv\Scripts\activate sous Windows
   ```

3. **Installer les dépendances**
   ```bash
   pip install -r requirement.txt
   ```

   Le code a été testé avec PyTorch 2.0+ et MPS (Mac). Pour GPU CUDA, ajustez l’argument `--device`.

4. **Télécharger les checkpoints pré‑entraînés**
   - Modèle enseignant final (`ViT_B_16-laion400m_teacher-marine_e15.pt`) : https://github.com/winycg/CLIP-KD
   - Checkpoints étudiants (RN50 OpenAI, RN50 pré‑distillé, ViT-T-16 pré‑distillé, etc.)
   - Placez‑les dans le dossier `pretrained_models/`.

## Structure du code

```
CLIP-KD/
├── script/                       # Scripts d’entraînement et de distillation et d'evaluation zero shot
├── training/                     
│   ├── main.py                   # Fine‑tuning du teacher
│   ├── main_kd.py                # Distillation du teacher vers le student
│   └── zero_shot_fathomnet_custom_v2.py  # Évaluation cross‑modal zero‑shot retrieval
├── csvfiles/                     # Fichiers CSV (train/val captions)
│   ├── train/
│   ├── val/
│   └── zero-shot/
├── logs/                         # TensorBoard logs et résultats
├── pretrained_models/            # Checkpoints (.pt)
└── README.md
```

## Utilisation

### 1. Fine‑tuning du modèle enseignant

Entraînez le teacher (ViT-B-16) sur le dataset marin (Version 2 des captions) :

```bash
python -m training.main \
    --train-data "csvfiles/train/captions_train_new-2.csv" \
    --val-data "csvfiles/val/captions_val_new-2.csv" \
    --csv-img-key filepath \
    --csv-caption-key caption \
    --model ViT-B-16 \
    --model-checkpoint pretrained_models/ViT_B_16-laion400m_e32.pt \
    --batch-size 8 \
    --lr 1e-5 \
    --wd 0.1 \
    --epochs 15 \
    --logs "../logs" \
    --name "teacher_marine_final"
```

Les meilleures performances sont obtenues à l’époque 13 (loss 0.508, R@1=51.3%).

### 2. Distillation

Pour distiller le teacher vers un student (par ex. RN50 pré‑entraîné OpenAI) :

```bash
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
```

La combinaison `FD + ICL + CKD` avec `alpha_fd=2000` donne les meilleurs résultats.

### 3. Évaluation zero‑shot

Testez la généralisation sur 14 espèces non vues (captions générées par VLM) :

```bash
python -m training.zero_shot_fathomnet_custom_v2 \
    --model ViT-B-16 \
    --checkpoint pretrained_models/teacher_marine_epoch13.pt \
    --csv csvfiles/zero-shot/captions_zero_shot2.csv \
    --logs ../logs/zero_shot_teacher
```

Pour évaluer un student :

```bash
python -m training.zero_shot_fathomnet_custom_v2 \
    --model RN50 \
    --checkpoint pretrained_models/rn50_distilled_student.pt \
    --csv csvfiles/zero-shot/captions_zero_shot2.csv
```

## Résultats clés

### Teacher (ViT-B-16 fine‑tuné)
- **R@1 (I→T)** : 51.3 %  
- **R@10 (I→T)** : 97.4 %  
- **Loss validation** : 0.508 (epoch 13)

### Student retenu (RN50 pré‑entraîné OpenAI)
- **R@1 (I→T)** : 25.6 %  
- **R@5 (I→T)** : 74.4 %  
- **Loss validation** : 0.837 (epoch 25)

### Évaluation zero‑shot (14 espèces)

Le tableau ci-dessous résume les accuracies des quatre modèles testés :

| Modèle | Accuracy |
|--------|----------|
| ViT-B-16 LAION-400M (baseline) | 57.1 % (8/14) |
| ViT-B-16 fine‑tuné marin | **85.7 %** (12/14) |
| RN50 student (distillé) | 50.0 % (7/14) |
| RN50 CC3M (non adapté) | 35.7 % (5/14) |

#### Détail des prédictions par modèle

**1. Modèle baseline (ViT-B-16 LAION-400M, non adapté)**  
- 8 bonnes réponses, 6 erreurs.  
- Confusions typiques : *Tarsastrocles verrilli* → *Paralomis* (score 43.0 %), *Sebastes levis* → *Scalicus engyceros* (91.3 %), *Hemicorallium abyssale* → *Sebastolobus* (63.3 %).  
- Exemple de bonne prédiction : *Deepstaria reticulum* (96.6 %).

**2. Modèle fine‑tuné marin (ViT-B-16 après adaptation)**  
- 12 bonnes réponses, seules 2 erreurs.  
- Corrige toutes les confusions de la baseline, sauf *Peribolaster* (confondu avec *Scalicus engyceros* à 71.4 %) et *Gaza* (confondu avec *Deepstaria reticulum* à 97.0 %).  
- Scores très élevés (>99 %) pour la plupart des bonnes prédictions.

**3. Student distillé (RN50 – checkpoint `OPENAI-RN50_Distilled_student.pt`)**  
- 7 bonnes réponses.  
- Forte tendance à prédire *Peribolaster* ou *Deepstaria reticulum* comme première hypothèse, avec des scores parfois élevés mais erronés (ex. *Tarsastrocles* → *Peribolaster* à 89.0 %).  
- Seul modèle (avec le marin) à bien classer *Hemicorallium abyssale*.

**4. RN50 non adapté (CC3M, sans fine‑tuning)**  
- 5 bonnes réponses seulement.  
- Confusions massives : *Deepstaria reticulum* → *Bathyraja trachura* (93.8 %), *Paralomis* → *Lophiodes beroe* (63.8 %), *Peribolaster* → *Gaza* (37.7 %), etc.


### Expérimentations notables
- Le **nettoyage du dataset** a fait passer le R@1 de 20.9 % à 46.2 %.
- La **Version 2 des captions** élimine le *shortcut learning* observé avec la Version 1.
- Le **batch size réduit à 8** et le **logit scale à 10** améliorent la stabilité.
- La perte **SigLIP** a donné un R@1 maximum de 56.4 % (gain de +5 points) mais n’a pas été retenue pour la distillation.

## Perspectives d’amélioration

- Augmenter la quantité de données.
- Utiliser un teacher ou un student plus puissant (ViT-L/14 ou ViT-H/14).
- Explorer systématiquement la perte SigLIP dans la distillation.
- Élargir le jeu de test zero‑shot.

## Références

- Yang et al., *CLIP-KD: An Empirical Study of CLIP Model Distillation*, CVPR 2024.
- Zhai et al., *Sigmoid Loss for Language Image Pre-Training*, ICCV 2023.
- FathomNet : [fathomnet.org](https://fathomnet.org/)
- OpenCLIP : [github.com/mlfoundations/open_clip](https://github.com/mlfoundations/open_clip)
- Code vers repository de génération des captions : https://github.com/RicaMl/knowledge_distillation_ter.

## Note

Ce code est distribué pour des fins académiques. Les modèles pré‑entraînés appartiennent à leurs auteurs respectifs.

## Auteurs

- Rica Mouele Yandza Itotoba  
- Azouaou Zouaoui 
