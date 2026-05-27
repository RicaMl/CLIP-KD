import argparse
import torch
import torch.nn.functional as F
from PIL import Image
import open_clip
import pandas as pd
import os
import json
from pathlib import Path

GROUND_TRUTH = {
    "0a81fd39-9158-4773-a19d-f321f067a9c9": "Tarsastrocles verrilli",
    "10b0dfdf-02e4-4b85-a4f8-f705075cbc17": "Deepstaria reticulum",
    "11ed725d-8f62-446a-ba6e-92af3b293b14": "Sebastes levis",
    "3bf47f2b-2915-410f-9f4f-57aa24529055": "Scalicus engyceros",
    "3fa6f63b-0b71-42cc-a751-10042004c761": "Paralomis",
    "50f9ba28-29aa-4c27-8c9b-064b881c6713": "Psilocalyx wilsoni",
    "5c9cba06-c647-4501-a56f-f8eda25330c5": "Hemicorallium abyssale",
    "68146dff-d779-49e9-93fa-60777487fdda": "Peribolaster",
    "74a6ba02-9b97-436a-a862-a9c126266462": "Bathyraja trachura",
    "766c9f8f-cc24-40b2-b02e-947459cef281": "Rhopalonematidae",
    "821052f3-75fa-4726-a176-7ca0e0f4ccf9": "Gonatopsis",
    "8e53d25f-1c5e-42e1-9077-3493970a66f0": "Lophiodes beroe",
    "984e8fbb-53b8-4ac8-8f10-a7c4f0aafe3f": "Gaza",
    "a91a18f6-69fc-48bc-b387-7ce3693c23c6": "Sebastolobus",
}

def extract_uuid(filepath):
    return Path(filepath).stem

def evaluate_zeroshot_classification(model, preprocess, tokenizer, device, csv_path, logs_dir):
    df = pd.read_csv(csv_path)

    # --- Correction des chemins : enlève le préfixe 'src/' si présent ---
    df['filepath'] = df['filepath'].apply(lambda p: p[4:] if p.startswith('src/') else p)

    df['uuid'] = df['filepath'].apply(extract_uuid)
    df = df[df['uuid'].isin(GROUND_TRUTH.keys())].drop_duplicates('uuid').reset_index(drop=True)

    if len(df) == 0:
        print("Aucune image correspondant au dictionnaire GROUND_TRUTH trouvée.")
        return None

    class_names = list(GROUND_TRUTH.values())
    prompts = [f"a photo of a {name}" for name in class_names]

    with torch.no_grad():
        text_tokens = tokenizer(prompts).to(device)
        text_feats = model.encode_text(text_tokens)
        text_feats = F.normalize(text_feats, dim=-1)
        class_emb_matrix = text_feats

    correct = 0
    total = len(df)
    print(f"\n{'Image UUID':<40} {'GT classe':<30} {'Prédite':<30} ✓")
    print("-" * 110)

    for idx, row in df.iterrows():
        img_path = row['filepath']   # chemin déjà corrigé (sans src/)
        uuid = row['uuid']
        gt_label = GROUND_TRUTH[uuid]

        # Vérification optionnelle : afficher le chemin pour debug
        # print("Tentative d'ouverture :", img_path)

        image = preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
        img_feat = model.encode_image(image)
        img_feat = F.normalize(img_feat, dim=-1)

        logits = 100.0 * img_feat @ class_emb_matrix.T
        probs = logits.softmax(dim=-1)[0]
        pred_idx = probs.argmax().item()
        pred_label = class_names[pred_idx]

        is_correct = (pred_label == gt_label)
        correct += int(is_correct)
        mark = "✓" if is_correct else "✗"
        print(f"{uuid:<40} {gt_label:<30} {pred_label:<30} {mark}")

    accuracy = correct / total
    print(f"\nZero-shot classification accuracy : {correct}/{total} ({accuracy:.1%})")

    os.makedirs(logs_dir, exist_ok=True)
    results = {"accuracy": accuracy, "correct": correct, "total": total}
    with open(os.path.join(logs_dir, "results_classification.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"Résultats sauvegardés dans {logs_dir}/results_classification.json")

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--logs", default="../logs/zero_shot_classification")
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = parser.parse_args()

    print(f"Modèle    : {args.model}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device    : {args.device}")

    model, _, preprocess = open_clip.create_model_and_transforms(args.model, pretrained=None)
    model = model.to(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    tokenizer = open_clip.get_tokenizer(args.model)

    evaluate_zeroshot_classification(model, preprocess, tokenizer, args.device, args.csv, args.logs)