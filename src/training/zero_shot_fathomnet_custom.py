import argparse
import torch
import torch.nn.functional as F
from PIL import Image
import open_clip
import pandas as pd
import os, json

GROUND_TRUTH = {
    "3fa6f63b-0b71-42cc-a751-10042004c761": "Paralomis",
    "821052f3-75fa-4726-a176-7ca0e0f4ccf9": "Gonatopsis",
    "a91a18f6-69fc-48bc-b387-7ce3693c23c6": "Sebastolobus",
    "50f9ba28-29aa-4c27-8c9b-064b881c6713": "Psilocalyx wilsoni",
    "74a6ba02-9b97-436a-a862-a9c126266462": "Bathyraja trachura",
    "11ed725d-8f62-446a-ba6e-92af3b293b14": "Sebastes levis",
    "5c9cba06-c647-4501-a56f-f8eda25330c5": "Hemicorallium abyssale",
    "984e8fbb-53b8-4ac8-8f10-a7c4f0aafe3f": "Gaza",
    "0a81fd39-9158-4773-a19d-f321f067a9c9": "Tarsastrocles verrilli",
    "8e53d25f-1c5e-42e1-9077-3493970a66f0": "Lophiodes beroe",
}


def fix_path(img_path):
    if img_path.startswith("src/"):
        return img_path[len("src/"):]
    return img_path


def evaluate(model, preprocess, tokenizer, device, csv_path, logs_dir):
    df = pd.read_csv(csv_path)

    # ── Garde uniquement les lignes dont l'UUID est dans GROUND_TRUTH ──
    df["img_id"] = df["filepath"].apply(
        lambda p: fix_path(p).split("/")[-1].replace(".jpg", "")
    )
    df = df[df["img_id"].isin(GROUND_TRUTH.keys())].drop_duplicates("img_id").reset_index(drop=True)

    captions  = df["caption"].tolist()
    filepaths = [fix_path(p) for p in df["filepath"].tolist()]
    img_ids   = df["img_id"].tolist()

    print(f"Images chargées : {len(df)}/10")

    with torch.no_grad():
        tokens     = tokenizer(captions).to(device)
        text_feats = model.encode_text(tokens)
        text_feats = F.normalize(text_feats, dim=-1)

        correct = 0
        print(f"\n{'Image':<45} {'GT':<30} {'Prédit':<30} ✓")
        print("-" * 115)

        for i, (img_path, img_id) in enumerate(zip(filepaths, img_ids)):
            gt_label = GROUND_TRUTH[img_id]

            image    = preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
            img_feat = F.normalize(model.encode_image(image), dim=-1)

            logits     = 100. * img_feat @ text_feats.T
            probs      = logits.softmax(dim=-1)[0]
            pred_idx   = probs.argmax().item()
            pred_label = GROUND_TRUTH[img_ids[pred_idx]]

            is_correct  = pred_label == gt_label
            correct    += int(is_correct)

            mark = "Great!!!" if is_correct else "Failed..."
            print(f"{img_id:<45} {gt_label:<30} {pred_label:<30} {mark}")
            print(f"  Caption prédite : {captions[pred_idx][:80]}...")
            print(f"  Score           : {probs[pred_idx]*100:.1f}%")

    accuracy = correct / len(df)
    print(f"\nAccuracy zero-shot : {correct}/{len(df)} ({accuracy*100:.0f}%)")

    os.makedirs(logs_dir, exist_ok=True)
    results = {"accuracy": accuracy, "correct": correct, "total": len(df)}
    results_path = os.path.join(logs_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Résultats sauvegardés dans : {results_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      required=True,  help="ex: ViT-B-16")
    parser.add_argument("--checkpoint", required=True,  help="chemin vers le .pt")
    parser.add_argument("--csv",        required=True,  help="chemin vers le CSV zero-shot")
    parser.add_argument("--logs",       default="../logs/zero_shot", help="dossier de logs")
    parser.add_argument("--device",     default="mps" if torch.backends.mps.is_available() else "cpu")
    args = parser.parse_args()

    print(f"Modèle    : {args.model}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device    : {args.device}")

    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model, pretrained=None
    )
    model = model.to(args.device)

    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)

    model     = model.eval()
    tokenizer = open_clip.get_tokenizer(args.model)

    evaluate(model, preprocess, tokenizer, args.device, args.csv, args.logs)