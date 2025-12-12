import os
import argparse
import yaml
import torch
import glob
import numpy as np
from PIL import Image
from tqdm import tqdm

# Add project root to path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ot_jepa.models.vjepa2_backbone import VJEPA2HubEncoder, VJEPA2VisionEncoder
from ot_jepa.models.encoders import VisionEncoder


def load_encoder(cfg, device):
    d = cfg["model"]["latent_dim"]
    v_patch = int(cfg["model"].get("patch_size", 16))
    v_depth = int(cfg["model"].get("vision_depth", 4))
    v_heads = int(cfg["model"].get("vision_heads", 4))
    vision_backbone = str(cfg["model"].get("vision_backbone", "internal")).lower()

    print(f"Loading backbone: {vision_backbone}")

    if vision_backbone == "vjepa2":
        model = VJEPA2VisionEncoder(
            latent_dim=d,
            img_size=tuple(cfg.get("data", {}).get("image_size", (256, 256))),
            patch_size=v_patch,
            depth=v_depth,
            heads=v_heads,
        )
    elif vision_backbone == "vjepa2_hub":
        v2 = cfg.get("vjepa2", {})
        model = VJEPA2HubEncoder(
            latent_dim=d,
            variant=str(v2.get("variant", "vjepa2_ac_vit_giant")),
            pretrained=bool(v2.get("pretrained", True)),
            freeze=True,
            img_size=tuple(cfg.get("data", {}).get("image_size", (256, 256))),
            patch_size=v_patch,
        )
    else:
        model = VisionEncoder(latent_dim=d, patch_size=v_patch, depth=v_depth, heads=v_heads)

    model.to(device)
    model.eval()
    return model


def process_episode(episode_path, model, device, image_size, batch_size=8):
    # Find all images
    front_files = sorted(glob.glob(os.path.join(episode_path, "front_*.png")))
    wrist_files = sorted(glob.glob(os.path.join(episode_path, "wrist_*.png")))

    if not front_files or not wrist_files:
        return False

    # Ensure matching counts (basic check)
    if len(front_files) != len(wrist_files):
        print(f"Mismatch in {episode_path}: {len(front_files)} front vs {len(wrist_files)} wrist")
        # Truncate to min
        n = min(len(front_files), len(wrist_files))
        front_files = front_files[:n]
        wrist_files = wrist_files[:n]

    def load_and_process(files):
        embeddings = []
        for i in range(0, len(files), batch_size):
            batch_files = files[i:i+batch_size]
            imgs = []
            for f in batch_files:
                img = Image.open(f).convert("RGB").resize(image_size)
                arr = np.array(img)
                arr = np.transpose(arr, (2,0,1)).astype(np.float32) / 255.0
                imgs.append(arr)

            batch_tensor = torch.tensor(np.stack(imgs), device=device)
            with torch.no_grad():
                emb = model(batch_tensor)
            embeddings.append(emb.cpu())
        return torch.cat(embeddings, dim=0)

    emb_front = load_and_process(front_files)
    emb_wrist = load_and_process(wrist_files)

    save_path = os.path.join(episode_path, "embeddings.pt")
    torch.save({
        "front": emb_front,
        "wrist": emb_wrist
    }, save_path)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    device = torch.device(args.device)

    model = load_encoder(cfg, device)

    # Safety check: ensure encoder is frozen
    trainable_params = sum(p.requires_grad for p in model.parameters())
    if trainable_params > 0:
        raise RuntimeError(
            f"Cannot precompute embeddings with trainable encoder!\n"
            f"  Found {trainable_params} trainable parameters in the vision encoder.\n"
            f"  Precomputed embeddings become stale when the encoder is updated during training.\n"
            f"  Please ensure the encoder is frozen in your config.\n"
            f"  Current architecture: {cfg.get('model', {}).get('architecture', 'unknown')}"
        )
    print(f"Encoder is frozen ({sum(p.numel() for p in model.parameters())} total params, 0 trainable) ✓")

    data_dir = cfg["data"]["episode_dir"]
    image_size = tuple(cfg["data"].get("image_size", (256, 256)))

    # Find episodes
    episode_files = glob.glob(os.path.join(data_dir, "**", "episode.parquet"), recursive=True)
    if episode_files:
        episodes = sorted({os.path.dirname(p) for p in episode_files})
    else:
        episodes = sorted([p for p in glob.glob(os.path.join(data_dir, "*")) if os.path.isdir(p)])

    print(f"Found {len(episodes)} episodes in {data_dir}")

    count = 0
    for ep in tqdm(episodes):
        if process_episode(ep, model, device, image_size, args.batch_size):
            count += 1

    print(f"Processed {count} episodes.")


if __name__ == "__main__":
    main()
