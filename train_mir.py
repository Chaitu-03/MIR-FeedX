# save as train_mir.py
import modal

app = modal.App("mir-training")

# Persistent volume for outputs
volume = modal.Volume.from_name("mir-storage", create_if_missing=True)

@app.function(
    image=modal.Image.debian_slim()
        .pip_install("torch", "torchvision", "torchaudio", "transformers", "pillow", "numpy")
        .pip_install("torch", extra_options="--index-url https://download.pytorch.org/whl/cu118"),
    gpu="t4",  # cheaper option; for more power use "a100"
    volumes={"/data": volume},
    timeout=1800,  # 30 min
)
def train_projection():
    import torch
    import numpy as np
    from pathlib import Path
    from torch.utils.data import DataLoader, TensorDataset
    import torch.nn as nn
    import torch.nn.functional as F
    import logging
    
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")
    log = logging.getLogger(__name__)
    
    # CONFIG
    CLIP_DIM = 512
    TEXT_DIM = 384
    VAL_SIZE = 500
    EPOCHS = 50
    BATCH_SIZE = 256
    LR = 1e-3
    TARGET_COS = 0.75
    BOOTSTRAP_SIZE = 6000
    PAIRS_PATH = Path("/data/projection_pairs.npz")
    OUT_PATH = Path("/data/clip_to_miniLM_projection.pt")
    
    # STEP 1: Generate pairs
    log.info(f"Generating {BOOTSTRAP_SIZE} bootstrap pairs...")
    
    from transformers import AutoTokenizer, AutoModel, CLIPTextModelWithProjection
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Using device: {device}")
    
    # Load CLIP text encoder (not the full model)
    from transformers import CLIPTextModel
    clip_tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    clip_text_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip_text_model.eval()
    
    text_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    text_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").to(device)
    text_model.eval()
    
    captions = [
        "a dog running in park",
        "sunset over ocean",
        "cat sleeping on couch",
    ] * (BOOTSTRAP_SIZE // 3)
    
    captions = captions[:BOOTSTRAP_SIZE]
    
    clip_embeddings = []
    text_embeddings = []
    
    with torch.no_grad():
        for i, caption in enumerate(captions):
            if (i + 1) % 500 == 0:
                log.info(f"  {i+1}/{BOOTSTRAP_SIZE}")
            
            clip_inputs = clip_tokenizer(caption, return_tensors="pt").to(device)
            clip_outputs = clip_text_model(**clip_inputs)
            clip_emb = clip_outputs.last_hidden_state[:, 0, :]  # Use [CLS] token
            clip_embeddings.append(clip_emb.cpu())
            
            text_inputs = text_tokenizer(caption, return_tensors="pt").to(device)
            text_outputs = text_model(**text_inputs)
            text_emb = text_outputs.last_hidden_state.mean(dim=1)  # Mean pooling
            text_embeddings.append(text_emb.cpu())
    
    clip_embeddings = torch.cat(clip_embeddings, dim=0)
    text_embeddings = torch.cat(text_embeddings, dim=0)
    
    log.info(f"CLIP embeddings: {clip_embeddings.shape}")
    log.info(f"Text embeddings: {text_embeddings.shape}")
    
    # Verify dimensions match config
    assert clip_embeddings.shape[1] == CLIP_DIM, f"CLIP dim mismatch: {clip_embeddings.shape[1]} != {CLIP_DIM}"
    assert text_embeddings.shape[1] == TEXT_DIM, f"Text dim mismatch: {text_embeddings.shape[1]} != {TEXT_DIM}"
    
    PAIRS_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        PAIRS_PATH,
        clip_embeddings=clip_embeddings.numpy(),
        caption_embeddings=text_embeddings.numpy(),
    )
    log.info(f"Saved pairs → {PAIRS_PATH}")
    
    # STEP 2: Train
    log.info("Loading pairs...")
    data = np.load(PAIRS_PATH)
    clip_embs = torch.from_numpy(data["clip_embeddings"]).float()
    text_embs = torch.from_numpy(data["caption_embeddings"]).float()
    
    n = len(clip_embs)
    log.info(f"Loaded {n} pairs")
    
    perm = torch.randperm(n)
    val_perm = perm[:VAL_SIZE]
    train_perm = perm[VAL_SIZE:]
    
    train_ds = TensorDataset(clip_embs[train_perm], text_embs[train_perm])
    val_clip = clip_embs[val_perm]
    val_text = text_embs[val_perm]
    
    loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    
    layer = nn.Linear(CLIP_DIM, TEXT_DIM, bias=False).to(device)
    nn.init.orthogonal_(layer.weight)
    optimizer = torch.optim.Adam(layer.parameters(), lr=LR)
    criterion = nn.MSELoss()
    
    def cosine_sim(a, b):
        a_n = F.normalize(a, dim=-1)
        b_n = F.normalize(b, dim=-1)
        return (a_n * b_n).sum(dim=-1).mean().item()
    
    log.info(f"Training: {len(train_perm)} train / {VAL_SIZE} val, {EPOCHS} epochs")
    
    for epoch in range(1, EPOCHS + 1):
        layer.train()
        epoch_loss = 0.0
        for clip_b, text_b in loader:
            clip_b, text_b = clip_b.to(device), text_b.to(device)
            optimizer.zero_grad()
            pred = layer(clip_b)
            loss = criterion(pred, text_b)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        if epoch % 10 == 0 or epoch == EPOCHS:
            layer.eval()
            with torch.no_grad():
                val_cos = cosine_sim(layer(val_clip.to(device)), val_text.to(device))
            log.info(f"Epoch {epoch:3d}/{EPOCHS}  loss={epoch_loss/len(loader):.5f}  val_cos={val_cos:.4f}")
    
    layer.eval()
    with torch.no_grad():
        final_cos = cosine_sim(layer(val_clip.to(device)), val_text.to(device))
    
    log.info(f"Complete. Val cosine: {final_cos:.4f} (target ≥ {TARGET_COS:.2f})")
    if final_cos < TARGET_COS:
        log.warning(f"Below target ({final_cos:.4f} < {TARGET_COS:.2f}).")
    
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(layer.state_dict(), str(OUT_PATH))
    log.info(f"Saved projection → {OUT_PATH}")
    
    return final_cos

@app.local_entrypoint()
def main():
    cos_sim = train_projection.remote()
    print(f"Final validation cosine: {cos_sim}")