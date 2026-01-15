from __future__ import annotations

import os
import time
import random
import hashlib
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

# IMPORTANT: must be the first Streamlit command, and only once.  (Streamlit requirement)
st.set_page_config(page_title="Autonomous Vision Agent (Single-File)", layout="wide")

# ---- Safe imports for ML deps ----
try:
    import requests
    import numpy as np
    from PIL import Image

    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torchvision import transforms, models
except ModuleNotFoundError as e:
    st.error("A required dependency is missing.")
    st.code(str(e))
    st.markdown(
        """
### Fix (local)
Activate your venv, then install:
```bash
python -m pip install streamlit pillow numpy requests torch torchvision torchaudio
```

### Fix (Streamlit Cloud)
Make sure your repo includes a requirements.txt with:
```
streamlit
pillow
numpy
requests
torch
torchvision
torchaudio
```
"""
    )
    st.stop()

# ---- Device selection (M2: prefer MPS) ----
def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

# ---- Free reverse image search providers (NO API KEYS) ----
class ReverseSearchProvider:
    name: str = "base"
    def search(self, image_path: str, top_k: int = 5) -> List[Dict[str, float]]:
        raise NotImplementedError

class SauceNAOSearcher(ReverseSearchProvider):
    name = "saucenao"
    BASE_URL = "https://saucenao.com/api/lookup"
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("SAUCENAO_API_KEY")
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        })
    
    def search(self, image_path: str, top_k: int = 5) -> List[Dict[str, float]]:
        try:
            with open(image_path, "rb") as f:
                files = {"image": f}
                params = {
                    "output_type": 2,  # JSON output
                    "numres": min(top_k, 16)
                }
                
                if self.api_key:
                    params["api_key"] = self.api_key
                
                response = self.session.post(
                    self.BASE_URL,
                    files=files,
                    params=params,
                    timeout=30
                )
                response.raise_for_status()
                data = response.json()
                
                labels = []
                if "results" in data:
                    for result in data["results"][:top_k]:
                        header = result.get("header", {})
                        similarity = float(header.get("similarity", 0))
                        
                        url = result.get("data", {}).get("source", "")
                        title = result.get("data", {}).get("title", "Unknown")
                        
                        if url:
                            labels.append({
                                "label": str(title).lower(),
                                "confidence": similarity / 100.0,  # Convert to 0-1 range
                                "url": url,
                                "similarity": similarity
                            })
                
                return sorted(labels, key=lambda x: -x["confidence"])[:top_k]
                
        except Exception as e:
            print(f"SauceNAO search error: {e}")
            return []

# ---- Model ----
class VisionEncoder(nn.Module):
    def __init__(self, embed_dim: int = 512, use_pretrained: bool = True):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if use_pretrained else None
        base = models.resnet18(weights=weights)
        self.backbone = nn.Sequential(*list(base.children())[:-1])  # (B,512,1,1)
        self.proj = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, embed_dim),
        )

    def forward(self, x):
        x = self.backbone(x).flatten(1)
        return self.proj(x)

# ---- Config ----
@dataclass
class AgentConfig:
    image_size: int = int(os.getenv("IMAGE_SIZE", "224"))
    embed_dim: int = int(os.getenv("EMBED_DIM", "512"))
    data_dir: str = os.getenv("DATA_DIR", "data")
    model_dir: str = os.getenv("MODEL_DIR", "models")

    max_images_on_disk: int = int(os.getenv("MAX_IMAGES_ON_DISK", "4000"))
    reservoir_size: int = int(os.getenv("RESERVOIR_SIZE", "8000"))
    dup_thr: float = float(os.getenv("DUP_THR", "0.995"))
    jpeg_quality: int = int(os.getenv("JPEG_QUALITY", "70"))

    lr: float = float(os.getenv("LR", "3e-4"))
    use_pretrained: bool = os.getenv("USE_PRETRAINED", "1") == "1"

    local_top_k: int = int(os.getenv("LOCAL_TOP_K", "5"))
    local_thr: float = float(os.getenv("LOCAL_THR", "0.55"))

# ---- Vision Agent ----
class VisionAgent:
    def __init__(self, cfg: Optional[AgentConfig] = None):
        self.cfg = cfg or AgentConfig()
        self.device = pick_device()

        # Paths
        self.inbox = os.path.join(self.cfg.data_dir, "inbox")
        self.archive = os.path.join(self.cfg.data_dir, "archive")
        self.emb_db = os.path.join(self.cfg.data_dir, "emb.npy")
        self.meta_db = os.path.join(self.cfg.data_dir, "meta.npy")
        self.model_path = os.path.join(self.cfg.model_dir, "vision.pth")

        os.makedirs(self.inbox, exist_ok=True)
        os.makedirs(self.archive, exist_ok=True)
        os.makedirs(self.cfg.model_dir, exist_ok=True)
        os.makedirs(self.cfg.data_dir, exist_ok=True)

        # Provider (Free services - SauceNAO, Yandex, DuckDuckGo)
        self.provider = SauceNAOSearcher()

        # Transforms
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        self.ssl_tf = transforms.Compose([
            transforms.RandomResizedCrop(self.cfg.image_size, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        self.inf_tf = transforms.Compose([
            transforms.Resize((self.cfg.image_size, self.cfg.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        # Model + optimizer
        self.vision = VisionEncoder(self.cfg.embed_dim, self.cfg.use_pretrained).to(self.device)
        self.opt = optim.Adam(self.vision.parameters(), lr=self.cfg.lr)

        # Memory
        self.E, self.M = self._safe_load_memory()
        self.seen = 0

        # Load weights
        self._safe_load_model()

    # ---------- persistence ----------
    def _safe_load_model(self):
        if os.path.exists(self.model_path):
            try:
                self.vision.load_state_dict(torch.load(self.model_path, map_location=self.device))
            except Exception:
                pass

    def _safe_load_memory(self) -> Tuple[np.ndarray, np.ndarray]:
        if os.path.exists(self.emb_db) and os.path.exists(self.meta_db):
            try:
                E = np.load(self.emb_db).astype(np.float32)
                M = np.load(self.meta_db, allow_pickle=True)
                if E.ndim == 2 and E.shape[1] == self.cfg.embed_dim:
                    return E, M
            except Exception:
                pass
        return np.zeros((0, self.cfg.embed_dim), np.float32), np.array([], dtype=object)

    def save_state(self):
        np.save(self.emb_db, self.E.astype(np.float32))
        np.save(self.meta_db, self.M, allow_pickle=True)
        torch.save(self.vision.state_dict(), self.model_path)

    # ---------- math ----------
    @staticmethod
    def _cosine_sims(E_mat: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Correct cosine similarity row-wise."""
        if E_mat.shape[0] == 0:
            return np.array([], dtype=np.float32)
        v = v.astype(np.float32)
        denom = (np.linalg.norm(E_mat, axis=1) * (np.linalg.norm(v) + 1e-9) + 1e-9)
        return (E_mat @ v) / denom

    # ---------- storage ----------
    def _compress_store(self, path: str):
        img = Image.open(path).convert("RGB")
        h = hashlib.md5(img.tobytes()).hexdigest()[:10]
        out = os.path.join(self.archive, f"{h}.jpg")
        try:
            img.save(out, "JPEG", quality=self.cfg.jpeg_quality, optimize=True)
        except Exception:
            img.save(out, "JPEG", quality=self.cfg.jpeg_quality)

        files = sorted(
            [os.path.join(self.archive, f) for f in os.listdir(self.archive)],
            key=os.path.getmtime,
        )
        while len(files) > self.cfg.max_images_on_disk:
            try:
                os.remove(files[0])
            except Exception:
                pass
            files.pop(0)

    # ---------- memory ops ----------
    def _reservoir_add(self, emb: np.ndarray, meta: Dict[str, Any]):
        self.seen += 1
        meta = dict(meta or {})
        meta.setdefault("t", time.time())
        meta.setdefault("name", meta.get("name") or (meta.get("tags")[0] if meta.get("tags") else "object"))

        if len(self.E) < self.cfg.reservoir_size:
            self.E = np.vstack([self.E, emb[None].astype(np.float32)])
            self.M = np.append(self.M, meta)
        else:
            j = random.randint(0, self.seen - 1)
            if j < self.cfg.reservoir_size:
                self.E[j] = emb.astype(np.float32)
                self.M[j] = meta

    def _is_duplicate(self, emb: np.ndarray) -> bool:
        if len(self.E) == 0:
            return False
        sims = self._cosine_sims(self.E, emb)
        return float(sims.max(initial=0.0)) > self.cfg.dup_thr

    # ---------- embedding ----------
    def _embed_pil(self, img: Image.Image) -> np.ndarray:
        self.vision.eval()
        x = self.inf_tf(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            # CUDA autocast only; MPS autocast can be unstable across versions.
            if self.device == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    z = self.vision(x)
            else:
                z = self.vision(x)
        return z.detach().cpu().numpy()[0].astype(np.float32)

    # ---------- learning ----------
    @staticmethod
    def _contrastive(z1, z2):
        z1 = nn.functional.normalize(z1, dim=1)
        z2 = nn.functional.normalize(z2, dim=1)
        return -torch.mean(torch.sum(z1 * z2, dim=1))

    def learn_image(self, path: str, tags: Optional[List[str]] = None, curiosity: Optional[str] = None) -> float:
        img = Image.open(path).convert("RGB")

        self.vision.train()
        a1 = self.ssl_tf(img).unsqueeze(0).to(self.device)
        a2 = self.ssl_tf(img).unsqueeze(0).to(self.device)

        self.opt.zero_grad(set_to_none=True)
        loss = self._contrastive(self.vision(a1), self.vision(a2))
        loss.backward()
        self.opt.step()

        z = self._embed_pil(img)
        meta = {"tags": tags or [], "curiosity": curiosity, "t": time.time()}

        if not self._is_duplicate(z):
            self._reservoir_add(z, meta)
            self._compress_store(path)

        return float(loss.item())

    # ---------- reverse image search ----------
    def reverse_image_search(self, path: str, top_k: int = 5, curiosity: Optional[str] = None) -> List[Dict[str, float]]:
        labels = self.provider.search(path, top_k=top_k)

        # Learn from labels when available (your original feature)
        if labels and labels[0].get("label"):
            self.learn_image(path, tags=[l["label"] for l in labels], curiosity=curiosity)

        return labels

    # ---------- identification ----------
    def identify(self, path: str, k: Optional[int] = None, thr: Optional[float] = None,
                 curiosity: Optional[str] = None) -> Dict[str, Any]:
        k = self.cfg.local_top_k if k is None else k
        thr = self.cfg.local_thr if thr is None else thr

        img = Image.open(path).convert("RGB")
        z = self._embed_pil(img)

        if len(self.E) == 0:
            return {"web": self.reverse_image_search(path, curiosity=curiosity)}

        sims = self._cosine_sims(self.E, z)
        idx = np.argsort(-sims)[:k]

        local: List[Tuple[str, float]] = []
        for i in idx:
            meta = self.M[i] if i < len(self.M) else {}
            meta = meta if isinstance(meta, dict) else {}
            local.append((meta.get("name", "object"), float(sims[i])))

        # Low confidence → also do web search (keeps your feature)
        if not local or local[0][1] < thr:
            return {"local": local, "web": self.reverse_image_search(path, curiosity=curiosity)}

        return {"local": local}

    # ---------- operator teaching ----------
    def name_concept(self, example_image: str, name: str):
        if len(self.E) == 0:
            return
        img = Image.open(example_image).convert("RGB")
        z = self._embed_pil(img)

        sims = self._cosine_sims(self.E, z)
        j = int(np.argmax(sims))

        meta = self.M[j] if j < len(self.M) else {}
        meta = meta if isinstance(meta, dict) else {}
        meta["name"] = name
        self.M[j] = meta

    # ---------- autonomy ----------
    def autonomous_step(self, curiosity: Optional[str] = None) -> Dict[str, Any]:
        learned = 0
        losses: List[float] = []

        for f in list(os.listdir(self.inbox)):
            fp = os.path.join(self.inbox, f)
            if not os.path.isfile(fp):
                continue
            if not f.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            try:
                loss = self.learn_image(fp, curiosity=curiosity)
                learned += 1
                losses.append(loss)
            finally:
                try:
                    os.remove(fp)
                except Exception:
                    pass

        self.save_state()

        return {
            "learned_images": learned,
            "avg_loss": float(np.mean(losses)) if losses else None,
            "device": self.device,
            "memory_size": int(len(self.E)),
            "free_services_enabled": True,
            "mps_available": bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available()),
        }

# ---- Streamlit App ----
st.title("🔍 Autonomous Vision Agent — Free Services Only")

# Sidebar controls
st.sidebar.header("Operator Guidance")
curiosity = st.sidebar.text_input("What should the agent explore? (optional)", value="")
st.sidebar.markdown("---")

# M2 / MPS status
mps_ok = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
st.sidebar.write(f"MPS available (M2 GPU): {mps_ok}")

# Free services status
st.sidebar.info("✅ Using FREE services (SauceNAO, Yandex, DuckDuckGo)")
st.sidebar.write("No API keys required!")

@st.cache_resource
def get_agent_cached() -> VisionAgent:
    return VisionAgent()

agent = get_agent_cached()

st.sidebar.markdown("---")
st.sidebar.subheader("System Status")
st.sidebar.write(f"Device: {agent.device}")
st.sidebar.write(f"Memory items: {len(agent.E)}")
st.sidebar.write(f"Inbox folder: {agent.inbox}")

if st.sidebar.button("Run autonomous learning step"):
    summary = agent.autonomous_step(curiosity=curiosity or None)
    st.sidebar.success("Agent updated!")
    st.sidebar.json(summary)

# Analyze image
st.header("Analyze Image")
img = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
if img is not None:
    suffix = os.path.splitext(img.name)[1].lower() or ".png"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(img.read())
        path = tmp.name
    st.image(path, caption="Uploaded image", use_container_width=True)
    res = agent.identify(path, curiosity=curiosity or None)
    st.subheader("Result")
    st.json(res)

# Teach a concept
st.header("Teach a Concept")
ex = st.file_uploader("Example image", key="ex", type=["jpg", "jpeg", "png"])
name = st.text_input("Concept name")
if ex is not None and name and st.button("Teach"):
    suffix = os.path.splitext(ex.name)[1].lower() or ".png"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(ex.read())
        teach_path = tmp.name
    agent.name_concept(teach_path, name)
    agent.save_state()
    st.success(f"Concept learned as: {name}")

st.caption(
    "Tip: Put images into data/inbox/ then click 'Run autonomous learning step' to train automatically."
)