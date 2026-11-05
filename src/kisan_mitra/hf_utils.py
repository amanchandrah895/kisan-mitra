"""Device / dtype helpers, GPU memory cleanup and Hugging Face access checks."""
from __future__ import annotations

import gc
import os


def get_device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def pick_dtype(pref: str = "auto"):
    """fp16 on pre-Ampere GPUs (T4), bf16 on Ampere+, fp32 on CPU."""
    import torch
    if pref and pref != "auto":
        return {"fp16": torch.float16, "float16": torch.float16, "bf16": torch.bfloat16,
                "bfloat16": torch.bfloat16, "fp32": torch.float32, "float32": torch.float32}[pref]
    if not torch.cuda.is_available():
        return torch.float32
    major, _ = torch.cuda.get_device_capability()
    return torch.bfloat16 if major >= 8 else torch.float16


def free_memory(*objs) -> None:
    for o in objs:
        del o
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def gpu_free_gb() -> float:
    try:
        import torch
        if not torch.cuda.is_available():
            return 0.0
        free, _ = torch.cuda.mem_get_info()
        return free / 1e9
    except Exception:
        return 0.0


def _token_from_dotenv() -> str | None:
    from pathlib import Path
    env = Path(__file__).resolve().parents[2] / ".env"
    if not env.exists():
        return None
    for line in env.read_text(encoding="utf-8").splitlines():
        key, _, val = line.strip().partition("=")
        if key.strip() in ("HF_TOKEN", "LLM_TOKEN", "HUGGING_FACE_HUB_TOKEN") and val.strip():
            return val.strip().strip('"').strip("'")
    return None


def hf_token() -> str | None:
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if tok:
        return tok
    tok = _token_from_dotenv()  # project-root .env: HF_TOKEN=... or LLM_TOKEN=...
    if tok:
        os.environ["HF_TOKEN"] = tok
        return tok
    try:  # Colab secrets
        from google.colab import userdata  # type: ignore
        tok = userdata.get("HF_TOKEN")
        if tok:
            os.environ["HF_TOKEN"] = tok
        return tok
    except Exception:
        return None


def check_model_access(model_ids: list[str]) -> dict[str, str]:
    """Return {model_id: 'ok' | error message}. Gated models need accepted terms + HF_TOKEN."""
    from huggingface_hub import hf_hub_download, list_repo_files
    tok = hf_token()
    status = {}
    for mid in model_ids:
        try:
            files = list_repo_files(mid, token=tok)       # allowed even for gated repos
            probe = "config.json" if "config.json" in files else next(
                (f for f in files if f.endswith(".json")), files[0])
            hf_hub_download(mid, probe, token=tok)       # fails if terms not accepted
            status[mid] = "ok"
        except Exception as e:  # noqa: BLE001
            msg = str(e).splitlines()[0][:200]
            if "gated" in msg.lower() or "401" in msg or "403" in msg:
                msg = ("GATED - open https://huggingface.co/" + mid +
                       " , click 'Agree and access', and make sure HF_TOKEN is set. (" + msg + ")")
            status[mid] = msg
    return status


def env_report() -> dict:
    info = {}
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["gpu_mem_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
    except ImportError:
        pass
    for mod in ("transformers", "sentence_transformers", "IndicTransToolkit", "gradio", "onnxruntime"):
        try:
            m = __import__(mod)
            info[mod] = getattr(m, "__version__", "installed")
        except Exception:
            info[mod] = "missing"
    return info
