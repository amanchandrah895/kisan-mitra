"""Speech recognition with AI4Bharat IndicConformer (600M, multilingual, CTC + RNNT heads)."""
from __future__ import annotations

import subprocess
from pathlib import Path


def load_audio_16k_mono(path: str | Path):
    """Returns a (1, N) float32 torch tensor at 16 kHz. Converts mp3/other formats with ffmpeg."""
    import soundfile as sf
    import torch
    path = Path(path)
    if path.suffix.lower() != ".wav":
        wav_path = path.with_suffix(".16k.wav")
        if not wav_path.exists():
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(path), "-ar", "16000", "-ac", "1",
                            str(wav_path)], check=True)
        path = wav_path
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)  # (N, C)
    wav = torch.from_numpy(data.T.copy())                            # (C, N)
    wav = torch.mean(wav, dim=0, keepdim=True)
    if sr != 16000:
        import torchaudio
        wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=16000)
    return wav


class IndicConformerASR:
    def __init__(self, model_id: str = "ai4bharat/indic-conformer-600m-multilingual", device: str = "cuda",
                 token: str | None = None):
        from transformers import AutoModel
        # Loaded exactly as on the model card (no .to(device)): the custom code runs ONNX sessions and
        # manages its own execution provider, and it expects a CPU waveform tensor.
        self.model = AutoModel.from_pretrained(model_id, trust_remote_code=True, token=token)
        self.device = device

    def transcribe(self, audio, lang: str, decoding: str = "ctc") -> str:
        wav = audio if not isinstance(audio, (str, Path)) else load_audio_16k_mono(audio)
        out = self.model(wav, lang, decoding)
        if isinstance(out, (list, tuple)):
            out = out[0] if out else ""
        return str(out).strip()
