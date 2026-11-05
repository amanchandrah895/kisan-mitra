"""Translation between English and Indic languages with AI4Bharat IndicTrans2.

`IndicTrans2Translator` is the real model; `IdentityTranslator` and `LookupTranslator`
are used for CPU tests / dry runs."""
from __future__ import annotations

from .lang import IT2_CODES, split_sentences


class BaseTranslator:
    name = "base"

    def translate(self, texts: list[str], src: str, tgt: str) -> list[str]:
        raise NotImplementedError

    def __call__(self, text: str, src: str, tgt: str) -> str:
        if src == tgt or not text.strip():
            return text
        return self.translate([text], src, tgt)[0]

    def translate_long(self, text: str, src: str, tgt: str) -> str:
        """Sentence-split long passages (the model is trained on sentence pairs)."""
        if src == tgt or not text.strip():
            return text
        sents = split_sentences(text)
        if not sents:
            return text
        return " ".join(self.translate(sents, src, tgt))


class IdentityTranslator(BaseTranslator):
    name = "identity"

    def translate(self, texts, src, tgt):
        return list(texts)


class LookupTranslator(BaseTranslator):
    """Uses known reference translations (e.g. test-set question_en); identity otherwise."""
    name = "lookup"

    def __init__(self, table: dict[str, str]):
        self.table = table

    def translate(self, texts, src, tgt):
        return [self.table.get(t, t) for t in texts]


def _import_indic_processor():
    try:
        from IndicTransToolkit.processor import IndicProcessor  # HF model card path
        return IndicProcessor
    except Exception:
        pass
    try:
        from IndicTransToolkit import IndicProcessor  # README path (Cython build)
        return IndicProcessor
    except Exception:
        pass
    from IndicTransToolkit.IndicTransToolkit import IndicProcessor  # README fallback
    return IndicProcessor


class IndicTrans2Translator(BaseTranslator):
    name = "indictrans2"

    def __init__(self, indic_en_model: str, en_indic_model: str, device: str = "cuda",
                 num_beams: int = 5, max_length: int = 256, batch_size: int = 16, token: str | None = None):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        self.torch = torch
        self.device = device
        self.num_beams, self.max_length, self.batch_size = num_beams, max_length, batch_size
        self.ip = _import_indic_processor()(inference=True)
        dtype = torch.float16 if device.startswith("cuda") else torch.float32
        self.models, self.toks = {}, {}
        for direction, name in (("indic-en", indic_en_model), ("en-indic", en_indic_model)):
            # NOTE: no flash_attention_2 — not supported on T4 (Turing)
            self.toks[direction] = AutoTokenizer.from_pretrained(name, trust_remote_code=True, token=token)
            m = AutoModelForSeq2SeqLM.from_pretrained(name, trust_remote_code=True, torch_dtype=dtype, token=token)
            self.models[direction] = m.to(device).eval()
        self.cache: dict[tuple[str, str, str], str] = {}
        self.use_cache = True

    def to(self, device: str):
        for m in self.models.values():
            m.to(device)
        self.device = device
        return self

    def translate(self, texts, src, tgt):
        if src == tgt:
            return list(texts)
        direction = "indic-en" if tgt == "en" else "en-indic"
        if direction == "en-indic" and src != "en":
            # Indic -> Indic: pivot through English
            return self.translate(self.translate(texts, src, "en"), "en", tgt)
        out: list[str | None] = [self.cache.get((t, src, tgt)) for t in texts]
        todo = [i for i, o in enumerate(out) if o is None]
        tok, model = self.toks[direction], self.models[direction]
        s_code, t_code = IT2_CODES[src], IT2_CODES[tgt]
        for b in range(0, len(todo), self.batch_size):
            idx = todo[b:b + self.batch_size]
            batch = self.ip.preprocess_batch([texts[i] for i in idx], src_lang=s_code, tgt_lang=t_code)
            enc = tok(batch, truncation=True, padding="longest", max_length=self.max_length,
                      return_tensors="pt", return_attention_mask=True).to(self.device)
            gen_kw = dict(min_length=0, max_length=self.max_length, num_beams=self.num_beams,
                          num_return_sequences=1)
            with self.torch.inference_mode():
                try:
                    gen = model.generate(**enc, use_cache=self.use_cache, **gen_kw)
                except Exception:  # some transformers versions break the custom cache code
                    if not self.use_cache:
                        raise
                    self.use_cache = False
                    gen = model.generate(**enc, use_cache=False, **gen_kw)
            dec = tok.batch_decode(gen, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            dec = self.ip.postprocess_batch(dec, lang=t_code)
            for i, d in zip(idx, dec):
                out[i] = d
                self.cache[(texts[i], src, tgt)] = d
        return [o or "" for o in out]
