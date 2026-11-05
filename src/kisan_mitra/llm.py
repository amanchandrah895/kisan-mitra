"""Answer generators.

- AgriParamLLM : BharatGen AgriParam (domain LLM) with its native <context>/<user>/<assistant> format
- Param1LLM    : BharatGen Param-1-2.9B-Instruct (general LLM) via its chat template
- ExtractiveLLM: no neural generation; picks the best-matching sentences from the context
                 (a strong, cheap baseline and the CPU stand-in used by the tests)
"""
from __future__ import annotations

import re
import time

from .lang import split_sentences
from .retriever import tokenize

_STOP_MARKERS = ("<user>", "<context>", "<assistant>", "<system_prompt>", "<|endoftext|>", "</s>")


def clean_answer(text: str, max_sentences: int = 6) -> str:
    text = text or ""
    for m in _STOP_MARKERS:
        if m in text:
            text = text.split(m)[0]
    text = re.sub(r"^\s*(answer|assistant)\s*[:\-]\s*", "", text.strip(), flags=re.I)
    sents, seen = [], set()
    for s in split_sentences(text):
        key = re.sub(r"\W+", " ", s.lower()).strip()
        if key and key not in seen:
            seen.add(key)
            sents.append(s)
    if len(sents) > 1 and not re.search(r"[.!?।)\]\"']$", sents[-1]):
        sents = sents[:-1]  # drop a truncated trailing sentence
    return " ".join(sents[:max_sentences]).strip()


class BaseLLM:
    name = "base"
    supports_context = True

    def generate(self, question: str, contexts: list[str] | None = None) -> str:
        raise NotImplementedError

    def unload(self):
        pass


class ExtractiveLLM(BaseLLM):
    name = "extractive"

    def __init__(self, n_sentences: int = 2):
        self.n = n_sentences

    def generate(self, question, contexts=None):
        if not contexts:
            return "I do not know."
        q = set(tokenize(question))
        cands = []
        for rank, ctx in enumerate(contexts):
            body = re.sub(r"^\[[^\]]*\]\s*", "", ctx)
            for s in split_sentences(body):
                toks = set(tokenize(s))
                overlap = len(q & toks)
                has_num = 1 if re.search(r"\d", s) else 0
                cands.append((overlap + 0.3 * has_num - 0.05 * rank, s))
        cands.sort(key=lambda x: -x[0])
        picked = [s for _, s in cands[: self.n]]
        return " ".join(picked) if picked else "I do not know."


class HFCausalLM(BaseLLM):
    """Shared loading / generation logic for the BharatGen Param-family models."""

    def __init__(self, model_id: str, device: str = "cuda", dtype: str = "auto", max_new_tokens: int = 200,
                 repetition_penalty: float = 1.1, max_context_tokens: int = 1100, token: str | None = None,
                 before_fp32_reload=None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from .hf_utils import pick_dtype
        self.torch = torch
        self.model_id, self.device = model_id, device
        self.max_new_tokens, self.repetition_penalty = max_new_tokens, repetition_penalty
        self.max_context_tokens = max_context_tokens
        try:
            self.tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=False, token=token)
        except Exception:
            self.tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, token=token)
        self.dtype = pick_dtype(dtype)
        self.model = self._load(AutoModelForCausalLM, self.dtype, token)
        self.use_cache = True
        if not self._finite_check():
            print(f"[{model_id}] non-finite logits in {self.dtype}; reloading in float32")
            del self.model
            from .hf_utils import free_memory
            free_memory()
            if before_fp32_reload is not None:
                before_fp32_reload()  # e.g. move auxiliary models to CPU to make room (fp32 3B = ~12 GB)
            self.dtype = torch.float32
            self.model = self._load(AutoModelForCausalLM, self.dtype, token)
        self.last_stats: dict = {}

    def _load(self, cls, dtype, token):
        kw = dict(trust_remote_code=True, torch_dtype=dtype, token=token, low_cpu_mem_usage=True)
        if self.device.startswith("cuda"):
            kw["device_map"] = {"": 0}
        m = cls.from_pretrained(self.model_id, **kw)
        if not self.device.startswith("cuda"):
            m = m.to(self.device)
        return m.eval()

    def _finite_check(self) -> bool:
        ids = self.tok("Wheat is sown in", return_tensors="pt").input_ids.to(self.model.device)
        with self.torch.inference_mode():
            logits = self.model(input_ids=ids).logits
        return bool(self.torch.isfinite(logits).all())

    def _fit_contexts(self, contexts: list[str]) -> str:
        kept, used = [], 0
        for c in contexts:
            n = len(self.tok(c, add_special_tokens=False).input_ids)
            if used + n > self.max_context_tokens:
                room = self.max_context_tokens - used
                if room > 60:
                    ids = self.tok(c, add_special_tokens=False).input_ids[:room]
                    kept.append(self.tok.decode(ids, skip_special_tokens=True))
                break
            kept.append(c)
            used += n
        return "\n".join(kept)

    def _generate_ids(self, input_ids, attention_mask=None) -> str:
        torch = self.torch
        input_ids = input_ids.to(self.model.device)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        attention_mask = attention_mask.to(self.model.device)
        kw = dict(max_new_tokens=self.max_new_tokens, do_sample=False,
                  repetition_penalty=self.repetition_penalty, eos_token_id=self.tok.eos_token_id,
                  pad_token_id=self.tok.pad_token_id if self.tok.pad_token_id is not None else self.tok.eos_token_id)
        t0 = time.time()
        with torch.inference_mode():
            try:
                out = self.model.generate(input_ids=input_ids, attention_mask=attention_mask,
                                          use_cache=self.use_cache, **kw)
            except Exception as e:  # the model cards use use_cache=False; fall back if the cache path breaks
                if not self.use_cache:
                    raise
                print(f"[{self.model_id}] generate(use_cache=True) failed ({type(e).__name__}: {e}); "
                      "retrying with use_cache=False")
                self.use_cache = False
                out = self.model.generate(input_ids=input_ids, attention_mask=attention_mask,
                                          use_cache=False, **kw)
        gen = out[0][input_ids.shape[-1]:]
        self.last_stats = {"prompt_tokens": int(input_ids.shape[-1]), "new_tokens": int(gen.shape[-1]),
                           "gen_seconds": round(time.time() - t0, 2), "use_cache": self.use_cache}
        return self.tok.decode(gen, skip_special_tokens=True)

    def unload(self):
        from .hf_utils import free_memory
        self.model = None
        free_memory()


class AgriParamLLM(HFCausalLM):
    name = "agriparam"

    def __init__(self, *a, rag_template: str = "<context> {context} <user> {question} <assistant>",
                 closed_template: str = "<user> {question} <assistant>", **kw):
        super().__init__(*a, **kw)
        self.rag_template, self.closed_template = rag_template, closed_template

    def generate(self, question, contexts=None):
        if contexts:
            prompt = self.rag_template.format(context=self._fit_contexts(contexts), question=question)
        else:
            prompt = self.closed_template.format(question=question)
        enc = self.tok(prompt, return_tensors="pt")
        return clean_answer(self._generate_ids(enc.input_ids, enc.get("attention_mask")))


class Param1LLM(HFCausalLM):
    name = "param1"

    def __init__(self, *a, system_prompt: str = "You are a helpful assistant.", **kw):
        super().__init__(*a, **kw)
        self.system_prompt = system_prompt

    def generate(self, question, contexts=None):
        user = (f"Context:\n{self._fit_contexts(contexts)}\n\nQuestion: {question}" if contexts
                else question)
        conv = [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": user}]
        ids = self.tok.apply_chat_template(conversation=conv, return_tensors="pt", add_generation_prompt=True)
        if isinstance(ids, dict) or hasattr(ids, "input_ids"):
            ids = ids["input_ids"]
        return clean_answer(self._generate_ids(ids))
