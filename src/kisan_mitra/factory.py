"""Builders for every component, so notebooks, the app and tests share one code path."""
from __future__ import annotations

from .corpus import chunk_documents, load_corpus
from .grounding import LexicalGrounder, NLIGrounder
from .hf_utils import get_device, gpu_free_gb, hf_token
from .llm import AgriParamLLM, ExtractiveLLM, Param1LLM
from .retriever import HybridRetriever
from .translate import IdentityTranslator, IndicTrans2Translator


def build_chunks(cfg):
    docs = load_corpus(cfg.path("corpus_dir"))
    return docs, chunk_documents(docs, cfg["retrieval"]["chunk_words"], cfg["retrieval"]["chunk_overlap"])


def build_retriever(cfg, chunks, neural: bool = True, device: str | None = None):
    device = device or get_device()
    m = cfg["models"]
    return HybridRetriever(chunks, embedder=m["embedder"] if neural else None,
                           reranker=m["reranker"] if neural else None, device=device,
                           candidate_k=cfg["retrieval"]["candidate_k"], rrf_k=cfg["retrieval"]["rrf_k"])


def build_translator(cfg, real: bool = True, device: str | None = None):
    if not real:
        return IdentityTranslator()
    t = cfg["translation"]
    return IndicTrans2Translator(cfg["models"]["indic_en"], cfg["models"]["en_indic"], device=device or get_device(),
                                 num_beams=t["num_beams"], max_length=t["max_length"], token=hf_token())


def build_grounder(cfg, real: bool = True, device: str | None = None):
    if not real:
        return LexicalGrounder()
    return NLIGrounder(cfg["models"]["nli"], device=device or get_device(),
                       threshold=cfg["grounding"]["entail_threshold"])


def load_llm(cfg, name: str, device: str | None = None, before_fp32_reload=None):
    g = cfg["generation"]
    if name == "extractive":
        return ExtractiveLLM()
    kw = dict(device=device or get_device(), dtype=g["dtype"], max_new_tokens=g["max_new_tokens"],
              repetition_penalty=g["repetition_penalty"], max_context_tokens=g["max_context_tokens"],
              token=hf_token(), before_fp32_reload=before_fp32_reload)
    if name == "agriparam":
        return AgriParamLLM(cfg["models"]["agriparam"], rag_template=g["agriparam_rag_template"],
                            closed_template=g["agriparam_closed_template"], **kw)
    if name == "param1":
        return Param1LLM(cfg["models"]["param1"], system_prompt=g["param1_system_prompt"], **kw)
    raise ValueError(name)


def load_asr(cfg, device: str | None = None):
    from .asr import IndicConformerASR
    return IndicConformerASR(cfg["models"]["asr"], device=device or get_device(), token=hf_token())


def offload_aux_to_cpu(agent):
    """Frees GPU memory for an fp32 LLM fallback by moving small models to CPU."""
    for obj in (agent.translator, agent.grounder):
        if hasattr(obj, "to"):
            obj.to("cpu")
    r = agent.retriever
    if getattr(r, "dense", None):
        r.dense.model.to("cpu")
    if getattr(r, "reranker", None):
        ce = r.reranker.model
        try:
            ce.to("cpu")             # sentence-transformers >= 4 (CrossEncoder is an nn.Module)
        except Exception:
            ce.model.to("cpu")       # older sentence-transformers
        if hasattr(ce, "_target_device"):
            import torch
            ce._target_device = torch.device("cpu")
    print(f"aux models moved to CPU; free GPU memory now {gpu_free_gb():.1f} GB")
