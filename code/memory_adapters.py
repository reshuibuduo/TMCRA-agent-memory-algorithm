from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path
import re
import tempfile
import time
import uuid
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from core.session_memory import SessionMemoryExtractor
from experiments.replacement.memory_graph import (
    SessionMemoryGraphV2,
    SessionMemoryEdgeV2,
    SessionMemoryRecordV2,
    SQLiteSessionMemoryStore,
    _clean_text,
    _estimate_tokens,
    _normalize,
    _public_query_subject,
    _public_slot_root,
    _public_subject_signature,
    stable_slot_key,
    _tokenize,
    guess_slot_key,
    infer_category_hints,
)
from experiments.replacement.node_memory import (
    _call_with_supported_kwargs,
    DEFAULT_MATRIX_EVENT_TOP_K,
    LoadedNodeMemoryScorer,
    MEMORY_ROUTER_LAYERS,
    build_default_path_templates,
    extract_question_features,
)
from experiments.replacement.public_event_signature import compute_public_event_signature
from experiments.replacement.memory_profiles import TMCRAProfile
from experiments.replacement.typed_tunnel_augmentation import (
    annotate_memory_record,
    merge_typed_metadata,
    typed_edge_tags_between,
    typed_tunnel_signature_text,
)
from experiments.replacement.profile_layer import infer_profile_query_intent, is_profile_layer_record, profile_query_score_delta
from experiments.replacement import injection_planner as injection_planner_runtime
from experiments.replacement.temporal_modeling_types import TemporalFrame, TemporalQueryPlan
from experiments.replacement.temporal_organizer import TemporalOrganizer
from experiments.replacement.temporal_query_planner import TemporalQueryPlanner
from experiments.replacement.temporal_router_runtime import LoadedTemporalRouter
from experiments.replacement.timeline_evidence_pack import TimelineEvidencePackBuilder
from experiments.replacement.timeline_state_layer import TimelineStateLayer

from .base import MemoryAdapter, MemoryHit, MemoryRetrieval


def _dedupe(items: Iterable[Any], *, max_items: int | None = None) -> List[str]:
    values: List[str] = []
    seen = set()
    for item in items:
        text = _clean_text(item)
        if not text:
            continue
        key = _normalize(text)
        if key in seen:
            continue
        seen.add(key)
        values.append(text)
        if max_items is not None and len(values) >= max_items:
            break
    return values


def _apply_typed_tunnel_annotations(records: List[SessionMemoryRecordV2], *, source_text: str = "") -> List[SessionMemoryRecordV2]:
    for record in records:
        annotate_memory_record(record, source_text=source_text)
    return records


def _estimate_tokens_from_hits(hits: Sequence[MemoryHit]) -> int:
    total = 0
    for hit in hits:
        total += _estimate_tokens(hit.value)
        total += sum(_estimate_tokens(anchor) for anchor in hit.anchors)
    return total


def _float_env(name: str, default: float) -> float:
    raw = _clean_text(os.getenv(name, ""))
    if not raw:
        return float(default)
    try:
        return float(raw)
    except ValueError:
        return float(default)


_GRAPH_PROMPT_MAX_CHARS = 12_000
_GRAPH_PROMPT_MAX_HITS = 8
_GRAPH_PROMPT_MAX_ACTIVE_SLOTS = 12
_GRAPH_PROMPT_MAX_RELATIONS = 10
_HYBRID_SELECTED_EVENT_FLOOR = 8
_HYBRID_SELECTED_PATH_CAP = 6
_HYBRID_TEMPORAL_PATH_CAP = 2
_HYBRID_PROFILE_PATH_CAP = 3
_MEMORY_ROUTER_GUIDED_MODES = {"guided", "route", "routing", "enforce"}
_MEMORY_ROUTER_FORCE_MODES = {"force", "forced"}
_MEMORY_ROUTER_DISABLED_MODES = {"", "off", "disabled", "none", "false", "0"}
_MEMORY_ROUTER_OBSERVE_MODES = {"observe", "observer", "telemetry", "shadow"}
_MEMORY_ROUTER_DEFAULT_THRESHOLD = 0.55
_MEMORY_ROUTER_DEFAULT_MARGIN = 0.08
_INJECTION_PLANNER_GUIDED_MODES = {"guided", "route", "routing", "enforce"}
_INJECTION_PLANNER_FORCE_MODES = {"force", "forced"}
_INJECTION_PLANNER_DISABLED_MODES = {"", "off", "disabled", "none", "false", "0"}
_INJECTION_PLANNER_OBSERVE_MODES = {"observe", "observer", "telemetry", "shadow"}
_TEMPORAL_LAYER_DISABLED_MODES = {"off", "disabled", "none", "false", "0"}
_TEMPORAL_ROUTER_DEFAULT_WRITER_MIN_CONFIDENCE = 0.72
_TEMPORAL_ROUTER_DEFAULT_QUERY_MIN_CONFIDENCE = 0.85
_TEMPORAL_ROUTER_DEFAULT_QUERY_INTENT_MIN_CONFIDENCE = 0.60
_EMBEDDER_INDEX_DISABLED_MODES = {"", "off", "disabled", "none", "false", "0"}
_EMBEDDER_INDEX_BGE_M3_MODES = {"bge", "bge_m3", "bge-m3", "baai_bge_m3", "baai/bge-m3"}
_EMBEDDER_INDEX_VERSION = "write_hash_sparse_v1"
_EMBEDDER_INDEX_BGE_M3_VERSION = "write_bge_m3_dense_sparse_v1"
_EMBEDDER_MODEL_CACHE: Dict[str, Any] = {}
_HYBRID_SYMBOLIC_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "did",
    "do",
    "does",
    "event",
    "for",
    "from",
    "had",
    "has",
    "have",
    "how",
    "in",
    "is",
    "of",
    "on",
    "or",
    "should",
    "the",
    "to",
    "turn",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "will",
    "with",
}


def _coerce_memory_router_scores(payload: Mapping[str, Any]) -> Dict[str, float]:
    raw_scores = dict(payload.get("memory_router_scores", {}) or {})
    scores: Dict[str, float] = {}
    for layer in MEMORY_ROUTER_LAYERS:
        try:
            scores[layer] = float(raw_scores.get(layer, 0.0))
        except (TypeError, ValueError):
            scores[layer] = 0.0
    return scores


def _memory_router_decision(
    payload: Mapping[str, Any],
    *,
    mode: str,
    threshold: float,
    margin: float,
) -> Dict[str, Any]:
    normalized_mode = _normalize(mode)
    scores = _coerce_memory_router_scores(payload)
    has_scores = bool(scores) and any(layer in dict(payload.get("memory_router_scores", {}) or {}) for layer in MEMORY_ROUTER_LAYERS)
    if not has_scores:
        return {
            "memory_router_enabled": False,
            "memory_router_guided": False,
            "memory_router_reason": "no_router_scores",
            "memory_router_scores": {},
            "memory_router_top_layers": [],
            "memory_router_active_layers": [],
            "memory_router_score_spread": 0.0,
            "memory_router_confidence": 0.0,
        }
    ranked_layers = [
        layer
        for layer, _ in sorted(
            scores.items(),
            key=lambda item: (-float(item[1]), item[0]),
        )
    ]
    score_values = list(scores.values())
    score_spread = max(score_values) - min(score_values)
    confidence = max(abs(float(score) - 0.5) for score in score_values)
    resolved_threshold = max(0.0, min(1.0, float(threshold or _MEMORY_ROUTER_DEFAULT_THRESHOLD)))
    resolved_margin = max(0.0, min(0.5, float(margin or _MEMORY_ROUTER_DEFAULT_MARGIN)))
    active_layers = [
        layer
        for layer in ranked_layers
        if float(scores.get(layer, 0.0)) >= resolved_threshold
    ]
    confident = bool(active_layers) and (score_spread >= resolved_margin or confidence >= resolved_margin)
    guided_requested = normalized_mode in _MEMORY_ROUTER_GUIDED_MODES
    forced = normalized_mode in _MEMORY_ROUTER_FORCE_MODES
    guided = bool(forced or (guided_requested and confident))
    if forced and not active_layers and ranked_layers:
        active_layers = [ranked_layers[0]]
    if active_layers and "event" not in active_layers:
        active_layers = ["event", *active_layers]
    reason = "observe"
    if normalized_mode in _MEMORY_ROUTER_DISABLED_MODES:
        reason = "disabled"
    elif normalized_mode in _MEMORY_ROUTER_OBSERVE_MODES:
        reason = "observe"
    elif forced:
        reason = "forced"
    elif guided_requested and not confident:
        reason = "low_confidence"
    elif guided:
        reason = "guided"
    return {
        "memory_router_enabled": True,
        "memory_router_guided": bool(guided and normalized_mode not in _MEMORY_ROUTER_DISABLED_MODES),
        "memory_router_reason": reason,
        "memory_router_scores": scores,
        "memory_router_top_layers": ranked_layers,
        "memory_router_active_layers": active_layers,
        "memory_router_score_spread": round(float(score_spread), 6),
        "memory_router_confidence": round(float(confidence), 6),
        "memory_router_threshold": round(float(resolved_threshold), 6),
        "memory_router_margin": round(float(resolved_margin), 6),
        "memory_router_mode": normalized_mode or "observe",
    }


def _memory_router_allows(decision: Mapping[str, Any], *layers: str) -> bool:
    if not bool(decision.get("memory_router_guided")):
        return True
    requested = {_normalize(layer) for layer in layers if _normalize(layer)}
    if not requested:
        return True
    if "event" in requested:
        return True
    active = {_normalize(layer) for layer in list(decision.get("memory_router_active_layers", []) or [])}
    return bool(active & requested)


def _hybrid_symbolic_tokens(value: Any) -> List[str]:
    return [
        token
        for token in _tokenize(value)
        if token and token not in _HYBRID_SYMBOLIC_STOPWORDS and not re.fullmatch(r"\d+", token)
    ]


_PATH_UTILITY_STOPWORDS = set(_HYBRID_SYMBOLIC_STOPWORDS) | {
    "actually",
    "also",
    "bit",
    "feel",
    "feels",
    "get",
    "guess",
    "i",
    "im",
    "it",
    "its",
    "just",
    "kind",
    "like",
    "maybe",
    "me",
    "more",
    "much",
    "my",
    "really",
    "something",
    "still",
    "think",
    "thats",
    "there",
    "this",
    "way",
    "would",
}


def _path_utility_tokens(value: Any) -> List[str]:
    return [
        token
        for token in _hybrid_symbolic_tokens(value)
        if token and token not in _PATH_UTILITY_STOPWORDS
    ]


_PROFILE_QUERY_ALIAS_GROUPS: tuple[set[str], ...] = (
    {
        "accessory",
        "accessories",
        "bag",
        "camera",
        "cameras",
        "equipment",
        "flash",
        "gear",
        "lens",
        "lenses",
        "photo",
        "photography",
        "sony",
        "tripod",
    },
    {
        "app",
        "apps",
        "dashboard",
        "interface",
        "layout",
        "panel",
        "software",
        "tool",
        "tools",
        "ui",
        "workflow",
    },
    {
        "diet",
        "drink",
        "food",
        "meal",
        "restaurant",
        "snack",
        "taste",
    },
    {
        "background",
        "career",
        "job",
        "occupation",
        "position",
        "previous",
        "profession",
        "role",
        "worked",
        "work",
    },
)


_PROFILE_QUERY_GENERIC_TOKENS = set(_PATH_UTILITY_STOPWORDS) | {
    "able",
    "about",
    "any",
    "anything",
    "based",
    "best",
    "can",
    "complement",
    "could",
    "current",
    "give",
    "help",
    "information",
    "looking",
    "make",
    "need",
    "please",
    "recommend",
    "recommendation",
    "recommendations",
    "should",
    "some",
    "suggest",
    "suggestion",
    "suggestions",
    "tell",
    "that",
    "using",
    "you",
}


def _profile_query_expanded_tokens(value: Any) -> set[str]:
    tokens = set(_path_utility_tokens(value))
    expanded = set(tokens)
    for group in _PROFILE_QUERY_ALIAS_GROUPS:
        if tokens & group:
            expanded.update(group)
    return expanded


def _profile_specific_tokens(tokens: Iterable[Any]) -> set[str]:
    return {
        _normalize(token)
        for token in tokens
        if _clean_text(token)
        and _normalize(token) not in _PROFILE_QUERY_GENERIC_TOKENS
        and not re.fullmatch(r"\d+", _normalize(token))
    }


def _profile_hit_match_score(query_tokens: set[str], expanded_query_tokens: set[str], hit: MemoryHit) -> tuple[float, List[str], List[str]]:
    metadata = dict(hit.metadata or {})
    payload_parts: List[Any] = [
        hit.category,
        hit.relation,
        hit.slot_key,
        hit.value,
        *list(hit.anchors or []),
        metadata.get("profile_summary", ""),
        metadata.get("profile_value", ""),
        metadata.get("profile_type", ""),
        metadata.get("profile_domain", ""),
        metadata.get("profile_domain_label", ""),
        metadata.get("semantic_slot", ""),
        metadata.get("subject", ""),
        metadata.get("extracted_subject", ""),
        metadata.get("profile_cluster_domains", []),
        metadata.get("profile_cluster_types", []),
        metadata.get("profile_cluster_route_terms", []),
        metadata.get("profile_route_terms", []),
        metadata.get("profile_support_values", []),
    ]
    payload_text = " ".join(str(item) for item in payload_parts)
    record_raw_tokens = set(_path_utility_tokens(payload_text))
    record_tokens = _profile_query_expanded_tokens(payload_text)
    specific_query_tokens = _profile_specific_tokens(query_tokens)
    raw_overlap_tokens = sorted(specific_query_tokens & _profile_specific_tokens(record_raw_tokens))
    expanded_overlap_tokens = sorted(_profile_specific_tokens(expanded_query_tokens) & _profile_specific_tokens(record_tokens))
    overlap_tokens = raw_overlap_tokens or expanded_overlap_tokens
    overlap_ratio = float(len(overlap_tokens)) / float(max(1, len(specific_query_tokens)))
    profile_type = _normalize(metadata.get("profile_type", ""))
    source_kind = _normalize(hit.source_kind)
    source_bonus = 0.0
    if source_kind in {"public_dialog_preference", "public_dialog_goal", "public_dialog_profile"}:
        source_bonus += 0.08
    if bool(metadata.get("profile_candidate_status") == "consolidated"):
        source_bonus += 0.04
    if bool(metadata.get("profile_cluster_node")):
        source_bonus -= 0.16
    type_bonus = 0.0
    if profile_type in {"preference", "goal", "setup", "usage_context"} and (
        {"recommend", "suggest", "suited", "accessory", "accessories", "gear", "equipment", "setup", "current"} & query_tokens
    ):
        type_bonus += 0.10
    if profile_type in {"setup", "usage_context"} and {"current", "setup", "profile"} & query_tokens:
        type_bonus += 0.08
    raw_bonus = 0.18 * len(raw_overlap_tokens)
    expanded_bonus = 0.08 * max(0, len(expanded_overlap_tokens) - len(raw_overlap_tokens))
    score = raw_bonus + expanded_bonus + (0.58 * overlap_ratio) + source_bonus + type_bonus
    return round(score, 6), overlap_tokens, raw_overlap_tokens


def _bounded_event_id_union(*groups: Iterable[Any], max_items: int) -> List[str]:
    return _dedupe((item for group in groups for item in group), max_items=max(1, int(max_items)))


def _symbolic_recall_event_ids(
    query: str,
    runtime_graph: Mapping[str, Any],
    *,
    grouped_hits: Mapping[str, Sequence[MemoryHit]],
    limit: int,
) -> List[str]:
    question_features = extract_question_features(query)
    query_tokens = set(_hybrid_symbolic_tokens(question_features.get("question_anchor_tokens", []) or query))
    if not query_tokens:
        return []
    nodes_by_id = {
        _clean_text(node.get("id", "")): dict(node)
        for node in list(runtime_graph.get("nodes", []) or [])
        if _clean_text(node.get("id", ""))
    }
    event_payloads: Dict[str, List[str]] = {}
    for node_id, node in nodes_by_id.items():
        if _clean_text(node.get("type", "")) == "event":
            metadata = dict(node.get("metadata", {}) or {})
            teacher_fields = dict(node.get("teacher_fields", {}) or {})
            event_payloads.setdefault(node_id, []).extend(
                [
                    node.get("text", ""),
                    node.get("speaker", ""),
                    node.get("slot_key", ""),
                    node.get("target_status", ""),
                    node.get("profile_type", ""),
                    node.get("profile_value", ""),
                    *teacher_fields.values(),
                    *metadata.values(),
                ]
            )
    for path in list(runtime_graph.get("paths", []) or []):
        event_id = _clean_text(path.get("event_id", ""))
        support_node = nodes_by_id.get(_clean_text(path.get("target", "")), {})
        if event_id and support_node:
            event_payloads.setdefault(event_id, []).append(support_node.get("text", ""))
    for event_id, group_hits in grouped_hits.items():
        payloads = event_payloads.setdefault(_clean_text(event_id), [])
        for hit in list(group_hits or []):
            metadata = dict(hit.metadata or {})
            payloads.extend([hit.value, hit.slot_key, hit.category, hit.relation, *hit.anchors, *metadata.values()])

    scored_events: List[tuple[str, float]] = []
    for event_id, payloads in event_payloads.items():
        event_tokens = set(_hybrid_symbolic_tokens(payloads))
        overlap = query_tokens & event_tokens
        if not overlap:
            continue
        overlap_ratio = float(len(overlap)) / float(max(1, len(query_tokens)))
        event_node = nodes_by_id.get(event_id, {})
        turn_index = int(event_node.get("turn_index", 0) or 0)
        scored_events.append((event_id, (len(overlap) * 4.0) + overlap_ratio + min(turn_index, 1000) * 0.000001))
    return [
        event_id
        for event_id, _ in sorted(scored_events, key=lambda item: (-float(item[1]), item[0]))
    ][: max(1, int(limit))]


_EMBEDDER_INDEX_METADATA_TEXT_KEYS = (
    "raw_text",
    "source_turn_text",
    "source_span",
    "event_phrase",
    "event_summary",
    "profile_value",
    "profile_summary",
    "profile_type",
    "profile_domain",
    "profile_domain_label",
    "semantic_slot",
    "target_status",
    "subject",
    "subject_signature",
    "canonical_slot_key",
    "resource_key",
    "resolved_date",
    "resolved_time_value",
    "time_value",
    "time_display_value",
    "time_granularity",
    "speaker",
    "session_name",
    "topic_label",
    "topic_bucket_id",
    "origin_query",
    "writeback_class",
    "depth_layer",
    "memory_chain_depth_layer",
)
_EMBEDDER_INDEX_METADATA_LIST_KEYS = (
    "topic_keywords",
    "profile_route_terms",
    "profile_cluster_route_terms",
    "profile_support_values",
    "evidence_anchors",
    "support_memory_ids",
    "support_fact_refs",
    "support_path_refs",
)


def _embedder_index_enabled(mode: Any) -> bool:
    return _normalize(mode) not in _EMBEDDER_INDEX_DISABLED_MODES


def _embedder_index_uses_bge_m3(mode: Any) -> bool:
    return _normalize(mode).replace("-", "_") in {item.replace("-", "_") for item in _EMBEDDER_INDEX_BGE_M3_MODES}


def _embedder_index_version_for_mode(mode: Any) -> str:
    return _EMBEDDER_INDEX_BGE_M3_VERSION if _embedder_index_uses_bge_m3(mode) else _EMBEDDER_INDEX_VERSION


def _embedder_index_text_items(value: Any, *, max_items: int = 64) -> List[str]:
    items: List[str] = []

    def visit(item: Any) -> None:
        if len(items) >= max_items:
            return
        if item is None:
            return
        if isinstance(item, Mapping):
            for key, nested in list(item.items()):
                if len(items) >= max_items:
                    break
                key_text = _clean_text(key)
                if isinstance(nested, (str, int, float, bool)):
                    value_text = _clean_text(nested)[:800]
                    if value_text:
                        items.append(f"{key_text} {value_text}".strip())
                else:
                    visit(nested)
            return
        if isinstance(item, (list, tuple, set)):
            for nested in list(item):
                if len(items) >= max_items:
                    break
                visit(nested)
            return
        text = _clean_text(item)[:800]
        if text:
            items.append(text)

    visit(value)
    return items[:max_items]


def _embedder_index_term_weights(value: Any, *, max_terms: int = 96) -> Dict[str, float]:
    text = _clean_text(value)
    if not text:
        return {}
    counts: Dict[str, float] = {}
    for token in _path_utility_tokens(text):
        token = _normalize(token)
        if not token or len(token) > 64:
            continue
        counts[token] = counts.get(token, 0.0) + 1.0
    normalized_text = _normalize(text)
    cjk_chars = [char for char in normalized_text if "\u4e00" <= char <= "\u9fff"]
    for width, weight in ((2, 1.35), (3, 1.15)):
        if len(cjk_chars) < width:
            continue
        for index in range(0, len(cjk_chars) - width + 1):
            gram = "".join(cjk_chars[index : index + width])
            if gram:
                counts[gram] = counts.get(gram, 0.0) + weight
    if not counts:
        return {}
    ranked = sorted(counts.items(), key=lambda item: (-float(item[1]), item[0]))[: max(1, int(max_terms or 1))]
    norm = math.sqrt(sum(float(weight) * float(weight) for _, weight in ranked)) or 1.0
    return {term: round(float(weight) / norm, 6) for term, weight in ranked}


def _embedder_dense_vectors_for_texts(texts: Sequence[str], *, mode: str) -> tuple[List[List[float]], Dict[str, Any]]:
    normalized_mode = _normalize(mode)
    if not _embedder_index_uses_bge_m3(normalized_mode):
        return [[] for _ in texts], {"write_embedder_dense_enabled": False}
    clean_texts = [_clean_text(text) for text in texts]
    metadata: Dict[str, Any] = {
        "write_embedder_dense_enabled": False,
        "write_embedder_dense_backend": "bge_m3_transformers",
        "write_embedder_dense_model": _clean_text(os.getenv("TMCRA_EMBEDDER_MODEL_PATH", "")) or "BAAI/bge-m3",
    }
    if not any(clean_texts):
        metadata["write_embedder_dense_error"] = "empty_texts"
        return [[] for _ in texts], metadata
    try:
        import torch  # type: ignore
        from transformers import AutoModel, AutoTokenizer  # type: ignore
    except Exception as exc:
        metadata["write_embedder_dense_error"] = f"dependency_unavailable:{exc.__class__.__name__}"
        return [[] for _ in texts], metadata
    model_name = metadata["write_embedder_dense_model"]
    device = _clean_text(os.getenv("TMCRA_EMBEDDER_DEVICE", ""))
    if not device:
        device = "cuda" if bool(getattr(torch, "cuda", None) and torch.cuda.is_available()) else "cpu"
    try:
        max_length = max(64, int(os.getenv("TMCRA_EMBEDDER_MODEL_MAX_LENGTH", "512") or 512))
    except (TypeError, ValueError):
        max_length = 512
    cache_key = f"bge_m3::{model_name}::{device}::{max_length}"
    try:
        cached = _EMBEDDER_MODEL_CACHE.get(cache_key)
        if cached is None:
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
            model.to(device)
            model.eval()
            cached = (tokenizer, model)
            _EMBEDDER_MODEL_CACHE[cache_key] = cached
        tokenizer, model = cached
        encoded = tokenizer(
            clean_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            output = model(**encoded)
            hidden = output.last_hidden_state
            mask = encoded.get("attention_mask")
            if mask is not None:
                mask = mask.unsqueeze(-1).expand(hidden.size()).float()
                pooled = torch.sum(hidden * mask, dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)
            else:
                pooled = hidden[:, 0]
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        vectors = [
            [round(float(value), 6) for value in row.detach().cpu().tolist()]
            for row in pooled
        ]
        metadata.update(
            {
                "write_embedder_dense_enabled": True,
                "write_embedder_dense_device": device,
                "write_embedder_dense_dim": int(len(vectors[0]) if vectors else 0),
                "write_embedder_dense_max_length": int(max_length),
            }
        )
        return vectors, metadata
    except Exception as exc:
        metadata["write_embedder_dense_error"] = f"{exc.__class__.__name__}:{_clean_text(exc)[:240]}"
        return [[] for _ in texts], metadata


def _prewarm_embedder_dense_if_requested(*, mode: str) -> Dict[str, Any]:
    flag = _normalize(os.getenv("TMCRA_EMBEDDER_PREWARM", ""))
    if flag in _EMBEDDER_INDEX_DISABLED_MODES or flag not in {"1", "true", "yes", "on", "auto"}:
        return {"embedder_prewarm_enabled": False}
    normalized_mode = _normalize(mode)
    if not _embedder_index_uses_bge_m3(normalized_mode):
        return {
            "embedder_prewarm_enabled": False,
            "embedder_prewarm_reason": "mode_not_dense",
            "embedder_prewarm_mode": normalized_mode or "off",
        }
    warmup_text = _clean_text(os.getenv("TMCRA_EMBEDDER_PREWARM_TEXT", "")) or "tmcra memory retrieval warmup"
    vectors, metadata = _embedder_dense_vectors_for_texts([warmup_text], mode=normalized_mode)
    return {
        "embedder_prewarm_enabled": bool(vectors and vectors[0]),
        "embedder_prewarm_mode": normalized_mode,
        "embedder_prewarm_dense_enabled": bool(metadata.get("write_embedder_dense_enabled")),
        "embedder_prewarm_dense_device": metadata.get("write_embedder_dense_device", ""),
        "embedder_prewarm_dense_error": metadata.get("write_embedder_dense_error", ""),
    }


def _embedder_index_record_text(record: SessionMemoryRecordV2, *, turn_text: str = "") -> str:
    metadata = dict(record.metadata or {})
    parts: List[str] = [
        record.category,
        record.slot_key,
        record.value,
        record.value,
        record.value,
        record.relation,
        *list(record.anchor_concepts or []),
        *list(record.anchor_concepts or []),
        *list(record.evidence_anchors or []),
    ]
    has_record_evidence_text = any(
        _clean_text(metadata.get(key, ""))
        for key in ("raw_text", "source_turn_text", "source_span", "event_phrase", "profile_value", "profile_summary")
    )
    if turn_text and not has_record_evidence_text:
        parts.append(_clean_text(turn_text)[:1000])
    for key in _EMBEDDER_INDEX_METADATA_TEXT_KEYS:
        value = metadata.get(key)
        if value:
            parts.append(f"{key} {' '.join(_embedder_index_text_items(value, max_items=12))}".strip())
    for key in _EMBEDDER_INDEX_METADATA_LIST_KEYS:
        value = metadata.get(key)
        if value:
            parts.append(f"{key} {' '.join(_embedder_index_text_items(value, max_items=24))}".strip())
    return _clean_text(" ".join(_clean_text(part) for part in parts if _clean_text(part)))


def _apply_write_embedder_index_to_graph(
    graph: SessionMemoryGraphV2,
    *,
    stored_ids: Sequence[str],
    turn_text: str,
    turn_index: int,
    mode: str,
    max_terms: int,
) -> Dict[str, Any]:
    normalized_mode = _normalize(mode)
    index_version = _embedder_index_version_for_mode(normalized_mode)
    metadata: Dict[str, Any] = {
        "write_embedder_index_enabled": False,
        "write_embedder_index_mode": normalized_mode or "off",
        "write_embedder_index_version": index_version,
        "write_embedder_index_record_count": 0,
    }
    if not _embedder_index_enabled(normalized_mode) or not stored_ids:
        return metadata
    indexed_ids: List[str] = []
    index_rows: List[tuple[SessionMemoryRecordV2, str, Dict[str, float]]] = []
    for memory_id in _dedupe(stored_ids):
        record = getattr(graph, "records_by_id", {}).get(memory_id)
        if record is None:
            continue
        index_text = _embedder_index_record_text(record, turn_text=turn_text)
        terms = _embedder_index_term_weights(
            index_text,
            max_terms=max_terms,
        )
        if not terms:
            continue
        index_rows.append((record, index_text, terms))
    dense_vectors, dense_metadata = _embedder_dense_vectors_for_texts(
        [index_text for _, index_text, _ in index_rows],
        mode=normalized_mode,
    )
    metadata.update(dense_metadata)
    for row_index, (record, _, terms) in enumerate(index_rows):
        dense_vector = dense_vectors[row_index] if row_index < len(dense_vectors) else []
        record_metadata = dict(record.metadata or {})
        record_metadata.update(
            {
                "write_embedder_index_enabled": True,
                "write_embedder_index_mode": normalized_mode,
                "write_embedder_index_version": index_version,
                "write_embedder_index_source_turn": int(turn_index),
                "write_embedder_index_term_count": int(len(terms)),
                "write_embedder_index_terms": dict(terms),
                "write_embedder_index_top_terms": list(terms.keys())[:24],
            }
        )
        if dense_vector:
            record_metadata.update(
                {
                    "write_embedder_dense_enabled": True,
                    "write_embedder_dense_backend": dense_metadata.get("write_embedder_dense_backend", ""),
                    "write_embedder_dense_model": dense_metadata.get("write_embedder_dense_model", ""),
                    "write_embedder_dense_dim": int(len(dense_vector)),
                    "write_embedder_dense_vector": list(dense_vector),
                }
            )
        elif _embedder_index_uses_bge_m3(normalized_mode):
            record_metadata.update(
                {
                    "write_embedder_dense_enabled": False,
                    "write_embedder_dense_error": dense_metadata.get("write_embedder_dense_error", "dense_vector_unavailable"),
                }
            )
        record.metadata = record_metadata
        indexed_ids.append(record.memory_id)
    metadata.update(
        {
            "write_embedder_index_enabled": bool(indexed_ids),
            "write_embedder_index_record_count": int(len(indexed_ids)),
            "write_embedder_index_record_ids": list(indexed_ids[:24]),
        }
    )
    return metadata


def _embedder_index_recall_event_ids(
    query: str,
    *,
    grouped_hits: Mapping[str, Sequence[MemoryHit]],
    mode: str,
    limit: int,
    max_terms: int,
) -> Dict[str, Any]:
    normalized_mode = _normalize(mode)
    index_version = _embedder_index_version_for_mode(normalized_mode)
    metadata: Dict[str, Any] = {
        "embedder_index_recall_enabled": False,
        "embedder_index_recall_mode": normalized_mode or "off",
        "embedder_index_recall_version": index_version,
        "embedder_index_event_ids": [],
        "embedder_index_event_scores": {},
        "embedder_index_record_count": 0,
    }
    if not _embedder_index_enabled(normalized_mode):
        return {"event_ids": [], "metadata": metadata}
    query_terms = _embedder_index_term_weights(query, max_terms=max_terms)
    query_vectors, query_dense_metadata = _embedder_dense_vectors_for_texts([query], mode=normalized_mode)
    query_vector = query_vectors[0] if query_vectors else []
    metadata.update(
        {
            "embedder_dense_recall_enabled": bool(query_vector),
            "embedder_dense_recall_backend": query_dense_metadata.get("write_embedder_dense_backend", ""),
            "embedder_dense_recall_model": query_dense_metadata.get("write_embedder_dense_model", ""),
            "embedder_dense_recall_error": query_dense_metadata.get("write_embedder_dense_error", ""),
        }
    )
    if not query_terms and not query_vector:
        metadata["embedder_index_recall_reason"] = "empty_query_terms"
        return {"event_ids": [], "metadata": metadata}
    scored_events: List[tuple[str, float, int]] = []
    indexed_record_count = 0
    dense_record_count = 0
    for event_id, group_hits in grouped_hits.items():
        event_score = 0.0
        event_turn = 0
        for hit in list(group_hits or []):
            hit_metadata = dict(hit.metadata or {})
            raw_terms = hit_metadata.get("write_embedder_index_terms")
            raw_vector = hit_metadata.get("write_embedder_dense_vector")
            if not isinstance(raw_terms, Mapping) and not isinstance(raw_vector, list):
                continue
            indexed_record_count += int(isinstance(raw_terms, Mapping))
            dense_record_count += int(isinstance(raw_vector, list) and bool(raw_vector))
            sparse_score = 0.0
            if isinstance(raw_terms, Mapping):
                for term, query_weight in query_terms.items():
                    try:
                        sparse_score += float(query_weight) * float(raw_terms.get(term, 0.0) or 0.0)
                    except (TypeError, ValueError):
                        continue
            dense_score = 0.0
            if query_vector and isinstance(raw_vector, list) and raw_vector:
                for query_value, record_value in zip(query_vector, raw_vector):
                    try:
                        dense_score += float(query_value) * float(record_value)
                    except (TypeError, ValueError):
                        continue
            score = max(float(sparse_score), float(dense_score) + (0.15 * float(sparse_score) if dense_score > 0.0 else 0.0))
            if score <= 0.0:
                continue
            event_score = max(event_score, float(score))
            event_turn = max(event_turn, int(hit.turn_index or 0))
        if event_score > 0.0:
            scored_events.append((_clean_text(event_id), round(event_score, 6), event_turn))
    scored_events.sort(key=lambda item: (-float(item[1]), -int(item[2]), item[0]))
    selected = scored_events[: max(1, int(limit or 1))]
    event_ids = [event_id for event_id, _, _ in selected if event_id]
    event_scores = {event_id: score for event_id, score, _ in selected if event_id}
    metadata.update(
        {
            "embedder_index_recall_enabled": True,
            "embedder_index_event_ids": list(event_ids),
            "embedder_index_event_scores": dict(event_scores),
            "embedder_index_query_terms": list(query_terms.keys())[:32],
            "embedder_index_record_count": int(indexed_record_count),
            "embedder_dense_record_count": int(dense_record_count),
        }
    )
    return {"event_ids": event_ids, "metadata": metadata}


_IDENTIFIER_GENERIC_TOKENS = {
    "api",
    "agent",
    "code",
    "codename",
    "context",
    "debug",
    "goal",
    "memory",
    "model",
    "name",
    "project",
    "retrieval",
    "runtime",
    "session",
    "target",
    "test",
    "turn",
}
_IDENTIFIER_REQUEST_RE = re.compile(
    r"\b(code\s*name|codename|identifier|alias|project\s+name|model\s+name|api\s+name)\b|"
    r"(代号|编号|标识符|名称|名字|别名|项目名|模型名|接口名)",
    flags=re.IGNORECASE,
)
_IDENTIFIER_TOKEN_RE = re.compile(r"\b[A-Za-z][A-Za-z0-9_-]{2,63}\b")


def _query_identifier_tokens(query: Any) -> List[str]:
    text = _clean_text(query)
    tokens: List[str] = []
    for match in _IDENTIFIER_TOKEN_RE.finditer(text):
        token = match.group(0)
        lowered = token.lower()
        if lowered in _IDENTIFIER_GENERIC_TOKENS:
            continue
        has_inner_upper = any(ch.isupper() for ch in token[1:])
        has_digit = any(ch.isdigit() for ch in token)
        has_joiner = "-" in token or "_" in token
        if has_inner_upper or has_digit or has_joiner:
            tokens.append(token)
    return _dedupe(tokens, max_items=8)


def _query_requests_identifier_fact(query: Any) -> bool:
    return bool(_IDENTIFIER_REQUEST_RE.search(_clean_text(query)))


def _hit_text_for_identifier_match(hit: MemoryHit) -> str:
    metadata = dict(hit.metadata or {})
    values = [
        hit.memory_id,
        hit.category,
        hit.value,
        hit.relation,
        hit.slot_key,
        hit.source_kind,
        *list(hit.anchors or []),
        metadata.get("raw_text", ""),
        metadata.get("source_turn_text", ""),
        metadata.get("source_span", ""),
        metadata.get("event_phrase", ""),
        metadata.get("profile_value", ""),
        metadata.get("target_status", ""),
    ]
    return " ".join(_clean_text(value) for value in values if _clean_text(value))


def _copy_hit_with_identifier_boost(hit: MemoryHit, *, score: float, reasons: Sequence[str], matched_tokens: Sequence[str]) -> MemoryHit:
    metadata = dict(hit.metadata or {})
    metadata.update(
        {
            "identifier_protected": True,
            "identifier_protected_score": round(float(score), 6),
            "identifier_protected_reasons": list(reasons)[:8],
            "identifier_protected_matched_tokens": list(matched_tokens)[:8],
            "identifier_protected_original_score": round(float(hit.score), 6),
        }
    )
    return MemoryHit(
        memory_id=hit.memory_id,
        category=hit.category,
        value=hit.value,
        relation=hit.relation,
        anchors=list(hit.anchors),
        score=max(float(hit.score), round(1.0 + float(score), 6)),
        source_kind=hit.source_kind,
        slot_key=hit.slot_key,
        state=hit.state,
        turn_index=int(hit.turn_index),
        metadata=metadata,
    )


def _identifier_protected_hits(
    *,
    query: str,
    final_hits: Sequence[MemoryHit],
    candidate_hits: Sequence[MemoryHit],
    top_k: int,
) -> Dict[str, Any]:
    identifier_tokens = _query_identifier_tokens(query)
    identifier_request = _query_requests_identifier_fact(query)
    if not identifier_tokens and not identifier_request:
        return {"enabled": False, "hits": list(final_hits), "promoted_hits": [], "metadata": {"identifier_protected_enabled": False}}

    query_text = _clean_text(query)
    pool: Dict[str, MemoryHit] = {}
    for hit in list(final_hits) + list(candidate_hits):
        key = hit.memory_id or f"{_hit_event_id(hit)}::{hit.slot_key}::{hit.value[:80]}"
        if key and key not in pool:
            pool[key] = hit

    scored: List[tuple[float, MemoryHit, List[str], List[str]]] = []
    for hit in pool.values():
        hit_text = _hit_text_for_identifier_match(hit)
        hit_lower = hit_text.lower()
        matched_tokens = [token for token in identifier_tokens if token.lower() in hit_lower]
        reasons: List[str] = []
        score = 0.0
        if matched_tokens:
            score += 12.0 + float(len(matched_tokens))
            reasons.append("exact_identifier_match")
        if identifier_request:
            if any(term in hit_lower for term in ("codename", "code name", "identifier", "alias", "project_codename")):
                score += 8.0
                reasons.append("identifier_field_match")
            if any(term in hit_text for term in ("代号", "编号", "标识符", "名称", "名字", "别名")):
                score += 8.0
                reasons.append("identifier_cjk_field_match")
            if "项目" in query_text and "项目" in hit_text:
                score += 1.5
                reasons.append("project_term_match")
        if score <= 0:
            continue
        scored.append((score, hit, matched_tokens, reasons))

    scored.sort(key=lambda item: (item[0], float(item[1].score), int(item[1].turn_index or 0)), reverse=True)
    promoted = [
        _copy_hit_with_identifier_boost(hit, score=score, reasons=reasons, matched_tokens=matched_tokens)
        for score, hit, matched_tokens, reasons in scored[:2]
    ]
    if not promoted:
        return {
            "enabled": True,
            "hits": list(final_hits),
            "promoted_hits": [],
            "metadata": {
                "identifier_protected_enabled": True,
                "identifier_query_tokens": identifier_tokens,
                "identifier_request": bool(identifier_request),
                "identifier_promoted_count": 0,
            },
        }

    promoted_keys = {hit.memory_id or f"{_hit_event_id(hit)}::{hit.slot_key}::{hit.value[:80]}" for hit in promoted}
    merged = list(promoted)
    for hit in final_hits:
        key = hit.memory_id or f"{_hit_event_id(hit)}::{hit.slot_key}::{hit.value[:80]}"
        if key in promoted_keys:
            continue
        merged.append(hit)
        if len(merged) >= max(1, int(top_k)):
            break
    return {
        "enabled": True,
        "hits": merged[: max(1, int(top_k))],
        "promoted_hits": promoted,
        "metadata": {
            "identifier_protected_enabled": True,
            "identifier_query_tokens": identifier_tokens,
            "identifier_request": bool(identifier_request),
            "identifier_promoted_count": len(promoted),
            "identifier_promoted_ids": [hit.memory_id for hit in promoted],
        },
    }


def _trim_prompt_text(value: Any, *, max_chars: int = 220) -> str:
    text = _clean_text(value)
    if len(text) <= max_chars:
        return text
    return f"{text[: max(0, max_chars - 3)].rstrip()}..."


def _prompt_hit_payload(hit: MemoryHit) -> Dict[str, Any]:
    return {
        "memory_id": hit.memory_id,
        "slot_key": hit.slot_key,
        "category": hit.category,
        "value": _trim_prompt_text(hit.value),
        "relation": hit.relation,
        "anchors": [_trim_prompt_text(anchor, max_chars=80) for anchor in list(hit.anchors)[:4]],
        "score": round(float(hit.score), 6),
        "state": hit.state,
        "turn_index": int(hit.turn_index),
        "source_kind": hit.source_kind,
    }


def _prompt_record_payload(record: SessionMemoryRecordV2) -> Dict[str, Any]:
    metadata = dict(record.metadata or {})
    return {
        "slot_key": record.slot_key,
        "category": record.category,
        "value": _trim_prompt_text(record.value),
        "state": record.state,
        "turn_index": int(record.turn_index),
        "anchors": [_trim_prompt_text(anchor, max_chars=80) for anchor in list(record.anchor_concepts)[:4]],
        "memory_role": _clean_text(metadata.get("memory_role", "")),
        "authority": _clean_text(metadata.get("authority", "")),
    }


def _graph_prompt_state_summary(graph: SessionMemoryGraphV2, retrieval: MemoryRetrieval) -> Dict[str, Any]:
    active_slots: List[Dict[str, Any]] = []
    for slot_key, record_id in list(graph.slot_heads.items())[:_GRAPH_PROMPT_MAX_ACTIVE_SLOTS]:
        record = graph.records_by_id.get(record_id)
        if record is None:
            continue
        active_slots.append(_prompt_record_payload(record))
    top_hits = [_prompt_hit_payload(hit) for hit in list(retrieval.hits)[:_GRAPH_PROMPT_MAX_HITS]]
    relation_preview = [
        {
            "from": _trim_prompt_text(item.get("from", ""), max_chars=72),
            "to": _trim_prompt_text(item.get("to", ""), max_chars=72),
            "relation": _clean_text(item.get("relation", "")),
        }
        for item in list(retrieval.relations)[:_GRAPH_PROMPT_MAX_RELATIONS]
    ]
    summary = {
        "records": len(graph.records_by_id),
        "active_slots": len(graph.slot_heads),
        "turn_index": int(graph.turn_index),
        "noise_turn_count": int(graph.noise_turn_count),
        "answer_support_events": len(graph.answer_support_log),
        "top_hits": top_hits,
        "active_slot_records": active_slots,
        "relation_preview": relation_preview,
        "context_truncated": False,
        "truncation_reason": "",
    }
    truncated = False
    truncation_reason = ""
    while len(json.dumps(summary, ensure_ascii=False)) > _GRAPH_PROMPT_MAX_CHARS:
        if len(summary["top_hits"]) > 1:
            summary["top_hits"] = summary["top_hits"][:-1]
            truncated = True
            truncation_reason = "trimmed_top_hits"
            continue
        if len(summary["active_slot_records"]) > 1:
            summary["active_slot_records"] = summary["active_slot_records"][:-1]
            truncated = True
            truncation_reason = "trimmed_active_slots"
            continue
        if len(summary["relation_preview"]) > 2:
            summary["relation_preview"] = summary["relation_preview"][:-1]
            truncated = True
            truncation_reason = "trimmed_relations"
            continue
        break
    summary["context_truncated"] = truncated
    summary["truncation_reason"] = truncation_reason
    return summary


def _state_stats(*, storage_bytes: int, retrieval_context_tokens: int, total_state_tokens: int, **extra: Any) -> Dict[str, Any]:
    return {
        **extra,
        "storage_bytes": int(storage_bytes),
        "context_token_estimate": int(retrieval_context_tokens),
        "retrieval_context_token_estimate": int(retrieval_context_tokens),
        "total_state_token_estimate": int(total_state_tokens),
    }


def _relation_hit(hit: MemoryHit, *, weight_bias: float = 0.0) -> Dict[str, Any]:
    if not hit.anchors:
        return {}
    anchor = hit.anchors[0]
    if not anchor or anchor == hit.value:
        return {}
    return {
        "from": anchor,
        "to": hit.value,
        "relation": hit.relation,
        "weight": round(max(0.25, min(0.98, 0.42 + hit.score * 0.4 + weight_bias)), 6),
        "source_kind": hit.source_kind,
        "memory_id": hit.memory_id,
    }


def _raw_hit_to_memory_hit(payload: Dict[str, Any]) -> MemoryHit:
    metadata = dict(payload.get("metadata", {}) or {})
    if payload.get("supersedes"):
        metadata["supersedes"] = list(payload.get("supersedes", []) or [])
    slot_key = stable_slot_key(
        category=str(payload.get("category", "")),
        value=str(payload.get("value", "")),
        anchors=[str(anchor) for anchor in payload.get("anchor_concepts", payload.get("anchors", [])) or [] if _clean_text(anchor)],
        slot_key=str(payload.get("slot_key", metadata.get("slot", ""))),
        relation=str(payload.get("relation", "related_to")),
        metadata=metadata,
    )
    return MemoryHit(
        memory_id=str(payload.get("memory_id", "")),
        category=str(payload.get("category", "")),
        value=str(payload.get("value", "")),
        relation=str(payload.get("relation", "related_to")),
        anchors=[str(anchor) for anchor in payload.get("anchor_concepts", payload.get("anchors", [])) or [] if _clean_text(anchor)],
        score=float(payload.get("score", payload.get("relevance", 0.0)) or 0.0),
        source_kind=str(payload.get("source_kind", "memory")),
        slot_key=slot_key,
        state=str(payload.get("state", payload.get("metadata", {}).get("state", "active")) or "active"),
        turn_index=int(payload.get("turn_index", 0) or 0),
        metadata=metadata,
    )


def _restore_hit_scores(hits: List[MemoryHit], scored_lookup: Dict[str, MemoryHit]) -> List[MemoryHit]:
    restored: List[MemoryHit] = []
    for hit in hits:
        scored = scored_lookup.get(hit.memory_id)
        if scored:
            hit.score = max(float(hit.score), float(scored.score))
            if not hit.anchors and scored.anchors:
                hit.anchors = list(scored.anchors)
        restored.append(hit)
    return restored


def _current_subject_query(query: str) -> bool:
    lowered = _normalize(query)
    return bool(
        _public_query_subject(query)
        and any(marker in lowered for marker in ("right now", "current", "currently", "active", "now", "当前", "现在"))
    )


def _record_subject_signatures(record: SessionMemoryRecordV2) -> set[str]:
    metadata = dict(record.metadata or {})
    signatures = {
        _normalize(metadata.get("subject_signature", "")).replace("-", "_"),
        _public_subject_signature(metadata.get("subject", "")),
    }
    canonical_slot_key = _clean_text(metadata.get("canonical_slot_key", "") or record.slot_key)
    if ".subject." in canonical_slot_key:
        signatures.add(_public_subject_signature(canonical_slot_key.split(".subject.", 1)[-1]))
    if ".subject." in record.slot_key:
        signatures.add(_public_subject_signature(record.slot_key.split(".subject.", 1)[-1]))
    signatures.discard("")
    return signatures


def _current_subject_protected_hits(
    *,
    query: str,
    graph: SessionMemoryGraphV2,
    final_hits: Sequence[MemoryHit],
    top_k: int,
) -> Dict[str, Any]:
    if not _current_subject_query(query):
        return {
            "hits": list(final_hits),
            "metadata": {"current_subject_resolver_enabled": False},
        }
    subject = _public_query_subject(query)
    subject_signature = _public_subject_signature(subject)
    if not subject_signature:
        return {
            "hits": list(final_hits),
            "metadata": {
                "current_subject_resolver_enabled": True,
                "current_subject_resolver_reason": "no_subject_signature",
            },
        }
    def _candidate_record(record: SessionMemoryRecordV2 | None) -> bool:
        return bool(
            record is not None
            and _clean_text(record.source_kind).startswith("public_dialog")
            and _normalize(record.category) != "question"
            and subject_signature in _record_subject_signatures(record)
        )

    slot_head_candidates = [
        record
        for slot_key, memory_id in graph.slot_heads.items()
        for record in [graph.records_by_id.get(memory_id)]
        if _candidate_record(record)
        and subject_signature in _record_subject_signatures(record)
    ]
    candidates = slot_head_candidates or [
        record
        for record in graph.records_by_id.values()
        if record.state == "active" and _candidate_record(record)
    ]
    candidates.sort(
        key=lambda record: (
            int(
                _normalize((record.metadata or {}).get("target_status", "")) == "current"
                or _normalize(record.relation) == "current_subject_value"
            ),
            int(record.turn_index),
            float(record.confidence),
            float(record.salience),
        ),
        reverse=True,
    )
    promoted: List[MemoryHit] = []
    for index, record in enumerate(candidates[: max(1, min(2, int(top_k or 1)))], start=1):
        metadata = dict(record.metadata or {})
        metadata.update(
            {
                "current_subject_resolver": True,
                "current_subject_resolver_rank": index,
                "current_subject_query_subject": subject,
                "current_subject_query_signature": subject_signature,
                "public_subject_match": True,
                "public_subject_overlap": 1.0,
            }
        )
        promoted.append(
            MemoryHit(
                memory_id=record.memory_id,
                category=record.category,
                value=record.value,
                relation=record.relation,
                anchors=list(record.anchor_concepts),
                score=max(float(record.confidence), float(record.salience), 1.75),
                source_kind=record.source_kind,
                slot_key=record.slot_key,
                state=record.state,
                turn_index=int(record.turn_index),
                metadata=metadata,
            )
        )
    if not promoted:
        return {
            "hits": list(final_hits),
            "metadata": {
                "current_subject_resolver_enabled": True,
                "current_subject_resolver_reason": "no_active_subject_head",
                "current_subject_query_subject": subject,
                "current_subject_query_signature": subject_signature,
            },
        }
    promoted_ids = {hit.memory_id for hit in promoted}
    merged_tail: List[MemoryHit] = []
    for hit in final_hits:
        if hit.memory_id in promoted_ids:
            continue
        metadata = dict(hit.metadata or {})
        hit_state = _normalize(hit.state)
        same_subject = subject_signature in {
            _normalize(metadata.get("subject_signature", "")).replace("-", "_"),
            _public_subject_signature(metadata.get("subject", "")),
            _public_subject_signature(hit.slot_key.split(".subject.", 1)[-1]) if ".subject." in hit.slot_key else "",
            _public_subject_signature(_clean_text(metadata.get("canonical_slot_key", "")).split(".subject.", 1)[-1])
            if ".subject." in _clean_text(metadata.get("canonical_slot_key", ""))
            else "",
        }
        if same_subject and hit_state in {"superseded", "evidence", "historical", "stale", "false"}:
            continue
        merged_tail.append(hit)
    merged = [*promoted, *merged_tail]
    limit = max(int(top_k or 1), len(promoted))
    return {
        "hits": merged[:limit],
        "metadata": {
            "current_subject_resolver_enabled": True,
            "current_subject_resolver_reason": "promoted_active_subject_head",
            "current_subject_query_subject": subject,
            "current_subject_query_signature": subject_signature,
            "current_subject_promoted_memory_ids": [hit.memory_id for hit in promoted],
        },
    }


def _depth_chain_protected_hits(
    *,
    query: str,
    graph: SessionMemoryGraphV2,
    final_hits: Sequence[MemoryHit],
    top_k: int,
) -> Dict[str, Any]:
    seed_memory_ids = [hit.memory_id for hit in final_hits if hit.memory_id]
    chain = graph.depth_chain_for_query(
        query,
        seed_memory_ids=seed_memory_ids,
        top_k=max(3, min(8, int(top_k or 1))),
    )
    if not chain.get("enabled") or not chain.get("nodes"):
        return {
            "hits": list(final_hits),
            "metadata": {
                "memory_chain_enabled": bool(chain.get("enabled", False)),
                "memory_chain_reason": _clean_text(chain.get("reason", "")),
                "memory_chain_node_count": 0,
                "memory_chain_edge_count": 0,
                "memory_chain": chain,
            },
        }
    seen = {hit.memory_id for hit in final_hits if hit.memory_id}
    chain_hits: List[MemoryHit] = []
    for rank, node in enumerate(list(chain.get("nodes", []) or []), start=1):
        if not isinstance(node, Mapping):
            continue
        memory_id = _clean_text(node.get("memory_id", ""))
        if not memory_id or memory_id in seen:
            continue
        payload = dict(node)
        metadata = dict(payload.get("metadata", {}) or {})
        metadata.update(
            {
                "memory_chain_protected": True,
                "memory_chain_rank": int(rank),
                "memory_chain_subject_signature": _clean_text(chain.get("subject_signature", "")),
                "memory_chain_depth_layer": _clean_text(metadata.get("depth_layer", "")) or "core_view",
            }
        )
        payload["metadata"] = metadata
        payload["score"] = max(float(payload.get("score", 0.0) or 0.0), 0.62 - (rank * 0.01))
        chain_hits.append(_raw_hit_to_memory_hit(payload))
        seen.add(memory_id)
    limit = max(int(top_k or 1), min(12, int(top_k or 1) + max(0, len(chain_hits))))
    merged = [*list(final_hits), *chain_hits]
    return {
        "hits": merged[:limit],
        "metadata": {
            "memory_chain_enabled": True,
            "memory_chain_reason": _clean_text(chain.get("reason", "")),
            "memory_chain_subject_signature": _clean_text(chain.get("subject_signature", "")),
            "memory_chain_node_count": int(chain.get("node_count", 0) or 0),
            "memory_chain_edge_count": int(chain.get("edge_count", 0) or 0),
            "memory_chain_depth_layers": list(chain.get("depth_layers", []) or []),
            "memory_chain": chain,
        },
    }


def _is_public_dialog_hit(hit: MemoryHit) -> bool:
    return _clean_text(hit.source_kind).startswith("public_dialog")


def _normalized_runtime_signature(prefix: str, value: str) -> str:
    normalized = _normalize(value)
    if not normalized:
        return ""
    return f"{prefix}{normalized.replace('|', '_').replace(':', '_')}"


def _runtime_event_key(hit: MemoryHit) -> str:
    metadata = dict(hit.metadata or {})
    explicit = _clean_text(metadata.get("event_id", ""))
    if explicit:
        return explicit
    dia_id = _clean_text(metadata.get("dia_id", ""))
    if dia_id:
        return f"event::{dia_id}"
    if not _is_public_dialog_hit(hit):
        state_signature = _clean_text(metadata.get("state_signature", ""))
        if state_signature:
            return _normalized_runtime_signature("event::state::", state_signature)
        memory_signature = _clean_text(metadata.get("memory_signature", ""))
        if memory_signature:
            return _normalized_runtime_signature("event::memory::", memory_signature)
    slot_root = _public_slot_root(_clean_text(hit.slot_key))
    if slot_root:
        return slot_root
    return _clean_text(hit.memory_id)


def _runtime_event_turn_index_from_id(event_id: str) -> int:
    text = _clean_text(event_id)
    if not text:
        return 0
    match = re.search(r"(?::|_)(\d+)$", text)
    if match:
        return int(match.group(1))
    matches = re.findall(r"\d+", text)
    return int(matches[-1]) if matches else 0


def _representative_event_hit(group_hits: Sequence[MemoryHit], *, query: str = "") -> MemoryHit | None:
    semantic_source_kinds = {
        "public_dialog_fact",
        "public_dialog_preference",
        "public_dialog_goal",
        "public_dialog_constraint",
        "public_dialog_status",
        "public_dialog_profile",
        "replacement_memory",
        "session_memory",
    }
    query_tokens = set(_path_utility_tokens(query))

    def rank(hit: MemoryHit) -> tuple[bool, float, bool, bool, bool, bool, float]:
        metadata = dict(hit.metadata or {})
        source_kind = _clean_text(hit.source_kind)
        text_parts: List[Any] = [hit.value, hit.slot_key, hit.category]
        if source_kind != "public_dialog_turn":
            text_parts.extend([metadata.get("source_turn_text", ""), metadata.get("raw_text", "")])
        text = " ".join(
            _clean_text(item)
            for item in text_parts
            if _clean_text(item)
        )
        hit_tokens = set(_path_utility_tokens(text))
        overlap = len(query_tokens & hit_tokens) if query_tokens else 0
        has_number = bool(re.search(r"\b\d+\b", text))
        is_semantic = source_kind in semantic_source_kinds or (
            source_kind != "public_dialog_turn" and bool(_clean_text(metadata.get("memory_writer_role", "")))
        )
        direct_semantic_answer = bool(is_semantic and has_number and overlap >= 2)
        query_score = float(overlap) + (0.75 if has_number and overlap else 0.0)
        return (
            direct_semantic_answer,
            query_score,
            is_semantic,
            source_kind == "public_dialog_event",
            source_kind == "public_dialog_turn",
            source_kind in {"replacement_memory", "session_memory"},
            float(hit.score),
        )

    ordered = sorted(
        list(group_hits),
        key=rank,
        reverse=True,
    )
    return ordered[0] if ordered else None


def _event_record_hits_from_graph(graph: SessionMemoryGraphV2, event_id: str) -> List[MemoryHit]:
    normalized_event_id = _clean_text(event_id)
    if not normalized_event_id:
        return []
    hits: List[MemoryHit] = []
    for record in graph.records_by_id.values():
        metadata = dict(record.metadata or {})
        if _clean_text(metadata.get("event_id", "")) != normalized_event_id:
            continue
        state = _normalize(record.state)
        if state not in {"active", "parallel_active", "evidence"}:
            continue
        hits.append(
            MemoryHit(
                memory_id=record.memory_id,
                category=record.category,
                value=record.value,
                relation=record.relation,
                anchors=list(record.anchor_concepts),
                score=max(float(record.confidence), float(record.salience), 0.01),
                source_kind=record.source_kind,
                slot_key=record.slot_key,
                state=record.state,
                turn_index=int(record.turn_index),
                metadata=metadata,
            )
        )
    return hits

def _hit_matches_path_support(path_type: str, hit: MemoryHit) -> bool:
    metadata = dict(hit.metadata or {})
    source_kind = _clean_text(hit.source_kind)
    category = _clean_text(hit.category)
    relation = _clean_text(hit.relation)
    if path_type == "speaker_event_time":
        return bool(
            source_kind == "public_dialog_time"
            or _clean_text(metadata.get("resolved_time_value", ""))
            or _clean_text(metadata.get("resolved_date", ""))
            or _clean_text(metadata.get("time_value", ""))
            or _clean_text(metadata.get("time_display_value", ""))
            or _clean_text(metadata.get("time_granularity", ""))
            or relation == "event_date"
            or category in {"time", "event_time"}
        )
    if path_type == "speaker_event_profile":
        semantic_slot = _clean_text(metadata.get("semantic_slot", "")) or _clean_text(metadata.get("profile_type", ""))
        return bool(
            source_kind == "public_dialog_profile"
            or semantic_slot in {"identity", "research_topic", "education", "occupation", "profile"}
            or _clean_text(metadata.get("profile_value", ""))
            or category == "profile"
        )
    if path_type == "speaker_event_status":
        return bool(
            _clean_text(metadata.get("target_status", ""))
            or category in {"status", "stage_state"}
            or relation == "status_of"
        )
    if path_type == "speaker_event_source_turn":
        return bool(
            source_kind in {"public_dialog_turn", "public_dialog_text", "public_dialog_auxiliary_evidence"}
            or _clean_text(metadata.get("raw_text", ""))
            or _clean_text(metadata.get("origin_query", ""))
            or _clean_text(metadata.get("source_turn_text", ""))
            or not _is_public_dialog_hit(hit)
        )
    return False


def _support_hit_for_path(path_type: str, group_hits: Sequence[MemoryHit]) -> MemoryHit | None:
    matching_hits = [hit for hit in group_hits if _hit_matches_path_support(path_type, hit)]
    if matching_hits:
        matching_hits.sort(key=lambda item: (float(item.score), int(item.turn_index)), reverse=True)
        return matching_hits[0]
    representative = _representative_event_hit(group_hits)
    return representative


def _path_support_node_id(path: Dict[str, Any]) -> str:
    node_ids = list(path.get("node_ids", []) or [])
    if len(node_ids) < 3:
        return ""
    return _clean_text(node_ids[2])


def _event_ids_from_hits(hits: Sequence[MemoryHit]) -> List[str]:
    return _dedupe(
        _clean_text(dict(hit.metadata or {}).get("event_id", ""))
        for hit in hits
        if _clean_text(dict(hit.metadata or {}).get("event_id", ""))
    )


def _dia_ids_from_hits(hits: Sequence[MemoryHit]) -> List[str]:
    return _dedupe(
        _clean_text(dict(hit.metadata or {}).get("dia_id", ""))
        for hit in hits
        if _clean_text(dict(hit.metadata or {}).get("dia_id", ""))
    )


def _final_hit_role_priority(hit: MemoryHit) -> int:
    metadata = dict(hit.metadata or {})
    source_kind = _clean_text(hit.source_kind)
    semantic_source_kinds = {
        "public_dialog_fact",
        "public_dialog_preference",
        "public_dialog_goal",
        "public_dialog_constraint",
        "public_dialog_status",
        "public_dialog_profile",
        "public_dialog_profile_cluster",
        "replacement_memory",
        "session_memory",
    }
    if source_kind in semantic_source_kinds or (
        source_kind != "public_dialog_turn" and bool(_clean_text(metadata.get("memory_writer_role", "")))
    ):
        return 0
    if bool(metadata.get("profile_first_source_support")):
        return 0
    role = _clean_text(metadata.get("evidence_snippet_role", ""))
    if role == "selected_path_support":
        return 1
    if role == "selected_event_representative":
        return 2
    if role == "selected_path_event":
        return 3
    return 4


def _coverage_preserving_final_hits(
    final_hits: Sequence[MemoryHit],
    *,
    selected_event_ids: Sequence[str],
    top_k: int,
) -> List[MemoryHit]:
    """Keep selected-event coverage before filling the remaining prompt budget.

    Learned selection can emit both path-support and event-representative snippets
    for the same event. A pure score sort can then drop another selected event at
    the top-k boundary, which hides recall/rerank successes from the answer head.
    """

    budget = max(1, int(top_k or 1))
    hits = list(final_hits)
    selected_order = [
        _clean_text(event_id)
        for event_id in selected_event_ids
        if _clean_text(event_id)
    ][:budget]
    if not selected_order:
        return sorted(hits, key=lambda item: float(item.score), reverse=True)[:budget]
    hits_by_event: Dict[str, List[MemoryHit]] = {}
    for hit in hits:
        event_id = _clean_text(dict(hit.metadata or {}).get("event_id", ""))
        if event_id:
            hits_by_event.setdefault(event_id, []).append(hit)
    selected: List[MemoryHit] = []
    used_memory_ids = set()
    for event_id in selected_order:
        candidates = [hit for hit in hits_by_event.get(event_id, []) if hit.memory_id not in used_memory_ids]
        if not candidates:
            continue
        candidates.sort(key=lambda item: (_final_hit_role_priority(item), -float(item.score), item.memory_id))
        chosen = candidates[0]
        selected.append(chosen)
        used_memory_ids.add(chosen.memory_id)
        if len(selected) >= budget:
            return selected
    remaining = [
        hit
        for hit in hits
        if hit.memory_id not in used_memory_ids
    ]
    remaining.sort(key=lambda item: (-float(item.score), _final_hit_role_priority(item), item.memory_id))
    for hit in remaining:
        selected.append(hit)
        if len(selected) >= budget:
            break
    return selected[:budget]


def _dominant_answer_type(question_analysis: Dict[str, Any], answer_type_scores: Dict[str, Any]) -> str:
    normalized_scores = {
        _clean_text(answer_type): float(value or 0.0)
        for answer_type, value in dict(answer_type_scores or {}).items()
        if _clean_text(answer_type)
    }
    if bool(question_analysis.get("is_temporal", False)) and normalized_scores.get("time", 0.0) >= (normalized_scores.get("abstain", 0.0) - 0.05):
        return "time"
    if not bool(question_analysis.get("is_temporal", False)) and normalized_scores:
        non_time_scores = {
            answer_type: score
            for answer_type, score in normalized_scores.items()
            if answer_type not in {"time", "abstain"}
        }
        if non_time_scores:
            return max(non_time_scores.items(), key=lambda item: (float(item[1]), item[0]))[0]
    if not normalized_scores:
        return "time" if bool(question_analysis.get("is_temporal", False)) else "event_text"
    return max(normalized_scores.items(), key=lambda item: (float(item[1]), item[0]))[0]


def _answer_type_preferred_path_types(question_analysis: Dict[str, Any], answer_type_scores: Dict[str, Any]) -> List[str]:
    dominant_answer_type = _dominant_answer_type(question_analysis, answer_type_scores)
    if dominant_answer_type == "time" or bool(question_analysis.get("is_temporal", False)):
        return ["speaker_event_time", "speaker_event_source_turn", "speaker_event_status", "speaker_event_profile"]
    if dominant_answer_type == "profile":
        return ["speaker_event_profile", "speaker_event_source_turn", "speaker_event_status", "speaker_event_time"]
    if dominant_answer_type == "multi_evidence":
        return ["speaker_event_source_turn", "speaker_event_time", "speaker_event_profile", "speaker_event_status"]
    return ["speaker_event_source_turn", "speaker_event_time", "speaker_event_profile", "speaker_event_status"]


def _reconciled_focused_answer_type(
    question_analysis: Dict[str, Any],
    answer_type_scores: Dict[str, Any],
    model_answer_type: str,
) -> str:
    model_type = _clean_text(model_answer_type)
    dominant_type = _dominant_answer_type(question_analysis, answer_type_scores)
    if bool(question_analysis.get("is_temporal", False)) and model_type not in {"", "time", "abstain"}:
        return "time"
    if model_type == "time" and not bool(question_analysis.get("is_temporal", False)):
        return dominant_type if dominant_type != "time" else "event_text"
    return model_type or dominant_type


def _path_type_is_focus_compatible(path_type: str, *, focused_answer_type: str, question_analysis: Dict[str, Any]) -> bool:
    normalized_path_type = _clean_text(path_type)
    normalized_answer_type = _clean_text(focused_answer_type)
    if normalized_answer_type == "time" or bool(question_analysis.get("is_temporal", False)):
        return normalized_path_type in {"speaker_event_time", "speaker_event_source_turn"}
    if normalized_answer_type == "profile":
        return normalized_path_type in {"speaker_event_profile", "speaker_event_source_turn"}
    return True


def _rank_focus_compatible_path_ids(
    *,
    runtime_paths: Mapping[str, Dict[str, Any]],
    selected_event_ids: Sequence[str],
    path_scores: Mapping[str, Any],
    event_scores: Mapping[str, Any],
    temporal_scores: Mapping[str, Any],
    question_analysis: Dict[str, Any],
    answer_type_scores: Dict[str, Any],
    focused_answer_type: str,
) -> List[str]:
    selected_event_id_set = {_clean_text(event_id) for event_id in selected_event_ids if _clean_text(event_id)}
    preferred_types = _answer_type_preferred_path_types(question_analysis, answer_type_scores)
    if _clean_text(focused_answer_type) == "profile" and "speaker_event_profile" not in preferred_types:
        preferred_types = ["speaker_event_profile", *preferred_types]
    if (_clean_text(focused_answer_type) == "time" or bool(question_analysis.get("is_temporal", False))) and "speaker_event_time" not in preferred_types:
        preferred_types = ["speaker_event_time", "speaker_event_source_turn", *preferred_types]
    ranked: List[str] = []
    seen = set()
    for preferred_type in preferred_types:
        candidates: List[tuple[str, float]] = []
        for path_id, path in runtime_paths.items():
            if _clean_text(path.get("type", "")) != preferred_type:
                continue
            event_id = _clean_text(path.get("event_id", ""))
            if selected_event_id_set and event_id not in selected_event_id_set:
                continue
            if not _path_type_is_focus_compatible(preferred_type, focused_answer_type=focused_answer_type, question_analysis=question_analysis):
                continue
            support_node_id = _path_support_node_id(path)
            score = (
                float(event_scores.get(event_id, 0.0) or 0.0)
                + (0.20 * float(path_scores.get(path_id, 0.0) or 0.0))
                + (0.15 * float(temporal_scores.get(support_node_id, 0.0) or 0.0))
            )
            candidates.append((path_id, score))
        for path_id, _ in sorted(candidates, key=lambda item: (-float(item[1]), item[0])):
            if path_id in seen:
                continue
            seen.add(path_id)
            ranked.append(path_id)
        if ranked and (_clean_text(focused_answer_type) == "time" or bool(question_analysis.get("is_temporal", False))):
            if preferred_type in {"speaker_event_time", "speaker_event_source_turn"}:
                break
    return ranked


def _repair_selected_paths_for_focus(
    selected_path_ids: Sequence[str],
    *,
    runtime_paths: Mapping[str, Dict[str, Any]],
    selected_event_ids: Sequence[str],
    path_scores: Mapping[str, Any],
    event_scores: Mapping[str, Any],
    temporal_scores: Mapping[str, Any],
    question_analysis: Dict[str, Any],
    answer_type_scores: Dict[str, Any],
    focused_answer_type: str,
    limit: int,
) -> tuple[List[str], bool, str]:
    normalized_selected = [_clean_text(path_id) for path_id in selected_path_ids if _clean_text(path_id)]
    if not normalized_selected:
        return [], False, ""
    incompatible = [
        path_id
        for path_id in normalized_selected
        if not _path_type_is_focus_compatible(
            _clean_text(runtime_paths.get(path_id, {}).get("type", "")),
            focused_answer_type=focused_answer_type,
            question_analysis=question_analysis,
        )
    ]
    if not incompatible:
        return normalized_selected, False, ""
    compatible_ranked = _rank_focus_compatible_path_ids(
        runtime_paths=runtime_paths,
        selected_event_ids=selected_event_ids,
        path_scores=path_scores,
        event_scores=event_scores,
        temporal_scores=temporal_scores,
        question_analysis=question_analysis,
        answer_type_scores=answer_type_scores,
        focused_answer_type=focused_answer_type,
    )
    if not compatible_ranked:
        return normalized_selected, False, ""
    repaired = _dedupe(
        [
            *[path_id for path_id in normalized_selected if path_id not in incompatible],
            *compatible_ranked,
        ]
    )[: max(1, limit)]
    if repaired == normalized_selected:
        return normalized_selected, False, ""
    return repaired, True, "replaced_focus_incompatible_model_paths"


def _runtime_node_by_id(runtime_graph: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {
        _clean_text(node.get("id", "")): dict(node)
        for node in list(runtime_graph.get("nodes", []) or [])
        if _clean_text(node.get("id", ""))
    }


def _runtime_event_subject_signature(runtime_nodes: Mapping[str, Dict[str, Any]], event_id: str) -> str:
    node = dict(runtime_nodes.get(_clean_text(event_id), {}) or {})
    metadata = dict(node.get("metadata", {}) or {})
    return _clean_text(node.get("subject_signature", "")) or _clean_text(metadata.get("subject_signature", ""))


def _path_utility_candidate_text(
    path: Mapping[str, Any],
    *,
    runtime_nodes: Mapping[str, Dict[str, Any]],
    grouped_hits: Mapping[str, Sequence[MemoryHit]],
) -> str:
    event_id = _clean_text(path.get("event_id", ""))
    support_node_id = _path_support_node_id(dict(path))
    node_texts = [
        _clean_text(dict(runtime_nodes.get(event_id, {}) or {}).get("text", "")),
        _clean_text(dict(runtime_nodes.get(support_node_id, {}) or {}).get("text", "")),
    ]
    path_type = _clean_text(path.get("type", ""))
    support_hit = _support_hit_for_path(path_type, grouped_hits.get(event_id, []))
    event_hit = _representative_event_hit(grouped_hits.get(event_id, []))
    hit_texts = [
        _clean_text(support_hit.value if support_hit is not None else ""),
        _clean_text(event_hit.value if event_hit is not None else ""),
        _runtime_source_turn_text(support_hit or event_hit, speaker=""),
    ]
    return " ".join(_dedupe([*node_texts, *hit_texts], max_items=6))


def _path_utility_gate(
    candidate_path_ids: Sequence[str],
    *,
    query: str,
    runtime_graph: Mapping[str, Any],
    runtime_paths: Mapping[str, Dict[str, Any]],
    grouped_hits: Mapping[str, Sequence[MemoryHit]],
    selected_path_ids: Sequence[str],
    selected_event_ids_from_model: Sequence[str],
    path_scores: Mapping[str, Any],
    path_tunnel_support_scores: Mapping[str, Any],
    question_analysis: Dict[str, Any],
    focused_answer_type: str,
    score_threshold: float,
    limit: int,
) -> Dict[str, Any]:
    runtime_nodes = _runtime_node_by_id(runtime_graph)
    query_tokens = set(_path_utility_tokens(query))
    anchor_event_ids = _dedupe(
        [
            *[
                _clean_text(runtime_paths.get(path_id, {}).get("event_id", ""))
                for path_id in selected_path_ids
                if _clean_text(runtime_paths.get(path_id, {}).get("event_id", ""))
            ],
            *[_clean_text(event_id) for event_id in selected_event_ids_from_model if _clean_text(event_id)],
        ]
    )
    anchor_subject_signatures = {
        signature
        for signature in (
            _runtime_event_subject_signature(runtime_nodes, event_id)
            for event_id in anchor_event_ids
        )
        if signature
    }
    direct_path_ids: List[str] = []
    contrast_path_ids: List[str] = []
    latent_path_ids: List[str] = []
    noise_path_ids: List[str] = []
    utility_scores: Dict[str, float] = {}
    utility_roles: Dict[str, str] = {}
    utility_reasons: Dict[str, str] = {}
    utility_overlap_tokens: Dict[str, List[str]] = {}
    for path_id in _dedupe(candidate_path_ids):
        path = dict(runtime_paths.get(path_id, {}) or {})
        if not path:
            continue
        path_type = _clean_text(path.get("type", ""))
        event_id = _clean_text(path.get("event_id", ""))
        candidate_text = _path_utility_candidate_text(
            path,
            runtime_nodes=runtime_nodes,
            grouped_hits=grouped_hits,
        )
        candidate_tokens = set(_path_utility_tokens(candidate_text))
        overlap_tokens = sorted(query_tokens & candidate_tokens)
        overlap_ratio = float(len(overlap_tokens)) / float(max(1, min(len(query_tokens), len(candidate_tokens))))
        support_score = float(path_tunnel_support_scores.get(path_id, 0.0) or 0.0)
        decision_score = float(path_scores.get(path_id, support_score) or 0.0)
        path_score = max(support_score, decision_score)
        subject_signature = _runtime_event_subject_signature(runtime_nodes, event_id)
        same_subject_chain = bool(subject_signature and subject_signature in anchor_subject_signatures)
        focus_compatible = _path_type_is_focus_compatible(
            path_type,
            focused_answer_type=focused_answer_type,
            question_analysis=question_analysis,
        )
        utility_score = path_score + (0.20 * overlap_ratio) + (0.04 if same_subject_chain else 0.0)
        utility_scores[path_id] = round(float(utility_score), 6)
        utility_overlap_tokens[path_id] = overlap_tokens[:12]
        if not focus_compatible:
            role = "drift_noise"
            reason = "focus_incompatible"
        elif overlap_ratio >= 0.18 and path_score >= score_threshold:
            role = "direct_support"
            reason = "query_overlap_and_tunnel_score"
        elif same_subject_chain and path_score >= score_threshold:
            role = "contrast_support"
            reason = "same_subject_deep_chain"
        elif path_score >= score_threshold:
            role = "latent_context"
            reason = "tunnel_score_without_current_turn_utility"
        else:
            role = "drift_noise"
            reason = "below_utility_threshold"
        utility_roles[path_id] = role
        utility_reasons[path_id] = reason
        if role == "direct_support":
            direct_path_ids.append(path_id)
        elif role == "contrast_support":
            contrast_path_ids.append(path_id)
        elif role == "latent_context":
            latent_path_ids.append(path_id)
        else:
            noise_path_ids.append(path_id)
    injected_path_ids = _dedupe([*direct_path_ids, *contrast_path_ids], max_items=max(0, int(limit)))
    overflow_latent_path_ids = [
        path_id
        for path_id in [*direct_path_ids, *contrast_path_ids]
        if path_id not in set(injected_path_ids)
    ]
    latent_path_ids = _dedupe([*latent_path_ids, *overflow_latent_path_ids])
    return {
        "enabled": True,
        "candidate_path_ids": list(_dedupe(candidate_path_ids)),
        "direct_support_path_ids": list(direct_path_ids),
        "contrast_support_path_ids": list(contrast_path_ids),
        "latent_context_path_ids": list(latent_path_ids),
        "drift_noise_path_ids": list(noise_path_ids),
        "injected_path_ids": list(injected_path_ids),
        "roles": dict(utility_roles),
        "reasons": dict(utility_reasons),
        "scores": dict(utility_scores),
        "overlap_tokens": dict(utility_overlap_tokens),
        "anchor_event_ids": list(anchor_event_ids),
        "anchor_subject_signatures": sorted(anchor_subject_signatures),
    }


def _calibrated_path_score(
    *,
    path: Dict[str, Any],
    base_score: float,
    temporal_scores: Dict[str, Any],
    question_analysis: Dict[str, Any],
    answer_type_scores: Dict[str, Any],
) -> float:
    normalized_path_type = _clean_text(path.get("type", ""))
    support_node_id = _path_support_node_id(path)
    temporal_score = float(temporal_scores.get(support_node_id, 0.0) or 0.0)
    normalized_answer_scores = {
        _clean_text(answer_type): float(value or 0.0)
        for answer_type, value in dict(answer_type_scores or {}).items()
        if _clean_text(answer_type)
    }
    dominant_answer_type = _dominant_answer_type(question_analysis, normalized_answer_scores)
    calibrated = float(base_score)
    if dominant_answer_type == "time" or bool(question_analysis.get("is_temporal", False)):
        if normalized_path_type == "speaker_event_time":
            calibrated += (0.30 * temporal_score) + (0.12 * normalized_answer_scores.get("time", 0.0))
        elif normalized_path_type == "speaker_event_source_turn":
            calibrated -= 0.08 + (0.04 * normalized_answer_scores.get("time", 0.0))
        elif normalized_path_type == "speaker_event_status":
            calibrated -= 0.12
        elif normalized_path_type == "speaker_event_profile":
            calibrated -= 0.22 + (0.06 * normalized_answer_scores.get("time", 0.0))
    elif dominant_answer_type == "profile":
        if normalized_path_type == "speaker_event_profile":
            calibrated += 0.12 * normalized_answer_scores.get("profile", 0.0)
        elif normalized_path_type != "speaker_event_source_turn":
            calibrated -= 0.08
    else:
        if normalized_path_type == "speaker_event_source_turn":
            calibrated += 0.08 * max(
                normalized_answer_scores.get("event_text", 0.0),
                normalized_answer_scores.get("multi_evidence", 0.0),
            )
        elif normalized_path_type == "speaker_event_profile":
            calibrated -= 0.04
    return calibrated


def _group_metadata_value(
    group_hits: Sequence[MemoryHit],
    key: str,
    *,
    source_kinds: Sequence[str] = (),
) -> str:
    source_kind_set = {_clean_text(item) for item in list(source_kinds) if _clean_text(item)}
    candidates = [
        hit
        for hit in group_hits
        if not source_kind_set or _clean_text(hit.source_kind) in source_kind_set
    ]
    candidates.sort(key=lambda item: (float(item.score), int(item.turn_index)), reverse=True)
    for hit in candidates:
        metadata = dict(hit.metadata or {})
        value = _clean_text(metadata.get(key, ""))
        if value:
            return value
    return ""


def _runtime_event_sequence_key(session_name: str, turn_index: int, event_id: str) -> tuple[Any, ...]:
    normalized_session = _clean_text(session_name)
    session_number_match = re.search(r"(\d+)$", normalized_session)
    if session_number_match:
        return (0, int(session_number_match.group(1)), int(turn_index), _clean_text(event_id))
    return (1, normalized_session, int(turn_index), _clean_text(event_id))


def _runtime_source_turn_text(hit: MemoryHit | None, *, speaker: str) -> str:
    if hit is None:
        return ""
    metadata = dict(hit.metadata or {})
    source_turn_text = _clean_text(metadata.get("source_turn_text", ""))
    if source_turn_text:
        return source_turn_text
    raw_text = _clean_text(metadata.get("raw_text", ""))
    if raw_text:
        auxiliary_text = _clean_text(metadata.get("auxiliary_evidence_text", ""))
        if auxiliary_text and auxiliary_text.lower() not in raw_text.lower():
            return f"{raw_text}\nAuxiliary evidence: {auxiliary_text}"
        return raw_text
    origin_query = _clean_text(metadata.get("origin_query", ""))
    if origin_query:
        return origin_query
    text = _clean_text(hit.value)
    if _clean_text(hit.source_kind) == "public_dialog_turn":
        text = re.sub(r"^\[[^\]]+\]\s*", "", text)
        if _clean_text(speaker):
            text = re.sub(rf"^{re.escape(_clean_text(speaker))}\s*:\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"^[A-Za-z][A-Za-z0-9_' -]{0,40}:\s*", "", text)
    return text


def _runtime_event_signature(
    *,
    group_hits: Sequence[MemoryHit],
    representative: MemoryHit,
    speaker: str,
    semantic_slot: str,
    source_turn_hit: MemoryHit | None,
) -> str:
    existing = _group_metadata_value(group_hits, "event_signature") or _clean_text(dict(representative.metadata or {}).get("event_signature", ""))
    if existing:
        return existing
    event_phrase = _group_metadata_value(group_hits, "event_phrase") or _clean_text(dict(representative.metadata or {}).get("event_phrase", ""))
    source_turn_text = _runtime_source_turn_text(source_turn_hit, speaker=speaker)
    base_text = event_phrase or _clean_text(representative.value) or source_turn_text
    if not base_text:
        return ""
    return compute_public_event_signature(
        base_text,
        speaker=_clean_text(speaker),
        semantic_slot=_clean_text(semantic_slot),
    ) or _clean_text(base_text)


def _build_runtime_graph_from_hits(query: str, hits: Sequence[MemoryHit]) -> Dict[str, Any]:
    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []
    paths: List[Dict[str, Any]] = []
    node_ids = set()
    grouped_hits: Dict[str, List[MemoryHit]] = {}
    ordered_events: List[tuple[Any, ...]] = []
    event_typed_metadata_by_id: Dict[str, Dict[str, Any]] = {}
    typed_tunnel_edges: List[Dict[str, Any]] = []

    def add_node(node: Dict[str, Any]) -> None:
        node_id = _clean_text(node.get("id", ""))
        if not node_id or node_id in node_ids:
            return
        node_ids.add(node_id)
        nodes.append(node)

    for hit in hits:
        event_id = _runtime_event_key(hit)
        grouped_hits.setdefault(event_id, []).append(hit)

    for event_id, group_hits in grouped_hits.items():
        representative = _representative_event_hit(group_hits)
        if representative is None:
            continue
        metadata = dict(representative.metadata or {})
        event_time_hit = _support_hit_for_path("speaker_event_time", group_hits)
        event_profile_hit = _support_hit_for_path("speaker_event_profile", group_hits)
        source_turn_hit = _support_hit_for_path("speaker_event_source_turn", group_hits)
        event_time_display_value = (
            _group_metadata_value(group_hits, "time_display_value", source_kinds=("public_dialog_time",))
            or _group_metadata_value(group_hits, "resolved_date")
            or _group_metadata_value(group_hits, "time_display_value")
        )
        event_time_value = (
            _group_metadata_value(group_hits, "resolved_time_value", source_kinds=("public_dialog_time",))
            or _group_metadata_value(group_hits, "resolved_time_value")
            or _group_metadata_value(group_hits, "time_value")
        )
        event_time_granularity = (
            _group_metadata_value(group_hits, "time_granularity", source_kinds=("public_dialog_time",))
            or _group_metadata_value(group_hits, "time_granularity")
        )
        event_profile_type = (
            _group_metadata_value(group_hits, "profile_type", source_kinds=("public_dialog_profile",))
            or _group_metadata_value(group_hits, "semantic_slot", source_kinds=("public_dialog_profile",))
            or _group_metadata_value(group_hits, "profile_type")
            or _group_metadata_value(group_hits, "semantic_slot")
        )
        profile_hit = _support_hit_for_path("speaker_event_profile", group_hits)
        event_profile_value = (
            _group_metadata_value(group_hits, "profile_value")
            or _clean_text(profile_hit.value if profile_hit is not None else "")
            or (_clean_text(representative.value) if event_profile_type else "")
        )
        event_target_status = _group_metadata_value(group_hits, "target_status")
        depth_layer = (
            _clean_text(metadata.get("depth_layer", ""))
            or _group_metadata_value(group_hits, "depth_layer")
            or _clean_text(metadata.get("memory_chain_depth_layer", ""))
            or _group_metadata_value(group_hits, "memory_chain_depth_layer")
        )
        subject_signature = (
            _clean_text(metadata.get("subject_signature", ""))
            or _group_metadata_value(group_hits, "subject_signature")
            or _clean_text(metadata.get("memory_chain_subject_signature", ""))
            or _group_metadata_value(group_hits, "memory_chain_subject_signature")
        )
        session_name = (
            _group_metadata_value(group_hits, "session_name")
            or _group_metadata_value(group_hits, "session_key")
            or _group_metadata_value(group_hits, "scope_id")
            or "runtime_session"
        )
        speaker = (
            _clean_text(metadata.get("speaker", ""))
            or _group_metadata_value(group_hits, "speaker")
            or _clean_text(metadata.get("subject_signature", ""))
            or (_clean_text(representative.anchors[0]) if representative.anchors else "")
            or "speaker"
        )
        event_turn_index = int(getattr(representative, "turn_index", 0) or 0)
        semantic_slot = (
            _group_metadata_value(group_hits, "semantic_slot")
            or _clean_text(metadata.get("semantic_slot", ""))
            or _clean_text(metadata.get("profile_type", ""))
            or ("profile" if event_profile_type else _clean_text(representative.category))
            or "event"
        )
        teacher_fields = {
            "event_phrase": _clean_text(metadata.get("event_phrase", "")) or _clean_text(representative.value),
            "semantic_slot": semantic_slot,
            "target_status": event_target_status,
            "time_expression_span": event_time_display_value,
            "time_granularity": event_time_granularity,
            "profile_type": event_profile_type,
        }
        base_event_signature = _runtime_event_signature(
            group_hits=group_hits,
            representative=representative,
            speaker=speaker,
            semantic_slot=event_profile_type or teacher_fields["semantic_slot"],
            source_turn_hit=source_turn_hit,
        )
        typed_tunnel_metadata = merge_typed_metadata([metadata, *[dict(hit.metadata or {}) for hit in group_hits]])
        event_typed_metadata_by_id[event_id] = typed_tunnel_metadata
        typed_signature = typed_tunnel_signature_text(typed_tunnel_metadata)
        event_text = _clean_text(metadata.get("event_phrase", "")) or representative.value
        depth_prefix = " ".join(
            item
            for item in (
                f"subject {subject_signature.replace('_', ' ')}" if subject_signature else "",
                f"depth layer {depth_layer.replace('_', ' ')}" if depth_layer else "",
            )
            if item
        )
        runtime_event_text = f"{depth_prefix} {event_text}".strip() if depth_prefix else event_text
        event_signature = (
            compute_public_event_signature(
                runtime_event_text,
                speaker=_clean_text(speaker),
                semantic_slot=_clean_text(event_profile_type or teacher_fields["semantic_slot"]),
            )
            or base_event_signature
        )
        if typed_signature and typed_signature not in event_signature:
            event_signature = f"{event_signature} {typed_signature}".strip()
        speaker_node_id = f"{event_id}:speaker:{_normalize(speaker).replace(' ', '_') or 'speaker'}"
        add_node({"id": speaker_node_id, "type": "speaker", "text": speaker, "metadata": {"speaker": speaker}})
        add_node(
            {
                "id": event_id,
                "type": "event",
                "text": runtime_event_text,
                "speaker": speaker,
                "turn_index": event_turn_index,
                "session_name": session_name,
                "dia_id": _clean_text(metadata.get("dia_id", "")),
                "event_signature": event_signature,
                "slot_key": _clean_text(representative.slot_key),
                "state_signature": _clean_text(metadata.get("state_signature", "")),
                "memory_signature": _clean_text(metadata.get("memory_signature", "")),
                "target_status": event_target_status,
                "time_granularity": event_time_granularity,
                "time_value": event_time_value,
                "time_display_value": event_time_display_value,
                "profile_type": event_profile_type,
                "profile_value": event_profile_value,
                "depth_layer": depth_layer,
                "subject_signature": subject_signature,
                "tmcra_node_tags": list(typed_tunnel_metadata.get("tmcra_node_tags", []) or []),
                "tmcra_path_tags": list(typed_tunnel_metadata.get("tmcra_path_tags", []) or []),
                "tmcra_tunnel_roles": list(typed_tunnel_metadata.get("tmcra_tunnel_roles", []) or []),
                "tmcra_tunnel_group_key": _clean_text(typed_tunnel_metadata.get("tmcra_tunnel_group_key", "")),
                "teacher_fields": teacher_fields,
                "metadata": {
                    "speaker": speaker,
                    "session_name": session_name,
                    "dia_id": _clean_text(metadata.get("dia_id", "")),
                    "slot_key": _clean_text(representative.slot_key),
                    "state_signature": _clean_text(metadata.get("state_signature", "")),
                    "memory_signature": _clean_text(metadata.get("memory_signature", "")),
                    "target_status": event_target_status,
                    "time_granularity": event_time_granularity,
                    "time_value": event_time_value,
                    "time_display_value": event_time_display_value,
                    "profile_type": event_profile_type,
                    "profile_value": event_profile_value,
                    "event_signature": event_signature,
                    "depth_layer": depth_layer,
                    "subject_signature": subject_signature,
                    **typed_tunnel_metadata,
                },
            }
        )
        edges.append({"id": f"{speaker_node_id}->{event_id}:speaker_of", "source": speaker_node_id, "target": event_id, "type": "speaker_of"})

        time_node_ids: List[str] = []
        profile_node_ids: List[str] = []
        status_node_ids: List[str] = []
        source_turn_node_ids: List[str] = []

        if event_time_display_value or event_time_value:
            time_node_id = f"{event_id}:time"
            time_hit_metadata = dict((event_time_hit or representative).metadata or {})
            add_node(
                {
                    "id": time_node_id,
                    "type": "time",
                    "text": event_time_display_value or event_time_value,
                    "turn_index": int(getattr(event_time_hit, "turn_index", 0) or 0),
                    "time_display_value": event_time_display_value,
                    "time_value": event_time_value,
                    "time_granularity": event_time_granularity,
                    "metadata": {
                        "time_display_value": event_time_display_value,
                        "time_value": event_time_value,
                        "time_granularity": event_time_granularity,
                        "resolved_date": _clean_text(time_hit_metadata.get("resolved_date", "")),
                    },
                }
            )
            edges.append({"id": f"{event_id}->{time_node_id}:time_of", "source": event_id, "target": time_node_id, "type": "time_of"})
            time_node_ids.append(time_node_id)
        if event_profile_value:
            profile_node_id = f"{event_id}:profile:{_normalize(event_profile_type).replace(' ', '_') or 'profile'}"
            add_node(
                {
                    "id": profile_node_id,
                    "type": "profile",
                    "text": event_profile_value,
                    "turn_index": int(getattr(event_profile_hit or representative, "turn_index", 0) or 0),
                    "profile_type": event_profile_type,
                    "profile_value": event_profile_value,
                    "metadata": {
                        "profile_type": event_profile_type,
                        "profile_value": event_profile_value,
                    },
                }
            )
            edges.append({"id": f"{event_id}->{profile_node_id}:profile_of", "source": event_id, "target": profile_node_id, "type": "profile_of"})
            profile_node_ids.append(profile_node_id)
        source_turn_text = _runtime_source_turn_text(source_turn_hit or representative, speaker=speaker)
        if source_turn_text:
            source_turn_node_id = f"{event_id}:source_turn"
            add_node(
                {
                    "id": source_turn_node_id,
                    "type": "source_turn",
                    "text": source_turn_text,
                    "turn_index": int(getattr(source_turn_hit or representative, "turn_index", 0) or 0),
                    "metadata": {
                        "speaker": speaker,
                        "dia_id": _clean_text(dict((source_turn_hit or representative).metadata or {}).get("dia_id", "")),
                    },
                }
            )
            edges.append({"id": f"{event_id}->{source_turn_node_id}:supported_by_turn", "source": event_id, "target": source_turn_node_id, "type": "supported_by_turn"})
            source_turn_node_ids.append(source_turn_node_id)
        if event_target_status:
            status_node_id = f"{event_id}:status"
            add_node({"id": status_node_id, "type": "status", "text": event_target_status, "metadata": {"target_status": event_target_status}})
            edges.append({"id": f"{event_id}->{status_node_id}:status_of", "source": event_id, "target": status_node_id, "type": "status_of"})
            status_node_ids.append(status_node_id)
        event_paths = build_default_path_templates(
            event_id=event_id,
            speaker_node_id=speaker_node_id,
            time_node_ids=time_node_ids,
            profile_node_ids=profile_node_ids,
            status_node_ids=status_node_ids,
            source_turn_node_ids=source_turn_node_ids,
        )
        for path in event_paths:
            path_metadata = dict(path.get("metadata", {}) or {})
            path_metadata["tmcra_path_tags"] = list(typed_tunnel_metadata.get("tmcra_path_tags", []) or [])
            path_metadata["tmcra_tunnel_group_key"] = _clean_text(typed_tunnel_metadata.get("tmcra_tunnel_group_key", ""))
            path["metadata"] = path_metadata
            path["tmcra_path_tags"] = path_metadata["tmcra_path_tags"]
        paths.extend(event_paths)
        ordered_events.append(_runtime_event_sequence_key(session_name, event_turn_index, event_id))
    ordered_event_ids = [event_id for _, _, _, event_id in sorted(ordered_events)]
    for previous_event_id, next_event_id in zip(ordered_event_ids, ordered_event_ids[1:]):
        typed_edge_tags = typed_edge_tags_between(
            event_typed_metadata_by_id.get(previous_event_id, {}),
            event_typed_metadata_by_id.get(next_event_id, {}),
        )
        edges.append(
            {
                "id": f"{previous_event_id}->{next_event_id}:same_session_next",
                "source": previous_event_id,
                "target": next_event_id,
                "type": "same_session_next",
                "metadata": {
                    "tmcra_edge_tags": typed_edge_tags,
                    "typed_tunnel_edge": bool(typed_edge_tags),
                },
            }
        )
    for index, source_event_id in enumerate(ordered_event_ids):
        for target_event_id in ordered_event_ids[index + 1 : index + 12]:
            typed_edge_tags = typed_edge_tags_between(
                event_typed_metadata_by_id.get(source_event_id, {}),
                event_typed_metadata_by_id.get(target_event_id, {}),
            )
            if not typed_edge_tags:
                continue
            typed_tunnel_edges.append(
                {
                    "id": f"{source_event_id}->{target_event_id}:typed_tunnel",
                    "source": source_event_id,
                    "target": target_event_id,
                    "type": "typed_tunnel_candidate",
                    "metadata": {
                        "tmcra_edge_tags": typed_edge_tags,
                        "typed_tunnel_edge": True,
                    },
                }
            )
            if len(typed_tunnel_edges) >= 64:
                break
        if len(typed_tunnel_edges) >= 64:
            break
    return {
        "conversation_id": "runtime",
        "query": query,
        "nodes": nodes,
        "edges": edges,
        "typed_tunnel_edges": typed_tunnel_edges,
        "paths": paths,
        "grouped_hits": grouped_hits,
    }


def _public_graph_hits(graph: SessionMemoryGraphV2) -> List[MemoryHit]:
    public_hits: List[MemoryHit] = []
    for record in graph.records_by_id.values():
        if record.state != "active":
            continue
        if not _clean_text(record.source_kind).startswith("public_dialog"):
            continue
        metadata = dict(record.metadata or {})
        public_hits.append(
            MemoryHit(
                memory_id=record.memory_id,
                category=record.category,
                value=record.value,
                relation=record.relation,
                anchors=list(record.anchor_concepts),
                score=max(float(record.confidence), float(record.salience), 0.01),
                source_kind=record.source_kind,
                slot_key=record.slot_key,
                state=record.state,
                turn_index=int(record.turn_index),
                metadata=metadata,
            )
        )
    public_hits.sort(key=lambda item: (int(item.turn_index), float(item.score)), reverse=True)
    return public_hits


_AUDIT_ANCHOR_QUERY_RE = re.compile(
    r"(?i)\b(?:remember|recall|earlier|previous|previously|old|before|mentioned|said|quote|verbatim|original|"
    r"that time|last time|bring back|return to|go back to|turn\s*\d+)\b|"
    r"(?:\u7b2c\s*\d+\s*[\u8f6e\u6b21]|\d+\s*\u8f6e|\u4e4b\u524d|\u4ee5\u524d|\u521a\u521a|"
    r"\u539f\u8bdd|\u90a3\u6b21|\u65e7\u8bdd\u9898|\u56de\u5230|\u63d0\u8d77|\u8bb0\u5f97)"
)

_AUDIT_TURN_ANCHOR_RE = re.compile(
    r"(?i)\bturn\s*#?\s*(\d{1,6})\b|"
    r"(?:\u7b2c\s*(\d{1,6})\s*[\u8f6e\u6b21]|\b(\d{1,6})\s*\u8f6e\b)"
)

_AUDIT_ANCHOR_STOPWORDS = set(_HYBRID_SYMBOLIC_STOPWORDS) | {
    "about",
    "again",
    "back",
    "bring",
    "can",
    "could",
    "did",
    "discuss",
    "discussed",
    "earlier",
    "from",
    "just",
    "keep",
    "mentioned",
    "old",
    "one",
    "previous",
    "previously",
    "really",
    "recall",
    "remember",
    "return",
    "said",
    "still",
    "that",
    "the",
    "thing",
    "think",
    "this",
    "turn",
    "what",
    "when",
    "where",
    "with",
    "you",
}

_AUDIT_ANCHOR_GENERIC_MATCH_TOKENS = {
    "body",
    "coherence",
    "continuity",
    "fiction",
    "fray",
    "memories",
    "memory",
    "narrative",
    "ourselves",
    "physical",
    "really",
    "stories",
    "story",
    "tell",
    "trust",
}

_AUDIT_ANCHOR_PHRASE_TOKEN_SETS = [
    {"spine", "book"},
    {"body", "spine"},
    {"stories", "pages"},
    {"story", "pages"},
    {"islands", "self"},
    {"archipelago", "self"},
    {"loom", "body"},
    {"braid", "body"},
]


def _audit_anchor_query(query: str) -> bool:
    return bool(_AUDIT_ANCHOR_QUERY_RE.search(_clean_text(query)))


def _audit_anchor_turn_numbers(query: str) -> List[int]:
    numbers: List[int] = []
    for match in _AUDIT_TURN_ANCHOR_RE.finditer(_clean_text(query)):
        for group in match.groups():
            if not group:
                continue
            try:
                value = int(group)
            except Exception:
                continue
            if value > 0 and value not in numbers:
                numbers.append(value)
    return numbers


def _hit_event_id(hit: MemoryHit) -> str:
    metadata = dict(hit.metadata or {})
    event_id = _clean_text(metadata.get("event_id", ""))
    if event_id:
        return event_id
    dia_id = _clean_text(metadata.get("dia_id", ""))
    if dia_id:
        return f"event::{dia_id}"
    if int(hit.turn_index or 0) > 0:
        return f"event::realchat:{int(hit.turn_index)}"
    return ""


def _audit_anchor_hit_text(hit: MemoryHit) -> str:
    metadata = dict(hit.metadata or {})
    values = [
        hit.value,
        hit.category,
        hit.source_kind,
        hit.slot_key,
        metadata.get("event_text", ""),
        metadata.get("source_span", ""),
        metadata.get("raw_text", ""),
    ]
    return " ".join(_clean_text(value) for value in values if _clean_text(value))


def _audit_anchor_content_tokens(text: str) -> set[str]:
    tokens = set()
    for token in _tokenize(text):
        norm = _normalize(token)
        if not norm or norm in _AUDIT_ANCHOR_STOPWORDS:
            continue
        if len(norm) < 3 and not any("\u4e00" <= ch <= "\u9fff" for ch in norm):
            continue
        tokens.add(norm)
    return tokens


def _audit_anchor_phrase_bonus(query_tokens: set[str], hit_tokens: set[str]) -> tuple[float, List[str]]:
    matched_phrases: List[str] = []
    bonus = 0.0
    for phrase_tokens in _AUDIT_ANCHOR_PHRASE_TOKEN_SETS:
        if phrase_tokens <= query_tokens and phrase_tokens <= hit_tokens:
            matched_phrases.append("+".join(sorted(phrase_tokens)))
            bonus += 5.0 if phrase_tokens == {"spine", "book"} else 3.0
    return bonus, matched_phrases


def _copy_hit_with_audit_anchor_boost(hit: MemoryHit, *, score: float, reason: str, matched_tokens: Sequence[str]) -> MemoryHit:
    metadata = dict(hit.metadata or {})
    metadata.update(
        {
            "audit_anchor_protected": True,
            "audit_anchor_reason": reason,
            "audit_anchor_original_score": round(float(hit.score), 6),
            "audit_anchor_score": round(float(score), 6),
            "audit_anchor_matched_tokens": list(matched_tokens)[:20],
        }
    )
    return MemoryHit(
        memory_id=hit.memory_id,
        category=hit.category,
        value=hit.value,
        relation=hit.relation,
        anchors=list(hit.anchors),
        score=max(float(hit.score), round(1.0 + float(score), 6)),
        source_kind=hit.source_kind,
        slot_key=hit.slot_key,
        state=hit.state,
        turn_index=int(hit.turn_index),
        metadata=metadata,
    )


def _audit_anchor_protected_hits(
    *,
    query: str,
    final_hits: Sequence[MemoryHit],
    candidate_hits: Sequence[MemoryHit],
    metadata: Mapping[str, Any],
    top_k: int,
) -> Dict[str, Any]:
    if not _audit_anchor_query(query):
        return {"enabled": False, "hits": list(final_hits), "promoted_hits": [], "metadata": {"audit_anchor_enabled": False}}
    query_tokens = _audit_anchor_content_tokens(query)
    if not query_tokens:
        return {"enabled": True, "hits": list(final_hits), "promoted_hits": [], "metadata": {"audit_anchor_enabled": True, "audit_anchor_reason": "no_content_tokens"}}

    explicit_turns = _audit_anchor_turn_numbers(query)
    explicit_turn_window = set()
    for number in explicit_turns:
        explicit_turn_window.update({number - 1, number, number + 1})

    symbolic_ids = list(dict.fromkeys(str(item) for item in dict(metadata or {}).get("symbolic_recall_event_ids", []) or []))
    learned_ids = list(dict.fromkeys(str(item) for item in dict(metadata or {}).get("learned_recall_event_ids", []) or []))
    selected_ids = set(str(item) for item in dict(metadata or {}).get("selected_event_ids", []) or [])
    symbolic_rank = {event_id: index for index, event_id in enumerate(symbolic_ids, start=1)}
    learned_rank = {event_id: index for index, event_id in enumerate(learned_ids, start=1)}

    pool: Dict[str, MemoryHit] = {}
    for hit in list(final_hits) + list(candidate_hits):
        key = hit.memory_id or f"{_hit_event_id(hit)}::{hit.slot_key}::{hit.value[:80]}"
        if key and key not in pool:
            pool[key] = hit

    scored: List[tuple[float, str, List[str], MemoryHit]] = []
    for hit in pool.values():
        event_id = _hit_event_id(hit)
        hit_tokens = _audit_anchor_content_tokens(_audit_anchor_hit_text(hit))
        matched = sorted(query_tokens & hit_tokens)
        distinctive_matched = [token for token in matched if token not in _AUDIT_ANCHOR_GENERIC_MATCH_TOKENS]
        phrase_bonus, matched_phrases = _audit_anchor_phrase_bonus(query_tokens, hit_tokens)
        if not matched and int(hit.turn_index or 0) not in explicit_turn_window:
            continue
        score = 0.0
        reason_parts: List[str] = []
        if int(hit.turn_index or 0) in explicit_turn_window:
            score += 8.0
            reason_parts.append("explicit_turn_anchor")
        if matched:
            weighted_overlap = 0.0
            for token in matched:
                weighted_overlap += 1.8 if token not in _AUDIT_ANCHOR_GENERIC_MATCH_TOKENS else 0.35
            score += min(8.0, weighted_overlap)
            score += len(matched) / max(1.0, float(min(len(query_tokens), len(hit_tokens))))
            reason_parts.append("lexical_anchor_overlap")
        if phrase_bonus > 0:
            score += phrase_bonus
            reason_parts.append("distinctive_phrase_anchor")
        if event_id in symbolic_rank:
            score += max(0.0, 2.0 - (symbolic_rank[event_id] - 1) * 0.08)
            reason_parts.append("symbolic_recall_anchor")
        if event_id in learned_rank:
            score += max(0.0, 1.0 - (learned_rank[event_id] - 1) * 0.02)
            reason_parts.append("learned_recall_anchor")
        if event_id in selected_ids:
            score -= 0.25
        # For non-numeric old-topic probes, require a real content overlap so generic
        # "earlier" phrasing does not promote unrelated old memories.
        if not explicit_turns and len(distinctive_matched) < 1 and phrase_bonus <= 0:
            continue
        if score > 0:
            scored.append((score, event_id, sorted(set(matched + matched_phrases)), hit))

    scored.sort(key=lambda item: (item[0], -abs(int(item[3].turn_index or 0))), reverse=True)
    max_promoted = 2 if not explicit_turns else 3
    promoted: List[MemoryHit] = []
    promoted_event_ids: set[str] = set()
    for score, event_id, matched, hit in scored:
        if event_id in promoted_event_ids:
            continue
        if any((hit.memory_id and hit.memory_id == current.memory_id) for current in final_hits[: max(1, top_k)]):
            continue
        reason = "explicit_turn_anchor" if int(hit.turn_index or 0) in explicit_turn_window else "old_topic_anchor"
        promoted.append(_copy_hit_with_audit_anchor_boost(hit, score=score, reason=reason, matched_tokens=matched))
        promoted_event_ids.add(event_id)
        if len(promoted) >= max_promoted:
            break

    if not promoted:
        return {
            "enabled": True,
            "hits": list(final_hits),
            "promoted_hits": [],
            "metadata": {
                "audit_anchor_enabled": True,
                "audit_anchor_turn_numbers": explicit_turns,
                "audit_anchor_query_tokens": sorted(query_tokens)[:50],
                "audit_anchor_promoted_event_ids": [],
            },
        }

    merged: List[MemoryHit] = []
    seen_keys: set[str] = set()
    for hit in promoted + list(final_hits):
        key = hit.memory_id or f"{_hit_event_id(hit)}::{hit.slot_key}::{hit.value[:80]}"
        if key in seen_keys:
            continue
        seen_keys.add(key)
        merged.append(hit)
        if len(merged) >= max(1, int(top_k)):
            break

    return {
        "enabled": True,
        "hits": merged,
        "promoted_hits": promoted,
        "metadata": {
            "audit_anchor_enabled": True,
            "audit_anchor_turn_numbers": explicit_turns,
            "audit_anchor_query_tokens": sorted(query_tokens)[:50],
            "audit_anchor_promoted_event_ids": [_hit_event_id(hit) for hit in promoted],
            "audit_anchor_promoted_turns": [int(hit.turn_index) for hit in promoted],
            "audit_anchor_promoted_hit_count": len(promoted),
        },
    }


def _learnable_graph_hits(graph: SessionMemoryGraphV2) -> List[MemoryHit]:
    learnable_hits: List[MemoryHit] = []
    for record in graph.records_by_id.values():
        state = _normalize(record.state)
        metadata = dict(record.metadata or {})
        is_source_grounded_evidence = (
            state == "evidence"
            and (
                _clean_text(record.source_kind).startswith("public_dialog")
                or _clean_text(metadata.get("content_variant", "")) in {"source_turn", "llm_semantic_write"}
                or _clean_text(metadata.get("write_path", "")) == "llm_semantic_writer_gate"
            )
        )
        if state not in {"active", "parallel_active"} and not is_source_grounded_evidence:
            continue
        if _clean_text(record.slot_key).startswith("noise."):
            continue
        learnable_hits.append(
            MemoryHit(
                memory_id=record.memory_id,
                category=record.category,
                value=record.value,
                relation=record.relation,
                anchors=list(record.anchor_concepts),
                score=max(float(record.confidence), float(record.salience), 0.01),
                source_kind=record.source_kind,
                slot_key=record.slot_key,
                state=record.state,
                turn_index=int(record.turn_index),
                metadata=metadata,
            )
        )
    learnable_hits.sort(
        key=lambda item: (
            _is_public_dialog_hit(item),
            int(item.turn_index),
            float(item.score),
        ),
        reverse=True,
    )
    return learnable_hits


def _parse_structured_records(
    payload: Dict[str, Any] | None,
    *,
    turn_index: int,
    profile: TMCRAProfile | None = None,
) -> List[SessionMemoryRecordV2]:
    profile = profile or TMCRAProfile()
    records: List[SessionMemoryRecordV2] = []
    structured_rows: List[tuple[str, Mapping[str, Any]]] = []
    for raw in (payload or {}).get("replacement_memory_records", []) or []:
        if isinstance(raw, Mapping):
            structured_rows.append(("formal", raw))
    for raw in (payload or {}).get("suspect_memory_records", []) or []:
        if isinstance(raw, Mapping):
            structured_rows.append(("suspect", raw))
    for index, (buffer_state, raw) in enumerate(structured_rows):
        if not isinstance(raw, dict):
            continue
        category = _clean_text(raw.get("category", "memory")) or "memory"
        value = _clean_text(raw.get("value", ""))
        if not value:
            continue
        anchors = _dedupe(raw.get("anchors", []) or [], max_items=8)
        metadata = dict(raw.get("metadata", {}) or {})
        slot_key = profile.stable_slot_key(
            category=category,
            value=value,
            anchors=anchors,
            slot_key=_clean_text(raw.get("slot_key", "")) or _clean_text(raw.get("slot", "")),
            relation=_clean_text(raw.get("relation", "")),
            metadata=metadata,
        )
        metadata = {
            **metadata,
            "memory_role": _clean_text(metadata.get("memory_role", "")) or "user",
            "authority": _clean_text(metadata.get("authority", "")) or "source",
            "canonical_slot_key": _clean_text(metadata.get("canonical_slot_key", "")) or slot_key,
            "writeback_class": _clean_text(metadata.get("writeback_class", "")),
            "origin_query": _clean_text(metadata.get("origin_query", "")),
            "origin_answer_id": _clean_text(metadata.get("origin_answer_id", "")),
            "support_memory_ids": _dedupe(metadata.get("support_memory_ids", []) or []),
            "support_fact_refs": _dedupe(metadata.get("support_fact_refs", []) or []),
            "support_path_refs": _dedupe(metadata.get("support_path_refs", []) or []),
            "promotion_state": _clean_text(metadata.get("promotion_state", "")) or "none",
            "memory_buffer_state": buffer_state,
        }
        memory_id = f"{slot_key}:{turn_index}:{index}"
        default_source_kind = "suspect_memory" if buffer_state == "suspect" else "replacement_memory"
        default_state = "suspect" if buffer_state == "suspect" else ("active" if bool(raw.get("active", True)) else "historical")
        records.append(
            SessionMemoryRecordV2(
                memory_id=memory_id,
                category=category,
                slot_key=slot_key,
                value=value,
                relation=_clean_text(raw.get("relation", "")) or f"{category}_memory",
                anchor_concepts=anchors,
                evidence_anchors=anchors,
                salience=float(raw.get("salience", 0.88 if category in {"goal", "constraint"} else 0.74) or 0.74),
                confidence=float(raw.get("confidence", 0.82) or 0.82),
                source_kind=_clean_text(raw.get("source_kind", "")) or default_source_kind,
                turn_index=int(raw.get("turn_index", turn_index) or turn_index),
                state=_clean_text(raw.get("state", "")) or default_state,
                metadata=metadata,
            )
        )
    return records


_WRITE_MARKERS = (
    "goal update",
    "goal seed",
    "constraint update",
    "constraint overwrite",
    "constraint seed",
    "preference update",
    "preference overwrite",
    "preference seed",
    "terminology:",
    "term seed",
    "term overwrite",
    "stage update",
    "stage overwrite",
    "path fact",
    "fact:",
    "memory update",
)


_OVERWRITE_MARKERS = (
    "overwrite",
    "replace",
    "supersede",
    "覆盖",
    "替换",
    "改成",
    "更新为",
)


_TOPIC_BUCKET_CUE_GROUPS: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    (
        (
            "食物",
            "饮品",
            "甜点",
            "早餐",
            "餐饮",
            "口味",
            "酸辣",
            "热食",
            "过敏",
            "芒果",
            "配料",
            "推荐",
            "吃",
            "food",
            "breakfast",
            "allergy",
            "mango",
        ),
        ("餐饮偏好安全", "食物偏好", "过敏约束", "口味偏好"),
    ),
    (
        (
            "界面",
            "首页",
            "首屏",
            "配色",
            "营销",
            "横幅",
            "电商",
            "网站",
            "移动端",
            "视觉",
            "工具型",
            "ui",
            "homepage",
            "ecommerce",
            "mobile",
            "design",
        ),
        ("界面产品设计", "电商页面", "视觉布局", "移动端体验"),
    ),
    (
        (
            "api",
            "writer",
            "延迟",
            "算法",
            "耗时",
            "评估",
            "指标",
            "调用",
            "服务",
            "key",
            "runtime",
            "latency",
            "metric",
        ),
        ("api评估运行", "算法服务", "调用指标", "writer延迟"),
    ),
)


_TOPIC_BUCKET_STOPWORDS = {
    "不要",
    "不用",
    "复述",
    "原话",
    "只说",
    "关键",
    "最关键",
    "以后",
    "需要",
    "必须",
    "可以",
    "时候",
    "这个",
    "那个",
    "现在",
    "做",
    "说",
    "的",
    "了",
    "和",
    "与",
    "to",
    "the",
    "and",
    "for",
}


def _topic_bucket_keywords(text: str, *, max_items: int = 18) -> List[str]:
    normalized = _normalize(text)
    tokens: List[str] = []
    for cues, anchors in _TOPIC_BUCKET_CUE_GROUPS:
        if any(cue and _normalize(cue) in normalized for cue in cues):
            tokens.extend(anchors)
            tokens.extend(cue for cue in cues if len(cue) >= 2)
    tokens.extend(re.findall(r"[\u4e00-\u9fff]{2,8}", str(text or "")))
    tokens.extend(_tokenize(text))
    cleaned = []
    for token in tokens:
        value = _clean_text(token)
        if not value:
            continue
        normalized_value = _normalize(value)
        if normalized_value in _TOPIC_BUCKET_STOPWORDS:
            continue
        if len(value) < 2:
            continue
        cleaned.append(value)
    return _dedupe(cleaned, max_items=max_items)


def _topic_bucket_id_from_keywords(keywords: Sequence[str]) -> str:
    basis = "|".join(_normalize(item) for item in keywords[:8] if _clean_text(item))
    if not basis:
        basis = "general"
    return "topic-" + str(uuid.uuid5(uuid.NAMESPACE_URL, f"tmcra-topic-bucket:{basis}"))[:12]


def _topic_bucket_label_from_keywords(keywords: Sequence[str]) -> str:
    visible = [
        _clean_text(item)
        for item in keywords
        if _clean_text(item) and not _normalize(item).startswith("topic:")
    ]
    return " / ".join(visible[:3]) if visible else "动态话题"


def _topic_bucket_overlap_score(left_keywords: Sequence[str], right_keywords: Sequence[str]) -> float:
    left = {_normalize(item) for item in left_keywords if _clean_text(item)}
    right = {_normalize(item) for item in right_keywords if _clean_text(item)}
    if not left or not right:
        return 0.0
    overlap = left & right
    if not overlap:
        return 0.0
    return len(overlap) / max(1, min(len(left), len(right)))


def _explicit_topic_bucket_from_payload(answer_payload: Dict[str, Any] | None) -> Dict[str, Any]:
    metadata = dict((answer_payload or {}).get("metadata", {}) or {})
    explicit = metadata.get("topic_bucket")
    if not isinstance(explicit, Mapping):
        return {}
    keywords = _dedupe(explicit.get("topic_keywords", explicit.get("keywords", [])) or [], max_items=24)
    label = _clean_text(explicit.get("topic_label", explicit.get("label", "")))
    if not keywords and label:
        keywords = _topic_bucket_keywords(label, max_items=12)
    bucket_id = _clean_text(explicit.get("topic_bucket_id", explicit.get("id", "")))
    if not bucket_id:
        bucket_id = _topic_bucket_id_from_keywords(keywords or [label])
    return {
        "topic_bucket_id": bucket_id,
        "topic_label": label or _topic_bucket_label_from_keywords(keywords),
        "topic_keywords": keywords,
        "topic_confidence": float(explicit.get("confidence", explicit.get("topic_confidence", 0.92)) or 0.92),
        "topic_assignment_source": "explicit_payload",
    }


def _coerce_topic_bucket(metadata: Mapping[str, Any]) -> Dict[str, Any]:
    bucket_id = _clean_text(metadata.get("topic_bucket_id", ""))
    if not bucket_id:
        nested = metadata.get("topic_bucket")
        if isinstance(nested, Mapping):
            bucket_id = _clean_text(nested.get("topic_bucket_id", nested.get("id", "")))
    if not bucket_id:
        return {}
    keywords = _dedupe(metadata.get("topic_keywords", []) or [], max_items=32)
    nested = metadata.get("topic_bucket")
    if isinstance(nested, Mapping):
        keywords = _dedupe([*keywords, *(nested.get("topic_keywords", nested.get("keywords", [])) or [])], max_items=32)
    label = _clean_text(metadata.get("topic_label", ""))
    if not label and isinstance(nested, Mapping):
        label = _clean_text(nested.get("topic_label", nested.get("label", "")))
    return {
        "topic_bucket_id": bucket_id,
        "topic_label": label or _topic_bucket_label_from_keywords(keywords),
        "topic_keywords": keywords,
    }


def _collect_topic_buckets(graph: SessionMemoryGraphV2) -> Dict[str, Dict[str, Any]]:
    buckets: Dict[str, Dict[str, Any]] = {}

    def merge(bucket: Mapping[str, Any], *, record_id: str = "", turn_index: int = 0) -> None:
        bucket_id = _clean_text(bucket.get("topic_bucket_id", bucket.get("id", "")))
        if not bucket_id:
            return
        current = buckets.setdefault(
            bucket_id,
            {
                "topic_bucket_id": bucket_id,
                "topic_label": _clean_text(bucket.get("topic_label", bucket.get("label", ""))),
                "topic_keywords": [],
                "record_ids": [],
                "last_turn_index": 0,
            },
        )
        current["topic_label"] = current.get("topic_label") or _clean_text(bucket.get("topic_label", bucket.get("label", "")))
        current["topic_keywords"] = _dedupe(
            [*list(current.get("topic_keywords", []) or []), *(bucket.get("topic_keywords", bucket.get("keywords", [])) or [])],
            max_items=32,
        )
        if record_id:
            current["record_ids"] = _dedupe([*list(current.get("record_ids", []) or []), record_id], max_items=200)
        current["last_turn_index"] = max(int(current.get("last_turn_index", 0) or 0), int(turn_index or 0))

    for record in getattr(graph, "records_by_id", {}).values():
        metadata = dict(record.metadata or {})
        bucket = _coerce_topic_bucket(metadata)
        if bucket:
            merge(bucket, record_id=record.memory_id, turn_index=int(record.turn_index))
    for turn in list(getattr(graph, "turn_log", []) or []):
        metadata = dict(turn.get("metadata", {}) if isinstance(turn, Mapping) else getattr(turn, "metadata", {}) or {})
        bucket = _coerce_topic_bucket(metadata)
        if bucket:
            merge(bucket, turn_index=int(turn.get("turn_index", 0) if isinstance(turn, Mapping) else getattr(turn, "turn_index", 0) or 0))
    for bucket in buckets.values():
        if not bucket.get("topic_label"):
            bucket["topic_label"] = _topic_bucket_label_from_keywords(bucket.get("topic_keywords", []) or [])
    return buckets


def _assign_topic_bucket_for_text(
    graph: SessionMemoryGraphV2,
    text: str,
    *,
    answer_payload: Dict[str, Any] | None = None,
    turn_index: int = 0,
    create: bool = True,
) -> Dict[str, Any]:
    explicit = _explicit_topic_bucket_from_payload(answer_payload)
    if explicit:
        explicit["turn_index"] = int(turn_index or 0)
        return explicit
    keywords = _topic_bucket_keywords(text, max_items=24)
    if not keywords:
        keywords = _dedupe([_clean_text(text)[:40]], max_items=1)
    buckets = _collect_topic_buckets(graph)
    best_bucket: Dict[str, Any] | None = None
    best_score = 0.0
    for bucket in buckets.values():
        score = _topic_bucket_overlap_score(keywords, bucket.get("topic_keywords", []) or [])
        if score > best_score:
            best_bucket = bucket
            best_score = score
    if best_bucket and best_score >= 0.22:
        merged_keywords = _dedupe([*list(best_bucket.get("topic_keywords", []) or []), *keywords], max_items=32)
        return {
            "topic_bucket_id": best_bucket["topic_bucket_id"],
            "topic_label": best_bucket.get("topic_label") or _topic_bucket_label_from_keywords(merged_keywords),
            "topic_keywords": merged_keywords,
            "topic_confidence": round(min(0.97, 0.68 + best_score * 0.24), 6),
            "topic_assignment_source": "reused_by_overlap",
            "topic_match_score": round(best_score, 6),
            "turn_index": int(turn_index or 0),
        }
    bucket_id = _topic_bucket_id_from_keywords(keywords)
    return {
        "topic_bucket_id": bucket_id,
        "topic_label": _topic_bucket_label_from_keywords(keywords),
        "topic_keywords": keywords,
        "topic_confidence": 0.72 if create else 0.58,
        "topic_assignment_source": "created_by_dialog" if create else "query_probe",
        "topic_match_score": round(best_score, 6),
        "turn_index": int(turn_index or 0),
    }


def _apply_topic_bucket_to_records(records: List[SessionMemoryRecordV2], topic_bucket: Mapping[str, Any]) -> None:
    if not records or not topic_bucket:
        return
    bucket_id = _clean_text(topic_bucket.get("topic_bucket_id", ""))
    if not bucket_id:
        return
    label = _clean_text(topic_bucket.get("topic_label", "")) or "动态话题"
    keywords = _dedupe(topic_bucket.get("topic_keywords", []) or [], max_items=32)
    for record in records:
        metadata = dict(record.metadata or {})
        metadata.update(
            {
                "topic_bucket_id": bucket_id,
                "topic_label": label,
                "topic_keywords": keywords,
                "topic_confidence": float(topic_bucket.get("topic_confidence", 0.72) or 0.72),
                "topic_assignment_source": _clean_text(topic_bucket.get("topic_assignment_source", "")) or "created_by_dialog",
            }
        )
        record.metadata = metadata
        record.anchor_concepts = _dedupe([*list(record.anchor_concepts or []), f"topic:{label}", *keywords[:8]], max_items=24)
        evidence_anchors = list(metadata.get("evidence_anchors", []) or [])
        metadata["evidence_anchors"] = _dedupe([*evidence_anchors, f"topic:{label}", *keywords[:8]], max_items=24)


def _last_topic_turn(graph: SessionMemoryGraphV2) -> Dict[str, Any]:
    for turn in reversed(list(getattr(graph, "turn_log", []) or [])):
        metadata = dict(turn.get("metadata", {}) if isinstance(turn, Mapping) else getattr(turn, "metadata", {}) or {})
        bucket = _coerce_topic_bucket(metadata)
        if bucket:
            bucket["turn_index"] = int(turn.get("turn_index", 0) if isinstance(turn, Mapping) else getattr(turn, "turn_index", 0) or 0)
            return bucket
    return {}


def _add_topic_bridge_edges(
    graph: SessionMemoryGraphV2,
    *,
    previous_topic: Mapping[str, Any],
    current_topic: Mapping[str, Any],
    current_record_ids: Sequence[str],
    turn_index: int,
    evidence: str,
) -> Dict[str, Any]:
    previous_id = _clean_text(previous_topic.get("topic_bucket_id", ""))
    current_id = _clean_text(current_topic.get("topic_bucket_id", ""))
    if not previous_id or not current_id or previous_id == current_id or not current_record_ids:
        return {"topic_bridge_edge_count": 0}
    buckets = _collect_topic_buckets(graph)
    previous_record_ids = list((buckets.get(previous_id, {}) or {}).get("record_ids", []) or [])[-4:]
    if not previous_record_ids:
        return {"topic_bridge_edge_count": 0}
    edge_count = 0
    evidence_text = _clean_text(evidence)[:240]
    for source_id in previous_record_ids:
        for target_id in list(current_record_ids)[:4]:
            if source_id == target_id:
                continue
            edge = SessionMemoryEdgeV2(
                edge_id=f"{source_id}->{target_id}:topic_bridge:{previous_id}->{current_id}",
                source_memory_id=source_id,
                target_memory_id=target_id,
                edge_type="topic_bridge",
                score=0.54,
                model_score=0.0,
                evidence_turn=int(turn_index or 0),
                evidence=evidence_text,
                metadata={
                    "from_topic_bucket_id": previous_id,
                    "to_topic_bucket_id": current_id,
                    "from_topic_label": _clean_text(previous_topic.get("topic_label", "")),
                    "to_topic_label": _clean_text(current_topic.get("topic_label", "")),
                    "bridge_reason": "adjacent_dialog_topic_transition",
                },
            )
            graph._upsert_memory_edge(edge)
            edge_count += 1
    return {
        "topic_bridge_edge_count": edge_count,
        "topic_bridge_from": previous_id,
        "topic_bridge_to": current_id,
    }


def _add_dialogue_tunnel_edges(
    graph: SessionMemoryGraphV2,
    *,
    current_topic: Mapping[str, Any],
    current_record_ids: Sequence[str],
    turn_index: int,
    evidence: str,
) -> Dict[str, Any]:
    current_id = _clean_text(current_topic.get("topic_bucket_id", ""))
    if not current_id or not current_record_ids:
        return {"dialogue_tunnel_edge_count": 0}
    buckets = _collect_topic_buckets(graph)
    edge_count = 0
    evidence_text = _clean_text(evidence)[:240]
    source_ids: List[tuple[str, str, str]] = []
    for bucket_id, bucket in sorted(
        buckets.items(),
        key=lambda item: int(item[1].get("last_turn_index", 0) or 0),
        reverse=True,
    ):
        if bucket_id == current_id:
            continue
        for source_id in list(bucket.get("record_ids", []) or [])[-2:]:
            if source_id not in current_record_ids:
                source_ids.append((bucket_id, _clean_text(bucket.get("topic_label", "")), source_id))
        if len(source_ids) >= 6:
            break
    for source_bucket_id, source_label, source_id in source_ids[:6]:
        for target_id in list(current_record_ids)[:2]:
            if source_id == target_id:
                continue
            edge = SessionMemoryEdgeV2(
                edge_id=f"{source_id}->{target_id}:dialogue_tunnel:{source_bucket_id}->{current_id}",
                source_memory_id=source_id,
                target_memory_id=target_id,
                edge_type="dialogue_tunnel",
                score=0.24,
                model_score=0.0,
                evidence_turn=int(turn_index or 0),
                evidence=evidence_text,
                metadata={
                    "from_topic_bucket_id": source_bucket_id,
                    "to_topic_bucket_id": current_id,
                    "from_topic_label": source_label,
                    "to_topic_label": _clean_text(current_topic.get("topic_label", "")),
                    "bridge_reason": "high_resistance_dialogue_level_tunnel",
                },
            )
            graph._upsert_memory_edge(edge)
            edge_count += 1
    return {"dialogue_tunnel_edge_count": edge_count}


def _topic_adjacent_bucket_ids(graph: SessionMemoryGraphV2, bucket_id: str) -> set[str]:
    adjacent: set[str] = set()
    if not bucket_id:
        return adjacent
    for edge in getattr(graph, "memory_edges", {}).values():
        if _normalize(edge.edge_type) != "topic_bridge":
            continue
        metadata = dict(edge.metadata or {})
        left = _clean_text(metadata.get("from_topic_bucket_id", ""))
        right = _clean_text(metadata.get("to_topic_bucket_id", ""))
        if left == bucket_id and right:
            adjacent.add(right)
        if right == bucket_id and left:
            adjacent.add(left)
    return adjacent


def _dialogue_tunnel_bucket_ids(graph: SessionMemoryGraphV2, bucket_id: str) -> set[str]:
    adjacent: set[str] = set()
    if not bucket_id:
        return adjacent
    for edge in getattr(graph, "memory_edges", {}).values():
        if _normalize(edge.edge_type) != "dialogue_tunnel":
            continue
        metadata = dict(edge.metadata or {})
        left = _clean_text(metadata.get("from_topic_bucket_id", ""))
        right = _clean_text(metadata.get("to_topic_bucket_id", ""))
        if left == bucket_id and right:
            adjacent.add(right)
        if right == bucket_id and left:
            adjacent.add(left)
    return adjacent


def _topic_bridge_requested(query: str) -> bool:
    text = _normalize(query)
    if not text:
        return False
    bridge_markers = (
        "关联",
        "联系",
        "链条",
        "脉络",
        "延展",
        "深入",
        "对比",
        "整合",
        "整体",
        "上下文",
        "刚才",
        "之前",
        "上面",
        "跨话题",
        "隧穿",
        "related",
        "connect",
        "compare",
        "context",
        "chain",
    )
    return any(marker in text for marker in bridge_markers)


def _dialogue_tunnel_requested(query: str) -> bool:
    text = _normalize(query)
    if not text:
        return False
    markers = (
        "跨话题",
        "跨对话",
        "不同话题",
        "不同对话",
        "历史对话",
        "长期脉络",
        "整体脉络",
        "所有相关",
        "全局",
        "全局记忆",
        "远一点",
        "更深",
        "深层关联",
        "对话级隧穿",
        "记忆隧穿",
        "cross topic",
        "cross-topic",
        "cross dialogue",
        "cross-session",
        "global context",
        "long range",
    )
    return any(marker in text for marker in markers)


def _topic_bucket_record_to_hit(
    record: SessionMemoryRecordV2,
    *,
    query_topic: Mapping[str, Any],
    rank: int,
    rescue_kind: str = "topic_bucket",
) -> MemoryHit:
    metadata = dict(record.metadata or {})
    dialogue_tunnel = _normalize(rescue_kind) == "dialogue_tunnel"
    metadata.update(
        {
            "topic_bucket_rescue": not dialogue_tunnel,
            "dialogue_tunnel_rescue": dialogue_tunnel,
            "topic_bucket_rescue_rank": int(rank),
            "topic_bucket_query_id": _clean_text(query_topic.get("topic_bucket_id", "")),
            "topic_bucket_query_label": _clean_text(query_topic.get("topic_label", "")),
            "topic_bucket_same": not dialogue_tunnel,
            "topic_bucket_bridge": False,
            "topic_bucket_bridge_allowed": False,
            "topic_bucket_dialogue_tunnel_allowed": dialogue_tunnel,
            "topic_bucket_overlap": 0.0 if dialogue_tunnel else 1.0,
        }
    )
    category = _normalize(record.category)
    value_text = _normalize(record.value)
    hardish = (
        category == "constraint"
        or _normalize(metadata.get("memory_type", "")) == "hard_constraint"
        or _normalize(metadata.get("durability", "")) == "hard"
        or _normalize(metadata.get("conflict_policy", "")) == "must_preserve"
        or any(marker in value_text for marker in ("过敏", "必须", "避开", "禁止", "不能", "must", "avoid", "allergy"))
    )
    base_score = max(float(record.confidence), float(record.salience), 0.62)
    if dialogue_tunnel:
        if hardish:
            base_score += 0.42
            metadata.setdefault("memory_type", "hard_constraint")
            metadata.setdefault("durability", "hard")
            metadata.setdefault("conflict_policy", "must_preserve")
        elif category == "preference":
            base_score += 0.22
            metadata.setdefault("memory_type", "durable_preference")
            metadata.setdefault("durability", "long_term")
        else:
            base_score += 0.12
        metadata["dialogue_tunnel_resistance"] = "high"
    elif hardish:
        base_score += 1.35
        metadata.setdefault("memory_type", "hard_constraint")
        metadata.setdefault("durability", "hard")
        metadata.setdefault("conflict_policy", "must_preserve")
    elif category == "preference":
        base_score += 0.72
        metadata.setdefault("memory_type", "durable_preference")
        metadata.setdefault("durability", "long_term")
    else:
        base_score += 0.38
    return MemoryHit(
        memory_id=record.memory_id,
        category=record.category,
        value=record.value,
        relation=record.relation,
        anchors=list(record.anchor_concepts),
        score=base_score,
        source_kind=record.source_kind,
        slot_key=record.slot_key,
        state=record.state,
        turn_index=int(record.turn_index),
        metadata=metadata,
    )


def _profile_query_rescue_hits(
    graph: SessionMemoryGraphV2,
    query: str,
    *,
    top_k: int,
) -> List[MemoryHit]:
    query_raw_tokens = set(_path_utility_tokens(query))
    query_tokens = _profile_query_expanded_tokens(query)
    intent = infer_profile_query_intent(query)
    if not bool(intent.get("enabled")):
        return []
    rescued: List[tuple[float, MemoryHit]] = []
    for record in getattr(graph, "records_by_id", {}).values():
        metadata = dict(record.metadata or {})
        if record.state != "active":
            continue
        if not is_profile_layer_record(
            category=record.category,
            source_kind=record.source_kind,
            semantic_slot=metadata.get("semantic_slot", ""),
            metadata=metadata,
        ):
            continue
        delta, reason = profile_query_score_delta(
            query=query,
            query_tokens=query_tokens,
            category=record.category,
            source_kind=record.source_kind,
            semantic_slot=metadata.get("semantic_slot", ""),
            value=record.value,
            anchors=record.anchor_concepts,
            metadata=metadata,
        )
        if delta <= 0:
            continue
        match_score, overlap_tokens, raw_overlap_tokens = _profile_hit_match_score(
            query_raw_tokens,
            query_tokens,
            MemoryHit(
                memory_id=record.memory_id,
                category=record.category,
                value=record.value,
                relation=record.relation,
                anchors=list(record.anchor_concepts),
                score=max(float(record.confidence), float(record.salience), 0.01),
                source_kind=record.source_kind,
                slot_key=record.slot_key,
                state=record.state,
                turn_index=int(record.turn_index),
                metadata=metadata,
            ),
        )
        if match_score <= 0.0:
            continue
        if match_score < 0.34 and not raw_overlap_tokens:
            continue
        metadata.update(
            {
                "profile_query_rescue": True,
                "profile_query_rescue_reason": reason or "profile_route",
                "profile_query_match_score": round(match_score, 6),
                "profile_query_overlap_tokens": list(overlap_tokens),
                "profile_query_raw_overlap_tokens": list(raw_overlap_tokens),
                "topic_bucket_profile_route_preserved": True,
                "match_reason": ",".join(_dedupe([metadata.get("match_reason", ""), reason or "profile_route"], max_items=4)),
            }
        )
        hit = MemoryHit(
            memory_id=record.memory_id,
            category=record.category,
            value=record.value,
            relation=record.relation,
            anchors=list(record.anchor_concepts),
            score=max(float(record.confidence), float(record.salience), 0.62) + float(delta) + float(match_score),
            source_kind=record.source_kind,
            slot_key=record.slot_key,
            state=record.state,
            turn_index=int(record.turn_index),
            metadata=metadata,
        )
        rescued.append((match_score, hit))
    rescued.sort(key=lambda item: (float(item[0]), float(item[1].score), int(item[1].turn_index)), reverse=True)
    return [
        hit
        for _, hit in rescued[: max(1, min(24, int(top_k or 1) * 3))]
    ]


def _memory_hit_from_record(record: SessionMemoryRecordV2, *, score: float | None = None, metadata: Mapping[str, Any] | None = None) -> MemoryHit:
    record_metadata = {**dict(record.metadata or {}), **dict(metadata or {})}
    return MemoryHit(
        memory_id=record.memory_id,
        category=record.category,
        value=record.value,
        relation=record.relation,
        anchors=list(record.anchor_concepts),
        score=max(float(record.confidence), float(record.salience), 0.01) if score is None else float(score),
        source_kind=record.source_kind,
        slot_key=record.slot_key,
        state=record.state,
        turn_index=int(record.turn_index),
        metadata=record_metadata,
    )


_FACET_NUMERIC_QUERY_TOKENS = {
    "amount",
    "count",
    "counts",
    "duration",
    "durations",
    "many",
    "much",
    "number",
    "quantity",
    "sum",
    "total",
    "totals",
    "weeks",
    "week",
    "hours",
    "hour",
    "dollars",
    "dollar",
    "tenants",
    "tickets",
}
_FACET_TEMPORAL_QUERY_TOKENS = {
    "after",
    "before",
    "date",
    "deadline",
    "end",
    "finish",
    "finished",
    "start",
    "started",
    "time",
    "when",
}


def _facet_query_pack_hits(
    graph: SessionMemoryGraphV2,
    query: str,
    final_hits: Sequence[MemoryHit],
    *,
    top_k: int,
) -> Dict[str, Any]:
    query_tokens = set(_path_utility_tokens(query))
    if not query_tokens:
        return {"hits": list(final_hits), "metadata": {"facet_query_pack_enabled": False, "facet_query_pack_reason": "empty_query_tokens"}}
    numeric_query = bool(query_tokens & _FACET_NUMERIC_QUERY_TOKENS)
    temporal_query = bool(query_tokens & _FACET_TEMPORAL_QUERY_TOKENS)
    if not numeric_query and not temporal_query and not any("facet" in _normalize(token) for token in query_tokens):
        return {"hits": list(final_hits), "metadata": {"facet_query_pack_enabled": False, "facet_query_pack_reason": "no_facet_intent"}}

    candidate_rows: List[tuple[float, SessionMemoryRecordV2, SessionMemoryRecordV2 | None, Dict[str, Any]]] = []
    for record in getattr(graph, "records_by_id", {}).values():
        metadata = dict(record.metadata or {})
        if _normalize(metadata.get("content_variant", "")) != "event_facet_write":
            continue
        if record.state not in {"active", "parallel_active", "evidence"}:
            continue
        facet_type = _normalize(metadata.get("facet_type", ""))
        parent_slot_key = _clean_text(metadata.get("facet_parent_slot_key", ""))
        parent = next(
            (
                candidate
                for candidate in getattr(graph, "records_by_id", {}).values()
                if _clean_text(candidate.slot_key).lower() == parent_slot_key.lower()
            ),
            None,
        )
        parent_text = " ".join(
            [
                _clean_text(parent.value if parent else ""),
                " ".join(parent.anchor_concepts if parent else []),
                _clean_text(dict(parent.metadata or {}).get("source_span", "") if parent else ""),
            ]
        )
        facet_text = " ".join(
            [
                record.value,
                " ".join(record.anchor_concepts or []),
                _clean_text(metadata.get("facet_type", "")),
                _clean_text(metadata.get("facet_role", "")),
                _clean_text(metadata.get("facet_source_span", "")),
                parent_text,
            ]
        )
        facet_tokens = set(_path_utility_tokens(facet_text))
        parent_tokens = set(_path_utility_tokens(parent_text))
        overlap_tokens = query_tokens & facet_tokens
        parent_overlap_tokens = query_tokens & parent_tokens
        unit_overlap = bool(query_tokens & set(_path_utility_tokens(record.value)))
        score = 0.0
        if overlap_tokens:
            score += min(0.72, len(overlap_tokens) / max(1.0, len(query_tokens)) * 1.2)
        if parent_overlap_tokens:
            score += min(0.72, len(parent_overlap_tokens) / max(1.0, len(query_tokens)) * 1.35)
        if numeric_query and facet_type == "numeric":
            score += 0.42
            if unit_overlap:
                score += 0.42
        if temporal_query and facet_type == "temporal":
            score += 0.32
        if facet_type == "entity" and parent_overlap_tokens:
            score += 0.26
        if parent is not None and _normalize(dict(parent.metadata or {}).get("content_variant", "")) == "llm_semantic_write":
            score += 0.08
        if score < 0.58:
            continue
        candidate_rows.append(
            (
                round(min(2.75, 1.18 + score), 6),
                record,
                parent,
                {
                    "facet_query_pack_overlap_tokens": sorted(overlap_tokens)[:12],
                    "facet_query_pack_parent_overlap_tokens": sorted(parent_overlap_tokens)[:12],
                    "facet_query_pack_unit_overlap": bool(unit_overlap),
                    "facet_query_pack_score": round(score, 6),
                },
            )
        )

    if not candidate_rows:
        return {
            "hits": list(final_hits),
            "metadata": {
                "facet_query_pack_enabled": True,
                "facet_query_pack_inserted_hit_count": 0,
                "facet_query_pack_candidate_count": 0,
            },
        }

    candidate_rows.sort(key=lambda item: (float(item[0]), int(item[1].turn_index)), reverse=True)
    selected = candidate_rows[: max(4, min(18, int(top_k or 1) * 2))]
    packed_hits: List[MemoryHit] = []
    for score, facet_record, parent, extra_metadata in selected:
        facet_metadata = {
            **extra_metadata,
            "facet_query_pack": True,
            "evidence_snippet_role": "facet_query_attribute",
        }
        packed_hits.append(_memory_hit_from_record(facet_record, score=score, metadata=facet_metadata))
        if parent is not None:
            packed_hits.append(
                _memory_hit_from_record(
                    parent,
                    score=max(1.12, score - 0.04),
                    metadata={
                        **extra_metadata,
                        "facet_query_pack": True,
                        "evidence_snippet_role": "facet_parent_event",
                        "facet_query_pack_child_id": facet_record.memory_id,
                    },
                )
            )

    merged: List[MemoryHit] = []
    seen_ids: set[str] = set()
    for hit in [*packed_hits, *list(final_hits)]:
        if hit.memory_id and hit.memory_id in seen_ids:
            continue
        if hit.memory_id:
            seen_ids.add(hit.memory_id)
        merged.append(hit)
    return {
        "hits": merged,
        "metadata": {
            "facet_query_pack_enabled": True,
            "facet_query_pack_candidate_count": len(candidate_rows),
            "facet_query_pack_inserted_hit_count": len(packed_hits),
            "facet_query_pack_numeric_query": bool(numeric_query),
            "facet_query_pack_temporal_query": bool(temporal_query),
        },
    }


_UNIT_COVERAGE_COUNT_TOKENS = {
    "amount",
    "amounts",
    "count",
    "counts",
    "cost",
    "costs",
    "dollar",
    "dollars",
    "how",
    "many",
    "much",
    "number",
    "minimum",
    "maximum",
    "paid",
    "percent",
    "percentage",
    "price",
    "prices",
    "sale",
    "sales",
    "sell",
    "sold",
    "total",
    "totals",
    "sum",
    "value",
    "valued",
    "values",
    "worth",
    "items",
    "projects",
    "events",
    "things",
}
_UNIT_COVERAGE_TEMPORAL_TOKENS = {
    "ago",
    "after",
    "before",
    "between",
    "consecutive",
    "date",
    "days",
    "day",
    "first",
    "last",
    "months",
    "month",
    "order",
    "passed",
    "since",
    "weeks",
    "week",
}
_MULTI_UNIT_CHAIN_TEMPORAL_COMPARISON_TOKENS = {
    "after",
    "before",
    "between",
    "consecutive",
    "earlier",
    "first",
    "later",
    "last",
    "order",
    "since",
}
_UNIT_COVERAGE_QUERY_DROP_TOKENS = {
    "date",
    "fri",
    "friday",
    "mon",
    "monday",
    "question",
    "sat",
    "saturday",
    "sun",
    "sunday",
    "thu",
    "thursday",
    "tue",
    "tuesday",
    "wed",
    "wednesday",
}
_MULTI_UNIT_CHAIN_DISABLED_MODES = {"", "off", "disabled", "none", "false", "0"}
_MULTI_UNIT_CHAIN_COUNT_TOKENS = {
    *_UNIT_COVERAGE_COUNT_TOKENS,
    "which",
    "each",
    "all",
    "both",
}
_MULTI_UNIT_CHAIN_FOCUS_DROP_TOKENS = {
    *_MULTI_UNIT_CHAIN_COUNT_TOKENS,
    *_UNIT_COVERAGE_TEMPORAL_TOKENS,
    "i",
    "me",
    "my",
    "mine",
    "am",
    "is",
    "are",
    "was",
    "were",
    "need",
    "needs",
    "needed",
    "currently",
    "did",
    "does",
    "fri",
    "friday",
    "mon",
    "monday",
    "or",
    "question",
    "sat",
    "saturday",
    "sun",
    "sunday",
    "thu",
    "thursday",
    "tue",
    "tuesday",
    "wed",
    "wednesday",
    "what",
    "which",
    "who",
}
_MULTI_UNIT_CHAIN_FACET_TYPES = {"action", "entity", "numeric", "role", "evidence_role", "state", "temporal"}
_MULTI_UNIT_CHAIN_UNIT_KINDS = {
    "action_unit",
    "target_entity",
    "numeric_quantity",
    "leadership_role",
    "participation_role",
    "state_status",
    "temporal_anchor",
    "evidence_role",
    "profile_shadow_unit",
}
_MULTI_UNIT_CHAIN_NUMERIC_VALUE_TOKENS = {
    "amount",
    "appraisal",
    "appraised",
    "cost",
    "costs",
    "dollar",
    "dollars",
    "minimum",
    "paid",
    "price",
    "prices",
    "sale",
    "sell",
    "sold",
    "total",
    "value",
    "valued",
    "values",
    "worth",
}


def _multi_unit_chain_numeric_signal(
    unit_kind: str,
    facet_type: str,
    text: str,
    tokens: set[str],
) -> float:
    compact_text = _clean_text(text)
    date_like_only = bool(
        re.fullmatch(
            r"(?:\d{4}[/-]\d{1,2}(?:[/-]\d{1,2})?|\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?|\d{4}/\d{2})",
            compact_text,
        )
    )
    if date_like_only:
        return 0.0
    signal = 0.0
    if unit_kind == "numeric_quantity" or facet_type == "numeric":
        signal += 0.55
    if tokens & _MULTI_UNIT_CHAIN_NUMERIC_VALUE_TOKENS:
        signal += 0.28
    if re.search(r"(?:[$€£¥]\s*\d|\b\d+(?:,\d{3})*(?:\.\d+)?\s*(?:dollars?|usd|bucks?|yuan|rmb)\b)", text, flags=re.IGNORECASE):
        signal += 0.72
    elif re.search(r"\b\d+(?:,\d{3})*(?:\.\d+)?\s*(?:comments?|items?|pieces?|kits?|projects?|doctors?|weddings?|hours?|weeks?|days?|months?|years?|miles?)\b", text, flags=re.IGNORECASE):
        signal += 0.42
    return min(1.1, signal)


def _multi_unit_chain_date_like_numeric(text: str) -> bool:
    compact_text = _clean_text(text)
    return bool(
        re.fullmatch(
            r"(?:\d{4}[/-]\d{1,2}(?:[/-]\d{1,2})?|\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?|\d{4}/\d{2})",
            compact_text,
        )
    )


def _multi_unit_chain_temporal_anchor_signal(text: str, tokens: set[str]) -> float:
    signal = 0.0
    if tokens & _UNIT_COVERAGE_TEMPORAL_TOKENS:
        signal += 0.22
    if re.search(
        r"\b(?:about|around|roughly|few|several|couple|last|previous|earlier)\s+"
        r"(?:a\s+)?(?:day|days|week|weeks|month|months|year|years)\s+ago\b|"
        r"\b(?:yesterday|today|tomorrow|last\s+week|last\s+month|a\s+few\s+months\s+ago)\b",
        text,
        flags=re.IGNORECASE,
    ):
        signal += 0.72
    return min(1.0, signal)


_PROFILE_SHADOW_EVENTLIKE_SLOT_HINTS = {
    "action",
    "appointment",
    "constraint",
    "deadline",
    "exchange",
    "obligation",
    "pickup",
    "plan",
    "preference",
    "return",
    "status",
    "task",
}
_PROFILE_SHADOW_EVENTLIKE_TEXT_HINTS = {
    "bought",
    "buy",
    "completed",
    "did",
    "exchange",
    "exchanged",
    "finish",
    "finished",
    "got",
    "have to",
    "need",
    "needed",
    "needs",
    "paid",
    "pick",
    "picked",
    "return",
    "returned",
    "should",
    "still",
    "took",
    "went",
}


def _profile_shadow_eventlike_record(record: SessionMemoryRecordV2, metadata: Mapping[str, Any]) -> bool:
    if _normalize(metadata.get("content_variant", "")) != "profile_shadow_from_writer":
        return False
    if record.state not in {"active", "parallel_active", "evidence"}:
        return False
    slot_text = " ".join(
        [
            _clean_text(record.category),
            _clean_text(record.relation),
            _clean_text(record.slot_key),
            _clean_text(metadata.get("semantic_slot", "")),
            _clean_text(metadata.get("profile_type", "")),
        ]
    ).lower()
    value_text = " ".join(
        [
            _clean_text(record.value),
            _clean_text(metadata.get("source_span", "")),
            _clean_text(metadata.get("raw_text", "")),
        ]
    ).lower()
    return bool(
        any(hint in slot_text for hint in _PROFILE_SHADOW_EVENTLIKE_SLOT_HINTS)
        or any(hint in value_text for hint in _PROFILE_SHADOW_EVENTLIKE_TEXT_HINTS)
    )


def _profile_shadow_unit_text(record: SessionMemoryRecordV2, metadata: Mapping[str, Any]) -> str:
    return " ".join(
        [
            _clean_text(record.value),
            _clean_text(record.category),
            _clean_text(record.relation),
            _clean_text(record.slot_key),
            _clean_text(metadata.get("semantic_slot", "")),
            _clean_text(metadata.get("profile_type", "")),
            _clean_text(metadata.get("profile_domain", "")),
            _clean_text(metadata.get("source_span", "")),
            _clean_text(metadata.get("raw_text", "")),
        ]
    )


def _unit_coverage_pack_hits(
    graph: SessionMemoryGraphV2,
    query: str,
    final_hits: Sequence[MemoryHit],
    *,
    top_k: int,
) -> Dict[str, Any]:
    raw_query_tokens = set(_path_utility_tokens(query))
    query_tokens = {token for token in raw_query_tokens if token not in _UNIT_COVERAGE_QUERY_DROP_TOKENS}
    if not query_tokens:
        return {"hits": list(final_hits), "metadata": {"unit_coverage_pack_enabled": False, "unit_coverage_reason": "empty_query"}}
    count_intent = bool(query_tokens & _UNIT_COVERAGE_COUNT_TOKENS)
    percentage_intent = bool(query_tokens & {"percent", "percentage"})
    temporal_intent = bool(raw_query_tokens & _MULTI_UNIT_CHAIN_TEMPORAL_COMPARISON_TOKENS)
    direct_unit_intent = not count_intent and not temporal_intent and len(query_tokens) >= 2
    if not count_intent and not temporal_intent and not direct_unit_intent:
        return {
            "hits": list(final_hits),
            "metadata": {"unit_coverage_pack_enabled": False, "unit_coverage_reason": "no_unit_intent"},
        }

    records = list(getattr(graph, "records_by_id", {}).values())
    parent_by_slot = {_clean_text(record.slot_key).lower(): record for record in records}
    candidates: List[tuple[float, SessionMemoryRecordV2, SessionMemoryRecordV2 | None, Dict[str, Any]]] = []
    for record in records:
        metadata = dict(record.metadata or {})
        if record.state not in {"active", "parallel_active", "evidence"}:
            continue
        profile_shadow_unit = _profile_shadow_eventlike_record(record, metadata)
        if profile_shadow_unit:
            unit_kind = "profile_shadow_unit"
            facet_type = "action"
            parent_slot = _clean_text(record.slot_key).lower()
            parent = None
            parent_text = ""
            unit_text = _profile_shadow_unit_text(record, metadata)
        else:
            if _normalize(metadata.get("content_variant", "")) != "event_facet_write":
                continue
            if _normalize(metadata.get("facet_layer_version", "")) not in {"event_unit_v1", "event_facet_v1"}:
                continue
            unit_kind = _normalize(metadata.get("unit_kind", ""))
            facet_type = _normalize(metadata.get("facet_type", ""))
            parent_slot = _clean_text(metadata.get("facet_parent_slot_key", "")).lower()
            parent = parent_by_slot.get(parent_slot)
            parent_text = " ".join(
                [
                    _clean_text(parent.value if parent else ""),
                    _clean_text(dict(parent.metadata or {}).get("source_span", "") if parent else ""),
                    " ".join(parent.anchor_concepts if parent else []),
                ]
            )
            unit_text = " ".join(
                [
                    _clean_text(record.value),
                    _clean_text(metadata.get("facet_role", "")),
                    _clean_text(metadata.get("unit_kind", "")),
                    _clean_text(metadata.get("action", "")),
                    _clean_text(metadata.get("target", "")),
                    _clean_text(metadata.get("quantity", "")),
                    _clean_text(metadata.get("unit", "")),
                    _clean_text(metadata.get("normalized_time", "")),
                    _clean_text(metadata.get("status", "")),
                    _clean_text(metadata.get("facet_source_span", "")),
                    parent_text,
                ]
            )
        unit_tokens = set(_path_utility_tokens(unit_text))
        semantic_event_unit = _normalize(record.memory_id).startswith("tmcra.event.") or _normalize(parent_slot).startswith("tmcra.event.")
        overlap = query_tokens & unit_tokens
        score = 0.0
        if overlap:
            score += min(0.92, len(overlap) / max(1.0, len(query_tokens)) * 1.55)
        if count_intent and facet_type in {"action", "entity", "numeric", "role", "evidence_role"}:
            score += 0.42
        if temporal_intent and facet_type in {"temporal", "action", "role", "entity"}:
            score += 0.38
        if direct_unit_intent and facet_type in {"action", "entity", "state", "numeric"}:
            score += 0.30
        if direct_unit_intent and len(overlap) >= 2:
            score += 0.36
        if unit_kind in {"action_unit", "target_entity", "leadership_role", "participation_role"}:
            score += 0.16
        if unit_kind in {"numeric_quantity", "temporal_anchor", "state_status"}:
            score += 0.12
        if profile_shadow_unit:
            score += 0.28
            if count_intent and facet_type in {"action", "entity"}:
                score += 0.24
        semantic_event_priority = bool(semantic_event_unit and (count_intent or percentage_intent))
        if semantic_event_priority:
            score += 0.46
        if percentage_intent and (
            unit_kind == "numeric_quantity"
            or facet_type == "numeric"
            or re.search(r"\b\d+(?:\.\d+)?\s*%|\b\d+(?:\.\d+)?\s*percent\b", unit_text, flags=re.IGNORECASE)
        ):
            score += 0.72
        if parent is not None:
            score += 0.08
        if score < 0.55:
            continue
        candidates.append(
            (
                round(min(3.2, 1.28 + score), 6),
                record,
                parent,
                {
                    "unit_coverage_overlap_tokens": sorted(overlap)[:14],
                    "unit_coverage_score": round(score, 6),
                    "unit_kind": unit_kind,
                    "facet_type": facet_type,
                    "unit_coverage_count_intent": bool(count_intent),
                    "unit_coverage_percentage_intent": bool(percentage_intent),
                    "unit_coverage_temporal_intent": bool(temporal_intent),
                    "unit_coverage_direct_unit_intent": bool(direct_unit_intent),
                    "unit_coverage_profile_shadow_unit": bool(profile_shadow_unit),
                    "unit_coverage_semantic_event_unit": bool(semantic_event_unit),
                    "unit_coverage_semantic_event_priority": bool(semantic_event_priority),
                },
            )
        )

    if not candidates:
        return {
            "hits": list(final_hits),
            "metadata": {
                "unit_coverage_pack_enabled": True,
                "unit_coverage_candidate_count": 0,
                "unit_coverage_inserted_hit_count": 0,
                "unit_coverage_count_intent": bool(count_intent),
                "unit_coverage_percentage_intent": bool(percentage_intent),
                "unit_coverage_temporal_intent": bool(temporal_intent),
                "unit_coverage_direct_unit_intent": bool(direct_unit_intent),
            },
        }
    candidates.sort(
        key=lambda item: (
            1 if bool(item[3].get("unit_coverage_semantic_event_priority", False)) else 0,
            1
            if bool(item[3].get("unit_coverage_percentage_intent"))
            and (
                _normalize(item[3].get("unit_kind", "")) == "numeric_quantity"
                or _normalize(item[3].get("facet_type", "")) == "numeric"
            )
            else 0,
            float(item[0]),
            int(item[1].turn_index),
        ),
        reverse=True,
    )
    selected: List[tuple[float, SessionMemoryRecordV2, SessionMemoryRecordV2 | None, Dict[str, Any]]] = []
    seen_unit_values: set[str] = set()
    try:
        max_selected_units = max(1, int(os.getenv("TMCRA_UNIT_COVERAGE_PACK_MAX_UNITS", "6") or 6))
    except (TypeError, ValueError):
        max_selected_units = 6
    for item in candidates:
        _, record, parent, metadata = item
        key = "|".join(
            [
                _normalize(metadata.get("unit_kind", "")),
                _normalize(record.value),
                _normalize(dict(record.metadata or {}).get("facet_parent_slot_key", "")),
            ]
        )
        if key in seen_unit_values:
            continue
        seen_unit_values.add(key)
        selected.append(item)
        if len(selected) >= max_selected_units:
            break
    if count_intent:
        try:
            max_profile_shadow_units = max(0, int(os.getenv("TMCRA_UNIT_COVERAGE_PROFILE_SHADOW_MAX_UNITS", "2") or 2))
        except (TypeError, ValueError):
            max_profile_shadow_units = 2
        selected_ids = {item[1].memory_id for item in selected}
        added_profile_shadow_units = 0
        for item in candidates:
            _, record, _parent, metadata = item
            if added_profile_shadow_units >= max_profile_shadow_units:
                break
            if record.memory_id in selected_ids:
                continue
            if not bool(metadata.get("unit_coverage_profile_shadow_unit", False)):
                continue
            selected.append(item)
            selected_ids.add(record.memory_id)
            added_profile_shadow_units += 1
    packed_hits: List[MemoryHit] = []
    for score, unit_record, parent, extra in selected:
        packed_hits.append(
            _memory_hit_from_record(
                unit_record,
                score=score,
                metadata={
                    **extra,
                    "unit_coverage_pack": True,
                    "evidence_snippet_role": "unit_coverage_evidence_unit",
                },
            )
        )
        if parent is not None:
            unit_metadata = dict(unit_record.metadata or {})
            parent_hit = _memory_hit_from_record(
                parent,
                score=max(1.12, score - 0.06),
                metadata={
                    **extra,
                    "unit_coverage_pack": True,
                    "evidence_snippet_role": "unit_coverage_parent_event",
                    "unit_coverage_parent_memory_id": parent.memory_id,
                    "unit_coverage_child_id": unit_record.memory_id,
                    "unit_coverage_child_value": unit_record.value,
                    "unit_coverage_child_source_span": _clean_text(
                        unit_metadata.get("facet_source_span", "") or unit_metadata.get("source_span", "")
                    ),
                },
            )
            parent_hit.memory_id = f"{parent.memory_id}#unit_parent:{unit_record.memory_id}"
            packed_hits.append(parent_hit)
    merged: List[MemoryHit] = []
    seen_ids: set[str] = set()
    try:
        insertion_index = max(0, min(len(final_hits), int(os.getenv("TMCRA_UNIT_COVERAGE_PACK_INSERT_AFTER", "2") or 2)))
    except (TypeError, ValueError):
        insertion_index = min(len(final_hits), 2)
    ordered_hits = [*list(final_hits[:insertion_index]), *packed_hits, *list(final_hits[insertion_index:])]
    for hit in ordered_hits:
        if hit.memory_id and hit.memory_id in seen_ids:
            continue
        if hit.memory_id:
            seen_ids.add(hit.memory_id)
        merged.append(hit)
    return {
        "hits": merged,
        "metadata": {
            "unit_coverage_pack_enabled": True,
            "unit_coverage_candidate_count": len(candidates),
            "unit_coverage_selected_unit_count": len(selected),
            "unit_coverage_inserted_hit_count": len(packed_hits),
            "unit_coverage_count_intent": bool(count_intent),
            "unit_coverage_percentage_intent": bool(percentage_intent),
            "unit_coverage_temporal_intent": bool(temporal_intent),
            "unit_coverage_direct_unit_intent": bool(direct_unit_intent),
        },
    }


def _multi_unit_chain_focus_tokens(query: str) -> set[str]:
    tokens = set(_path_utility_tokens(query))
    return {
        normalized
        for token in tokens
        for normalized in [_multi_unit_chain_normalize_token(token)]
        if normalized and normalized not in _MULTI_UNIT_CHAIN_FOCUS_DROP_TOKENS and len(normalized) > 2
    }


def _multi_unit_chain_normalize_token(token: str) -> str:
    raw = _normalize(token)
    if not raw:
        return ""
    aliases = {
        "bought": "buy",
        "buying": "buy",
        "purchased": "buy",
        "purchasing": "buy",
        "worked": "work",
        "working": "work",
        "started": "start",
        "starting": "start",
        "finished": "finish",
        "finishing": "finish",
        "completed": "complete",
        "completing": "complete",
        "returned": "return",
        "returning": "return",
        "met": "meet",
        "meeting": "meet",
        "picked": "pick",
        "picking": "pick",
        "led": "lead",
        "leading": "lead",
        "kits": "kit",
        "items": "item",
        "projects": "project",
        "events": "event",
        "clothes": "clothing",
    }
    if raw in aliases:
        return aliases[raw]
    if len(raw) > 4 and raw.endswith("ies"):
        return raw[:-3] + "y"
    if len(raw) > 4 and raw.endswith("ing"):
        stem = raw[:-3]
        if len(stem) > 3 and stem[-1] == stem[-2]:
            stem = stem[:-1]
        return stem
    if len(raw) > 3 and raw.endswith("ed"):
        stem = raw[:-2]
        if len(stem) > 3 and stem[-1] == stem[-2]:
            stem = stem[:-1]
        return stem
    if len(raw) > 3 and raw.endswith("s") and not raw.endswith("ss"):
        return raw[:-1]
    return raw


def _multi_unit_chain_normalized_tokens(text: str) -> set[str]:
    return {
        normalized
        for token in _path_utility_tokens(text)
        for normalized in [_multi_unit_chain_normalize_token(token)]
        if normalized
    }


def _multi_unit_chain_hit_text(record: SessionMemoryRecordV2, parent: SessionMemoryRecordV2 | None) -> str:
    metadata = dict(record.metadata or {})
    parent_metadata = dict(parent.metadata or {}) if parent is not None else {}
    return " ".join(
        [
            _clean_text(record.value),
            _clean_text(record.category),
            _clean_text(record.relation),
            _clean_text(metadata.get("facet_type", "")),
            _clean_text(metadata.get("unit_kind", "")),
            _clean_text(metadata.get("facet_role", "")),
            _clean_text(metadata.get("facet_value", "")),
            _clean_text(metadata.get("facet_source_span", "")),
            _clean_text(metadata.get("action", "")),
            _clean_text(metadata.get("target", "")),
            _clean_text(metadata.get("quantity", "")),
            _clean_text(metadata.get("status", "")),
            _clean_text(parent.value if parent else ""),
            _clean_text(parent_metadata.get("source_span", "")),
            _clean_text(parent_metadata.get("raw_text", "")),
        ]
    )


def _multi_unit_chain_local_hit_text(record: SessionMemoryRecordV2, parent: SessionMemoryRecordV2 | None) -> str:
    metadata = dict(record.metadata or {})
    parent_metadata = dict(parent.metadata or {}) if parent is not None else {}
    return " ".join(
        [
            _clean_text(record.value),
            _clean_text(record.category),
            _clean_text(record.relation),
            _clean_text(metadata.get("facet_type", "")),
            _clean_text(metadata.get("unit_kind", "")),
            _clean_text(metadata.get("facet_role", "")),
            _clean_text(metadata.get("facet_value", "")),
            _clean_text(metadata.get("facet_source_span", "")),
            _clean_text(metadata.get("action", "")),
            _clean_text(metadata.get("target", "")),
            _clean_text(metadata.get("quantity", "")),
            _clean_text(metadata.get("status", "")),
            _clean_text(parent.value if parent else ""),
            _clean_text(parent_metadata.get("source_span", "")),
        ]
    )


def _multi_unit_chain_slot_hits(
    graph: SessionMemoryGraphV2,
    query: str,
    final_hits: Sequence[MemoryHit],
    *,
    top_k: int,
) -> Dict[str, Any]:
    mode = _normalize(os.getenv("TMCRA_MULTI_UNIT_CHAIN_SLOT_MODE", "on"))
    if mode in _MULTI_UNIT_CHAIN_DISABLED_MODES:
        return {
            "hits": list(final_hits),
            "metadata": {"multi_unit_chain_slot_enabled": False, "multi_unit_chain_slot_reason": "disabled"},
        }
    query_tokens = set(_path_utility_tokens(query))
    focus_tokens = _multi_unit_chain_focus_tokens(query)
    temporal_comparison_intent = bool(query_tokens & _MULTI_UNIT_CHAIN_TEMPORAL_COMPARISON_TOKENS) and len(focus_tokens) >= 2
    aggregation_or_joiner_intent = bool(
        re.search(r"\b(?:and|or|both|each|total|sum|minimum|maximum|amount|count|many|number)\b", str(query), flags=re.IGNORECASE)
    )
    numeric_aggregation_intent = bool(
        query_tokens
        & (
            _MULTI_UNIT_CHAIN_NUMERIC_VALUE_TOKENS
            | {"comment", "comments", "number", "percent", "percentage", "sum", "total"}
        )
    )
    count_or_aggregation_intent = bool(query_tokens & _MULTI_UNIT_CHAIN_COUNT_TOKENS) and (aggregation_or_joiner_intent or len(focus_tokens) >= 3)
    multi_intent = count_or_aggregation_intent or temporal_comparison_intent
    if not query_tokens or not multi_intent:
        return {
            "hits": list(final_hits),
            "metadata": {
                "multi_unit_chain_slot_enabled": False,
                "multi_unit_chain_slot_reason": "no_multi_intent",
            },
        }
    if not focus_tokens:
        return {
            "hits": list(final_hits),
            "metadata": {
                "multi_unit_chain_slot_enabled": True,
                "multi_unit_chain_slot_formed": False,
                "multi_unit_chain_slot_reason": "no_focus_tokens",
            },
        }

    records = list(getattr(graph, "records_by_id", {}).values())
    parent_by_slot = {_clean_text(record.slot_key).lower(): record for record in records}
    candidates: List[tuple[float, SessionMemoryRecordV2, SessionMemoryRecordV2 | None, Dict[str, Any]]] = []
    for record in records:
        metadata = dict(record.metadata or {})
        if record.state not in {"active", "parallel_active", "evidence"}:
            continue
        profile_shadow_unit = _profile_shadow_eventlike_record(record, metadata)
        if profile_shadow_unit:
            facet_type = "action"
            unit_kind = "profile_shadow_unit"
            parent_slot = _clean_text(record.slot_key).lower()
            parent = None
            text = _profile_shadow_unit_text(record, metadata)
            local_text = text
        else:
            if _normalize(metadata.get("content_variant", "")) != "event_facet_write":
                continue
            if _normalize(metadata.get("facet_layer_version", "")) != "event_unit_v1":
                continue
            facet_type = _normalize(metadata.get("facet_type", ""))
            unit_kind = _normalize(metadata.get("unit_kind", ""))
            if facet_type not in _MULTI_UNIT_CHAIN_FACET_TYPES and unit_kind not in _MULTI_UNIT_CHAIN_UNIT_KINDS:
                continue
            parent_slot = _clean_text(metadata.get("facet_parent_slot_key", "")).lower()
            parent = parent_by_slot.get(parent_slot)
            text = _multi_unit_chain_hit_text(record, parent)
            local_text = _multi_unit_chain_local_hit_text(record, parent)
        semantic_event_unit = _normalize(record.memory_id).startswith("tmcra.event.") or _normalize(parent_slot).startswith("tmcra.event.")
        unit_tokens = _multi_unit_chain_normalized_tokens(text)
        local_tokens = _multi_unit_chain_normalized_tokens(local_text)
        focus_overlap = focus_tokens & unit_tokens
        if not focus_overlap:
            continue
        score = min(1.2, len(focus_overlap) / max(1.0, len(focus_tokens)) * 1.6)
        local_numeric_signal = _multi_unit_chain_numeric_signal(unit_kind, facet_type, local_text, local_tokens)
        if _multi_unit_chain_date_like_numeric(local_text):
            numeric_signal = 0.0
        else:
            numeric_signal = local_numeric_signal
            if numeric_signal <= 0.0:
                numeric_signal = _multi_unit_chain_numeric_signal(unit_kind, facet_type, text, unit_tokens)
        temporal_anchor_signal = _multi_unit_chain_temporal_anchor_signal(text, unit_tokens)
        local_temporal_anchor_signal = _multi_unit_chain_temporal_anchor_signal(local_text, local_tokens)
        if facet_type in {"action", "entity", "role", "numeric"}:
            score += 0.34
        if unit_kind in {"action_unit", "target_entity", "leadership_role", "participation_role", "numeric_quantity"}:
            score += 0.24
        if profile_shadow_unit:
            score += 0.30
        semantic_event_priority = bool(semantic_event_unit and (count_or_aggregation_intent or numeric_aggregation_intent))
        if semantic_event_priority:
            score += 0.48
        if numeric_signal and numeric_aggregation_intent:
            score += numeric_signal
        if count_or_aggregation_intent and not numeric_aggregation_intent and facet_type in {"action", "entity", "role"}:
            score += 0.36
        if count_or_aggregation_intent and not numeric_aggregation_intent and unit_kind in {"action_unit", "target_entity", "leadership_role", "participation_role"}:
            score += 0.22
        if parent is not None:
            score += 0.12
        if score < 0.72:
            continue
        candidates.append(
            (
                round(min(4.0, 2.15 + score), 6),
                record,
                parent,
                {
                    "multi_unit_chain_focus_overlap_tokens": sorted(focus_overlap)[:16],
                    "multi_unit_chain_score": round(score, 6),
                    "multi_unit_chain_numeric_signal": round(numeric_signal, 6),
                    "multi_unit_chain_temporal_anchor_signal": round(temporal_anchor_signal, 6),
                    "multi_unit_chain_local_temporal_anchor_signal": round(local_temporal_anchor_signal, 6),
                    "multi_unit_chain_temporal_comparison": bool(temporal_comparison_intent),
                    "multi_unit_chain_numeric_aggregation": bool(numeric_aggregation_intent),
                    "unit_kind": unit_kind,
                    "facet_type": facet_type,
                    "multi_unit_chain_parent_slot_key": parent_slot,
                    "multi_unit_chain_profile_shadow_unit": bool(profile_shadow_unit),
                    "multi_unit_chain_semantic_event_unit": bool(semantic_event_unit),
                    "multi_unit_chain_semantic_event_priority": bool(semantic_event_priority),
                    "multi_unit_chain_count_or_aggregation": bool(count_or_aggregation_intent),
                },
            )
        )

    if not candidates:
        return {
            "hits": list(final_hits),
            "metadata": {
                "multi_unit_chain_slot_enabled": True,
                "multi_unit_chain_slot_formed": False,
                "multi_unit_chain_slot_reason": "no_matching_units",
                "multi_unit_chain_candidate_count": 0,
                "multi_unit_chain_focus_tokens": sorted(focus_tokens)[:24],
            },
        }

    if temporal_comparison_intent:
        candidates.sort(
            key=lambda item: (
                float(item[3].get("multi_unit_chain_local_temporal_anchor_signal", 0.0) or 0.0),
                float(item[3].get("multi_unit_chain_temporal_anchor_signal", 0.0) or 0.0),
                1 if bool(item[3].get("multi_unit_chain_semantic_event_priority", False)) else 0,
                0
                if _normalize(item[3].get("unit_kind", "")) == "numeric_quantity"
                or _normalize(item[3].get("facet_type", "")) == "numeric"
                else 1,
                float(item[0]),
                int(item[1].turn_index),
            ),
            reverse=True,
        )
    else:
        candidates.sort(
            key=lambda item: (
                float(item[3].get("multi_unit_chain_numeric_signal", 0.0) or 0.0)
                if bool(item[3].get("multi_unit_chain_numeric_aggregation", False))
                else (
                    1.2
                    if bool(item[3].get("multi_unit_chain_semantic_event_priority", False))
                    else
                    0.0
                    if _normalize(item[3].get("unit_kind", "")) == "numeric_quantity"
                    or _normalize(item[3].get("facet_type", "")) == "numeric"
                    else 1.0
                ),
                float(item[0]),
                int(item[1].turn_index),
            ),
            reverse=True,
        )
    selected: List[tuple[float, SessionMemoryRecordV2, SessionMemoryRecordV2 | None, Dict[str, Any]]] = []
    seen_parent_slots: set[str] = set()
    seen_values: set[str] = set()
    max_units = max(2, min(8, int(os.getenv("TMCRA_MULTI_UNIT_CHAIN_SLOT_MAX_UNITS", "6") or 6)))
    for item in candidates:
        _, record, parent, metadata = item
        parent_slot = _clean_text(metadata.get("multi_unit_chain_parent_slot_key", ""))
        value_key = "|".join(
            [
                _normalize(record.value),
                _normalize(metadata.get("unit_kind", "")),
                parent_slot,
            ]
        )
        if value_key in seen_values:
            continue
        if parent_slot and parent_slot in seen_parent_slots:
            # Keep the chain broad: one strongest unit per parent event first.
            continue
        seen_values.add(value_key)
        if parent_slot:
            seen_parent_slots.add(parent_slot)
        selected.append(item)
        if len(selected) >= max_units:
            break
    if count_or_aggregation_intent:
        try:
            max_profile_shadow_units = max(0, int(os.getenv("TMCRA_MULTI_UNIT_CHAIN_PROFILE_SHADOW_MAX_UNITS", "2") or 2))
        except (TypeError, ValueError):
            max_profile_shadow_units = 2
        selected_ids = {item[1].memory_id for item in selected}
        added_profile_shadow_units = 0
        for item in candidates:
            _, record, _parent, metadata = item
            if added_profile_shadow_units >= max_profile_shadow_units:
                break
            if record.memory_id in selected_ids:
                continue
            if not bool(metadata.get("multi_unit_chain_profile_shadow_unit", False)):
                continue
            selected.append(item)
            selected_ids.add(record.memory_id)
            parent_slot = _clean_text(metadata.get("multi_unit_chain_parent_slot_key", ""))
            if parent_slot:
                seen_parent_slots.add(parent_slot)
            added_profile_shadow_units += 1

    min_parents = max(2, min(4, int(os.getenv("TMCRA_MULTI_UNIT_CHAIN_SLOT_MIN_PARENTS", "2") or 2)))
    if len(seen_parent_slots) < min_parents:
        return {
            "hits": list(final_hits),
            "metadata": {
                "multi_unit_chain_slot_enabled": True,
                "multi_unit_chain_slot_formed": False,
                "multi_unit_chain_slot_reason": "insufficient_parent_coverage",
                "multi_unit_chain_candidate_count": len(candidates),
                "multi_unit_chain_selected_unit_count": len(selected),
                "multi_unit_chain_parent_count": len(seen_parent_slots),
                "multi_unit_chain_focus_tokens": sorted(focus_tokens)[:24],
            },
        }

    chain_memory_ids: List[str] = []
    packed_hits: List[MemoryHit] = []
    for score, unit_record, parent, extra in selected:
        chain_memory_ids.append(unit_record.memory_id)
        packed_hits.append(
            _memory_hit_from_record(
                unit_record,
                score=score,
                metadata={
                    **extra,
                    "multi_unit_chain_slot": True,
                    "multi_unit_chain_bundle": True,
                    "evidence_snippet_role": "multi_unit_chain_evidence_unit",
                },
            )
        )
        if parent is not None:
            unit_metadata = dict(unit_record.metadata or {})
            chain_memory_ids.append(parent.memory_id)
            parent_hit = _memory_hit_from_record(
                parent,
                score=max(1.9, score - 0.08),
                metadata={
                    **extra,
                    "multi_unit_chain_slot": True,
                    "multi_unit_chain_bundle": True,
                    "evidence_snippet_role": "multi_unit_chain_parent_event",
                    "multi_unit_chain_child_id": unit_record.memory_id,
                    "multi_unit_chain_child_value": unit_record.value,
                    "multi_unit_chain_child_source_span": _clean_text(
                        unit_metadata.get("facet_source_span", "") or unit_metadata.get("source_span", "")
                    ),
                },
            )
            parent_hit.memory_id = f"{parent.memory_id}#multi_parent:{unit_record.memory_id}"
            packed_hits.append(parent_hit)

    seen_ids: set[str] = set()
    merged: List[MemoryHit] = []
    insertion_index = max(0, min(len(final_hits), int(os.getenv("TMCRA_MULTI_UNIT_CHAIN_SLOT_INSERT_AFTER", "1") or 1)))
    ordered = [*list(final_hits[:insertion_index]), *packed_hits, *list(final_hits[insertion_index:])]
    for hit in ordered:
        if hit.memory_id and hit.memory_id in seen_ids:
            continue
        if hit.memory_id:
            seen_ids.add(hit.memory_id)
        merged.append(hit)
    return {
        "hits": merged,
        "metadata": {
            "multi_unit_chain_slot_enabled": True,
            "multi_unit_chain_slot_formed": True,
            "multi_unit_chain_slot_reason": "formed",
            "multi_unit_chain_candidate_count": len(candidates),
            "multi_unit_chain_selected_unit_count": len(selected),
            "multi_unit_chain_parent_count": len(seen_parent_slots),
            "multi_unit_chain_inserted_hit_count": len(packed_hits),
            "multi_unit_chain_memory_ids": _dedupe(chain_memory_ids, max_items=32),
            "multi_unit_chain_focus_tokens": sorted(focus_tokens)[:24],
        },
    }


def _profile_support_source_hits(
    graph: SessionMemoryGraphV2,
    profile_hit: MemoryHit,
    *,
    grouped_hits: Mapping[str, Sequence[MemoryHit]],
    query: str,
    limit: int,
) -> List[MemoryHit]:
    metadata = dict(profile_hit.metadata or {})
    support_ids = _dedupe(
        [
            *list(metadata.get("profile_support_ids", []) or []),
            *list(metadata.get("support_memory_ids", []) or []),
        ],
        max_items=24,
    )
    if not support_ids:
        return []
    query_raw_tokens = set(_path_utility_tokens(query))
    query_tokens = _profile_query_expanded_tokens(query)
    profile_event_id = _clean_text(metadata.get("profile_first_hybrid_event_id", "")) or _runtime_event_key(profile_hit)
    candidates: List[tuple[float, MemoryHit]] = []
    for support_id in support_ids:
        record = getattr(graph, "records_by_id", {}).get(support_id)
        if record is None or record.state not in {"active", "parallel_active", "evidence"}:
            continue
        support_hit = _memory_hit_from_record(record)
        event_id = _runtime_event_key(support_hit)
        if not event_id or event_id == profile_event_id or event_id not in grouped_hits:
            continue
        match_score, overlap_tokens, raw_overlap_tokens = _profile_hit_match_score(query_raw_tokens, query_tokens, support_hit)
        source_score = max(0.80, float(profile_hit.score) - 0.02) + min(0.28, match_score * 0.12)
        support_metadata = dict(support_hit.metadata or {})
        profile_metadata = dict(profile_hit.metadata or {})
        support_metadata.update(
            {
                "profile_first_hybrid_rescue": True,
                "profile_first_source_support": True,
                "profile_first_parent_memory_id": profile_hit.memory_id,
                "profile_first_hybrid_event_id": event_id,
                "profile_first_parent_summary": _clean_text(profile_metadata.get("profile_summary", "")) or _clean_text(profile_metadata.get("profile_value", "")) or _clean_text(profile_hit.value),
                "profile_type": _clean_text(support_metadata.get("profile_type", "")) or _clean_text(profile_metadata.get("profile_type", "")),
                "profile_domain": _clean_text(support_metadata.get("profile_domain", "")) or _clean_text(profile_metadata.get("profile_domain", "")),
                "profile_domain_label": _clean_text(support_metadata.get("profile_domain_label", "")) or _clean_text(profile_metadata.get("profile_domain_label", "")),
                "profile_value": _clean_text(support_metadata.get("profile_value", "")) or _clean_text(profile_metadata.get("profile_value", "")) or _clean_text(profile_hit.value),
                "profile_summary": _clean_text(support_metadata.get("profile_summary", "")) or _clean_text(profile_metadata.get("profile_summary", "")) or _clean_text(profile_hit.value),
                "profile_query_match_score": round(match_score, 6),
                "profile_query_overlap_tokens": list(overlap_tokens),
                "profile_query_raw_overlap_tokens": list(raw_overlap_tokens),
                "profile_source_pack_role": "source_event_support",
            }
        )
        candidates.append(
            (
                source_score,
                MemoryHit(
                    memory_id=support_hit.memory_id,
                    category=support_hit.category,
                    value=support_hit.value,
                    relation=support_hit.relation,
                    anchors=list(support_hit.anchors),
                    score=round(source_score, 6),
                    source_kind=support_hit.source_kind,
                    slot_key=support_hit.slot_key,
                    state=support_hit.state,
                    turn_index=int(support_hit.turn_index),
                    metadata=support_metadata,
                ),
            )
        )
    candidates.sort(
        key=lambda item: (
            bool((item[1].metadata or {}).get("profile_query_raw_overlap_tokens")),
            float(item[0]),
            int(item[1].turn_index),
        ),
        reverse=True,
    )
    return [hit for _, hit in candidates[: max(0, int(limit))]]


def _profile_same_event_source_hit(profile_hit: MemoryHit, group_hits: Sequence[MemoryHit], *, event_id: str) -> MemoryHit | None:
    source_hit = _support_hit_for_path("speaker_event_source_turn", group_hits) or _representative_event_hit(group_hits)
    if source_hit is None or source_hit.memory_id == profile_hit.memory_id:
        return None
    metadata = dict(source_hit.metadata or {})
    profile_metadata = dict(profile_hit.metadata or {})
    metadata.update(
        {
            "profile_first_hybrid_rescue": True,
            "profile_first_source_support": True,
            "profile_first_same_event_support": True,
            "profile_first_parent_memory_id": profile_hit.memory_id,
            "profile_first_hybrid_event_id": event_id,
            "profile_first_parent_summary": _clean_text(profile_metadata.get("profile_summary", "")) or _clean_text(profile_metadata.get("profile_value", "")) or _clean_text(profile_hit.value),
            "profile_type": _clean_text(metadata.get("profile_type", "")) or _clean_text(profile_metadata.get("profile_type", "")),
            "profile_domain": _clean_text(metadata.get("profile_domain", "")) or _clean_text(profile_metadata.get("profile_domain", "")),
            "profile_domain_label": _clean_text(metadata.get("profile_domain_label", "")) or _clean_text(profile_metadata.get("profile_domain_label", "")),
            "profile_value": _clean_text(metadata.get("profile_value", "")) or _clean_text(profile_metadata.get("profile_value", "")) or _clean_text(profile_hit.value),
            "profile_summary": _clean_text(metadata.get("profile_summary", "")) or _clean_text(profile_metadata.get("profile_summary", "")) or _clean_text(profile_hit.value),
            "profile_source_pack_role": "same_event_source_turn",
            "evidence_snippet_role": "profile_source_support",
        }
    )
    return MemoryHit(
        memory_id=source_hit.memory_id,
        category=source_hit.category,
        value=source_hit.value,
        relation=source_hit.relation,
        anchors=list(source_hit.anchors),
        score=max(float(source_hit.score), float(profile_hit.score) - 0.01),
        source_kind=source_hit.source_kind,
        slot_key=source_hit.slot_key,
        state=source_hit.state,
        turn_index=int(source_hit.turn_index),
        metadata=metadata,
    )


def _profile_first_hybrid_rescue(
    graph: SessionMemoryGraphV2,
    query: str,
    *,
    grouped_hits: Mapping[str, Sequence[MemoryHit]],
    top_k: int,
) -> Dict[str, Any]:
    intent = infer_profile_query_intent(query)
    if not bool(intent.get("enabled")):
        return {"hits": [], "event_ids": []}
    raw_hits = _profile_query_rescue_hits(graph, query, top_k=max(4, int(top_k or 1) * 3))
    limit = max(2, min(4, int(top_k or 1)))
    selected_hits: List[MemoryHit] = []
    selected_event_ids: List[str] = []
    seen_event_ids: set[str] = set()
    for rank, hit in enumerate(raw_hits, start=1):
        event_id = _runtime_event_key(hit)
        if not event_id or event_id not in grouped_hits or event_id in seen_event_ids:
            continue
        metadata = dict(hit.metadata or {})
        metadata.update(
            {
                "profile_first_hybrid_rescue": True,
                "profile_first_hybrid_rank": rank,
                "profile_first_hybrid_event_id": event_id,
            }
        )
        hit.metadata = metadata
        hit.score = round(max(float(hit.score), 0.88) + max(0.0, 0.08 - (0.01 * len(selected_hits))), 6)
        selected_hits.append(hit)
        selected_event_ids.append(event_id)
        seen_event_ids.add(event_id)
        same_event_source = _profile_same_event_source_hit(hit, grouped_hits.get(event_id, []), event_id=event_id)
        if same_event_source is not None:
            selected_hits.append(same_event_source)
        if len(selected_hits) >= limit:
            break
        support_hits = _profile_support_source_hits(
            graph,
            hit,
            grouped_hits=grouped_hits,
            query=query,
            limit=max(1, min(3, limit - len(selected_hits) + 1)),
        )
        for support_hit in support_hits:
            support_event_id = _clean_text((support_hit.metadata or {}).get("profile_first_hybrid_event_id", "")) or _runtime_event_key(support_hit)
            if not support_event_id or support_event_id in seen_event_ids:
                continue
            selected_hits.append(support_hit)
            selected_event_ids.append(support_event_id)
            seen_event_ids.add(support_event_id)
            if len(selected_hits) >= limit:
                break
        if len(selected_hits) >= limit:
            break
    return {
        "hits": selected_hits,
        "event_ids": selected_event_ids,
        "memory_ids": [hit.memory_id for hit in selected_hits],
    }


def _inject_profile_first_hits(
    final_hits: Sequence[MemoryHit],
    profile_first_hits: Sequence[MemoryHit],
    *,
    selected_event_ids: Sequence[str],
    selected_path_ids: Sequence[str],
) -> List[MemoryHit]:
    merged: List[MemoryHit] = []
    seen_memory_ids: set[str] = set()
    for hit in list(profile_first_hits) + list(final_hits):
        if not hit or not hit.memory_id or hit.memory_id in seen_memory_ids:
            continue
        metadata = dict(hit.metadata or {})
        if bool(metadata.get("profile_first_hybrid_rescue")):
            event_id = _clean_text(metadata.get("profile_first_hybrid_event_id", "")) or _runtime_event_key(hit)
            metadata.update(
                {
                    "event_id": event_id,
                    "path_id": "",
                    "evidence_snippet_role": "selected_event_representative",
                    "hybrid_score_source": "profile_first_hybrid_rescue",
                    "selected_event_ids": list(selected_event_ids),
                    "selected_path_ids": list(selected_path_ids),
                }
            )
            hit = MemoryHit(
                memory_id=hit.memory_id,
                category=hit.category,
                value=hit.value,
                relation=hit.relation,
                anchors=list(hit.anchors),
                score=float(hit.score),
                source_kind=hit.source_kind,
                slot_key=hit.slot_key,
                state=hit.state,
                turn_index=int(hit.turn_index),
                metadata=metadata,
            )
        merged.append(hit)
        seen_memory_ids.add(hit.memory_id)
    return merged


def _profile_focused_pack_hits(
    graph: SessionMemoryGraphV2,
    query: str,
    final_hits: Sequence[MemoryHit],
    *,
    top_k: int,
) -> Dict[str, Any]:
    intent = infer_profile_query_intent(query)
    if not bool(intent.get("enabled")):
        return {
            "hits": list(final_hits),
            "metadata": {
                "profile_focused_pack_enabled": False,
                "profile_focused_pack_reason": "profile_intent_not_requested",
            },
        }
    source_hits = _learnable_graph_hits(graph)
    if not source_hits:
        return {
            "hits": list(final_hits),
            "metadata": {
                "profile_focused_pack_enabled": False,
                "profile_focused_pack_reason": "no_learnable_hits",
            },
        }
    runtime_graph = _build_runtime_graph_from_hits(query, source_hits)
    grouped_hits = dict(runtime_graph.get("grouped_hits", {}) or {})
    profile_first_payload = _profile_first_hybrid_rescue(
        graph,
        query,
        grouped_hits=grouped_hits,
        top_k=max(2, min(max(1, int(top_k or 1)), 4)),
    )
    profile_first_hits = list(profile_first_payload.get("hits", []) or [])
    profile_first_event_ids = list(profile_first_payload.get("event_ids", []) or [])
    profile_first_memory_ids = list(profile_first_payload.get("memory_ids", []) or [])
    if not profile_first_hits:
        return {
            "hits": list(final_hits),
            "metadata": {
                "profile_focused_pack_enabled": True,
                "profile_focused_pack_reason": "no_profile_hits",
                "profile_focused_pack_event_ids": [],
                "profile_focused_pack_memory_ids": [],
            },
        }
    selected_event_ids = _dedupe([*profile_first_event_ids, *_event_ids_from_hits(final_hits)], max_items=max(1, int(top_k or 1) * 2))
    selected_path_ids = _dedupe(
        [
            _clean_text((hit.metadata or {}).get("path_id", ""))
            for hit in final_hits
            if _clean_text((hit.metadata or {}).get("path_id", ""))
        ],
        max_items=max(1, int(top_k or 1)),
    )
    merged_hits = _inject_profile_first_hits(
        final_hits,
        profile_first_hits,
        selected_event_ids=selected_event_ids,
        selected_path_ids=selected_path_ids,
    )
    merged_hits = _coverage_preserving_final_hits(merged_hits, selected_event_ids=selected_event_ids, top_k=top_k)
    return {
        "hits": merged_hits,
        "metadata": {
            "profile_focused_pack_enabled": True,
            "profile_focused_pack_reason": "profile_first_pack_injected",
            "profile_focused_pack_event_ids": list(profile_first_event_ids),
            "profile_focused_pack_memory_ids": list(profile_first_memory_ids),
            "profile_focused_pack_hit_count": len(profile_first_hits),
            "profile_first_hybrid_enabled": True,
            "profile_first_event_ids": list(profile_first_event_ids),
            "profile_first_memory_ids": list(profile_first_memory_ids),
        },
    }


def _topic_bucket_rerank_hits(
    graph: SessionMemoryGraphV2,
    query: str,
    hits: Sequence[MemoryHit],
    *,
    top_k: int,
) -> Dict[str, Any]:
    if not hits:
        profile_rescue_hits = _profile_query_rescue_hits(graph, query, top_k=top_k)
        if profile_rescue_hits:
            return {
                "hits": profile_rescue_hits[: max(1, int(top_k or 1))],
                "metadata": {
                    "topic_bucket_rerank_enabled": True,
                    "topic_bucket_rerank_reason": "profile_query_rescue_from_empty_hits",
                    "topic_bucket_profile_query_rescue_count": len(profile_rescue_hits),
                },
            }
        return {
            "hits": [],
            "metadata": {"topic_bucket_rerank_enabled": True, "topic_bucket_rerank_reason": "no_hits"},
        }
    query_topic = _assign_topic_bucket_for_text(graph, query, turn_index=0, create=False)
    query_bucket_id = _clean_text(query_topic.get("topic_bucket_id", ""))
    adjacent_ids = _topic_adjacent_bucket_ids(graph, query_bucket_id)
    dialogue_tunnel_ids = _dialogue_tunnel_bucket_ids(graph, query_bucket_id)
    query_keywords = list(query_topic.get("topic_keywords", []) or [])
    bridge_requested = _topic_bridge_requested(query)
    dialogue_requested = _dialogue_tunnel_requested(query)
    profile_query_requested = bool(infer_profile_query_intent(query).get("enabled"))
    reranked: List[MemoryHit] = []
    stats = {
        "same_bucket": 0,
        "bridge_bucket": 0,
        "blocked_bridge_bucket": 0,
        "dialogue_tunnel_bucket": 0,
        "blocked_dialogue_tunnel_bucket": 0,
        "profile_route_preserved": 0,
        "profile_query_rescue": 0,
        "overlap_bucket": 0,
        "off_topic": 0,
    }
    for hit in list(hits):
        metadata = dict(hit.metadata or {})
        hit_bucket_id = _clean_text(metadata.get("topic_bucket_id", ""))
        hit_keywords = _dedupe(metadata.get("topic_keywords", []) or [], max_items=32)
        overlap = _topic_bucket_overlap_score(query_keywords, hit_keywords)
        same_bucket = bool(query_bucket_id and hit_bucket_id and query_bucket_id == hit_bucket_id)
        bridge_bucket = bool(hit_bucket_id and hit_bucket_id in adjacent_ids)
        dialogue_tunnel_bucket = bool(hit_bucket_id and hit_bucket_id in dialogue_tunnel_ids)
        overlap_bucket = overlap >= 0.22
        bridge_allowed = bridge_bucket and bridge_requested
        dialogue_allowed = dialogue_tunnel_bucket and dialogue_requested
        profile_route_preserved = bool(
            profile_query_requested
            and (
                bool(metadata.get("profile_layer"))
                or "profile_route" in _normalize(metadata.get("match_reason", ""))
            )
        )
        current_subject_preserved = bool(metadata.get("current_subject_resolver") or metadata.get("public_subject_match"))
        delta = 0.0
        if same_bucket:
            delta += 1.35
            stats["same_bucket"] += 1
        elif current_subject_preserved:
            delta += 0.18
            stats["profile_route_preserved"] += 1
        elif bridge_allowed:
            delta += 0.42
            stats["bridge_bucket"] += 1
        elif dialogue_allowed:
            delta += 0.16
            stats["dialogue_tunnel_bucket"] += 1
        elif overlap_bucket:
            delta += 0.28 + overlap
            stats["overlap_bucket"] += 1
        elif bridge_bucket:
            delta -= 1.05
            stats["blocked_bridge_bucket"] += 1
        elif dialogue_tunnel_bucket:
            delta -= 1.35
            stats["blocked_dialogue_tunnel_bucket"] += 1
        elif profile_route_preserved:
            delta += 0.08
            stats["profile_route_preserved"] += 1
        else:
            delta -= 0.72
            stats["off_topic"] += 1
        memory_type = _normalize(metadata.get("memory_type", ""))
        durability = _normalize(metadata.get("durability", ""))
        conflict_policy = _normalize(metadata.get("conflict_policy", ""))
        if (memory_type == "hard_constraint" or durability == "hard" or conflict_policy == "must_preserve") and (
            same_bucket or bridge_allowed or dialogue_allowed or overlap_bucket
        ):
            delta += 1.10
        match_reason = _clean_text(metadata.get("match_reason", ""))
        if profile_route_preserved and "profile_route" not in _normalize(match_reason):
            match_reason = ",".join(_dedupe([match_reason, "profile_route"], max_items=4))
        metadata.update(
            {
                "topic_bucket_rerank": True,
                "topic_bucket_query_id": query_bucket_id,
                "topic_bucket_query_label": _clean_text(query_topic.get("topic_label", "")),
                "topic_bucket_overlap": round(overlap, 6),
                "topic_bucket_delta": round(delta, 6),
                "topic_bucket_same": same_bucket,
                "topic_bucket_bridge": bridge_bucket,
                "topic_bucket_bridge_allowed": bridge_allowed,
                "topic_bucket_dialogue_tunnel": dialogue_tunnel_bucket,
                "topic_bucket_dialogue_tunnel_allowed": dialogue_allowed,
                "topic_bucket_profile_route_preserved": profile_route_preserved,
                "topic_bucket_current_subject_preserved": current_subject_preserved,
                "match_reason": match_reason,
            }
        )
        hit.metadata = metadata
        hit.score = float(hit.score) + delta
        reranked.append(hit)
    seen_ids = {hit.memory_id for hit in reranked if hit.memory_id}
    rescue_records = [
        record
        for record in getattr(graph, "records_by_id", {}).values()
        if record.memory_id not in seen_ids
        and record.state == "active"
        and _clean_text((record.metadata or {}).get("topic_bucket_id", "")) == query_bucket_id
        and _normalize(record.category) != "question"
    ]
    rescue_records.sort(
        key=lambda record: (
            int(any(marker in _normalize(record.value) for marker in ("过敏", "必须", "避开", "禁止", "不能", "must", "avoid", "allergy"))),
            int(_normalize(record.category) in {"constraint", "preference"}),
            float(record.confidence),
            float(record.salience),
            -int(record.turn_index),
        ),
        reverse=True,
    )
    rescue_hits = [
        _topic_bucket_record_to_hit(record, query_topic=query_topic, rank=index)
        for index, record in enumerate(rescue_records[: max(2, min(12, int(top_k or 1) * 2))], start=1)
    ]
    reranked.extend(rescue_hits)
    dialogue_rescue_hits: List[MemoryHit] = []
    if dialogue_requested and dialogue_tunnel_ids:
        seen_ids.update(hit.memory_id for hit in rescue_hits if hit.memory_id)
        dialogue_records = [
            record
            for record in getattr(graph, "records_by_id", {}).values()
            if record.memory_id not in seen_ids
            and record.state == "active"
            and _clean_text((record.metadata or {}).get("topic_bucket_id", "")) in dialogue_tunnel_ids
            and _normalize(record.category) != "question"
        ]
        def dialogue_record_rank(record: SessionMemoryRecordV2) -> tuple[int, int, float, float, int]:
            return (
                int(any(marker in _normalize(record.value) for marker in ("过敏", "必须", "避开", "禁止", "不能", "must", "avoid", "allergy"))),
                int(_normalize(record.category) in {"constraint", "preference"}),
                float(record.confidence),
                float(record.salience),
                -int(record.turn_index),
            )

        dialogue_records.sort(key=dialogue_record_rank, reverse=True)
        selected_dialogue_records: List[SessionMemoryRecordV2] = []
        selected_dialogue_ids: set[str] = set()
        for bucket_id in sorted(dialogue_tunnel_ids):
            bucket_records = [
                record
                for record in dialogue_records
                if _clean_text((record.metadata or {}).get("topic_bucket_id", "")) == bucket_id
            ]
            if not bucket_records:
                continue
            best = bucket_records[0]
            selected_dialogue_records.append(best)
            selected_dialogue_ids.add(best.memory_id)
        target_dialogue_rescue = max(len(selected_dialogue_records), max(1, min(4, int(top_k or 1) // 2 or 1)))
        for record in dialogue_records:
            if len(selected_dialogue_records) >= target_dialogue_rescue:
                break
            if record.memory_id in selected_dialogue_ids:
                continue
            selected_dialogue_records.append(record)
            selected_dialogue_ids.add(record.memory_id)
        dialogue_rescue_hits = [
            _topic_bucket_record_to_hit(record, query_topic=query_topic, rank=index, rescue_kind="dialogue_tunnel")
            for index, record in enumerate(selected_dialogue_records, start=1)
        ]
        reranked.extend(dialogue_rescue_hits)
    profile_rescue_hits: List[MemoryHit] = []
    if profile_query_requested:
        seen_ids.update(hit.memory_id for hit in reranked if hit.memory_id)
        for index, hit in enumerate(_profile_query_rescue_hits(graph, query, top_k=top_k), start=1):
            if hit.memory_id in seen_ids:
                continue
            metadata = dict(hit.metadata or {})
            metadata.update(
                {
                    "profile_query_rescue_rank": index,
                    "topic_bucket_query_id": query_bucket_id,
                    "topic_bucket_query_label": _clean_text(query_topic.get("topic_label", "")),
                    "topic_bucket_profile_route_preserved": True,
                }
            )
            hit.metadata = metadata
            hit.score = float(hit.score) + 0.08
            profile_rescue_hits.append(hit)
            seen_ids.add(hit.memory_id)
        stats["profile_query_rescue"] = len(profile_rescue_hits)
        reranked.extend(profile_rescue_hits)
    reranked.sort(key=lambda item: (float(item.score), int(item.turn_index)), reverse=True)
    limit = max(1, int(top_k or 1))
    focused_hits = [
        hit
        for hit in reranked
        if bool((hit.metadata or {}).get("topic_bucket_same"))
        or float((hit.metadata or {}).get("topic_bucket_overlap", 0.0) or 0.0) >= 0.22
        or bool((hit.metadata or {}).get("topic_bucket_bridge_allowed"))
        or bool((hit.metadata or {}).get("topic_bucket_dialogue_tunnel_allowed"))
        or bool((hit.metadata or {}).get("topic_bucket_profile_route_preserved"))
        or bool((hit.metadata or {}).get("topic_bucket_current_subject_preserved"))
        or bool((hit.metadata or {}).get("current_subject_resolver"))
    ]
    model_path_fallback = False
    no_bucket_model_fallback = False
    if not focused_hits:
        model_supported_hits = [
            hit
            for hit in reranked
            if _clean_text((hit.metadata or {}).get("path_id", ""))
            or _clean_text((hit.metadata or {}).get("hybrid_score_source", ""))
        ]
        if model_supported_hits:
            focused_hits = model_supported_hits
            model_path_fallback = True
    generic_memory_fallback = False
    if not focused_hits and any(not _is_public_dialog_hit(hit) for hit in reranked):
        focused_hits = list(reranked)
        generic_memory_fallback = True
    if not focused_hits and not query_bucket_id:
        focused_hits = list(reranked)
        no_bucket_model_fallback = True
    time_focused_model_path = any(
        "speaker_event_time" in _clean_text((hit.metadata or {}).get("path_id", ""))
        or _clean_text((hit.metadata or {}).get("model_focused_answer_type", "")) == "time"
        for hit in focused_hits
    )
    if time_focused_model_path:
        focused_hits = [
            hit
            for hit in focused_hits
            if not (
                _clean_text(hit.source_kind) == "public_dialog_profile"
                and not _clean_text((hit.metadata or {}).get("path_id", ""))
            )
        ]
    filtered_count = max(0, len(reranked) - len(focused_hits))
    query_subject_signature = _public_subject_signature(_public_query_subject(query))

    def _hit_subject_signatures(hit: MemoryHit) -> set[str]:
        metadata = dict(hit.metadata or {})
        signatures = {
            _normalize(metadata.get("subject_signature", "")).replace("-", "_"),
            _public_subject_signature(metadata.get("subject", "")),
        }
        canonical_slot_key = _clean_text(metadata.get("canonical_slot_key", ""))
        if ".subject." in canonical_slot_key:
            signatures.add(_public_subject_signature(canonical_slot_key.split(".subject.", 1)[-1]))
        if ".subject." in hit.slot_key:
            signatures.add(_public_subject_signature(hit.slot_key.split(".subject.", 1)[-1]))
        signatures.discard("")
        return signatures

    protected_current_hits = [
        hit
        for hit in focused_hits
        if bool((hit.metadata or {}).get("current_subject_resolver"))
    ]
    protected_model_path_hits = [
        hit
        for hit in focused_hits
        if _clean_text((hit.metadata or {}).get("path_id", ""))
        and _clean_text((hit.metadata or {}).get("hybrid_score_source", ""))
    ]
    protected_generic_hits = [
        hit
        for hit in focused_hits
        if not _is_public_dialog_hit(hit)
        and not bool((hit.metadata or {}).get("profile_layer"))
    ]
    if protected_current_hits and query_subject_signature:
        protected_ids = {hit.memory_id for hit in protected_current_hits if hit.memory_id}
        compacted_focused_hits: List[MemoryHit] = []
        for hit in focused_hits:
            if hit.memory_id in protected_ids:
                compacted_focused_hits.append(hit)
                continue
            same_current_subject = query_subject_signature in _hit_subject_signatures(hit)
            inactive_state = _normalize(hit.state) in {"superseded", "evidence", "historical", "stale", "false"}
            if same_current_subject and inactive_state:
                continue
            compacted_focused_hits.append(hit)
        focused_hits = compacted_focused_hits
        filtered_count = max(0, len(reranked) - len(focused_hits))

    selected_hits: List[MemoryHit] = []
    selected_keys: set[str] = set()
    for hit in [*protected_current_hits, *protected_model_path_hits, *protected_generic_hits, *focused_hits]:
        key = hit.memory_id or f"{_hit_event_id(hit)}::{hit.slot_key}::{hit.value[:80]}"
        if key in selected_keys:
            continue
        selected_hits.append(hit)
        selected_keys.add(key)
        if len(selected_hits) >= limit:
            break
    dialogue_reserved_count = 0
    if dialogue_requested and dialogue_tunnel_ids:
        selected_ids = {hit.memory_id for hit in selected_hits if hit.memory_id}
        selected_bucket_ids = {
            _clean_text((hit.metadata or {}).get("topic_bucket_id", ""))
            for hit in selected_hits
            if _clean_text((hit.metadata or {}).get("topic_bucket_id", ""))
        }
        for bucket_id in sorted(dialogue_tunnel_ids):
            if bucket_id in selected_bucket_ids:
                continue
            candidate = next(
                (
                    hit
                    for hit in focused_hits
                    if hit.memory_id not in selected_ids
                    and _clean_text((hit.metadata or {}).get("topic_bucket_id", "")) == bucket_id
                    and bool((hit.metadata or {}).get("topic_bucket_dialogue_tunnel_allowed"))
                ),
                None,
            )
            if candidate is None:
                continue
            if len(selected_hits) < limit:
                selected_hits.append(candidate)
            else:
                replace_index = len(selected_hits) - 1
                for index in range(len(selected_hits) - 1, -1, -1):
                    metadata = dict(selected_hits[index].metadata or {})
                    hardish = (
                        _normalize(metadata.get("memory_type", "")) == "hard_constraint"
                        or _normalize(metadata.get("durability", "")) == "hard"
                        or _normalize(metadata.get("conflict_policy", "")) == "must_preserve"
                    )
                    if not hardish and not bool(metadata.get("topic_bucket_dialogue_tunnel_allowed")):
                        replace_index = index
                        break
                selected_hits[replace_index] = candidate
            selected_ids.add(candidate.memory_id)
            selected_bucket_ids.add(bucket_id)
            dialogue_reserved_count += 1
        selected_hits.sort(
            key=lambda item: (
                int(bool((item.metadata or {}).get("current_subject_resolver"))),
                float(item.score),
                int(item.turn_index),
            ),
            reverse=True,
        )
    return {
        "hits": selected_hits[:limit],
        "metadata": {
            "topic_bucket_rerank_enabled": True,
            "topic_bucket_no_fill_policy": True,
            "topic_bucket_query_id": query_bucket_id,
            "topic_bucket_query_label": _clean_text(query_topic.get("topic_label", "")),
            "topic_bucket_query_keywords": query_keywords,
            "topic_bucket_adjacent_ids": sorted(adjacent_ids),
            "dialogue_tunnel_adjacent_ids": sorted(dialogue_tunnel_ids),
            "topic_bucket_bridge_requested": bridge_requested,
            "dialogue_tunnel_requested": dialogue_requested,
            "topic_bucket_model_path_fallback": model_path_fallback,
            "topic_bucket_generic_memory_fallback": generic_memory_fallback,
            "topic_bucket_no_bucket_model_fallback": no_bucket_model_fallback,
            "topic_bucket_candidate_count": len(hits),
            "topic_bucket_rescue_count": len(rescue_hits),
            "dialogue_tunnel_rescue_count": len(dialogue_rescue_hits),
            "profile_query_rescue_count": len(profile_rescue_hits),
            "dialogue_tunnel_reserved_count": dialogue_reserved_count,
            "topic_bucket_focused_count": len(focused_hits),
            "topic_bucket_final_count": len(selected_hits[:limit]),
            "topic_bucket_filtered_count": filtered_count,
            "topic_bucket_stats": stats,
        },
    }


def _looks_like_write_turn(user_text: str, *, answer_payload: Dict[str, Any] | None = None) -> bool:
    text = _normalize(user_text)
    if not text:
        return False
    if "?" in text:
        return False
    if any(marker in text for marker in _WRITE_MARKERS):
        return True
    metadata = dict((answer_payload or {}).get("metadata", {}) or {})
    return bool(metadata.get("memory_write"))


def _explicit_overwrite_requested(user_text: str) -> bool:
    text = _normalize(user_text)
    return bool(text) and any(marker in text for marker in _OVERWRITE_MARKERS)


def _apply_turn_write_intent(records: List[SessionMemoryRecordV2], *, user_text: str) -> List[SessionMemoryRecordV2]:
    if not records or not _explicit_overwrite_requested(user_text):
        return records
    for record in records:
        metadata = dict(record.metadata or {})
        metadata.setdefault("write_intent", "overwrite")
        metadata.setdefault("memory_gate_decision", "explicit_overwrite")
        metadata["allow_parallel_state"] = False
        record.metadata = metadata
    return records


def _records_from_extractor(
    extractor: SessionMemoryExtractor,
    *,
    query: str,
    answer_payload: Dict[str, Any] | None,
    extraction_result: Dict[str, Any] | None,
    turn_index: int,
    profile: TMCRAProfile | None = None,
) -> List[SessionMemoryRecordV2]:
    profile = profile or TMCRAProfile()
    raw_records = extractor.extract(
        query=query,
        extraction_result=extraction_result,
        answer_bundle=answer_payload,
        answer_mode=str((answer_payload or {}).get("answer_mode", "transparent")),
        turn_index=turn_index,
    )
    results: List[SessionMemoryRecordV2] = []
    for index, record in enumerate(raw_records):
        slot_key = profile.stable_slot_key(
            category=record.category,
            value=record.value,
            anchors=record.anchor_concepts,
            slot_key=record.metadata.get("slot_key", "") if isinstance(record.metadata, dict) else "",
            relation=record.relation,
            metadata=dict(record.metadata or {}),
        )
        metadata = {
            **dict(record.metadata or {}),
            "memory_role": _clean_text(dict(record.metadata or {}).get("memory_role", "")) or "user",
            "authority": _clean_text(dict(record.metadata or {}).get("authority", "")) or "source",
            "canonical_slot_key": _clean_text(dict(record.metadata or {}).get("canonical_slot_key", "")) or slot_key,
            "writeback_class": _clean_text(dict(record.metadata or {}).get("writeback_class", "")),
            "origin_query": _clean_text(dict(record.metadata or {}).get("origin_query", "")) or _clean_text(query),
            "origin_answer_id": _clean_text(dict(record.metadata or {}).get("origin_answer_id", "")),
            "support_memory_ids": _dedupe(dict(record.metadata or {}).get("support_memory_ids", []) or []),
            "support_fact_refs": _dedupe(dict(record.metadata or {}).get("support_fact_refs", []) or []),
            "support_path_refs": _dedupe(dict(record.metadata or {}).get("support_path_refs", []) or []),
            "promotion_state": _clean_text(dict(record.metadata or {}).get("promotion_state", "")) or "none",
        }
        results.append(
            SessionMemoryRecordV2(
                memory_id=f"{slot_key}:{turn_index}:auto:{index}",
                category=record.category,
                slot_key=slot_key,
                value=_clean_text(record.value),
                relation=_clean_text(record.relation) or f"{record.category}_memory",
                anchor_concepts=_dedupe(record.anchor_concepts, max_items=8),
                evidence_anchors=_dedupe(record.anchor_concepts, max_items=8),
                salience=float(record.salience),
                confidence=float(record.confidence),
                source_kind=_clean_text(record.source_kind) or "session_memory",
                turn_index=int(record.turn_index),
                state="active",
                metadata=metadata,
            )
        )
    return results


def _build_turn_records(
    extractor: SessionMemoryExtractor,
    *,
    user_text: str,
    answer_payload: Dict[str, Any] | None,
    extraction_result: Dict[str, Any] | None,
    turn_index: int,
    allow_auto_extract: bool,
    profile: TMCRAProfile | None = None,
) -> List[SessionMemoryRecordV2]:
    profile = profile or TMCRAProfile()
    structured_records = _parse_structured_records(answer_payload, turn_index=turn_index, profile=profile)
    if structured_records:
        return _apply_typed_tunnel_annotations(
            _apply_turn_write_intent(structured_records, user_text=user_text),
            source_text=user_text,
        )
    structured_records = _parse_structured_records(extraction_result, turn_index=turn_index, profile=profile)
    if structured_records:
        return _apply_typed_tunnel_annotations(
            _apply_turn_write_intent(structured_records, user_text=user_text),
            source_text=user_text,
        )
    if not allow_auto_extract and not _looks_like_write_turn(user_text, answer_payload=answer_payload):
        return []
    return _apply_typed_tunnel_annotations(
        _records_from_extractor(
            extractor,
            query=user_text,
            answer_payload=None,
            extraction_result=extraction_result,
            turn_index=turn_index,
            profile=profile,
        ),
        source_text=user_text,
    )


class NullMemoryAdapter(MemoryAdapter):
    name = "null_memory"

    def reset(self) -> None:
        return None

    def ingest_turn(
        self,
        user_text: str,
        assistant_text: str = "",
        *,
        answer_payload: Dict[str, Any] | None = None,
        extraction_result: Dict[str, Any] | None = None,
    ) -> None:
        _ = user_text, assistant_text, answer_payload, extraction_result

    def retrieve(self, query: str, *, top_k: int = 6) -> MemoryRetrieval:
        _ = query, top_k
        return MemoryRetrieval()

    def stats(self) -> Dict[str, Any]:
        return _state_stats(storage_bytes=0, retrieval_context_tokens=0, total_state_tokens=0, records=0)

    def storage_bytes(self) -> int:
        return 0

    def build_prompt_context(self, query: str, *, top_k: int = 8) -> Dict[str, Any]:
        _ = top_k
        return {
            "mode": "null_memory",
            "query": query,
            "retrieval": MemoryRetrieval().to_dict(),
            "stats": self.stats(),
            "state": {},
        }


class GraphSessionMemoryAdapter(MemoryAdapter):
    name = "graph_session_memory_v2"

    def __init__(
        self,
        *,
        auto_extract: bool = False,
        storage_backend: str = "sqlite",
        storage_path: str = "",
        scope_id: str = "",
        audit_retention: int = 256,
        lightweight_stats: bool = True,
        retrieval_mode: str = "heuristic",
        node_model_path: str = "",
        path_model_path: str = "",
        node_model_device: str = "",
        candidate_event_k: int = 24,
        support_path_k: int = 3,
        path_tunnel_rescue_k: int = 0,
        path_tunnel_rescue_score_floor: float = 0.0,
        path_tunnel_rescue_min_age: int = 0,
        path_tunnel_rescue_min_score_margin: float = 0.0,
        event_rerank_mode: str = "matrix",
        matrix_event_top_k: int = DEFAULT_MATRIX_EVENT_TOP_K,
        memory_router_mode: str = "",
        memory_router_threshold: float = _MEMORY_ROUTER_DEFAULT_THRESHOLD,
        memory_router_margin: float = _MEMORY_ROUTER_DEFAULT_MARGIN,
        injection_planner_mode: str = "",
        injection_planner_model_path: str = "",
        injection_planner_latest_path: str = "",
        injection_planner_device: str = "",
        injection_planner_selection_threshold: float = -1.0,
        injection_planner_row_threshold: float = -1.0,
        injection_planner_logic_threshold: float = -1.0,
        temporal_layer_mode: str = "",
        temporal_router_mode: str = "",
        temporal_router_dir: str = "",
        temporal_router_latest_path: str = "",
        temporal_router_device: str = "",
    ) -> None:
        prewarm_embedder_mode = (
            _normalize(os.getenv("TMCRA_EMBEDDER_INDEX_RECALL_MODE", ""))
            or _normalize(os.getenv("TMCRA_WRITE_EMBEDDER_INDEX_MODE", ""))
        )
        self._embedder_prewarm_metadata = _prewarm_embedder_dense_if_requested(mode=prewarm_embedder_mode)
        self.extractor = SessionMemoryExtractor()
        self.profile = TMCRAProfile()
        self.temporal_organizer = TemporalOrganizer()
        self.temporal_query_planner = TemporalQueryPlanner()
        self.timeline_evidence_builder = TimelineEvidencePackBuilder()
        self.auto_extract = bool(auto_extract)
        self.storage_backend = _normalize(storage_backend) or "sqlite"
        self.audit_retention = max(1, int(audit_retention))
        self.lightweight_stats = bool(lightweight_stats)
        self.scope_id = _clean_text(scope_id) or f"graph-session-{uuid.uuid4().hex}"
        self._store: SQLiteSessionMemoryStore | None = None
        self.storage_path = ""
        if self.storage_backend == "sqlite":
            resolved_storage_path = _clean_text(storage_path) or str((Path(tempfile.gettempdir()) / "tmcra_graph_session_memory.sqlite3").resolve())
            self._store = SQLiteSessionMemoryStore(resolved_storage_path, audit_retention=self.audit_retention)
            self.storage_path = str(self._store.storage_path)
            self.graph = self._store.load_graph(self.scope_id)
        elif self.storage_backend == "memory":
            self.storage_path = _clean_text(storage_path)
            self.graph = SessionMemoryGraphV2(
                audit_retention=self.audit_retention,
                persistence_backend="memory",
                persistence_path=self.storage_path,
            )
        else:
            raise ValueError(f"Unsupported storage backend: {self.storage_backend}")
        self._last_retrieval_context_tokens = 0
        self._last_writeback_summary: Dict[str, Any] = {}
        self.retrieval_mode = _normalize(retrieval_mode) or "heuristic"
        self.node_model_path = _clean_text(node_model_path)
        self.path_model_path = _clean_text(path_model_path)
        self.node_model_device = _clean_text(node_model_device)
        self.candidate_event_k = max(1, int(candidate_event_k))
        self.support_path_k = max(1, int(support_path_k))
        self.path_tunnel_rescue_k = max(0, int(path_tunnel_rescue_k or 0))
        self.path_tunnel_rescue_score_floor = max(0.0, float(path_tunnel_rescue_score_floor or 0.0))
        self.path_tunnel_rescue_min_age = max(0, int(path_tunnel_rescue_min_age or 0))
        self.path_tunnel_rescue_min_score_margin = max(0.0, float(path_tunnel_rescue_min_score_margin or 0.0))
        self.event_rerank_mode = _normalize(event_rerank_mode) or "matrix"
        self.matrix_event_top_k = max(1, int(matrix_event_top_k or DEFAULT_MATRIX_EVENT_TOP_K))
        self.write_embedder_index_mode = _normalize(os.getenv("TMCRA_WRITE_EMBEDDER_INDEX_MODE", ""))
        if not self.write_embedder_index_mode:
            self.write_embedder_index_mode = "off"
        try:
            self.write_embedder_index_max_terms = max(
                8,
                int(os.getenv("TMCRA_WRITE_EMBEDDER_INDEX_MAX_TERMS", "96") or 96),
            )
        except (TypeError, ValueError):
            self.write_embedder_index_max_terms = 96
        env_embedder_recall_mode = _normalize(os.getenv("TMCRA_EMBEDDER_INDEX_RECALL_MODE", ""))
        self.embedder_index_recall_mode = env_embedder_recall_mode or self.write_embedder_index_mode
        try:
            self.embedder_index_recall_k = max(
                0,
                int(os.getenv("TMCRA_EMBEDDER_INDEX_RECALL_K", "0") or 0),
            )
        except (TypeError, ValueError):
            self.embedder_index_recall_k = 0
        self.embedder_pre_recall_mode = _normalize(os.getenv("TMCRA_EMBEDDER_PRE_RECALL_MODE", ""))
        if not self.embedder_pre_recall_mode:
            self.embedder_pre_recall_mode = "off"
        try:
            self.embedder_pre_recall_k = max(
                0,
                int(os.getenv("TMCRA_EMBEDDER_PRE_RECALL_K", "0") or 0),
            )
        except (TypeError, ValueError):
            self.embedder_pre_recall_k = 0
        self.embedder_fusion_mode = _normalize(os.getenv("TMCRA_EMBEDDER_FUSION_MODE", ""))
        try:
            self.embedder_fusion_weight = max(
                0.0,
                float(os.getenv("TMCRA_EMBEDDER_FUSION_WEIGHT", "0.35") or 0.35),
            )
        except (TypeError, ValueError):
            self.embedder_fusion_weight = 0.35
        try:
            self.embedder_fusion_score_floor = max(
                0.0,
                float(os.getenv("TMCRA_EMBEDDER_FUSION_SCORE_FLOOR", "0.62") or 0.62),
            )
        except (TypeError, ValueError):
            self.embedder_fusion_score_floor = 0.62
        try:
            self.embedder_fusion_top_k = max(
                0,
                int(os.getenv("TMCRA_EMBEDDER_FUSION_TOP_K", "16") or 16),
            )
        except (TypeError, ValueError):
            self.embedder_fusion_top_k = 16
        try:
            self.embedder_fusion_select_k = max(
                0,
                int(os.getenv("TMCRA_EMBEDDER_FUSION_SELECT_K", "4") or 4),
            )
        except (TypeError, ValueError):
            self.embedder_fusion_select_k = 4
        try:
            self.embedder_fusion_max_boost = max(
                0.0,
                float(os.getenv("TMCRA_EMBEDDER_FUSION_MAX_BOOST", "0.42") or 0.42),
            )
        except (TypeError, ValueError):
            self.embedder_fusion_max_boost = 0.42
        env_router_mode = _clean_text(os.getenv("TMCRA_MEMORY_ROUTER_MODE", ""))
        self.memory_router_mode = _normalize(memory_router_mode) or _normalize(env_router_mode) or "observe"
        try:
            self.memory_router_threshold = float(
                os.getenv("TMCRA_MEMORY_ROUTER_THRESHOLD", "")
                or memory_router_threshold
                or _MEMORY_ROUTER_DEFAULT_THRESHOLD
            )
        except (TypeError, ValueError):
            self.memory_router_threshold = _MEMORY_ROUTER_DEFAULT_THRESHOLD
        try:
            self.memory_router_margin = float(
                os.getenv("TMCRA_MEMORY_ROUTER_MARGIN", "")
                or memory_router_margin
                or _MEMORY_ROUTER_DEFAULT_MARGIN
            )
        except (TypeError, ValueError):
            self.memory_router_margin = _MEMORY_ROUTER_DEFAULT_MARGIN
        self._loaded_node_scorer: LoadedNodeMemoryScorer | None = None
        self._node_scorer_error = ""
        env_planner_mode = _clean_text(os.getenv("TMCRA_INJECTION_PLANNER_MODE", ""))
        self.injection_planner_mode = _normalize(injection_planner_mode) or _normalize(env_planner_mode) or "observe"
        self.injection_planner_model_path = _clean_text(
            injection_planner_model_path or os.getenv("TMCRA_INJECTION_PLANNER_MODEL_PATH", "")
        )
        self.injection_planner_latest_path = _clean_text(
            injection_planner_latest_path or os.getenv("TMCRA_INJECTION_PLANNER_LATEST", "")
        )
        self.injection_planner_device = _clean_text(
            injection_planner_device or os.getenv("TMCRA_INJECTION_PLANNER_DEVICE", "")
        )
        self.injection_planner_selection_threshold_override = float(injection_planner_selection_threshold)
        self.injection_planner_row_threshold_override = float(injection_planner_row_threshold)
        self.injection_planner_logic_threshold_override = float(injection_planner_logic_threshold)
        self._loaded_injection_planner: Any | None = None
        self._injection_planner_config: Any | None = None
        self._injection_planner_payload: Dict[str, Any] = {}
        self._injection_planner_thresholds: Dict[str, float] = {}
        self._injection_planner_resolved_path = ""
        self._injection_planner_error = ""
        self._injection_planner_evidence_role_supported = False
        env_temporal_layer_mode = _clean_text(os.getenv("TMCRA_TEMPORAL_LAYER_MODE", ""))
        self.temporal_layer_mode = _normalize(temporal_layer_mode) or _normalize(env_temporal_layer_mode) or "observe"
        env_temporal_router_mode = _clean_text(os.getenv("TMCRA_TEMPORAL_ROUTER_MODE", ""))
        self.temporal_router_mode = _normalize(temporal_router_mode) or _normalize(env_temporal_router_mode) or "observe"
        self.temporal_router_dir = _clean_text(
            temporal_router_dir or os.getenv("TMCRA_TEMPORAL_ROUTER_DIR", "")
        )
        self.temporal_router_latest_path = _clean_text(
            temporal_router_latest_path
            or os.getenv("TMCRA_TEMPORAL_ROUTER_LATEST", "")
            or "/home/wuyou/tmcra_training/temporal_router_v1_latest.txt"
        )
        self.temporal_router_device = _clean_text(
            temporal_router_device or os.getenv("TMCRA_TEMPORAL_ROUTER_DEVICE", "")
        ) or "cpu"
        self._loaded_temporal_router: Any | None = None
        self._temporal_router_resolved_dir = ""
        self._temporal_router_error = ""

    def _empty_graph(self) -> SessionMemoryGraphV2:
        return SessionMemoryGraphV2(
            audit_retention=self.audit_retention,
            persistence_backend=self.storage_backend,
            persistence_path=self.storage_path,
        )

    def _reload_graph(self) -> None:
        if self._store is not None:
            self.graph = self._store.load_graph(self.scope_id)
        else:
            self.graph.configure_persistence(
                backend=self.storage_backend,
                path=self.storage_path,
                audit_retention=self.audit_retention,
            )

    def _persist_graph(self) -> None:
        self.graph.configure_persistence(
            backend=self.storage_backend,
            path=self.storage_path,
            audit_retention=self.audit_retention,
        )
        if self._store is not None:
            self._store.save_graph(self.scope_id, self.graph)

    def replace_graph(self, graph: SessionMemoryGraphV2) -> None:
        self.graph = graph
        self._persist_graph()

    def _storage_breakdown(self) -> Dict[str, int]:
        core_payload = self.graph._core_payload()
        audit_payload_full = self.graph._audit_payload()
        if self.lightweight_stats:
            audit_token_payload = {
                "totals": dict(self.graph.audit_event_totals),
                "trimmed": dict(self.graph.audit_trimmed_counts),
                "retained": {
                    "turn_log": len(self.graph.turn_log),
                    "retrieval_log": len(self.graph.retrieval_log),
                    "answer_support_log": len(self.graph.answer_support_log),
                },
                "audit_retention": int(self.graph.audit_retention),
            }
        else:
            audit_token_payload = audit_payload_full
        core_storage_bytes = len(json.dumps(core_payload, ensure_ascii=False).encode("utf-8"))
        audit_storage_bytes = len(json.dumps(audit_payload_full, ensure_ascii=False).encode("utf-8"))
        core_state_token_estimate = _estimate_tokens(json.dumps(core_payload, ensure_ascii=False))
        audit_state_token_estimate = _estimate_tokens(json.dumps(audit_token_payload, ensure_ascii=False))
        return {
            "core_storage_bytes": int(core_storage_bytes),
            "audit_storage_bytes": int(audit_storage_bytes),
            "storage_bytes": int(core_storage_bytes + audit_storage_bytes),
            "core_state_token_estimate": int(core_state_token_estimate),
            "audit_state_token_estimate": int(audit_state_token_estimate),
            "total_state_token_estimate": int(core_state_token_estimate + audit_state_token_estimate),
        }

    def reset(self) -> None:
        if self._store is not None:
            self._store.clear_scope(self.scope_id)
            self.graph = self._store.load_graph(self.scope_id)
        else:
            self.graph = self._empty_graph()
        self._last_retrieval_context_tokens = 0
        self._last_writeback_summary = {}

    def _temporal_layer_enabled(self) -> bool:
        return _normalize(self.temporal_layer_mode) not in _TEMPORAL_LAYER_DISABLED_MODES

    def _temporal_router_enabled(self) -> bool:
        return self._temporal_layer_enabled() and _normalize(self.temporal_router_mode) not in _TEMPORAL_LAYER_DISABLED_MODES

    def _resolve_temporal_router_dir(self) -> str:
        direct_dir = _clean_text(self.temporal_router_dir)
        if direct_dir:
            root = Path(direct_dir)
            if (root / "writer_temporal_router.pt").exists() and (root / "query_temporal_router.pt").exists():
                return str(root)
        latest_path = _clean_text(self.temporal_router_latest_path)
        if latest_path:
            pointer = Path(latest_path)
            if pointer.exists():
                lines = pointer.read_text(encoding="utf-8").splitlines()
                candidate = _clean_text(lines[0] if lines else "")
                if candidate:
                    root = Path(candidate)
                    if (root / "writer_temporal_router.pt").exists() and (root / "query_temporal_router.pt").exists():
                        return str(root)
        return ""

    def _load_temporal_router(self) -> LoadedTemporalRouter | None:
        if not self._temporal_router_enabled():
            self._temporal_router_error = "disabled"
            return None
        model_dir = self._resolve_temporal_router_dir()
        self._temporal_router_resolved_dir = model_dir
        if not model_dir:
            self._temporal_router_error = "model_dir_missing"
            return None
        if self._loaded_temporal_router is not None:
            return self._loaded_temporal_router
        try:
            self._loaded_temporal_router = LoadedTemporalRouter.from_dir(
                model_dir,
                device=self.temporal_router_device or "cpu",
            )
            self._temporal_router_error = ""
        except Exception as exc:  # pragma: no cover - defensive runtime path
            self._loaded_temporal_router = None
            self._temporal_router_error = f"{type(exc).__name__}: {exc}"
        return self._loaded_temporal_router

    def _temporal_router_status_metadata(self) -> Dict[str, Any]:
        router = self._load_temporal_router()
        return {
            "temporal_router_enabled": router is not None,
            "temporal_router_mode": _normalize(self.temporal_router_mode) or "observe",
            "temporal_router_model_dir": self._temporal_router_resolved_dir,
            "temporal_router_error": self._temporal_router_error,
        }

    def _session_timestamp_from_payloads(self, *payloads: Mapping[str, Any] | None) -> str:
        for payload in payloads:
            data = dict(payload or {})
            for key in ("session_timestamp", "timestamp", "created_at", "turn_timestamp"):
                value = _clean_text(data.get(key, ""))
                if value:
                    return value
            metadata = data.get("metadata")
            if isinstance(metadata, Mapping):
                for key in ("session_timestamp", "timestamp", "created_at", "turn_timestamp"):
                    value = _clean_text(metadata.get(key, ""))
                    if value:
                        return value
        return ""

    def _temporal_frame_for_turn(
        self,
        *,
        user_text: str,
        previous_turn: str = "",
        answer_payload: Dict[str, Any] | None = None,
        extraction_result: Dict[str, Any] | None = None,
    ) -> TemporalFrame | None:
        if not self._temporal_layer_enabled():
            return None
        session_timestamp = self._session_timestamp_from_payloads(answer_payload, extraction_result)
        model_frame = None
        for payload in (answer_payload, extraction_result):
            data = dict(payload or {})
            candidate = data.get("temporal_frame") or dict(data.get("metadata", {}) or {}).get("temporal_frame")
            if isinstance(candidate, Mapping):
                model_frame = candidate
                break
        fallback_frame = self.temporal_organizer.organize_turn(
            current_turn=user_text,
            previous_turn=previous_turn,
            session_timestamp=session_timestamp,
            speaker="user",
        )
        if model_frame is None:
            router = self._load_temporal_router()
            if router is not None and router.writer_available():
                predicted = router.predict_writer_frame(
                    current_turn=user_text,
                    previous_turn=previous_turn,
                    session_timestamp=session_timestamp,
                )
                writer_confidence = float(predicted.get("confidence", 0.0) or 0.0) if predicted else 0.0
                if predicted and writer_confidence >= _float_env("TMCRA_TEMPORAL_ROUTER_WRITER_MIN_CONFIDENCE", _TEMPORAL_ROUTER_DEFAULT_WRITER_MIN_CONFIDENCE):
                    frame_payload = fallback_frame.to_dict()
                    frame_payload.update(
                        {
                            key: value
                            for key, value in predicted.items()
                            if key in {"temporal_intent", "anchor_type", "granularity", "state_operation"}
                            and _clean_text(value)
                        }
                    )
                    if "should_create_timeline_edge" in predicted:
                        frame_payload["should_create_timeline_edge"] = bool(predicted.get("should_create_timeline_edge", False))
                    if writer_confidence > 0.0:
                        frame_payload["confidence"] = writer_confidence
                    frame_payload["metadata"] = {
                        **dict(frame_payload.get("metadata", {}) or {}),
                        **dict(predicted.get("metadata", {}) or {}),
                        **self._temporal_router_status_metadata(),
                    }
                    model_frame = frame_payload
        if model_frame is None:
            return fallback_frame
        return self.temporal_organizer.organize_turn(
            current_turn=user_text,
            previous_turn=previous_turn,
            session_timestamp=session_timestamp,
            speaker="user",
            model_frame=model_frame,
        )

    def _temporal_turn_metadata(self, frame: TemporalFrame | None) -> Dict[str, Any]:
        if frame is None:
            return {
                "temporal_layer_enabled": False,
                "temporal_layer_mode": _normalize(self.temporal_layer_mode) or "observe",
                **self._temporal_router_status_metadata(),
            }
        return {
            "temporal_layer_enabled": True,
            "temporal_layer_mode": _normalize(self.temporal_layer_mode) or "observe",
            "temporal_frame": frame.to_dict(),
            "temporal_intent": frame.temporal_intent,
            "temporal_subject_key": frame.subject_key,
            "temporal_state_operation": frame.state_operation,
            **self._temporal_router_status_metadata(),
        }

    def _apply_temporal_frame_to_records(self, records: Sequence[SessionMemoryRecordV2], frame: TemporalFrame | None) -> None:
        if frame is None:
            return
        if frame.temporal_intent == "non_temporal" and not frame.subject_key:
            return
        for record in records:
            metadata = dict(record.metadata or {})
            metadata.update(
                {
                    "temporal_layer": True,
                    "temporal_frame": frame.to_dict(),
                    "temporal_intent": frame.temporal_intent,
                    "temporal_subject": frame.subject,
                    "temporal_subject_key": frame.subject_key,
                    "temporal_state_operation": frame.state_operation,
                    "temporal_event_time": frame.event_time,
                    "temporal_state_valid_from": frame.state_valid_from,
                    "temporal_state_valid_to": frame.state_valid_to,
                }
            )
            record.metadata = metadata

    def _build_timeline_state_layer(self) -> TimelineStateLayer:
        layer = TimelineStateLayer()
        for turn in sorted(list(getattr(self.graph, "turn_log", []) or []), key=lambda item: int(item.get("turn_index", 0) or 0)):
            metadata = dict(turn.get("metadata", {}) or {})
            frame_payload = metadata.get("temporal_frame")
            if not isinstance(frame_payload, Mapping):
                continue
            frame = TemporalFrame.from_mapping(frame_payload)
            if frame.temporal_intent == "non_temporal" and not frame.new_state:
                continue
            record_ids = [item for item in list(turn.get("record_ids", []) or []) if _clean_text(item)]
            source_event_id = _clean_text(metadata.get("temporal_source_record_id", "")) or (record_ids[0] if record_ids else "")
            frame.metadata = {
                **dict(frame.metadata or {}),
                "source_text": _clean_text(turn.get("text", "")) or frame.evidence_span,
            }
            layer.apply_frame(
                frame,
                source_event_id=source_event_id,
                source_turn_id=_clean_text(turn.get("turn_id", "")),
                state_type=_clean_text(metadata.get("temporal_state_type", "")) or "profile",
            )
        return layer

    def _temporal_runtime_pack(self, query: str) -> Dict[str, Any]:
        metadata: Dict[str, Any] = {
            "temporal_runtime_enabled": False,
            "temporal_layer_mode": _normalize(self.temporal_layer_mode) or "observe",
            **self._temporal_router_status_metadata(),
        }
        if not self._temporal_layer_enabled():
            metadata["temporal_runtime_reason"] = "disabled"
            return {"metadata": metadata}
        plan = self.temporal_query_planner.plan(query)
        router = self._load_temporal_router()
        if router is not None and router.query_available():
            predicted_plan = router.predict_query_plan(query=query)
            query_confidence = float(predicted_plan.get("confidence", 0.0) or 0.0) if predicted_plan else 0.0
            router_confidences = dict(dict(predicted_plan.get("metadata", {}) or {}).get("temporal_router_confidences", {}) or {}) if predicted_plan else {}
            intent_confidence = float(router_confidences.get("query_temporal_intent", 0.0) or 0.0)
            if (
                predicted_plan
                and query_confidence >= _float_env("TMCRA_TEMPORAL_ROUTER_QUERY_MIN_CONFIDENCE", _TEMPORAL_ROUTER_DEFAULT_QUERY_MIN_CONFIDENCE)
                and intent_confidence >= _float_env("TMCRA_TEMPORAL_ROUTER_QUERY_INTENT_MIN_CONFIDENCE", _TEMPORAL_ROUTER_DEFAULT_QUERY_INTENT_MIN_CONFIDENCE)
            ):
                plan_payload = plan.to_dict()
                plan_payload.update(
                    {
                        key: value
                        for key, value in predicted_plan.items()
                        if key in {"query_temporal_intent", "timeline_operation"} and _clean_text(value)
                    }
                )
                for key in ("prefer_current_state", "prefer_previous_state", "requires_ordered_chain", "requires_comparison"):
                    if key in predicted_plan:
                        plan_payload[key] = bool(predicted_plan.get(key, False))
                if float(predicted_plan.get("confidence", 0.0) or 0.0) > 0.0:
                    plan_payload["confidence"] = float(predicted_plan.get("confidence", 0.0) or 0.0)
                plan_payload["metadata"] = {
                    **dict(plan_payload.get("metadata", {}) or {}),
                    **dict(predicted_plan.get("metadata", {}) or {}),
                    **self._temporal_router_status_metadata(),
                }
                plan = TemporalQueryPlan(**{key: value for key, value in plan_payload.items() if key in TemporalQueryPlan.__dataclass_fields__})
        metadata["temporal_query_plan"] = plan.to_dict()
        if plan.query_temporal_intent == "non_temporal" or plan.timeline_operation == "none":
            metadata["temporal_runtime_reason"] = "non_temporal_query"
            return {"plan": plan, "metadata": metadata}
        if plan.timeline_operation in {"query_current", "query_previous"} and not _clean_text(plan.target_subject_key):
            metadata["temporal_runtime_reason"] = "missing_target_subject"
            return {"plan": plan, "metadata": metadata}
        timeline_layer = self._build_timeline_state_layer()
        pack = self.timeline_evidence_builder.build(plan=plan, timeline_layer=timeline_layer)
        metadata.update(
            {
                "temporal_runtime_enabled": True,
                "temporal_runtime_reason": "ok",
                "temporal_evidence_pack": pack.to_dict(),
                "temporal_selected_answer_value": _clean_text(pack.selected_evidence.get("answer_value", "")),
                "temporal_selected_state_id": _clean_text(pack.selected_evidence.get("state_id", "")),
                "temporal_timeline_state_count": len(pack.timeline),
            }
        )
        return {"plan": plan, "pack": pack, "metadata": metadata}

    def _temporal_state_hit(
        self,
        *,
        state_payload: Mapping[str, Any],
        plan_payload: Mapping[str, Any],
        selected: bool,
        rank: int,
    ) -> MemoryHit | None:
        state_value = _clean_text(state_payload.get("state", ""))
        state_id = _clean_text(state_payload.get("state_id", ""))
        if not state_value or not state_id:
            return None
        source_event_id = _clean_text(state_payload.get("source_event_id", ""))
        source_record = self.graph.records_by_id.get(source_event_id)
        source_metadata = dict(source_record.metadata or {}) if source_record is not None else {}
        score = 0.98 if selected else max(0.75, 0.92 - (rank * 0.03))
        return MemoryHit(
            memory_id=f"temporal_state:{state_id}",
            category="time",
            value=state_value,
            relation="temporal_state",
            anchors=_dedupe([state_payload.get("time", ""), state_payload.get("source_text", ""), plan_payload.get("target_subject", "")], max_items=6),
            score=round(score, 6),
            source_kind="temporal_state_layer",
            slot_key=f"temporal.{_clean_text(plan_payload.get('target_subject_key', 'general'))}",
            state="active" if bool(state_payload.get("is_current", False)) else "history",
            turn_index=int(source_record.turn_index) if source_record is not None else 0,
            metadata={
                **source_metadata,
                "temporal_runtime_hit": True,
                "temporal_runtime_selected": bool(selected),
                "temporal_state_id": state_id,
                "temporal_state_value": state_value,
                "temporal_state_time": _clean_text(state_payload.get("time", "")),
                "temporal_state_valid_to": _clean_text(state_payload.get("valid_to", "")),
                "temporal_state_is_current": bool(state_payload.get("is_current", False)),
                "temporal_source_event_id": source_event_id,
                "temporal_source_turn_id": _clean_text(state_payload.get("source_turn_id", "")),
                "temporal_query_plan": dict(plan_payload),
            },
        )

    def _apply_temporal_evidence_pack_to_hits(
        self,
        hits: Sequence[MemoryHit],
        temporal_payload: Mapping[str, Any],
        *,
        top_k: int,
    ) -> Dict[str, Any]:
        metadata = dict(temporal_payload.get("metadata", {}) or {})
        pack = temporal_payload.get("pack")
        plan = temporal_payload.get("plan")
        if pack is None or plan is None or not bool(metadata.get("temporal_runtime_enabled", False)):
            return {"hits": list(hits), "metadata": metadata}
        pack_payload = pack.to_dict() if hasattr(pack, "to_dict") else dict(pack)
        plan_payload = plan.to_dict() if hasattr(plan, "to_dict") else dict(plan)
        selected_state_id = _clean_text(dict(pack_payload.get("selected_evidence", {}) or {}).get("state_id", ""))
        selected_answer_value = _clean_text(dict(pack_payload.get("selected_evidence", {}) or {}).get("answer_value", ""))
        timeline = list(pack_payload.get("timeline", []) or [])
        synthetic_hits: List[MemoryHit] = []
        if selected_state_id:
            selected_state = next((dict(item) for item in timeline if _clean_text(dict(item).get("state_id", "")) == selected_state_id), None)
            if selected_state is None and selected_answer_value:
                selected_state = {
                    "state_id": selected_state_id,
                    "state": selected_answer_value,
                    "source_event_id": dict(pack_payload.get("selected_evidence", {}) or {}).get("source_event_id", ""),
                    "source_turn_id": dict(pack_payload.get("selected_evidence", {}) or {}).get("source_turn_id", ""),
                    "is_current": bool(plan_payload.get("prefer_current_state", False)),
                }
            if selected_state:
                hit = self._temporal_state_hit(state_payload=selected_state, plan_payload=plan_payload, selected=True, rank=0)
                if hit is not None:
                    synthetic_hits.append(hit)
        if bool(plan_payload.get("requires_ordered_chain", False)) or bool(plan_payload.get("requires_comparison", False)):
            for index, state_payload in enumerate(timeline):
                if _clean_text(dict(state_payload).get("state_id", "")) == selected_state_id:
                    continue
                hit = self._temporal_state_hit(
                    state_payload=dict(state_payload),
                    plan_payload=plan_payload,
                    selected=False,
                    rank=index + 1,
                )
                if hit is not None:
                    synthetic_hits.append(hit)
        existing: List[MemoryHit] = []
        synthetic_source_ids = {
            _clean_text((hit.metadata or {}).get("temporal_source_event_id", ""))
            for hit in synthetic_hits
            if _clean_text((hit.metadata or {}).get("temporal_source_event_id", ""))
        }
        for hit in hits:
            hit_metadata = dict(hit.metadata or {})
            source_event_id = _clean_text(hit_metadata.get("event_id", hit.memory_id))
            if source_event_id in synthetic_source_ids:
                hit_metadata.update(
                    {
                        "temporal_runtime_support": True,
                        "temporal_query_plan": dict(plan_payload),
                        "temporal_selected_answer_value": selected_answer_value,
                        "temporal_selected_state_id": selected_state_id,
                    }
                )
                hit = MemoryHit(
                    memory_id=hit.memory_id,
                    category=hit.category,
                    value=hit.value,
                    relation=hit.relation,
                    anchors=list(hit.anchors),
                    score=hit.score,
                    source_kind=hit.source_kind,
                    slot_key=hit.slot_key,
                    state=hit.state,
                    turn_index=int(hit.turn_index),
                    metadata=hit_metadata,
                )
            existing.append(hit)
        merged: List[MemoryHit] = []
        seen = set()
        for hit in [*synthetic_hits, *existing]:
            key = _clean_text(hit.memory_id)
            if key and key in seen:
                continue
            if key:
                seen.add(key)
            merged.append(hit)
        metadata.update(
            {
                "temporal_runtime_injected_hit_count": len(synthetic_hits),
                "temporal_runtime_injected_hit_ids": [hit.memory_id for hit in synthetic_hits],
            }
        )
        return {"hits": merged[: max(len(merged), top_k)], "metadata": metadata}

    def ingest_turn(
        self,
        user_text: str,
        assistant_text: str = "",
        *,
        answer_payload: Dict[str, Any] | None = None,
        extraction_result: Dict[str, Any] | None = None,
    ) -> None:
        self._reload_graph()
        turn_index = self.graph.next_turn()
        previous_topic = _last_topic_turn(self.graph)
        previous_turn_text = _clean_text(self.graph.turn_log[-1].get("text", "")) if self.graph.turn_log else ""
        temporal_frame = self._temporal_frame_for_turn(
            user_text=user_text,
            previous_turn=previous_turn_text,
            answer_payload=answer_payload,
            extraction_result=extraction_result,
        )
        topic_bucket = _assign_topic_bucket_for_text(
            self.graph,
            user_text,
            answer_payload=answer_payload,
            turn_index=turn_index,
            create=True,
        )
        records = _build_turn_records(
            self.extractor,
            user_text=user_text,
            answer_payload=answer_payload,
            extraction_result=extraction_result,
            turn_index=turn_index,
            allow_auto_extract=self.auto_extract,
            profile=self.profile,
        )
        self._apply_temporal_frame_to_records(records, temporal_frame)
        _apply_topic_bucket_to_records(records, topic_bucket)
        stored_ids = self.graph.add_records(records)
        write_embedder_metadata = _apply_write_embedder_index_to_graph(
            self.graph,
            stored_ids=stored_ids,
            turn_text=user_text,
            turn_index=turn_index,
            mode=self.write_embedder_index_mode,
            max_terms=self.write_embedder_index_max_terms,
        )
        topic_bridge_metadata = _add_topic_bridge_edges(
            self.graph,
            previous_topic=previous_topic,
            current_topic=topic_bucket,
            current_record_ids=stored_ids,
            turn_index=turn_index,
            evidence=user_text,
        )
        dialogue_tunnel_metadata = _add_dialogue_tunnel_edges(
            self.graph,
            current_topic=topic_bucket,
            current_record_ids=stored_ids,
            turn_index=turn_index,
            evidence=user_text,
        )
        turn_kind = "memory_write" if stored_ids else "noise"
        self.graph.record_turn(
            turn_kind=turn_kind,
            text=user_text,
            turn_index=turn_index,
            record_ids=stored_ids,
            speaker="user",
            assistant_text=assistant_text,
            metadata={
                "source": "user_turn",
                "auto_extract": bool(self.auto_extract),
                "topic_bucket_id": _clean_text(topic_bucket.get("topic_bucket_id", "")),
                "topic_label": _clean_text(topic_bucket.get("topic_label", "")),
                "topic_keywords": list(topic_bucket.get("topic_keywords", []) or []),
                "temporal_source_record_id": stored_ids[0] if stored_ids else "",
                **self._temporal_turn_metadata(temporal_frame),
                **write_embedder_metadata,
                **topic_bridge_metadata,
                **dialogue_tunnel_metadata,
            },
        )
        self._persist_graph()

    def ingest_answer_writeback(
        self,
        *,
        query_text: str,
        answer_text: str,
        answer_id: str,
        writeback_records: List[Dict[str, Any]],
        trace: Dict[str, Any] | None = None,
    ) -> List[str]:
        self._reload_graph()
        if not writeback_records:
            self._last_writeback_summary = {"stored_record_ids": [], "promotion_events": []}
            return []
        turn_index = self.graph.next_turn()
        records: List[SessionMemoryRecordV2] = []
        writeback_classes: List[str] = []
        for index, raw in enumerate(writeback_records):
            if not isinstance(raw, dict):
                continue
            category = _clean_text(raw.get("category", "fact")) or "fact"
            value = _clean_text(raw.get("value", ""))
            raw_slot_key = _clean_text(raw.get("slot_key", "")) or _clean_text(raw.get("slot", ""))
            anchors = _dedupe(raw.get("anchors", []) or [], max_items=8)
            slot_key = self.profile.stable_slot_key(
                category=category,
                value=value,
                anchors=anchors,
                slot_key=raw_slot_key,
                relation=_clean_text(raw.get("relation", "")),
                metadata=dict(raw.get("metadata", {}) or {}),
            )
            if not value or not slot_key:
                continue
            raw_metadata = dict(raw.get("metadata", {}) or {})
            writeback_class = _clean_text(raw_metadata.get("writeback_class", "")) or "fact"
            writeback_classes.append(writeback_class)
            metadata = {
                **raw_metadata,
                "memory_role": _clean_text(raw_metadata.get("memory_role", "")) or "assistant",
                "authority": _clean_text(raw_metadata.get("authority", "")) or "derived",
                "canonical_slot_key": _clean_text(raw_metadata.get("canonical_slot_key", "")) or slot_key.removeprefix("assistant.").split(".fact", 1)[0].split(".state_change", 1)[0].split(".high_conf_conclusion", 1)[0],
                "writeback_class": writeback_class,
                "origin_query": _clean_text(raw_metadata.get("origin_query", "")) or _clean_text(query_text),
                "origin_answer_id": _clean_text(raw_metadata.get("origin_answer_id", "")) or answer_id,
                "origin_answer_ids": _dedupe([*(raw_metadata.get("origin_answer_ids", []) or []), _clean_text(raw_metadata.get("origin_answer_id", "")) or answer_id]),
                "support_memory_ids": _dedupe(raw_metadata.get("support_memory_ids", []) or []),
                "support_fact_refs": _dedupe(raw_metadata.get("support_fact_refs", []) or []),
                "support_path_refs": _dedupe(raw_metadata.get("support_path_refs", []) or []),
                "promotion_state": _clean_text(raw_metadata.get("promotion_state", "")) or "candidate",
                "answer_id": answer_id,
            }
            record = SessionMemoryRecordV2(
                memory_id=f"{slot_key}:{turn_index}:assistant:{index}",
                category=category,
                slot_key=slot_key,
                value=value,
                relation=_clean_text(raw.get("relation", "")) or "assistant_memory",
                anchor_concepts=anchors,
                evidence_anchors=anchors,
                salience=float(raw.get("salience", 0.62) or 0.62),
                confidence=float(raw.get("confidence", 0.88) or 0.88),
                source_kind=_clean_text(raw.get("source_kind", "")) or "assistant_writeback",
                turn_index=turn_index,
                state=_clean_text(raw.get("state", "")) or "active",
                metadata=metadata,
            )
            records.append(record)
        stored_ids = self.graph.add_records(records)
        write_embedder_metadata = _apply_write_embedder_index_to_graph(
            self.graph,
            stored_ids=stored_ids,
            turn_text=" ".join([_clean_text(query_text), _clean_text(answer_text)]),
            turn_index=turn_index,
            mode=self.write_embedder_index_mode,
            max_terms=self.write_embedder_index_max_terms,
        )
        promotion_events = self._apply_writeback_promotions(stored_ids)
        writeback_class = writeback_classes[0] if len(set(writeback_classes)) == 1 and writeback_classes else ("mixed" if writeback_classes else "")
        self.graph.record_turn(
            turn_kind="assistant_writeback" if stored_ids else "assistant_writeback_empty",
            text=query_text,
            turn_index=turn_index,
            record_ids=stored_ids,
            speaker="assistant",
            assistant_text=answer_text,
            writeback_class=writeback_class,
            metadata={
                "source": "assistant_writeback",
                "answer_id": answer_id,
                "trace": dict(trace or {}),
                **write_embedder_metadata,
            },
        )
        self._last_writeback_summary = {
            "stored_record_ids": list(stored_ids),
            "promotion_events": list(promotion_events),
            **write_embedder_metadata,
        }
        self._persist_graph()
        return stored_ids

    def last_writeback_summary(self) -> Dict[str, Any]:
        return dict(self._last_writeback_summary)

    def _resolve_injection_planner_model_path(self) -> str:
        explicit_path = _clean_text(self.injection_planner_model_path)
        if explicit_path:
            path = Path(explicit_path).expanduser()
            if path.is_dir():
                path = path / "injection_planner.pt"
            return str(path)
        latest_path = _clean_text(self.injection_planner_latest_path)
        if not latest_path:
            return ""
        latest = Path(latest_path).expanduser()
        if latest.is_dir():
            return str(latest / "injection_planner.pt")
        if latest.suffix == ".pt":
            return str(latest)
        try:
            target = Path(latest.read_text(encoding="utf-8").strip()).expanduser()
        except Exception as exc:
            self._injection_planner_error = f"latest_pointer_read_failed: {exc}"
            return ""
        if target.is_dir():
            target = target / "injection_planner.pt"
        return str(target)

    def _load_injection_planner(self) -> Any | None:
        normalized_mode = _normalize(self.injection_planner_mode)
        if normalized_mode in _INJECTION_PLANNER_DISABLED_MODES:
            self._injection_planner_error = "disabled"
            return None
        if self._loaded_injection_planner is not None:
            return self._loaded_injection_planner
        torch_module = getattr(injection_planner_runtime, "torch", None)
        if torch_module is None:
            self._injection_planner_error = "torch_unavailable"
            return None
        model_path_text = self._resolve_injection_planner_model_path()
        if not model_path_text:
            if not self._injection_planner_error:
                self._injection_planner_error = "model_path_missing"
            return None
        model_path = Path(model_path_text)
        if not model_path.exists():
            self._injection_planner_error = f"model_path_not_found: {model_path}"
            return None
        try:
            device = torch_module.device(self.injection_planner_device or "cpu")
            payload = torch_module.load(model_path, map_location=device, weights_only=False)
            config = injection_planner_runtime.InjectionPlannerConfig.from_dict(dict(payload.get("config", {}) or {}))
            model = injection_planner_runtime.InjectionPlannerModel(config).to(device)
            state_dict = dict(payload.get("state_dict", {}) or {})
            model.load_state_dict(state_dict, strict=False)
            model.eval()
            self._loaded_injection_planner = model
            self._injection_planner_config = config
            self._injection_planner_payload = dict(payload)
            self._injection_planner_resolved_path = str(model_path)
            self._injection_planner_error = ""
            self._injection_planner_evidence_role_supported = any(
                str(key).startswith("evidence_role_head.") for key in state_dict
            )
            thresholds = {"selection_threshold": 0.5, "row_threshold": 0.5, "logic_threshold": 0.5}
            summary_path = model_path.parent / "train_summary.json"
            if summary_path.exists():
                try:
                    summary = json.loads(summary_path.read_text(encoding="utf-8"))
                    calibration = dict(summary.get("calibration", {}) or {})
                    logic_calibration = dict(summary.get("logic_calibration", {}) or {})
                    if calibration.get("selection_threshold") is not None:
                        thresholds["selection_threshold"] = float(calibration.get("selection_threshold"))
                    if calibration.get("row_threshold") is not None:
                        thresholds["row_threshold"] = float(calibration.get("row_threshold"))
                    if logic_calibration.get("logic_threshold") is not None:
                        thresholds["logic_threshold"] = float(logic_calibration.get("logic_threshold"))
                except Exception:
                    pass
            if self.injection_planner_selection_threshold_override >= 0.0:
                thresholds["selection_threshold"] = float(self.injection_planner_selection_threshold_override)
            if self.injection_planner_row_threshold_override >= 0.0:
                thresholds["row_threshold"] = float(self.injection_planner_row_threshold_override)
            if self.injection_planner_logic_threshold_override >= 0.0:
                thresholds["logic_threshold"] = float(self.injection_planner_logic_threshold_override)
            self._injection_planner_thresholds = thresholds
        except Exception as exc:
            self._loaded_injection_planner = None
            self._injection_planner_config = None
            self._injection_planner_payload = {}
            self._injection_planner_resolved_path = str(model_path)
            self._injection_planner_error = str(exc)
            self._injection_planner_evidence_role_supported = False
        return self._loaded_injection_planner

    def _injection_planner_candidate_from_hit(
        self,
        query: str,
        hit: MemoryHit,
        *,
        index: int,
        current_turn_index: int,
    ) -> Dict[str, Any]:
        metadata = dict(hit.metadata or {})
        category = _normalize(hit.category)
        source_kind = _normalize(hit.source_kind)
        logic_roles = [
            _normalize(item)
            for item in (
                metadata.get("logic_roles", [])
                if isinstance(metadata.get("logic_roles", []), list)
                else [metadata.get("logic_roles", "")]
            )
            if _normalize(item) in set(injection_planner_runtime.LOGIC_ROLES)
        ]
        if category in set(injection_planner_runtime.LOGIC_ROLES):
            logic_roles.append(category)
        if _normalize(metadata.get("writeback_class", "")) in set(injection_planner_runtime.LOGIC_ROLES):
            logic_roles.append(_normalize(metadata.get("writeback_class", "")))
        if not logic_roles:
            logic_roles = ["negative"] if _normalize(hit.state) in {"stale", "superseded", "false"} else ["evidence"]
        evidence_snippet_role = _normalize(metadata.get("evidence_snippet_role", ""))
        if bool(metadata.get("profile_layer")) or category == "profile" or source_kind.endswith("profile"):
            layer = "profile"
        elif category == "time" or float(metadata.get("temporal_score", 0.0) or 0.0) > 0.0:
            layer = "temporal"
        elif "resource" in logic_roles or _clean_text(metadata.get("resource_key", "")):
            layer = "resource"
        elif (
            source_kind in {"path_tunnel", "path_support", "public_dialog_path"}
            or evidence_snippet_role in {"selected_path_support", "path_tunnel_support"}
            or bool(metadata.get("path_tunnel_node"))
        ):
            layer = "path_tunnel"
        elif bool(metadata.get("topic_bucket_rerank")) or bool(metadata.get("topic_bucket_dialogue_tunnel_allowed")):
            layer = "topic_tunnel"
        else:
            layer = "event"
        normalized_state = _normalize(hit.state)
        if normalized_state in {"stale", "superseded", "false"}:
            temporal_state = "superseded"
        elif bool(metadata.get("topic_bucket_current_subject_preserved")) or bool(metadata.get("current_subject_protected")):
            temporal_state = "current"
        elif current_turn_index and int(hit.turn_index or 0) and current_turn_index - int(hit.turn_index) <= 1:
            temporal_state = "current"
        elif normalized_state == "active":
            temporal_state = "stable"
        else:
            temporal_state = "historical"
        query_tokens = set(_tokenize(query))
        hit_tokens = set(_tokenize(" ".join([hit.value, " ".join(hit.anchors), _clean_text(metadata.get("topic_label", ""))])))
        overlap = len(query_tokens & hit_tokens) / max(1, len(query_tokens | hit_tokens))
        candidate_id = _clean_text(hit.memory_id) or f"hit:{index}"
        topic_label = _clean_text(metadata.get("topic_label", "")) or _clean_text(metadata.get("topic_bucket_query_label", ""))
        profile_key = _clean_text(metadata.get("profile_type", "")) or _clean_text(metadata.get("semantic_slot", ""))
        if not profile_key and layer in {"profile", "temporal", "topic_tunnel"}:
            profile_key = topic_label
        semantic_similarity = max(
            0.0,
            min(
                1.0,
                float(
                    metadata.get(
                        "semantic_similarity",
                        metadata.get(
                            "answer_window_semantic_similarity",
                            metadata.get(
                                "embedder_similarity",
                                metadata.get("dense_similarity", metadata.get("bge_m3_similarity", hit.score)),
                            ),
                        ),
                    )
                    or 0.0
                ),
            ),
        )
        return {
            "id": candidate_id,
            "text": hit.value,
            "summary": _clean_text(metadata.get("source_turn_text", "")) or _clean_text(metadata.get("event_summary", "")),
            "topic": topic_label,
            "profile_key": profile_key,
            "resource_key": _clean_text(metadata.get("resource_key", "")) or (_clean_text(hit.slot_key) if layer == "resource" else ""),
            "layer": layer,
            "temporal_state": temporal_state,
            "logic_roles": _dedupe(logic_roles),
            "query_overlap": round(float(overlap), 6),
            "retrieval_score": max(0.0, min(1.0, float(hit.score or 0.0))),
            "graph_score": max(
                0.0,
                min(
                    1.0,
                    float(
                        metadata.get(
                            "event_score",
                            metadata.get("recall_score", metadata.get("hybrid_score", hit.score)),
                        )
                        or 0.0
                    ),
                ),
            ),
            "tunnel_score": max(
                0.0,
                min(
                    1.0,
                    max(
                        float(metadata.get("path_tunnel_support_score", 0.0) or 0.0),
                        float(metadata.get("event_tunnel_support_score", 0.0) or 0.0),
                        float(metadata.get("path_chain_extension_delta_score", 0.0) or 0.0),
                    ),
                ),
            ),
            "topic_similarity": max(0.0, min(1.0, float(metadata.get("topic_bucket_overlap", 0.0) or 0.0))),
            "semantic_similarity": semantic_similarity,
            "confidence": max(0.0, min(1.0, float(metadata.get("confidence", hit.score or 0.0) or 0.0))),
            "rank_score": round(1.0 / float(index + 1), 6),
            "age_turns": max(0, int(current_turn_index) - int(hit.turn_index or 0)) if current_turn_index else 0,
            "branch_depth": max(0, len(list(metadata.get("selected_path_ids", []) or []))),
            "contradicts_current": normalized_state in {"stale", "superseded", "false"} or bool(metadata.get("contradicts_current")),
            "is_current": temporal_state == "current",
        }

    def _apply_injection_planner_to_hits(
        self,
        query: str,
        hits: Sequence[MemoryHit],
        *,
        top_k: int,
    ) -> Dict[str, Any]:
        normalized_mode = _normalize(self.injection_planner_mode)
        base_metadata: Dict[str, Any] = {
            "injection_planner_enabled": False,
            "injection_planner_mode": normalized_mode or "observe",
        }
        if normalized_mode in _INJECTION_PLANNER_DISABLED_MODES:
            base_metadata["injection_planner_reason"] = "disabled"
            return {"hits": list(hits), "metadata": base_metadata}
        if not hits:
            base_metadata["injection_planner_reason"] = "no_hits"
            return {"hits": list(hits), "metadata": base_metadata}
        model = self._load_injection_planner()
        if model is None or self._injection_planner_config is None:
            base_metadata.update(
                {
                    "injection_planner_reason": self._injection_planner_error or "load_failed",
                    "injection_planner_model_path": self._injection_planner_resolved_path,
                }
            )
            return {"hits": list(hits), "metadata": base_metadata}
        torch_module = getattr(injection_planner_runtime, "torch", None)
        if torch_module is None:
            base_metadata["injection_planner_reason"] = "torch_unavailable"
            return {"hits": list(hits), "metadata": base_metadata}
        current_turn_index = max(
            [int(getattr(self.graph, "turn_index", 0) or 0), *[int(hit.turn_index or 0) for hit in hits]],
            default=0,
        )
        candidates = [
            self._injection_planner_candidate_from_hit(query, hit, index=index, current_turn_index=current_turn_index)
            for index, hit in enumerate(hits)
        ]
        row = {"id": "runtime_injection_plan", "query": query, "candidates": candidates, "gold": {}}
        try:
            dataset = injection_planner_runtime.InjectionPlannerDataset([row], self._injection_planner_config)
            batch = injection_planner_runtime.collate_injection_batch([dataset[0]])
            device = next(model.parameters()).device
            model_batch = {
                key: value.to(device) if hasattr(value, "to") else value
                for key, value in dict(batch).items()
            }
            with torch_module.no_grad():
                outputs = model(model_batch["features"], model_batch["valid_mask"])
                selection_scores = torch_module.sigmoid(outputs["selection_logits"])[0].detach().cpu().tolist()
                should_inject_score = float(torch_module.sigmoid(outputs["should_inject_logits"])[0].detach().cpu().item())
                mode_probs = torch_module.softmax(outputs["injection_mode_logits"], dim=-1)[0].detach().cpu()
                mode_index = int(torch_module.argmax(mode_probs).item())
                temporal_indices = torch_module.argmax(outputs["temporal_logits"], dim=-1)[0].detach().cpu().tolist()
                logic_scores = torch_module.sigmoid(outputs["logic_logits"])[0].detach().cpu().tolist()
                if bool(self._injection_planner_evidence_role_supported) and "evidence_role_logits" in outputs:
                    evidence_role_indices = torch_module.argmax(outputs["evidence_role_logits"], dim=-1)[0].detach().cpu().tolist()
                else:
                    evidence_role_indices = [
                        injection_planner_runtime.EVIDENCE_ROLES.index("direct_answer")
                        for _ in candidates
                    ]
        except Exception as exc:
            base_metadata.update(
                {
                    "injection_planner_reason": f"inference_failed: {exc}",
                    "injection_planner_model_path": self._injection_planner_resolved_path,
                }
            )
            return {"hits": list(hits), "metadata": base_metadata}
        thresholds = {
            "selection_threshold": float(self._injection_planner_thresholds.get("selection_threshold", 0.5)),
            "row_threshold": float(self._injection_planner_thresholds.get("row_threshold", 0.5)),
            "logic_threshold": float(self._injection_planner_thresholds.get("logic_threshold", 0.5)),
        }
        injection_mode = injection_planner_runtime.INJECTION_MODES[mode_index]
        row_allows_injection = should_inject_score >= thresholds["row_threshold"] and injection_mode != "none"
        predictions: Dict[str, Dict[str, Any]] = {}
        for index, candidate in enumerate(candidates):
            logic_roles = [
                role
                for role, score in zip(injection_planner_runtime.LOGIC_ROLES, logic_scores[index])
                if float(score) >= thresholds["logic_threshold"]
            ]
            evidence_role = injection_planner_runtime.EVIDENCE_ROLES[int(evidence_role_indices[index])]
            role_allows_selection = evidence_role not in {"noise", "negative_evidence"}
            predictions[candidate["id"]] = {
                "selection_score": float(selection_scores[index]),
                "selected": bool(
                    row_allows_injection
                    and role_allows_selection
                    and float(selection_scores[index]) >= thresholds["selection_threshold"]
                ),
                "temporal_state": injection_planner_runtime.TEMPORAL_STATES[int(temporal_indices[index])],
                "evidence_role": evidence_role,
                "logic_roles": logic_roles or ["evidence"],
                "candidate_layer": candidate.get("layer", ""),
                "role_allows_selection": bool(role_allows_selection),
            }
        annotated_hits: List[MemoryHit] = []
        for index, hit in enumerate(hits):
            candidate_id = _clean_text(hit.memory_id) or f"hit:{index}"
            prediction = predictions.get(candidate_id, {})
            metadata = dict(hit.metadata or {})
            metadata.update(
                {
                    "injection_planner_enabled": True,
                    "injection_planner_mode": normalized_mode or "observe",
                    "injection_planner_model_path": self._injection_planner_resolved_path,
                    "injection_planner_candidate_id": candidate_id,
                    "injection_planner_score": round(float(prediction.get("selection_score", 0.0)), 6),
                    "injection_planner_selected": bool(prediction.get("selected", False)),
                    "injection_planner_temporal_state": _clean_text(prediction.get("temporal_state", "")),
                    "injection_planner_evidence_role": _clean_text(prediction.get("evidence_role", "")),
                    "injection_planner_role_allows_selection": bool(prediction.get("role_allows_selection", False)),
                    "injection_planner_logic_roles": list(prediction.get("logic_roles", []) or []),
                    "injection_planner_candidate_layer": _clean_text(prediction.get("candidate_layer", "")),
                    "injection_planner_injection_mode": injection_mode,
                    "injection_planner_should_inject_score": round(float(should_inject_score), 6),
                    "injection_planner_thresholds": dict(thresholds),
                    "injection_planner_evidence_role_supported": bool(self._injection_planner_evidence_role_supported),
                }
            )
            planner_score = float(prediction.get("selection_score", 0.0))
            next_score = max(float(hit.score), planner_score) if bool(prediction.get("selected", False)) else float(hit.score)
            annotated_hits.append(
                MemoryHit(
                    memory_id=hit.memory_id,
                    category=hit.category,
                    value=hit.value,
                    relation=hit.relation,
                    anchors=list(hit.anchors),
                    score=round(next_score, 6),
                    source_kind=hit.source_kind,
                    slot_key=hit.slot_key,
                    state=hit.state,
                    turn_index=int(hit.turn_index),
                    metadata=metadata,
                )
            )
        selected_count = sum(1 for item in annotated_hits if bool((item.metadata or {}).get("injection_planner_selected")))
        if normalized_mode in _INJECTION_PLANNER_FORCE_MODES and selected_count:
            selected_ids = {hit.memory_id for hit in annotated_hits if bool((hit.metadata or {}).get("injection_planner_selected"))}
            planned_hits = [hit for hit in annotated_hits if hit.memory_id in selected_ids]
        elif normalized_mode in _INJECTION_PLANNER_GUIDED_MODES and selected_count:
            planned_hits = sorted(
                annotated_hits,
                key=lambda hit: (
                    not bool((hit.metadata or {}).get("injection_planner_selected")),
                    -float((hit.metadata or {}).get("injection_planner_score", 0.0) or 0.0),
                    -float(hit.score or 0.0),
                ),
            )
        else:
            planned_hits = annotated_hits
        base_metadata.update(
            {
                "injection_planner_enabled": True,
                "injection_planner_reason": "ok",
                "injection_planner_model_path": self._injection_planner_resolved_path,
                "injection_planner_candidate_count": len(candidates),
                "injection_planner_selected_count": int(selected_count),
                "injection_planner_should_inject_score": round(float(should_inject_score), 6),
                "injection_planner_injection_mode": injection_mode,
                "injection_planner_thresholds": dict(thresholds),
                "injection_planner_evidence_role_supported": bool(self._injection_planner_evidence_role_supported),
                "injection_planner_guided": bool(normalized_mode in _INJECTION_PLANNER_GUIDED_MODES | _INJECTION_PLANNER_FORCE_MODES),
                "injection_planner_prediction_ids": [
                    candidate_id
                    for candidate_id, prediction in predictions.items()
                    if bool(prediction.get("selected", False))
                ],
            }
        )
        return {"hits": planned_hits[: max(len(planned_hits), top_k)], "metadata": base_metadata}

    def _node_scorer(self) -> LoadedNodeMemoryScorer | None:
        if self.retrieval_mode != "hybrid_node_scored":
            return None
        if self._loaded_node_scorer is not None:
            return self._loaded_node_scorer
        if not self.node_model_path:
            self._node_scorer_error = "node_model_path_missing"
            return None
        try:
            self._loaded_node_scorer = LoadedNodeMemoryScorer(
                node_model_path=Path(self.node_model_path),
                path_model_path=Path(self.path_model_path) if self.path_model_path else None,
                device=self.node_model_device or None,
            )
        except Exception as exc:
            self._node_scorer_error = str(exc)
            self._loaded_node_scorer = None
        return self._loaded_node_scorer

    def _apply_writeback_promotions(self, stored_ids: Sequence[str]) -> List[Dict[str, Any]]:
        promotion_events: List[Dict[str, Any]] = []
        for memory_id in stored_ids:
            record = self.graph.records_by_id.get(memory_id)
            if record is None or not isinstance(record.metadata, dict):
                continue
            if _normalize(record.metadata.get("memory_role", "")) != "assistant" or _normalize(record.metadata.get("authority", "")) != "derived":
                continue
            canonical_slot_key = _clean_text(record.metadata.get("canonical_slot_key", ""))
            writeback_class = _clean_text(record.metadata.get("writeback_class", ""))
            if not canonical_slot_key or not writeback_class:
                continue
            support_refs = self._support_ref_union(record.metadata)
            confidence = float(record.confidence or 0.0)
            same_records = [
                item
                for item in self.graph.records_by_id.values()
                if isinstance(item.metadata, dict)
                and _normalize(item.metadata.get("memory_role", "")) == "assistant"
                and _normalize(item.metadata.get("canonical_slot_key", "")) == _normalize(canonical_slot_key)
                and _normalize(item.metadata.get("writeback_class", "")) == _normalize(writeback_class)
                and _normalize(item.value) == _normalize(record.value)
            ]
            qualifying = [item for item in same_records if float(item.confidence or 0.0) >= 0.9]
            distinct_answers = {
                _clean_text(answer_id)
                for item in qualifying
                for answer_id in [*list(item.metadata.get("origin_answer_ids", []) or []), _clean_text(item.metadata.get("origin_answer_id", ""))]
                if _clean_text(answer_id)
            }
            aggregated_support = set()
            for item in same_records:
                aggregated_support.update(self._support_ref_union(item.metadata))
            fast_promotion = writeback_class in {"fact", "state_change"} and confidence >= 0.97 and len(support_refs) >= 3
            standard_promotion = len(distinct_answers) >= 2 and len(aggregated_support) >= 2
            if not (fast_promotion or standard_promotion):
                record.metadata["promotion_state"] = "candidate"
                continue
            source_head = self._source_head_for_canonical(canonical_slot_key)
            blocked_conflict = source_head is not None and _normalize(source_head.value) != _normalize(record.value)
            promoted_slot = f"promoted.{canonical_slot_key}"
            promoted_metadata = {
                **dict(record.metadata or {}),
                "memory_role": "assistant",
                "authority": "promoted",
                "canonical_slot_key": canonical_slot_key,
                "writeback_class": writeback_class,
                "promotion_state": "blocked_conflict" if blocked_conflict else "promoted",
                "support_memory_ids": sorted({*list(record.metadata.get("support_memory_ids", []) or []), *[ref for ref in aggregated_support if ref.startswith("fact:") is False and ref.startswith("path:") is False]}),
                "support_fact_refs": sorted({*list(record.metadata.get("support_fact_refs", []) or []), *[ref for ref in aggregated_support if ref.startswith("fact:")]}),
                "support_path_refs": sorted({*list(record.metadata.get("support_path_refs", []) or []), *[ref for ref in aggregated_support if ref.startswith("path:")]}),
            }
            promoted_record = SessionMemoryRecordV2(
                memory_id=f"{promoted_slot}:{record.turn_index}:promoted",
                category=record.category,
                slot_key=promoted_slot,
                value=record.value,
                relation=record.relation,
                anchor_concepts=list(record.anchor_concepts),
                evidence_anchors=list(record.evidence_anchors),
                salience=max(float(record.salience), 0.78),
                confidence=max(float(record.confidence), 0.9),
                source_kind=f"promoted_{record.source_kind}",
                turn_index=int(record.turn_index),
                state="active",
                metadata=promoted_metadata,
            )
            promoted_ids = self.graph.add_records([promoted_record])
            record.metadata["promotion_state"] = "blocked_conflict" if blocked_conflict else "promoted"
            promotion_events.append(
                {
                    "source_memory_id": record.memory_id,
                    "promoted_record_ids": list(promoted_ids),
                    "canonical_slot_key": canonical_slot_key,
                    "writeback_class": writeback_class,
                    "promotion_state": record.metadata["promotion_state"],
                    "blocked_conflict": bool(blocked_conflict),
                }
            )
        return promotion_events

    def _source_head_for_canonical(self, canonical_slot_key: str) -> SessionMemoryRecordV2 | None:
        head_id = self.graph.slot_heads.get(canonical_slot_key)
        if not head_id:
            return None
        record = self.graph.records_by_id.get(head_id)
        if record is None or not isinstance(record.metadata, dict):
            return None
        if _normalize(record.metadata.get("memory_role", "")) != "user" or _normalize(record.metadata.get("authority", "")) != "source":
            return None
        return record

    def _support_ref_union(self, metadata: Dict[str, Any]) -> set[str]:
        return {
            *[_clean_text(item) for item in list(metadata.get("support_memory_ids", []) or []) if _clean_text(item)],
            *[_clean_text(item) for item in list(metadata.get("support_fact_refs", []) or []) if _clean_text(item)],
            *[_clean_text(item) for item in list(metadata.get("support_path_refs", []) or []) if _clean_text(item)],
        }

    def _hybrid_node_scored_hits(
        self,
        query: str,
        hits: Sequence[MemoryHit],
        *,
        top_k: int,
        public_hits: Sequence[MemoryHit] | None = None,
    ) -> Dict[str, Any]:
        scorer = self._node_scorer()
        if scorer is None:
            return {
                "hits": list(hits),
                "metadata": {
                    "retrieval_mode": "heuristic",
                    "hybrid_enabled": False,
                    "hybrid_error": self._node_scorer_error,
                },
            }
        source_hits = _learnable_graph_hits(self.graph)
        if not source_hits:
            return {
                "hits": list(hits),
                "metadata": {
                    "retrieval_mode": "heuristic",
                    "hybrid_enabled": False,
                    "hybrid_error": "no_learnable_hits",
                },
            }
        has_public_hits = any(
            _is_public_dialog_hit(hit)
            and not bool((hit.metadata or {}).get("profile_layer"))
            for hit in source_hits
        )
        has_generic_hits = any(not _is_public_dialog_hit(hit) for hit in source_hits)
        hybrid_source = "mixed_full_graph"
        if has_public_hits and not has_generic_hits:
            hybrid_source = "public_full_graph"
        elif has_generic_hits and not has_public_hits:
            hybrid_source = "generic_full_graph"
        runtime_graph = _build_runtime_graph_from_hits(query, source_hits)
        grouped_hits = dict(runtime_graph.get("grouped_hits", {}) or {})
        profile_first_payload = _profile_first_hybrid_rescue(
            self.graph,
            query,
            grouped_hits=grouped_hits,
            top_k=top_k,
        )
        profile_first_hits = list(profile_first_payload.get("hits", []) or [])
        profile_first_event_ids = list(profile_first_payload.get("event_ids", []) or [])
        profile_first_memory_ids = list(profile_first_payload.get("memory_ids", []) or [])
        candidate_event_ids = sorted(grouped_hits.keys())
        if not candidate_event_ids:
            return {
                "hits": list(hits),
                "metadata": {
                    "retrieval_mode": "heuristic",
                    "hybrid_enabled": False,
                    "hybrid_error": "no_event_candidates",
                },
            }
        question_analysis = extract_question_features(query)
        hybrid_candidate_limit = min(
            len(candidate_event_ids),
            max(
                _HYBRID_SELECTED_EVENT_FLOOR,
                int(self.candidate_event_k) * 4,
                int(self.support_path_k) * 4,
                int(top_k) * 4,
            ),
        )
        embedder_pre_recall_mode = _normalize(getattr(self, "embedder_pre_recall_mode", ""))
        embedder_pre_recall_enabled = embedder_pre_recall_mode not in _EMBEDDER_INDEX_DISABLED_MODES
        embedder_pre_recall_index_mode = self.embedder_index_recall_mode
        if embedder_pre_recall_mode not in {"1", "true", "yes", "on", "auto", "seed", "candidate", "candidates"}:
            embedder_pre_recall_index_mode = embedder_pre_recall_mode
        pre_embedder_index_payload: Dict[str, Any] = {
            "event_ids": [],
            "metadata": {
                "embedder_pre_recall_enabled": False,
                "embedder_pre_recall_mode": embedder_pre_recall_mode or "off",
                "embedder_pre_recall_index_mode": embedder_pre_recall_index_mode or "off",
                "embedder_pre_recall_event_ids": [],
            },
        }
        pre_embedder_event_ids: List[str] = []
        pre_candidate_event_ids: List[str] = []
        if embedder_pre_recall_enabled:
            pre_embedder_index_payload = _embedder_index_recall_event_ids(
                query,
                grouped_hits=grouped_hits,
                mode=embedder_pre_recall_index_mode,
                limit=self.embedder_pre_recall_k or self.embedder_index_recall_k or hybrid_candidate_limit,
                max_terms=self.write_embedder_index_max_terms,
            )
            pre_embedder_event_ids = list(pre_embedder_index_payload.get("event_ids", []) or [])
            pre_candidate_event_ids = _bounded_event_id_union(
                pre_embedder_event_ids,
                max_items=hybrid_candidate_limit,
            )
        pre_score_kwargs = {
            "graph": runtime_graph,
            "question": query,
            "question_features": question_analysis,
            "rerank_top_k": self.candidate_event_k,
            "event_rerank_mode": self.event_rerank_mode,
            "matrix_event_top_k": self.matrix_event_top_k,
            "support_path_k": self.support_path_k,
            "top_k": top_k,
        }
        if pre_candidate_event_ids:
            pre_score_kwargs["candidate_event_ids"] = pre_candidate_event_ids
        scored = _call_with_supported_kwargs(
            scorer.score_runtime,
            **pre_score_kwargs,
        )
        memory_router_decision = _memory_router_decision(
            scored,
            mode=self.memory_router_mode,
            threshold=self.memory_router_threshold,
            margin=self.memory_router_margin,
        )
        profile_first_router_suppressed = False
        if not _memory_router_allows(memory_router_decision, "profile", "resource"):
            profile_first_hits = []
            profile_first_event_ids = []
            profile_first_memory_ids = []
            profile_first_router_suppressed = True
        initial_recall_event_scores = dict(scored.get("recall_event_scores", {}) or {})
        model_recall_event_ids = _bounded_event_id_union(
            list(scored.get("recall_event_ids", []) or []),
            [
                event_id
                for event_id, _ in sorted(
                    initial_recall_event_scores.items(),
                    key=lambda item: (-float(item[1]), item[0]),
                )
            ],
            list(scored.get("rerank_candidate_event_ids", []) or []),
            max_items=max(1, len(candidate_event_ids)),
        )
        learned_recall_event_ids = list(model_recall_event_ids)
        symbolic_recall_event_ids = _symbolic_recall_event_ids(
            query,
            runtime_graph,
            grouped_hits=grouped_hits,
            limit=hybrid_candidate_limit,
        )
        symbolic_recall_event_ids = _bounded_event_id_union(
            profile_first_event_ids,
            symbolic_recall_event_ids,
            max_items=hybrid_candidate_limit,
        )
        if pre_embedder_event_ids:
            embedder_index_payload = pre_embedder_index_payload
        else:
            embedder_index_payload = _embedder_index_recall_event_ids(
                query,
                grouped_hits=grouped_hits,
                mode=self.embedder_index_recall_mode,
                limit=self.embedder_index_recall_k or hybrid_candidate_limit,
                max_terms=self.write_embedder_index_max_terms,
            )
        embedder_index_event_ids = list(embedder_index_payload.get("event_ids", []) or [])
        embedder_index_metadata = dict(embedder_index_payload.get("metadata", {}) or {})
        embedder_index_metadata.update(
            {
                "embedder_pre_recall_enabled": bool(pre_candidate_event_ids),
                "embedder_pre_recall_mode": embedder_pre_recall_mode or "off",
                "embedder_pre_recall_index_mode": embedder_pre_recall_index_mode or "off",
                "embedder_pre_recall_event_ids": list(pre_embedder_event_ids),
                "embedder_pre_recall_candidate_event_ids": list(pre_candidate_event_ids),
                "embedder_pre_recall_candidate_count": int(len(pre_candidate_event_ids)),
            }
        )
        hybrid_candidate_event_ids = _bounded_event_id_union(
            profile_first_event_ids,
            embedder_index_event_ids,
            learned_recall_event_ids,
            symbolic_recall_event_ids,
            max_items=hybrid_candidate_limit,
        )
        learned_candidate_event_ids = _bounded_event_id_union(
            learned_recall_event_ids,
            max_items=hybrid_candidate_limit,
        )
        learned_recall_event_id_set = set(learned_candidate_event_ids)
        hybrid_candidate_union_added_event_ids = [
            event_id
            for event_id in hybrid_candidate_event_ids
            if event_id not in learned_recall_event_id_set
        ]
        hybrid_candidate_union_priority_changed = list(hybrid_candidate_event_ids) != list(learned_candidate_event_ids)
        hybrid_candidate_union_rescored = False
        if hybrid_candidate_union_added_event_ids or (embedder_index_event_ids and hybrid_candidate_union_priority_changed):
            scored = _call_with_supported_kwargs(
                scorer.score_runtime,
                graph=runtime_graph,
                question=query,
                question_features=question_analysis,
                candidate_event_ids=hybrid_candidate_event_ids,
                rerank_top_k=self.candidate_event_k,
                event_rerank_mode=self.event_rerank_mode,
                matrix_event_top_k=self.matrix_event_top_k,
                support_path_k=self.support_path_k,
                top_k=top_k,
            )
            hybrid_candidate_union_rescored = True
            memory_router_decision = _memory_router_decision(
                scored,
                mode=self.memory_router_mode,
                threshold=self.memory_router_threshold,
                margin=self.memory_router_margin,
            )
        recall_event_scores = dict(scored.get("recall_event_scores", {}) or {})
        rerank_candidate_event_ids = list(scored.get("rerank_candidate_event_ids", []) or [])
        base_event_scores = dict(scored.get("base_event_scores", {}) or {})
        rerank_event_scores = dict(scored.get("rerank_event_scores", {}) or {})
        calibrated_event_scores = dict(scored.get("calibrated_event_scores", {}) or {})
        matrix_event_scores = dict(scored.get("matrix_event_scores", {}) or {})
        event_fusion_delta_scores = dict(scored.get("event_fusion_delta_scores", {}) or {})
        event_tunnel_support_scores = dict(scored.get("event_tunnel_support_scores", {}) or {})
        event_tunnel_delta_scores = dict(scored.get("event_tunnel_delta_scores", {}) or {})
        tri_maze_event_reverse_scores = dict(scored.get("tri_maze_event_reverse_scores", {}) or {})
        tri_maze_event_boundary_scores = dict(scored.get("tri_maze_event_boundary_scores", {}) or {})
        tri_maze_event_reverse_relations = dict(scored.get("tri_maze_event_reverse_relations", {}) or {})
        matrix_rerank_event_ids = list(scored.get("matrix_rerank_event_ids", []) or [])
        matrix_enabled = bool(scored.get("matrix_enabled", False))
        rerank_path_scores = dict(scored.get("rerank_path_scores", {}) or {})
        matrix_path_scores = dict(scored.get("matrix_path_scores", {}) or {})
        tri_maze_path_reverse_scores = dict(scored.get("tri_maze_path_reverse_scores", {}) or {})
        tri_maze_path_boundary_scores = dict(scored.get("tri_maze_path_boundary_scores", {}) or {})
        tri_maze_path_reverse_relations = dict(scored.get("tri_maze_path_reverse_relations", {}) or {})
        matrix_path_rerank_ids = list(scored.get("matrix_path_rerank_ids", []) or [])
        matrix_path_enabled = bool(scored.get("matrix_path_enabled", False))
        fusion_enabled = bool(scored.get("fusion_enabled", False))
        event_fusion_enabled = bool(scored.get("event_fusion_enabled", fusion_enabled))
        path_fusion_enabled = bool(scored.get("path_fusion_enabled", fusion_enabled))
        event_calibration_enabled = bool(scored.get("event_calibration_enabled", False))
        path_calibration_enabled = bool(scored.get("path_calibration_enabled", False))
        event_tunnel_enabled = bool(scored.get("event_tunnel_enabled", False))
        path_tunnel_enabled = bool(scored.get("path_tunnel_enabled", False))
        final_event_fusion_enabled = bool(scored.get("final_event_fusion_enabled", False))
        final_path_fusion_enabled = bool(scored.get("final_path_fusion_enabled", False))
        decision_fusion_enabled = bool(scored.get("decision_fusion_enabled", False))
        decision_score_source = _clean_text(scored.get("decision_score_source", ""))
        event_scores = dict(scored.get("event_scores", {}) or {})
        base_path_scores = dict(scored.get("base_path_scores", {}) or {})
        calibrated_path_scores = dict(scored.get("calibrated_path_scores", {}) or {})
        path_fusion_delta_scores = dict(scored.get("path_fusion_delta_scores", {}) or {})
        path_tunnel_support_scores = dict(scored.get("path_tunnel_support_scores", {}) or {})
        path_tunnel_delta_scores = dict(scored.get("path_tunnel_delta_scores", {}) or {})
        path_model_scores = dict(scored.get("path_model_scores", {}) or {})
        path_chain_extension_delta_scores = dict(scored.get("path_chain_extension_delta_scores", {}) or {})
        path_chain_extended_scores = dict(scored.get("path_chain_extended_scores", {}) or {})
        path_chain_extension_enabled = bool(scored.get("path_chain_extension_enabled", False))
        answer_type_scores = dict(scored.get("answer_type_scores", {}) or {})
        selected_event_ids_from_model = list(scored.get("selected_event_ids", []) or [])
        selected_path_ids_from_model = list(scored.get("selected_path_ids", []) or [])
        focused_answer_type_from_model = _clean_text(scored.get("focused_answer_type", ""))
        path_scores = dict(scored.get("path_scores", {}) or {})
        temporal_scores = dict(scored.get("temporal_scores", {}) or {})
        runtime_paths = {_clean_text(path.get("id", "")): dict(path) for path in list(runtime_graph.get("paths", []) or [])}
        answer_plan_scores_raw = dict(scored.get("answer_plan_scores", {}) or {})

        def _answer_plan_score_map(role: str) -> Dict[str, float]:
            raw_scores = answer_plan_scores_raw.get(role, {})
            if not isinstance(raw_scores, Mapping):
                return {}
            normalized: Dict[str, float] = {}
            for raw_event_id, raw_score in raw_scores.items():
                event_id = _clean_text(raw_event_id)
                if not event_id:
                    continue
                try:
                    normalized[event_id] = float(raw_score or 0.0)
                except (TypeError, ValueError):
                    normalized[event_id] = 0.0
            return normalized

        answer_plan_selected_scores = _answer_plan_score_map("selected")
        answer_plan_current_scores = _answer_plan_score_map("current")
        answer_plan_historical_scores = _answer_plan_score_map("historical")
        answer_plan_suppressed_scores = _answer_plan_score_map("suppressed")
        answer_plan_scores = {
            "selected": dict(answer_plan_selected_scores),
            "current": dict(answer_plan_current_scores),
            "historical": dict(answer_plan_historical_scores),
            "suppressed": dict(answer_plan_suppressed_scores),
        }
        try:
            answer_plan_event_selection_threshold = float(
                os.getenv("TMCRA_ANSWER_PLAN_EVENT_SELECTION_THRESHOLD", "0.50") or 0.50
            )
        except (TypeError, ValueError):
            answer_plan_event_selection_threshold = 0.50
        try:
            answer_plan_event_selection_top_k = int(os.getenv("TMCRA_ANSWER_PLAN_EVENT_SELECTION_TOP_K", "0") or 0)
        except (TypeError, ValueError):
            answer_plan_event_selection_top_k = 0
        if answer_plan_event_selection_top_k <= 0:
            answer_plan_event_selection_top_k = max(top_k, self.support_path_k * 2, _HYBRID_SELECTED_EVENT_FLOOR)
        answer_plan_available_event_ids = {
            _clean_text(event_id)
            for event_id in [
                *list(grouped_hits.keys()),
                *[_clean_text(path.get("event_id", "")) for path in runtime_paths.values()],
            ]
            if _clean_text(event_id)
        }
        answer_plan_event_rows: List[tuple[str, float, float, float, float, float]] = []
        answer_plan_event_ids = set(answer_plan_selected_scores) | set(answer_plan_current_scores) | set(answer_plan_historical_scores)
        for event_id in answer_plan_event_ids:
            if answer_plan_available_event_ids and event_id not in answer_plan_available_event_ids:
                continue
            selected_score = float(answer_plan_selected_scores.get(event_id, 0.0) or 0.0)
            current_score = float(answer_plan_current_scores.get(event_id, 0.0) or 0.0)
            historical_score = float(answer_plan_historical_scores.get(event_id, 0.0) or 0.0)
            suppressed_score = float(answer_plan_suppressed_scores.get(event_id, 0.0) or 0.0)
            support_score = max(selected_score, current_score)
            adjusted_score = support_score - max(0.0, suppressed_score - support_score) * 0.5
            if support_score < answer_plan_event_selection_threshold or suppressed_score > support_score:
                continue
            answer_plan_event_rows.append(
                (event_id, adjusted_score, support_score, selected_score, current_score, historical_score)
            )
        answer_plan_event_rows.sort(
            key=lambda row: (-float(row[1]), -float(row[2]), -float(row[3]), -float(row[4]), row[0])
        )
        answer_plan_raw_ranked_event_ids = [
            event_id for event_id, *_ in answer_plan_event_rows[: max(1, answer_plan_event_selection_top_k)]
        ]
        try:
            answer_plan_promotion_min_margin = float(
                os.getenv("TMCRA_ANSWER_PLAN_PROMOTION_MIN_MARGIN", "0.02") or 0.02
            )
        except (TypeError, ValueError):
            answer_plan_promotion_min_margin = 0.02
        answer_plan_promotion_score_margin = 0.0
        if len(answer_plan_event_rows) == 1:
            answer_plan_promotion_score_margin = float(answer_plan_event_rows[0][1])
            answer_plan_promotion_enabled = True
        elif len(answer_plan_event_rows) > 1:
            comparison_index = min(len(answer_plan_event_rows) - 1, max(1, min(answer_plan_event_selection_top_k, 5) - 1))
            answer_plan_promotion_score_margin = float(answer_plan_event_rows[0][1]) - float(answer_plan_event_rows[comparison_index][1])
            answer_plan_promotion_enabled = answer_plan_promotion_score_margin >= answer_plan_promotion_min_margin
        else:
            answer_plan_promotion_enabled = False
        answer_plan_ranked_event_ids = list(answer_plan_raw_ranked_event_ids if answer_plan_promotion_enabled else [])
        answer_plan_selected_event_ids = [
            event_id
            for event_id in answer_plan_ranked_event_ids
            if float(answer_plan_selected_scores.get(event_id, 0.0) or 0.0)
            >= answer_plan_event_selection_threshold
        ]
        answer_plan_current_event_ids = [
            event_id
            for event_id in answer_plan_ranked_event_ids
            if float(answer_plan_current_scores.get(event_id, 0.0) or 0.0)
            >= answer_plan_event_selection_threshold
        ]
        answer_plan_support_scores = {
            event_id: round(float(support_score), 6)
            for event_id, _, support_score, *_ in answer_plan_event_rows
        }
        answer_plan_adjusted_scores = {
            event_id: round(float(adjusted_score), 6)
            for event_id, adjusted_score, *_ in answer_plan_event_rows
        }
        answer_plan_rank_lookup = {
            event_id: rank for rank, event_id in enumerate(answer_plan_ranked_event_ids, start=1)
        }
        answer_plan_ranked_event_id_set = set(answer_plan_ranked_event_ids)

        def _answer_plan_hit_metadata(event_id: str) -> Dict[str, Any]:
            clean_event_id = _clean_text(event_id)
            selected_score = float(answer_plan_selected_scores.get(clean_event_id, 0.0) or 0.0)
            current_score = float(answer_plan_current_scores.get(clean_event_id, 0.0) or 0.0)
            historical_score = float(answer_plan_historical_scores.get(clean_event_id, 0.0) or 0.0)
            suppressed_score = float(answer_plan_suppressed_scores.get(clean_event_id, 0.0) or 0.0)
            support_score = max(selected_score, current_score)
            return {
                "answer_plan_score": round(float(support_score), 6),
                "answer_plan_selected_score": round(float(selected_score), 6),
                "answer_plan_current_score": round(float(current_score), 6),
                "answer_plan_historical_score": round(float(historical_score), 6),
                "answer_plan_suppressed_score": round(float(suppressed_score), 6),
                "answer_plan_adjusted_score": round(float(answer_plan_adjusted_scores.get(clean_event_id, 0.0)), 6),
                "answer_plan_selected": bool(clean_event_id in answer_plan_ranked_event_id_set),
                "answer_plan_rank": int(answer_plan_rank_lookup.get(clean_event_id, 0) or 0),
            }

        embedder_fusion_mode = _normalize(self.embedder_fusion_mode)
        embedder_fusion_enabled = (
            embedder_fusion_mode not in _EMBEDDER_INDEX_DISABLED_MODES
            and bool(embedder_index_event_ids)
            and bool(self.embedder_fusion_top_k > 0)
            and bool(self.embedder_fusion_weight > 0.0)
        )
        embedder_fusion_applied_event_scores: Dict[str, float] = {}
        embedder_fusion_boosts: Dict[str, float] = {}
        embedder_event_scores = dict(embedder_index_metadata.get("embedder_index_event_scores", {}) or {})
        if embedder_fusion_enabled and embedder_event_scores:
            ranked_embedder_events = [
                (event_id, float(score))
                for event_id, score in sorted(
                    embedder_event_scores.items(),
                    key=lambda item: (-float(item[1]), item[0]),
                )
                if _clean_text(event_id)
            ][: max(1, int(self.embedder_fusion_top_k))]
            for rank, (event_id, embedder_score) in enumerate(ranked_embedder_events, start=1):
                if embedder_score < float(self.embedder_fusion_score_floor):
                    continue
                rank_bonus = max(0.0, 0.08 - (rank - 1) * 0.006)
                boost = min(
                    float(self.embedder_fusion_max_boost),
                    (float(self.embedder_fusion_weight) * embedder_score) + rank_bonus,
                )
                current_score = max(
                    float(recall_event_scores.get(event_id, 0.0) or 0.0),
                    float(event_scores.get(event_id, 0.0) or 0.0),
                    float(base_event_scores.get(event_id, 0.0) or 0.0),
                    float(rerank_event_scores.get(event_id, 0.0) or 0.0),
                    float(calibrated_event_scores.get(event_id, 0.0) or 0.0),
                )
                fused_score = current_score + boost
                recall_event_scores[event_id] = max(float(recall_event_scores.get(event_id, 0.0) or 0.0), fused_score)
                event_scores[event_id] = max(float(event_scores.get(event_id, 0.0) or 0.0), fused_score)
                base_event_scores[event_id] = max(float(base_event_scores.get(event_id, 0.0) or 0.0), fused_score)
                rerank_event_scores[event_id] = max(float(rerank_event_scores.get(event_id, 0.0) or 0.0), fused_score)
                calibrated_event_scores[event_id] = max(float(calibrated_event_scores.get(event_id, 0.0) or 0.0), fused_score)
                embedder_fusion_applied_event_scores[event_id] = round(fused_score, 6)
                embedder_fusion_boosts[event_id] = round(boost, 6)
            if embedder_fusion_applied_event_scores:
                decision_score_source = f"{decision_score_source or 'learned_decision_fusion'}+embedder_fusion"
        recall_event_ids = [
            event_id
            for event_id, _ in sorted(recall_event_scores.items(), key=lambda item: (-float(item[1]), item[0]))
        ][: self.candidate_event_k]
        if profile_first_event_ids:
            profile_first_score_by_event = {
                _clean_text((hit.metadata or {}).get("profile_first_hybrid_event_id", "")) or _runtime_event_key(hit): float(hit.score)
                for hit in profile_first_hits
            }
            for rank, event_id in enumerate(profile_first_event_ids, start=1):
                if not event_id:
                    continue
                floor = max(0.92, float(profile_first_score_by_event.get(event_id, 0.0))) + max(0.0, 0.04 - (rank * 0.005))
                recall_event_scores[event_id] = max(float(recall_event_scores.get(event_id, 0.0)), floor)
                event_scores[event_id] = max(float(event_scores.get(event_id, 0.0)), floor)
                base_event_scores[event_id] = max(float(base_event_scores.get(event_id, 0.0)), floor)
                rerank_event_scores[event_id] = max(float(rerank_event_scores.get(event_id, 0.0)), floor)
                calibrated_event_scores[event_id] = max(float(calibrated_event_scores.get(event_id, 0.0)), floor)
            for path_id, path in runtime_paths.items():
                event_id = _clean_text(path.get("event_id", ""))
                path_type = _clean_text(path.get("type", ""))
                if event_id not in set(profile_first_event_ids):
                    continue
                if path_type not in {"speaker_event_profile", "speaker_event_source_turn", "speaker_event_status"}:
                    continue
                floor = max(0.90, float(event_scores.get(event_id, 0.0)))
                path_scores[path_id] = max(float(path_scores.get(path_id, 0.0)), floor)
                base_path_scores[path_id] = max(float(base_path_scores.get(path_id, 0.0)), floor)
                calibrated_path_scores[path_id] = max(float(calibrated_path_scores.get(path_id, 0.0)), floor)
                path_model_scores[path_id] = max(float(path_model_scores.get(path_id, 0.0)), floor)
        selection_consistency_repaired = False
        selection_consistency_reason = ""
        model_focused_answer_type = _clean_text(focused_answer_type_from_model)
        embedder_fusion_selected_event_ids: List[str] = []
        embedder_fusion_selected_path_ids: List[str] = []
        if decision_fusion_enabled and (selected_path_ids_from_model or selected_event_ids_from_model):
            base_selected_path_limit = max(1, min(max(1, self.support_path_k), max(1, top_k)))
            selected_path_ids = [
                path_id
                for path_id in selected_path_ids_from_model
                if _clean_text(path_id) in runtime_paths
            ]
            if not selected_path_ids:
                selected_path_ids = [
                    path_id
                    for path_id, _ in sorted(
                        ((path_id, float(score or 0.0)) for path_id, score in path_scores.items()),
                        key=lambda item: (-float(item[1]), item[0]),
                    )
                ][:base_selected_path_limit]
            focused_answer_type = _reconciled_focused_answer_type(question_analysis, answer_type_scores, focused_answer_type_from_model)
            tunnel_rescue_path_ids: List[str] = []
            tunnel_rescue_pre_filter_path_ids: List[str] = []
            path_utility_direct_support_path_ids: List[str] = []
            path_utility_contrast_support_path_ids: List[str] = []
            path_utility_latent_context_path_ids: List[str] = []
            path_utility_drift_noise_path_ids: List[str] = []
            path_utility_roles: Dict[str, str] = {}
            path_utility_reasons: Dict[str, str] = {}
            path_utility_scores: Dict[str, float] = {}
            path_utility_overlap_tokens: Dict[str, List[str]] = {}
            path_utility_anchor_event_ids: List[str] = []
            path_utility_anchor_subject_signatures: List[str] = []
            tunnel_rescue_score_threshold = float(self.path_tunnel_rescue_score_floor)
            tunnel_rescue_candidate_count = 0
            tunnel_rescue_filtered_count = 0
            if (
                self.path_tunnel_rescue_k > 0
                and path_tunnel_support_scores
                and _memory_router_allows(memory_router_decision, "path_tunnel", "topic_tunnel")
            ):
                selected_path_id_set = set(selected_path_ids)
                candidate_scores = [
                    float(score or 0.0)
                    for path_id, score in path_tunnel_support_scores.items()
                    if _clean_text(path_id) in runtime_paths
                ]
                tunnel_rescue_candidate_count = len(candidate_scores)
                if candidate_scores and self.path_tunnel_rescue_min_score_margin > 0.0:
                    sorted_scores = sorted(candidate_scores)
                    median_score = sorted_scores[len(sorted_scores) // 2]
                    tunnel_rescue_score_threshold = max(
                        tunnel_rescue_score_threshold,
                        median_score + float(self.path_tunnel_rescue_min_score_margin),
                    )
                event_turn_indices = [
                    _runtime_event_turn_index_from_id(_clean_text(path.get("event_id", "")))
                    for path in runtime_paths.values()
                ]
                current_turn_index = max(
                    [int(getattr(self.graph, "turn_index", 0) or 0), *[turn for turn in event_turn_indices if turn > 0]],
                    default=0,
                )
                ranked_tunnel_path_ids = []
                for path_id, score in sorted(
                    (
                        (path_id, float(score or 0.0))
                        for path_id, score in path_tunnel_support_scores.items()
                        if _clean_text(path_id) in runtime_paths
                    ),
                    key=lambda item: (-float(item[1]), item[0]),
                ):
                    if path_id in selected_path_id_set or score < tunnel_rescue_score_threshold:
                        continue
                    path_event_turn = _runtime_event_turn_index_from_id(
                        _clean_text(runtime_paths.get(path_id, {}).get("event_id", ""))
                    )
                    path_age = max(0, current_turn_index - path_event_turn) if current_turn_index and path_event_turn else 0
                    if self.path_tunnel_rescue_min_age > 0 and path_age < self.path_tunnel_rescue_min_age:
                        continue
                    ranked_tunnel_path_ids.append(path_id)
                tunnel_rescue_filtered_count = len(ranked_tunnel_path_ids)
                tunnel_rescue_pre_filter_path_ids = ranked_tunnel_path_ids[: max(self.path_tunnel_rescue_k, self.path_tunnel_rescue_k * 4)]
                utility_gate = _path_utility_gate(
                    tunnel_rescue_pre_filter_path_ids,
                    query=query,
                    runtime_graph=runtime_graph,
                    runtime_paths=runtime_paths,
                    grouped_hits=grouped_hits,
                    selected_path_ids=selected_path_ids,
                    selected_event_ids_from_model=selected_event_ids_from_model,
                    path_scores=path_scores,
                    path_tunnel_support_scores=path_tunnel_support_scores,
                    question_analysis=question_analysis,
                    focused_answer_type=focused_answer_type,
                    score_threshold=tunnel_rescue_score_threshold,
                    limit=self.path_tunnel_rescue_k,
                )
                tunnel_rescue_path_ids = list(utility_gate.get("injected_path_ids", []) or [])
                path_utility_direct_support_path_ids = list(utility_gate.get("direct_support_path_ids", []) or [])
                path_utility_contrast_support_path_ids = list(utility_gate.get("contrast_support_path_ids", []) or [])
                path_utility_latent_context_path_ids = list(utility_gate.get("latent_context_path_ids", []) or [])
                path_utility_drift_noise_path_ids = list(utility_gate.get("drift_noise_path_ids", []) or [])
                path_utility_roles = dict(utility_gate.get("roles", {}) or {})
                path_utility_reasons = dict(utility_gate.get("reasons", {}) or {})
                path_utility_scores = dict(utility_gate.get("scores", {}) or {})
                path_utility_overlap_tokens = dict(utility_gate.get("overlap_tokens", {}) or {})
                path_utility_anchor_event_ids = list(utility_gate.get("anchor_event_ids", []) or [])
                path_utility_anchor_subject_signatures = list(utility_gate.get("anchor_subject_signatures", []) or [])
                if tunnel_rescue_path_ids:
                    selected_path_ids = _dedupe([*selected_path_ids, *tunnel_rescue_path_ids])
            effective_selected_path_limit = base_selected_path_limit + len(tunnel_rescue_path_ids)
            selected_event_ids = _dedupe(
                [
                    *[
                        _clean_text(runtime_paths.get(path_id, {}).get("event_id", ""))
                        for path_id in selected_path_ids
                        if _clean_text(runtime_paths.get(path_id, {}).get("event_id", ""))
                    ],
                    *[
                        _clean_text(event_id)
                        for event_id in selected_event_ids_from_model
                        if _clean_text(event_id)
                    ],
                ]
            )
            if answer_plan_ranked_event_ids:
                selected_event_ids = _dedupe([*selected_event_ids, *answer_plan_ranked_event_ids])
            if not selected_event_ids:
                selected_event_ids = [
                    event_id
                    for event_id, _ in sorted(
                        ((event_id, float(score or 0.0)) for event_id, score in event_scores.items()),
                        key=lambda item: (-float(item[1]), item[0]),
                    )
                ][: max(1, min(self.candidate_event_k, max(top_k, _HYBRID_SELECTED_EVENT_FLOOR)))]
            repaired_path_ids, selection_consistency_repaired, selection_consistency_reason = _repair_selected_paths_for_focus(
                selected_path_ids,
                runtime_paths=runtime_paths,
                selected_event_ids=selected_event_ids,
                path_scores=path_scores,
                event_scores=event_scores,
                temporal_scores=temporal_scores,
                question_analysis=question_analysis,
                answer_type_scores=answer_type_scores,
                focused_answer_type=focused_answer_type,
                limit=max(1, effective_selected_path_limit),
            )
            if selection_consistency_repaired:
                selected_path_ids = repaired_path_ids
                selected_event_ids = _dedupe(
                    [
                        *[
                            _clean_text(runtime_paths.get(path_id, {}).get("event_id", ""))
                            for path_id in selected_path_ids
                            if _clean_text(runtime_paths.get(path_id, {}).get("event_id", ""))
                        ],
                        *[
                            _clean_text(event_id)
                            for event_id in selected_event_ids_from_model
                            if _clean_text(event_id)
                        ],
                    ]
                )
                if answer_plan_ranked_event_ids:
                    selected_event_ids = _dedupe([*selected_event_ids, *answer_plan_ranked_event_ids])
            if profile_first_event_ids:
                selected_event_ids = _dedupe([*profile_first_event_ids, *selected_event_ids])
            if embedder_fusion_applied_event_scores and self.embedder_fusion_select_k > 0:
                ranked_fusion_event_ids = [
                    event_id
                    for event_id in embedder_index_event_ids
                    if event_id in embedder_fusion_applied_event_scores
                ][: max(1, int(self.embedder_fusion_select_k))]
                selected_event_ids = _dedupe([*ranked_fusion_event_ids, *selected_event_ids])
                selected_path_id_set = set(selected_path_ids)
                selected_embedder_path_ids: List[str] = []
                for event_id in ranked_fusion_event_ids:
                    candidate_paths = [
                        (path_id, path)
                        for path_id, path in runtime_paths.items()
                        if _clean_text(path.get("event_id", "")) == event_id
                    ]
                    candidate_paths.sort(
                        key=lambda item: (
                            int(_clean_text(item[1].get("type", "")) in {"speaker_event_source_turn", "speaker_event_profile", "speaker_event_status", "speaker_event_time"}),
                            float(path_scores.get(item[0], 0.0) or 0.0),
                            item[0],
                        ),
                        reverse=True,
                    )
                    for path_id, _ in candidate_paths:
                        if path_id in selected_path_id_set:
                            continue
                        selected_embedder_path_ids.append(path_id)
                        selected_path_id_set.add(path_id)
                        embedder_fusion_selected_path_ids.append(path_id)
                        break
                if selected_embedder_path_ids:
                    selected_path_ids = _dedupe([*selected_embedder_path_ids, *selected_path_ids])
                embedder_fusion_selected_event_ids = list(ranked_fusion_event_ids)
            if answer_plan_ranked_event_ids:
                selected_event_ids = _dedupe([*selected_event_ids, *answer_plan_ranked_event_ids])
            final_hits: List[MemoryHit] = []
            seen_memory_ids = set()
            selected_event_id_set = set(selected_event_ids)
            for path_id in selected_path_ids:
                path = runtime_paths.get(path_id, {})
                event_id = _clean_text(path.get("event_id", ""))
                if event_id not in selected_event_id_set:
                    continue
                path_type = _clean_text(path.get("type", ""))
                support_node_id = _path_support_node_id(path)
                support_hit = _support_hit_for_path(path_type, grouped_hits.get(event_id, []))
                event_hit = _representative_event_hit(
                    [*grouped_hits.get(event_id, []), *_event_record_hits_from_graph(self.graph, event_id)],
                    query=query,
                )
                decision_score = round(float(path_scores.get(path_id, 0.0)), 6)
                candidate_hits: List[tuple[MemoryHit | None, float]] = [(support_hit, decision_score)]
                if event_hit is not None and (support_hit is None or event_hit.memory_id != support_hit.memory_id):
                    candidate_hits.append((event_hit, max(0.0, decision_score - 0.0001)))
                for raw_hit, hit_score in candidate_hits:
                    if raw_hit is None or raw_hit.memory_id in seen_memory_ids:
                        continue
                    seen_memory_ids.add(raw_hit.memory_id)
                    recall_score = float(recall_event_scores.get(event_id, event_scores.get(event_id, 0.0)))
                    event_score = float(event_scores.get(event_id, raw_hit.score))
                    temporal_score = float(temporal_scores.get(support_node_id, 0.0))
                    metadata = dict(raw_hit.metadata or {})
                    metadata.update(
                        {
                            "event_id": event_id,
                            **_answer_plan_hit_metadata(event_id),
                            "path_id": path_id,
                            "base_event_score": round(float(base_event_scores.get(event_id, event_score)), 6),
                            "rerank_event_score": round(float(rerank_event_scores.get(event_id, event_score)), 6),
                            "calibrated_event_score": round(float(calibrated_event_scores.get(event_id, event_score)), 6),
                            "matrix_event_score": round(float(matrix_event_scores.get(event_id, 0.0)), 6),
                            "event_fusion_delta_score": round(float(event_fusion_delta_scores.get(event_id, 0.0)), 6),
                            "event_tunnel_support_score": round(float(event_tunnel_support_scores.get(event_id, 0.0)), 6),
                            "event_tunnel_delta_score": round(float(event_tunnel_delta_scores.get(event_id, 0.0)), 6),
                            "tri_maze_event_reverse_score": round(float(tri_maze_event_reverse_scores.get(event_id, 0.0)), 6),
                            "tri_maze_event_boundary_score": round(float(tri_maze_event_boundary_scores.get(event_id, 0.0)), 6),
                            "tri_maze_event_reverse_relation": round(float(tri_maze_event_reverse_relations.get(event_id, 0.0)), 6),
                            "matrix_enabled": matrix_enabled,
                            "event_calibration_enabled": event_calibration_enabled,
                            "path_calibration_enabled": path_calibration_enabled,
                            "event_tunnel_enabled": event_tunnel_enabled,
                            "path_tunnel_enabled": path_tunnel_enabled,
                            "final_event_fusion_enabled": final_event_fusion_enabled,
                            "final_path_fusion_enabled": final_path_fusion_enabled,
                            "decision_fusion_enabled": True,
                            "event_fusion_enabled": event_fusion_enabled,
                            "path_fusion_enabled": path_fusion_enabled,
                            "event_score": round(event_score, 6),
                            "recall_score": round(recall_score, 6),
                            "path_score": round(float(path_scores.get(path_id, hit_score)), 6),
                            "base_path_score": round(float(base_path_scores.get(path_id, path_scores.get(path_id, hit_score))), 6),
                            "calibrated_path_score": round(float(calibrated_path_scores.get(path_id, path_scores.get(path_id, hit_score))), 6),
                            "path_fusion_delta_score": round(float(path_fusion_delta_scores.get(path_id, 0.0)), 6),
                            "path_tunnel_support_score": round(float(path_tunnel_support_scores.get(path_id, 0.0)), 6),
                            "path_tunnel_delta_score": round(float(path_tunnel_delta_scores.get(path_id, 0.0)), 6),
                            "path_model_score": round(float(path_model_scores.get(path_id, path_scores.get(path_id, hit_score))), 6),
                            "path_chain_extension_enabled": path_chain_extension_enabled,
                            "path_chain_extension_delta_score": round(float(path_chain_extension_delta_scores.get(path_id, 0.0)), 6),
                            "path_chain_extended_score": round(float(path_chain_extended_scores.get(path_id, path_scores.get(path_id, hit_score))), 6),
                            "tri_maze_path_reverse_score": round(float(tri_maze_path_reverse_scores.get(path_id, 0.0)), 6),
                            "tri_maze_path_boundary_score": round(float(tri_maze_path_boundary_scores.get(path_id, 0.0)), 6),
                            "tri_maze_path_reverse_relation": round(float(tri_maze_path_reverse_relations.get(path_id, 0.0)), 6),
                            "effective_path_score": round(float(path_scores.get(path_id, hit_score)), 6),
                            "temporal_score": round(temporal_score, 6),
                            "raw_public_score": round(float(raw_hit.score), 6),
                            "hybrid_score": round(hit_score, 6),
                            "hybrid_score_source": decision_score_source or "learned_decision_fusion",
                            "evidence_snippet_role": "selected_path_support" if raw_hit is support_hit else "selected_path_event",
                            "selected_event_ids": list(selected_event_ids),
                            "selected_path_ids": list(selected_path_ids),
                            "path_tunnel_rescue_enabled": bool(self.path_tunnel_rescue_k > 0),
                            "path_tunnel_rescue_k": int(self.path_tunnel_rescue_k),
                            "path_tunnel_rescue_score_floor": round(float(self.path_tunnel_rescue_score_floor), 6),
                            "path_tunnel_rescue_min_age": int(self.path_tunnel_rescue_min_age),
                            "path_tunnel_rescue_min_score_margin": round(float(self.path_tunnel_rescue_min_score_margin), 6),
                            "path_tunnel_rescue_score_threshold": round(float(tunnel_rescue_score_threshold), 6),
                            "path_tunnel_rescue_candidate_count": int(tunnel_rescue_candidate_count),
                            "path_tunnel_rescue_filtered_count": int(tunnel_rescue_filtered_count),
                            "path_tunnel_rescue_path_ids": list(tunnel_rescue_path_ids),
                            "path_utility_gate_enabled": bool(self.path_tunnel_rescue_k > 0),
                            "path_utility_pre_filter_path_ids": list(tunnel_rescue_pre_filter_path_ids),
                            "path_utility_injected_path_ids": list(tunnel_rescue_path_ids),
                            "path_utility_direct_support_path_ids": list(path_utility_direct_support_path_ids),
                            "path_utility_contrast_support_path_ids": list(path_utility_contrast_support_path_ids),
                            "path_utility_latent_context_path_ids": list(path_utility_latent_context_path_ids),
                            "path_utility_drift_noise_path_ids": list(path_utility_drift_noise_path_ids),
                            "path_utility_roles": dict(path_utility_roles),
                            "path_utility_reasons": dict(path_utility_reasons),
                            "path_utility_scores": dict(path_utility_scores),
                            "path_utility_overlap_tokens": dict(path_utility_overlap_tokens),
                            "path_utility_anchor_event_ids": list(path_utility_anchor_event_ids),
                            "path_utility_anchor_subject_signatures": list(path_utility_anchor_subject_signatures),
                            "tunnel_recall_pre_filter_count": int(len(tunnel_rescue_pre_filter_path_ids)),
                            "tunnel_usable_post_filter_count": int(len(tunnel_rescue_path_ids)),
                            "model_focused_answer_type": model_focused_answer_type,
                            "selection_consistency_repaired": bool(selection_consistency_repaired),
                            "selection_consistency_reason": selection_consistency_reason,
                        }
                    )
                    final_hits.append(
                        MemoryHit(
                            memory_id=raw_hit.memory_id,
                            category=raw_hit.category,
                            value=raw_hit.value,
                            relation=raw_hit.relation,
                            anchors=list(raw_hit.anchors),
                            score=round(hit_score, 6),
                            source_kind=raw_hit.source_kind,
                            slot_key=raw_hit.slot_key,
                            state=raw_hit.state,
                            turn_index=int(raw_hit.turn_index),
                            metadata=metadata,
                        )
                    )
            for event_rank, event_id in enumerate(selected_event_ids, start=1):
                event_hit = _representative_event_hit(
                    [*grouped_hits.get(event_id, []), *_event_record_hits_from_graph(self.graph, event_id)],
                    query=query,
                )
                if event_hit is None or event_hit.memory_id in seen_memory_ids:
                    continue
                seen_memory_ids.add(event_hit.memory_id)
                recall_score = float(recall_event_scores.get(event_id, event_hit.score))
                event_score = float(event_scores.get(event_id, event_hit.score))
                metadata = dict(event_hit.metadata or {})
                metadata.update(
                    {
                        "event_id": event_id,
                        **_answer_plan_hit_metadata(event_id),
                        "path_id": "",
                        "base_event_score": round(float(base_event_scores.get(event_id, event_score)), 6),
                        "rerank_event_score": round(float(rerank_event_scores.get(event_id, event_score)), 6),
                        "calibrated_event_score": round(float(calibrated_event_scores.get(event_id, event_score)), 6),
                        "matrix_event_score": round(float(matrix_event_scores.get(event_id, 0.0)), 6),
                        "event_fusion_delta_score": round(float(event_fusion_delta_scores.get(event_id, 0.0)), 6),
                        "event_tunnel_support_score": round(float(event_tunnel_support_scores.get(event_id, 0.0)), 6),
                        "event_tunnel_delta_score": round(float(event_tunnel_delta_scores.get(event_id, 0.0)), 6),
                        "tri_maze_event_reverse_score": round(float(tri_maze_event_reverse_scores.get(event_id, 0.0)), 6),
                        "tri_maze_event_boundary_score": round(float(tri_maze_event_boundary_scores.get(event_id, 0.0)), 6),
                        "tri_maze_event_reverse_relation": round(float(tri_maze_event_reverse_relations.get(event_id, 0.0)), 6),
                        "matrix_enabled": matrix_enabled,
                        "event_calibration_enabled": event_calibration_enabled,
                        "path_calibration_enabled": path_calibration_enabled,
                        "event_tunnel_enabled": event_tunnel_enabled,
                        "path_tunnel_enabled": path_tunnel_enabled,
                        "final_event_fusion_enabled": final_event_fusion_enabled,
                        "final_path_fusion_enabled": final_path_fusion_enabled,
                        "decision_fusion_enabled": True,
                        "event_fusion_enabled": event_fusion_enabled,
                        "path_fusion_enabled": path_fusion_enabled,
                        "event_score": round(event_score, 6),
                        "recall_score": round(recall_score, 6),
                        "path_score": 0.0,
                        "base_path_score": 0.0,
                        "calibrated_path_score": 0.0,
                        "path_fusion_delta_score": 0.0,
                        "path_tunnel_support_score": 0.0,
                        "path_tunnel_delta_score": 0.0,
                        "path_model_score": 0.0,
                        "path_chain_extension_enabled": path_chain_extension_enabled,
                        "path_chain_extension_delta_score": 0.0,
                        "path_chain_extended_score": 0.0,
                        "effective_path_score": 0.0,
                        "temporal_score": 0.0,
                        "raw_public_score": round(float(event_hit.score), 6),
                        "hybrid_score": round(event_score, 6),
                        "hybrid_score_source": decision_score_source or "learned_final_event_fusion",
                        "evidence_snippet_role": "selected_event_representative",
                        "selected_event_rank": int(event_rank),
                        "selected_event_ids": list(selected_event_ids),
                        "selected_path_ids": list(selected_path_ids),
                        "path_tunnel_rescue_enabled": bool(self.path_tunnel_rescue_k > 0),
                        "path_tunnel_rescue_k": int(self.path_tunnel_rescue_k),
                        "path_tunnel_rescue_score_floor": round(float(self.path_tunnel_rescue_score_floor), 6),
                        "path_tunnel_rescue_min_age": int(self.path_tunnel_rescue_min_age),
                        "path_tunnel_rescue_min_score_margin": round(float(self.path_tunnel_rescue_min_score_margin), 6),
                        "path_tunnel_rescue_score_threshold": round(float(tunnel_rescue_score_threshold), 6),
                        "path_tunnel_rescue_candidate_count": int(tunnel_rescue_candidate_count),
                        "path_tunnel_rescue_filtered_count": int(tunnel_rescue_filtered_count),
                        "path_tunnel_rescue_path_ids": list(tunnel_rescue_path_ids),
                        "path_utility_gate_enabled": bool(self.path_tunnel_rescue_k > 0),
                        "path_utility_pre_filter_path_ids": list(tunnel_rescue_pre_filter_path_ids),
                        "path_utility_injected_path_ids": list(tunnel_rescue_path_ids),
                        "path_utility_direct_support_path_ids": list(path_utility_direct_support_path_ids),
                        "path_utility_contrast_support_path_ids": list(path_utility_contrast_support_path_ids),
                        "path_utility_latent_context_path_ids": list(path_utility_latent_context_path_ids),
                        "path_utility_drift_noise_path_ids": list(path_utility_drift_noise_path_ids),
                        "path_utility_roles": dict(path_utility_roles),
                        "path_utility_reasons": dict(path_utility_reasons),
                        "path_utility_scores": dict(path_utility_scores),
                        "path_utility_overlap_tokens": dict(path_utility_overlap_tokens),
                        "path_utility_anchor_event_ids": list(path_utility_anchor_event_ids),
                        "path_utility_anchor_subject_signatures": list(path_utility_anchor_subject_signatures),
                        "tunnel_recall_pre_filter_count": int(len(tunnel_rescue_pre_filter_path_ids)),
                        "tunnel_usable_post_filter_count": int(len(tunnel_rescue_path_ids)),
                        "model_focused_answer_type": model_focused_answer_type,
                        "selection_consistency_repaired": bool(selection_consistency_repaired),
                        "selection_consistency_reason": selection_consistency_reason,
                    }
                )
                final_hits.append(
                    MemoryHit(
                        memory_id=event_hit.memory_id,
                        category=event_hit.category,
                        value=event_hit.value,
                        relation=event_hit.relation,
                        anchors=list(event_hit.anchors),
                        score=round(event_score, 6),
                        source_kind=event_hit.source_kind,
                        slot_key=event_hit.slot_key,
                        state=event_hit.state,
                        turn_index=int(event_hit.turn_index),
                        metadata=metadata,
                    )
                )
            if not final_hits:
                final_hits = list(hits)
            if profile_first_hits:
                final_hits = _inject_profile_first_hits(
                    final_hits,
                    profile_first_hits,
                    selected_event_ids=selected_event_ids,
                    selected_path_ids=selected_path_ids,
                )
            final_hits = _coverage_preserving_final_hits(final_hits, selected_event_ids=selected_event_ids, top_k=top_k)
            final_hit_event_ids = _event_ids_from_hits(final_hits)
            final_missing_selected_event_ids = [event_id for event_id in selected_event_ids if event_id not in set(final_hit_event_ids)]
            return {
                "hits": final_hits,
                "metadata": {
                    "retrieval_mode": "hybrid_node_scored",
                    "hybrid_enabled": True,
                    "hybrid_source": hybrid_source,
                    **memory_router_decision,
                    "profile_first_router_suppressed": bool(profile_first_router_suppressed),
                    "recall_event_ids": list(recall_event_ids),
                    "learned_recall_event_ids": list(learned_recall_event_ids),
                    "model_recall_event_ids": list(model_recall_event_ids),
                    "symbolic_recall_event_ids": list(symbolic_recall_event_ids),
                    "embedder_index_recall_event_ids": list(embedder_index_event_ids),
                    **embedder_index_metadata,
                    "embedder_fusion_mode": embedder_fusion_mode or "off",
                    "embedder_fusion_enabled": bool(embedder_fusion_enabled),
                    "embedder_fusion_weight": round(float(self.embedder_fusion_weight), 6),
                    "embedder_fusion_score_floor": round(float(self.embedder_fusion_score_floor), 6),
                    "embedder_fusion_top_k": int(self.embedder_fusion_top_k),
                    "embedder_fusion_select_k": int(self.embedder_fusion_select_k),
                    "embedder_fusion_max_boost": round(float(self.embedder_fusion_max_boost), 6),
                    "embedder_fusion_event_scores": dict(embedder_fusion_applied_event_scores),
                    "embedder_fusion_boosts": dict(embedder_fusion_boosts),
                    "embedder_fusion_selected_event_ids": list(embedder_fusion_selected_event_ids),
                    "embedder_fusion_selected_path_ids": list(embedder_fusion_selected_path_ids),
                    "profile_first_hybrid_enabled": bool(profile_first_event_ids),
                    "profile_first_event_ids": list(profile_first_event_ids),
                    "profile_first_memory_ids": list(profile_first_memory_ids),
                    "hybrid_candidate_event_ids": list(hybrid_candidate_event_ids),
                    "hybrid_candidate_union_enabled": True,
                    "hybrid_candidate_union_rescored": bool(hybrid_candidate_union_rescored),
                    "hybrid_candidate_union_added_event_ids": list(hybrid_candidate_union_added_event_ids),
                    "hybrid_candidate_union_priority_changed": bool(hybrid_candidate_union_priority_changed),
                    "rerank_candidate_event_ids": list(rerank_candidate_event_ids),
                    "base_event_scores": dict(base_event_scores),
                    "rerank_event_scores": dict(rerank_event_scores),
                    "calibrated_event_scores": dict(calibrated_event_scores),
                    "matrix_event_scores": dict(matrix_event_scores),
                    "event_fusion_delta_scores": dict(event_fusion_delta_scores),
                    "event_tunnel_support_scores": dict(event_tunnel_support_scores),
                    "event_tunnel_delta_scores": dict(event_tunnel_delta_scores),
                    "tri_maze_event_reverse_scores": dict(tri_maze_event_reverse_scores),
                    "tri_maze_event_boundary_scores": dict(tri_maze_event_boundary_scores),
                    "tri_maze_event_reverse_relations": dict(tri_maze_event_reverse_relations),
                    "matrix_rerank_event_ids": list(matrix_rerank_event_ids),
                    "matrix_enabled": matrix_enabled,
                    "rerank_path_scores": dict(rerank_path_scores),
                    "matrix_path_scores": dict(matrix_path_scores),
                    "tri_maze_path_reverse_scores": dict(tri_maze_path_reverse_scores),
                    "tri_maze_path_boundary_scores": dict(tri_maze_path_boundary_scores),
                    "tri_maze_path_reverse_relations": dict(tri_maze_path_reverse_relations),
                    "matrix_path_rerank_ids": list(matrix_path_rerank_ids),
                    "matrix_path_enabled": matrix_path_enabled,
                    "fusion_enabled": fusion_enabled,
                    "event_calibration_enabled": event_calibration_enabled,
                    "path_calibration_enabled": path_calibration_enabled,
                    "event_tunnel_enabled": event_tunnel_enabled,
                    "path_tunnel_enabled": path_tunnel_enabled,
                    "final_event_fusion_enabled": final_event_fusion_enabled,
                    "final_path_fusion_enabled": final_path_fusion_enabled,
                    "decision_fusion_enabled": True,
                    "decision_score_source": decision_score_source or "learned_decision_fusion",
                    "event_fusion_enabled": event_fusion_enabled,
                    "path_fusion_enabled": path_fusion_enabled,
                    "selected_event_ids": list(selected_event_ids),
                    "path_rescue_event_ids": [],
                    "selected_path_ids": list(selected_path_ids),
                    "path_tunnel_rescue_enabled": bool(self.path_tunnel_rescue_k > 0),
                    "path_tunnel_rescue_k": int(self.path_tunnel_rescue_k),
                    "path_tunnel_rescue_score_floor": round(float(self.path_tunnel_rescue_score_floor), 6),
                    "path_tunnel_rescue_min_age": int(self.path_tunnel_rescue_min_age),
                    "path_tunnel_rescue_min_score_margin": round(float(self.path_tunnel_rescue_min_score_margin), 6),
                    "path_tunnel_rescue_score_threshold": round(float(tunnel_rescue_score_threshold), 6),
                    "path_tunnel_rescue_candidate_count": int(tunnel_rescue_candidate_count),
                    "path_tunnel_rescue_filtered_count": int(tunnel_rescue_filtered_count),
                    "path_tunnel_rescue_path_ids": list(tunnel_rescue_path_ids),
                    "path_utility_gate_enabled": bool(self.path_tunnel_rescue_k > 0),
                    "path_utility_pre_filter_path_ids": list(tunnel_rescue_pre_filter_path_ids),
                    "path_utility_injected_path_ids": list(tunnel_rescue_path_ids),
                    "path_utility_direct_support_path_ids": list(path_utility_direct_support_path_ids),
                    "path_utility_contrast_support_path_ids": list(path_utility_contrast_support_path_ids),
                    "path_utility_latent_context_path_ids": list(path_utility_latent_context_path_ids),
                    "path_utility_drift_noise_path_ids": list(path_utility_drift_noise_path_ids),
                    "path_utility_roles": dict(path_utility_roles),
                    "path_utility_reasons": dict(path_utility_reasons),
                    "path_utility_scores": dict(path_utility_scores),
                    "path_utility_overlap_tokens": dict(path_utility_overlap_tokens),
                    "path_utility_anchor_event_ids": list(path_utility_anchor_event_ids),
                    "path_utility_anchor_subject_signatures": list(path_utility_anchor_subject_signatures),
                    "tunnel_recall_pre_filter_count": int(len(tunnel_rescue_pre_filter_path_ids)),
                    "tunnel_usable_post_filter_count": int(len(tunnel_rescue_path_ids)),
                    "final_hit_event_ids": list(final_hit_event_ids),
                    "final_hit_dia_ids": _dia_ids_from_hits(final_hits),
                    "final_missing_selected_event_ids": list(final_missing_selected_event_ids),
                    "selected_event_count": int(len(selected_event_ids)),
                    "selected_path_count": int(len(selected_path_ids)),
                    "temporal_scores": dict(temporal_scores),
                    "recall_event_scores": dict(recall_event_scores),
                    "event_scores": dict(event_scores),
                    "base_path_scores": dict(base_path_scores),
                    "calibrated_path_scores": dict(calibrated_path_scores),
                    "path_fusion_delta_scores": dict(path_fusion_delta_scores),
                    "path_tunnel_support_scores": dict(path_tunnel_support_scores),
                    "path_tunnel_delta_scores": dict(path_tunnel_delta_scores),
                    "path_model_scores": dict(path_model_scores),
                    "path_chain_extension_enabled": path_chain_extension_enabled,
                    "path_chain_extension_delta_scores": dict(path_chain_extension_delta_scores),
                    "path_chain_extended_scores": dict(path_chain_extended_scores),
                    "effective_path_scores": dict(path_scores),
                    "path_scores": dict(path_scores),
                    "answer_type_scores": dict(answer_type_scores),
                    "answer_plan_scores": dict(answer_plan_scores),
                    "answer_plan_support_scores": dict(answer_plan_support_scores),
                    "answer_plan_adjusted_scores": dict(answer_plan_adjusted_scores),
                    "answer_plan_raw_ranked_event_ids": list(answer_plan_raw_ranked_event_ids),
                    "answer_plan_ranked_event_ids": list(answer_plan_ranked_event_ids),
                    "answer_plan_selected_event_ids": list(answer_plan_selected_event_ids),
                    "answer_plan_current_event_ids": list(answer_plan_current_event_ids),
                    "answer_plan_promotion_enabled": bool(answer_plan_promotion_enabled),
                    "answer_plan_promotion_score_margin": round(float(answer_plan_promotion_score_margin), 6),
                    "answer_plan_promotion_min_margin": round(float(answer_plan_promotion_min_margin), 6),
                    "answer_plan_event_selection_threshold": round(float(answer_plan_event_selection_threshold), 6),
                    "answer_plan_event_selection_top_k": int(answer_plan_event_selection_top_k),
                    "focused_answer_type": focused_answer_type,
                    "model_focused_answer_type": model_focused_answer_type,
                    "selection_consistency_repaired": bool(selection_consistency_repaired),
                    "selection_consistency_reason": selection_consistency_reason,
                    "preferred_path_types": [],
                },
            }
        dominant_answer_type = _dominant_answer_type(question_analysis, answer_type_scores)
        preferred_path_types = _answer_type_preferred_path_types(question_analysis, answer_type_scores)
        if bool(memory_router_decision.get("memory_router_guided")):
            temporal_focus = _memory_router_allows(memory_router_decision, "temporal")
        else:
            temporal_focus = dominant_answer_type == "time" or bool(question_analysis.get("is_temporal", False))
        router_profile_focus = bool(memory_router_decision.get("memory_router_guided")) and _memory_router_allows(
            memory_router_decision,
            "profile",
            "resource",
        )
        learned_event_available = bool(event_scores)
        learned_path_available = bool(path_scores)
        ranked_events = [
            event_id
            for event_id, _ in sorted(
                (
                    (event_id, event_scores.get(event_id, recall_event_scores.get(event_id, 0.0)))
                    for event_id in recall_event_ids
                ),
                key=lambda item: (-float(item[1]), item[0]),
            )
        ]
        base_selected_event_count = min(
            len(ranked_events),
            max(
                self.support_path_k * 2,
                min(self.candidate_event_k, max(top_k, _HYBRID_SELECTED_EVENT_FLOOR)),
            ),
        )
        effective_path_scores = dict(path_scores)
        if not learned_path_available:
            effective_path_scores = {
                path_id: _calibrated_path_score(
                    path=runtime_paths.get(path_id, {}),
                    base_score=float(score),
                    temporal_scores=temporal_scores,
                    question_analysis=question_analysis,
                    answer_type_scores=answer_type_scores,
                )
                for path_id, score in path_scores.items()
            }
        ranked_path_ids_all = [
            path_id
            for path_id, _ in sorted(
                ((path_id, float(score)) for path_id, score in effective_path_scores.items()),
                key=lambda item: (-float(item[1]), item[0]),
            )
        ]
        path_rescue_count = min(
            len(ranked_path_ids_all),
            max(self.support_path_k * 2, min(self.candidate_event_k, max(1, top_k))),
        )
        path_rescue_event_ids = _dedupe(
            _clean_text(runtime_paths.get(path_id, {}).get("event_id", ""))
            for path_id in ranked_path_ids_all[: max(1, path_rescue_count)]
        )
        selected_event_ids = _dedupe(
            [
                *path_rescue_event_ids,
                *ranked_events[: max(1, base_selected_event_count)],
            ]
        )
        if profile_first_event_ids:
            selected_event_ids = _dedupe([*profile_first_event_ids, *selected_event_ids])
        if answer_plan_ranked_event_ids:
            selected_event_ids = _dedupe([*selected_event_ids, *answer_plan_ranked_event_ids])
        selected_event_id_set = set(selected_event_ids)
        ranked_path_ids = [
            path_id
            for path_id in ranked_path_ids_all
            if _clean_text(runtime_paths.get(path_id, {}).get("event_id", "")) in selected_event_id_set
        ]
        if temporal_focus:
            focused_time_path_ids = [
                path_id
                for path_id in ranked_path_ids
                if _clean_text(runtime_paths.get(path_id, {}).get("type", "")) == "speaker_event_time"
            ]
            if focused_time_path_ids:
                ranked_path_ids = focused_time_path_ids
        path_limit_cap = _HYBRID_SELECTED_PATH_CAP
        if temporal_focus:
            path_limit_cap = _HYBRID_TEMPORAL_PATH_CAP
        elif dominant_answer_type == "profile" or router_profile_focus:
            path_limit_cap = _HYBRID_PROFILE_PATH_CAP
        selected_path_count = min(
            len(ranked_path_ids),
            max(1, min(max(1, self.support_path_k), min(max(1, top_k), path_limit_cap))),
        )
        selected_path_ids = ranked_path_ids[: max(1, selected_path_count)]
        final_hits: List[MemoryHit] = []
        seen_memory_ids = set()
        temporal_event_hits_added = 0
        for path_id in selected_path_ids:
            path = runtime_paths.get(path_id, {})
            event_id = _clean_text(path.get("event_id", ""))
            support_node_id = _path_support_node_id(path)
            path_type = _clean_text(path.get("type", ""))
            support_hit = _support_hit_for_path(path_type, grouped_hits.get(event_id, []))
            event_hit = _representative_event_hit(
                [*grouped_hits.get(event_id, []), *_event_record_hits_from_graph(self.graph, event_id)],
                query=query,
            )
            hit_pairs: List[tuple[MemoryHit | None, float]] = [(support_hit, path_scores.get(path_id, 0.0))]
            if not temporal_focus:
                hit_pairs.append((event_hit, path_scores.get(path_id, 0.0)))
            elif support_hit is None and event_hit is not None:
                hit_pairs.append((event_hit, path_scores.get(path_id, 0.0)))
            elif path_type == "speaker_event_time" and event_hit is not None and temporal_event_hits_added < 1:
                hit_pairs.append((event_hit, path_scores.get(path_id, 0.0)))
                temporal_event_hits_added += 1
            for raw_hit, path_score in hit_pairs:
                if raw_hit is None or raw_hit.memory_id in seen_memory_ids:
                    continue
                seen_memory_ids.add(raw_hit.memory_id)
                recall_score = float(recall_event_scores.get(event_id, event_scores.get(event_id, 0.0)))
                event_score = float(event_scores.get(event_id, raw_hit.score))
                temporal_score = float(temporal_scores.get(support_node_id, 0.0))
                effective_path_score = float(effective_path_scores.get(path_id, path_score))
                hybrid_score = round(effective_path_score, 6) if learned_path_available else round(
                    (0.55 * event_score)
                    + (0.20 * float(path_score))
                    + (0.15 * recall_score)
                    + (0.10 * temporal_score),
                    6,
                )
                metadata = dict(raw_hit.metadata or {})
                metadata.update(
                    {
                        "event_id": event_id,
                        **_answer_plan_hit_metadata(event_id),
                        "path_id": path_id,
                        "base_event_score": round(float(base_event_scores.get(event_id, event_score)), 6),
                        "rerank_event_score": round(float(rerank_event_scores.get(event_id, event_score)), 6),
                        "calibrated_event_score": round(float(calibrated_event_scores.get(event_id, event_score)), 6),
                        "matrix_event_score": round(float(matrix_event_scores.get(event_id, 0.0)), 6),
                        "event_fusion_delta_score": round(float(event_fusion_delta_scores.get(event_id, 0.0)), 6),
                        "event_tunnel_support_score": round(float(event_tunnel_support_scores.get(event_id, 0.0)), 6),
                        "event_tunnel_delta_score": round(float(event_tunnel_delta_scores.get(event_id, 0.0)), 6),
                        "matrix_enabled": matrix_enabled,
                        "event_calibration_enabled": event_calibration_enabled,
                        "path_calibration_enabled": path_calibration_enabled,
                        "event_tunnel_enabled": event_tunnel_enabled,
                        "path_tunnel_enabled": path_tunnel_enabled,
                        "final_event_fusion_enabled": final_event_fusion_enabled,
                        "final_path_fusion_enabled": final_path_fusion_enabled,
                        "decision_fusion_enabled": False,
                        "event_fusion_enabled": event_fusion_enabled,
                        "path_fusion_enabled": path_fusion_enabled,
                        "event_score": round(event_score, 6),
                        "recall_score": round(recall_score, 6),
                        "path_score": round(float(path_score), 6),
                        "base_path_score": round(float(base_path_scores.get(path_id, path_score)), 6),
                        "calibrated_path_score": round(float(calibrated_path_scores.get(path_id, effective_path_score)), 6),
                        "path_fusion_delta_score": round(float(path_fusion_delta_scores.get(path_id, 0.0)), 6),
                        "path_tunnel_support_score": round(float(path_tunnel_support_scores.get(path_id, 0.0)), 6),
                        "path_tunnel_delta_score": round(float(path_tunnel_delta_scores.get(path_id, 0.0)), 6),
                        "path_model_score": round(float(path_model_scores.get(path_id, path_score)), 6),
                        "path_chain_extension_enabled": path_chain_extension_enabled,
                        "path_chain_extension_delta_score": round(float(path_chain_extension_delta_scores.get(path_id, 0.0)), 6),
                        "path_chain_extended_score": round(float(path_chain_extended_scores.get(path_id, effective_path_score)), 6),
                        "effective_path_score": round(effective_path_score, 6),
                        "temporal_score": round(temporal_score, 6),
                        "raw_public_score": round(float(raw_hit.score), 6),
                        "hybrid_score": hybrid_score,
                        "hybrid_score_source": (
                            "learned_path_fusion"
                            if path_fusion_enabled
                            else "learned_path_score"
                            if learned_path_available
                            else "heuristic_mix"
                        ),
                        "evidence_snippet_role": "selected_path_support" if raw_hit is support_hit else "selected_path_event",
                        "selected_event_ids": list(selected_event_ids),
                        "selected_path_ids": list(selected_path_ids),
                    }
                )
                final_hits.append(
                    MemoryHit(
                        memory_id=raw_hit.memory_id,
                        category=raw_hit.category,
                        value=raw_hit.value,
                        relation=raw_hit.relation,
                        anchors=list(raw_hit.anchors),
                        score=hybrid_score,
                        source_kind=raw_hit.source_kind,
                        slot_key=raw_hit.slot_key,
                        state=raw_hit.state,
                        turn_index=int(raw_hit.turn_index),
                        metadata=metadata,
                    )
                )
        for event_rank, event_id in enumerate(selected_event_ids, start=1):
            event_hit = _representative_event_hit(
                [*grouped_hits.get(event_id, []), *_event_record_hits_from_graph(self.graph, event_id)],
                query=query,
            )
            if event_hit is None or event_hit.memory_id in seen_memory_ids:
                continue
            seen_memory_ids.add(event_hit.memory_id)
            recall_score = float(recall_event_scores.get(event_id, event_hit.score))
            event_score = float(event_scores.get(event_id, event_hit.score))
            hybrid_score = round(event_score, 6) if learned_event_available else round(
                (0.7 * event_score) + (0.3 * recall_score),
                6,
            )
            metadata = dict(event_hit.metadata or {})
            metadata.update(
                {
                    "event_id": event_id,
                    **_answer_plan_hit_metadata(event_id),
                    "path_id": "",
                    "base_event_score": round(float(base_event_scores.get(event_id, event_score)), 6),
                    "rerank_event_score": round(float(rerank_event_scores.get(event_id, event_score)), 6),
                    "calibrated_event_score": round(float(calibrated_event_scores.get(event_id, event_score)), 6),
                    "matrix_event_score": round(float(matrix_event_scores.get(event_id, 0.0)), 6),
                    "event_fusion_delta_score": round(float(event_fusion_delta_scores.get(event_id, 0.0)), 6),
                    "event_tunnel_support_score": round(float(event_tunnel_support_scores.get(event_id, 0.0)), 6),
                    "event_tunnel_delta_score": round(float(event_tunnel_delta_scores.get(event_id, 0.0)), 6),
                    "matrix_enabled": matrix_enabled,
                    "event_calibration_enabled": event_calibration_enabled,
                    "path_calibration_enabled": path_calibration_enabled,
                    "event_tunnel_enabled": event_tunnel_enabled,
                    "path_tunnel_enabled": path_tunnel_enabled,
                    "final_event_fusion_enabled": final_event_fusion_enabled,
                    "final_path_fusion_enabled": final_path_fusion_enabled,
                    "decision_fusion_enabled": False,
                    "event_fusion_enabled": event_fusion_enabled,
                    "path_fusion_enabled": path_fusion_enabled,
                    "event_score": round(event_score, 6),
                    "recall_score": round(recall_score, 6),
                    "path_score": 0.0,
                    "base_path_score": 0.0,
                    "calibrated_path_score": 0.0,
                    "path_fusion_delta_score": 0.0,
                    "path_tunnel_support_score": 0.0,
                    "path_tunnel_delta_score": 0.0,
                    "path_model_score": 0.0,
                    "path_chain_extension_enabled": path_chain_extension_enabled,
                    "path_chain_extension_delta_score": 0.0,
                    "path_chain_extended_score": 0.0,
                    "effective_path_score": 0.0,
                    "temporal_score": 0.0,
                    "raw_public_score": round(float(event_hit.score), 6),
                    "hybrid_score": hybrid_score,
                    "hybrid_score_source": (
                        "learned_event_fusion"
                        if event_fusion_enabled
                        else "learned_event_score"
                        if learned_event_available
                        else "heuristic_event_mix"
                    ),
                    "evidence_snippet_role": "selected_event_representative",
                    "selected_event_rank": int(event_rank),
                    "selected_event_ids": list(selected_event_ids),
                    "selected_path_ids": list(selected_path_ids),
                }
            )
            final_hits.append(
                MemoryHit(
                    memory_id=event_hit.memory_id,
                    category=event_hit.category,
                    value=event_hit.value,
                    relation=event_hit.relation,
                    anchors=list(event_hit.anchors),
                    score=hybrid_score,
                    source_kind=event_hit.source_kind,
                    slot_key=event_hit.slot_key,
                    state=event_hit.state,
                    turn_index=int(event_hit.turn_index),
                    metadata=metadata,
                )
            )
        if not final_hits:
            final_hits = list(hits)
        if profile_first_hits:
            final_hits = _inject_profile_first_hits(
                final_hits,
                profile_first_hits,
                selected_event_ids=selected_event_ids,
                selected_path_ids=selected_path_ids,
            )
        final_hits = _coverage_preserving_final_hits(final_hits, selected_event_ids=selected_event_ids, top_k=top_k)
        final_hit_event_ids = _event_ids_from_hits(final_hits)
        final_missing_selected_event_ids = [event_id for event_id in selected_event_ids if event_id not in set(final_hit_event_ids)]
        return {
            "hits": final_hits,
            "metadata": {
                "retrieval_mode": "hybrid_node_scored",
                "hybrid_enabled": True,
                "hybrid_source": hybrid_source,
                **memory_router_decision,
                "profile_first_router_suppressed": bool(profile_first_router_suppressed),
                "recall_event_ids": list(recall_event_ids),
                "learned_recall_event_ids": list(learned_recall_event_ids),
                "model_recall_event_ids": list(model_recall_event_ids),
                "symbolic_recall_event_ids": list(symbolic_recall_event_ids),
                "embedder_index_recall_event_ids": list(embedder_index_event_ids),
                **embedder_index_metadata,
                "embedder_fusion_mode": embedder_fusion_mode or "off",
                "embedder_fusion_enabled": bool(embedder_fusion_enabled),
                "embedder_fusion_weight": round(float(self.embedder_fusion_weight), 6),
                "embedder_fusion_score_floor": round(float(self.embedder_fusion_score_floor), 6),
                "embedder_fusion_top_k": int(self.embedder_fusion_top_k),
                "embedder_fusion_select_k": int(self.embedder_fusion_select_k),
                "embedder_fusion_max_boost": round(float(self.embedder_fusion_max_boost), 6),
                "embedder_fusion_event_scores": dict(embedder_fusion_applied_event_scores),
                "embedder_fusion_boosts": dict(embedder_fusion_boosts),
                "embedder_fusion_selected_event_ids": list(embedder_fusion_selected_event_ids),
                "embedder_fusion_selected_path_ids": list(embedder_fusion_selected_path_ids),
                "profile_first_hybrid_enabled": bool(profile_first_event_ids),
                "profile_first_event_ids": list(profile_first_event_ids),
                "profile_first_memory_ids": list(profile_first_memory_ids),
                "hybrid_candidate_event_ids": list(hybrid_candidate_event_ids),
                "hybrid_candidate_union_enabled": True,
                "hybrid_candidate_union_rescored": bool(hybrid_candidate_union_rescored),
                "hybrid_candidate_union_added_event_ids": list(hybrid_candidate_union_added_event_ids),
                "hybrid_candidate_union_priority_changed": bool(hybrid_candidate_union_priority_changed),
                "rerank_candidate_event_ids": list(rerank_candidate_event_ids),
                "base_event_scores": dict(base_event_scores),
                "rerank_event_scores": dict(rerank_event_scores),
                "calibrated_event_scores": dict(calibrated_event_scores),
                "matrix_event_scores": dict(matrix_event_scores),
                "event_fusion_delta_scores": dict(event_fusion_delta_scores),
                "event_tunnel_support_scores": dict(event_tunnel_support_scores),
                "event_tunnel_delta_scores": dict(event_tunnel_delta_scores),
                "tri_maze_event_reverse_scores": dict(tri_maze_event_reverse_scores),
                "tri_maze_event_boundary_scores": dict(tri_maze_event_boundary_scores),
                "tri_maze_event_reverse_relations": dict(tri_maze_event_reverse_relations),
                "matrix_rerank_event_ids": list(matrix_rerank_event_ids),
                "matrix_enabled": matrix_enabled,
                "rerank_path_scores": dict(rerank_path_scores),
                "matrix_path_scores": dict(matrix_path_scores),
                "tri_maze_path_reverse_scores": dict(tri_maze_path_reverse_scores),
                "tri_maze_path_boundary_scores": dict(tri_maze_path_boundary_scores),
                "tri_maze_path_reverse_relations": dict(tri_maze_path_reverse_relations),
                "matrix_path_rerank_ids": list(matrix_path_rerank_ids),
                "matrix_path_enabled": matrix_path_enabled,
                "fusion_enabled": fusion_enabled,
                "event_calibration_enabled": event_calibration_enabled,
                "path_calibration_enabled": path_calibration_enabled,
                "event_tunnel_enabled": event_tunnel_enabled,
                "path_tunnel_enabled": path_tunnel_enabled,
                "final_event_fusion_enabled": final_event_fusion_enabled,
                "final_path_fusion_enabled": final_path_fusion_enabled,
                "decision_fusion_enabled": False,
                "decision_score_source": "",
                "event_fusion_enabled": event_fusion_enabled,
                "path_fusion_enabled": path_fusion_enabled,
                "selected_event_ids": list(selected_event_ids),
                "path_rescue_event_ids": list(path_rescue_event_ids),
                "selected_path_ids": list(selected_path_ids),
                "final_hit_event_ids": list(final_hit_event_ids),
                "final_hit_dia_ids": _dia_ids_from_hits(final_hits),
                "final_missing_selected_event_ids": list(final_missing_selected_event_ids),
                "selected_event_count": int(len(selected_event_ids)),
                "selected_path_count": int(len(selected_path_ids)),
                "temporal_scores": dict(temporal_scores),
                "recall_event_scores": dict(recall_event_scores),
                "event_scores": dict(event_scores),
                "base_path_scores": dict(base_path_scores),
                "calibrated_path_scores": dict(calibrated_path_scores),
                "path_fusion_delta_scores": dict(path_fusion_delta_scores),
                "path_tunnel_support_scores": dict(path_tunnel_support_scores),
                "path_tunnel_delta_scores": dict(path_tunnel_delta_scores),
                "path_model_scores": dict(path_model_scores),
                "path_chain_extension_enabled": path_chain_extension_enabled,
                "path_chain_extension_delta_scores": dict(path_chain_extension_delta_scores),
                "path_chain_extended_scores": dict(path_chain_extended_scores),
                "effective_path_scores": dict(effective_path_scores),
                "path_scores": dict(path_scores),
                "answer_type_scores": dict(answer_type_scores),
                "answer_plan_scores": dict(answer_plan_scores),
                "answer_plan_support_scores": dict(answer_plan_support_scores),
                "answer_plan_adjusted_scores": dict(answer_plan_adjusted_scores),
                "answer_plan_raw_ranked_event_ids": list(answer_plan_raw_ranked_event_ids),
                "answer_plan_ranked_event_ids": list(answer_plan_ranked_event_ids),
                "answer_plan_selected_event_ids": list(answer_plan_selected_event_ids),
                "answer_plan_current_event_ids": list(answer_plan_current_event_ids),
                "answer_plan_promotion_enabled": bool(answer_plan_promotion_enabled),
                "answer_plan_promotion_score_margin": round(float(answer_plan_promotion_score_margin), 6),
                "answer_plan_promotion_min_margin": round(float(answer_plan_promotion_min_margin), 6),
                "answer_plan_event_selection_threshold": round(float(answer_plan_event_selection_threshold), 6),
                "answer_plan_event_selection_top_k": int(answer_plan_event_selection_top_k),
                "focused_answer_type": dominant_answer_type,
                "preferred_path_types": list(preferred_path_types),
            },
        }

    def retrieve(self, query: str, *, top_k: int = 6) -> MemoryRetrieval:
        self._reload_graph()
        start = time.perf_counter()
        candidate_top_k = max(top_k, self.candidate_event_k if self.retrieval_mode == "hybrid_node_scored" else min(max(top_k, 18), 48))
        payload = self.graph.retrieve(query, top_k=candidate_top_k)
        hits = [_raw_hit_to_memory_hit(item) for item in payload.get("hits", []) or []]
        scored_lookup = {hit.memory_id: hit for hit in hits if hit.memory_id}
        active_hits = _restore_hit_scores([_raw_hit_to_memory_hit(item) for item in payload.get("active_hits", []) or []], scored_lookup)
        history_hits = _restore_hit_scores([_raw_hit_to_memory_hit(item) for item in payload.get("history_hits", []) or []], scored_lookup)
        stale_hits = _restore_hit_scores([_raw_hit_to_memory_hit(item) for item in payload.get("stale_hits", []) or []], scored_lookup)
        overwrite_hits = _restore_hit_scores([_raw_hit_to_memory_hit(item) for item in payload.get("overwrite_hits", []) or []], scored_lookup)
        false_hits = _restore_hit_scores([_raw_hit_to_memory_hit(item) for item in payload.get("false_hits", []) or []], scored_lookup)
        public_hits = _public_graph_hits(self.graph) if self.retrieval_mode == "hybrid_node_scored" else []
        hybrid_payload = self._hybrid_node_scored_hits(query, hits, top_k=top_k, public_hits=public_hits)
        hits = list(hybrid_payload.get("hits", []) or hits)
        hybrid_metadata = dict(hybrid_payload.get("metadata", {}) or {})
        memory_router_decision = _memory_router_decision(
            hybrid_metadata,
            mode=self.memory_router_mode,
            threshold=self.memory_router_threshold,
            margin=self.memory_router_margin,
        )
        current_subject_payload = _current_subject_protected_hits(
            query=query,
            graph=self.graph,
            final_hits=hits,
            top_k=top_k,
        )
        hits = list(current_subject_payload.get("hits", hits) or hits)
        current_subject_metadata = dict(current_subject_payload.get("metadata", {}) or {})
        audit_anchor_payload = _audit_anchor_protected_hits(
            query=query,
            final_hits=hits,
            candidate_hits=list(public_hits) + list(active_hits) + list(history_hits),
            metadata={**dict(payload.get("metadata", {}) or {}), **hybrid_metadata, **current_subject_metadata},
            top_k=top_k,
        )
        hits = list(audit_anchor_payload.get("hits", hits) or hits)
        audit_anchor_metadata = dict(audit_anchor_payload.get("metadata", {}) or {})
        identifier_payload = _identifier_protected_hits(
            query=query,
            final_hits=hits,
            candidate_hits=list(public_hits) + list(active_hits) + list(history_hits),
            top_k=top_k,
        )
        hits = list(identifier_payload.get("hits", hits) or hits)
        identifier_metadata = dict(identifier_payload.get("metadata", {}) or {})
        if _memory_router_allows(memory_router_decision, "path_tunnel", "topic_tunnel"):
            depth_chain_payload = _depth_chain_protected_hits(
                query=query,
                graph=self.graph,
                final_hits=hits,
                top_k=top_k,
            )
            hits = list(depth_chain_payload.get("hits", hits) or hits)
            depth_chain_metadata = dict(depth_chain_payload.get("metadata", {}) or {})
        else:
            depth_chain_metadata = {
                "depth_chain_protected_enabled": False,
                "depth_chain_router_suppressed": True,
            }
        if _memory_router_allows(memory_router_decision, "profile", "resource"):
            profile_focused_payload = _profile_focused_pack_hits(
                self.graph,
                query,
                hits,
                top_k=top_k,
            )
            hits = list(profile_focused_payload.get("hits", hits) or hits)
            profile_focused_metadata = dict(profile_focused_payload.get("metadata", {}) or {})
        else:
            profile_focused_metadata = {
                "profile_focused_pack_enabled": False,
                "profile_focused_router_suppressed": True,
            }
        if _memory_router_allows(memory_router_decision, "topic_tunnel"):
            topic_bucket_payload = _topic_bucket_rerank_hits(self.graph, query, hits, top_k=top_k)
            topic_bucket_hits = topic_bucket_payload.get("hits", hits)
            hits = list(hits if topic_bucket_hits is None else topic_bucket_hits)
            topic_bucket_metadata = dict(topic_bucket_payload.get("metadata", {}) or {})
        else:
            topic_bucket_metadata = {
                "topic_bucket_rerank_enabled": False,
                "topic_bucket_router_suppressed": True,
            }
        temporal_runtime_payload = self._temporal_runtime_pack(query)
        temporal_evidence_payload = self._apply_temporal_evidence_pack_to_hits(
            hits,
            temporal_runtime_payload,
            top_k=top_k,
        )
        hits = list(temporal_evidence_payload.get("hits", hits) or hits)
        temporal_runtime_metadata = dict(temporal_evidence_payload.get("metadata", {}) or {})
        injection_planner_payload = self._apply_injection_planner_to_hits(query, hits, top_k=top_k)
        hits = list(injection_planner_payload.get("hits", hits) or hits)
        injection_planner_metadata = dict(injection_planner_payload.get("metadata", {}) or {})
        facet_query_pack_payload = _facet_query_pack_hits(
            self.graph,
            query,
            hits,
            top_k=top_k,
        )
        hits = list(facet_query_pack_payload.get("hits", hits) or hits)
        facet_query_pack_metadata = dict(facet_query_pack_payload.get("metadata", {}) or {})
        unit_coverage_mode = _normalize(os.getenv("TMCRA_UNIT_COVERAGE_PACK_MODE", "on"))
        if unit_coverage_mode in _MULTI_UNIT_CHAIN_DISABLED_MODES:
            unit_coverage_metadata = {
                "unit_coverage_pack_enabled": False,
                "unit_coverage_reason": "disabled",
            }
        else:
            unit_coverage_payload = _unit_coverage_pack_hits(
                self.graph,
                query,
                hits,
                top_k=top_k,
            )
            hits = list(unit_coverage_payload.get("hits", hits) or hits)
            unit_coverage_metadata = dict(unit_coverage_payload.get("metadata", {}) or {})
        if _normalize(os.getenv("TMCRA_MULTI_UNIT_CHAIN_SLOT_MODE", "on")) not in _MULTI_UNIT_CHAIN_DISABLED_MODES:
            multi_unit_chain_slot_payload = _multi_unit_chain_slot_hits(
                self.graph,
                query,
                hits,
                top_k=top_k,
            )
            hits = list(multi_unit_chain_slot_payload.get("hits", hits) or hits)
            multi_unit_chain_slot_metadata = dict(multi_unit_chain_slot_payload.get("metadata", {}) or {})
        else:
            multi_unit_chain_slot_metadata = {
                "multi_unit_chain_slot_enabled": False,
                "multi_unit_chain_slot_reason": "disabled",
            }
        profile_protected_reinserted_count = 0
        profile_protected_ids = _dedupe(
            [
                *list(profile_focused_metadata.get("profile_focused_pack_memory_ids", []) or []),
                *list(profile_focused_metadata.get("profile_first_memory_ids", []) or []),
            ],
            max_items=max(1, min(6, int(top_k or 1))),
        )
        if profile_protected_ids:
            existing_hits_by_id = {hit.memory_id: hit for hit in hits if hit.memory_id}
            protected_hits: List[MemoryHit] = []
            for memory_id in profile_protected_ids:
                hit = existing_hits_by_id.get(memory_id)
                if hit is None:
                    record = getattr(self.graph, "records_by_id", {}).get(memory_id)
                    if record is None:
                        continue
                    hit = _memory_hit_from_record(record)
                metadata = dict(hit.metadata or {})
                metadata.update(
                    {
                        "profile_first_hybrid_rescue": True,
                        "profile_protected_slot": True,
                        "evidence_snippet_role": "profile_protected_slot",
                    }
                )
                protected_hits.append(
                    MemoryHit(
                        memory_id=hit.memory_id,
                        category=hit.category,
                        value=hit.value,
                        relation=hit.relation,
                        anchors=list(hit.anchors),
                        score=max(float(hit.score), 4.8),
                        source_kind=hit.source_kind,
                        slot_key=hit.slot_key,
                        state=hit.state,
                        turn_index=int(hit.turn_index),
                        metadata=metadata,
                    )
                )
            if protected_hits:
                profile_protected_reinserted_count = len(protected_hits)
                seen_profile_ids = {hit.memory_id for hit in protected_hits if hit.memory_id}
                hits = [*protected_hits, *[hit for hit in hits if hit.memory_id not in seen_profile_ids]]
        embedder_fusion_output_event_ids = [
            _clean_text(event_id)
            for event_id in list(hybrid_metadata.get("embedder_fusion_selected_event_ids", []) or [])
            if _clean_text(event_id)
        ]
        embedder_fusion_output_reordered = False
        if embedder_fusion_output_event_ids:
            before_order = _event_ids_from_hits(hits)
            seen_output_memory_ids = {hit.memory_id for hit in hits}
            for event_id in embedder_fusion_output_event_ids:
                event_hit = _representative_event_hit(_event_record_hits_from_graph(self.graph, event_id), query=query)
                if event_hit is None or event_hit.memory_id in seen_output_memory_ids:
                    continue
                metadata = dict(event_hit.metadata or {})
                metadata.update(
                    {
                        "event_id": event_id,
                        "evidence_snippet_role": "embedder_fusion_event_representative",
                        "embedder_fusion_event_representative": True,
                    }
                )
                hits.append(
                    MemoryHit(
                        memory_id=event_hit.memory_id,
                        category=event_hit.category,
                        value=event_hit.value,
                        relation=event_hit.relation,
                        anchors=list(event_hit.anchors),
                        score=float(event_hit.score),
                        source_kind=event_hit.source_kind,
                        slot_key=event_hit.slot_key,
                        state=event_hit.state,
                        turn_index=int(event_hit.turn_index),
                        metadata=metadata,
                    )
                )
                seen_output_memory_ids.add(event_hit.memory_id)
            hits = _coverage_preserving_final_hits(
                hits,
                selected_event_ids=_dedupe([*embedder_fusion_output_event_ids, *before_order]),
                top_k=top_k,
            )
            embedder_fusion_output_reordered = before_order != _event_ids_from_hits(hits)
        retrieval_context_tokens = int(payload.get("context_token_estimate", _estimate_tokens_from_hits(hits)))
        self._last_retrieval_context_tokens = retrieval_context_tokens
        result = MemoryRetrieval(
            concepts=list(payload.get("concepts", []) or []),
            relations=list(payload.get("relations", []) or []),
            hits=hits,
            active_hits=active_hits,
            history_hits=history_hits,
            stale_hits=stale_hits,
            overwrite_hits=overwrite_hits,
            false_hits=false_hits,
            retrieval_seconds=time.perf_counter() - start,
            context_token_estimate=retrieval_context_tokens,
            retrieval_context_token_estimate=retrieval_context_tokens,
            metadata={
                "query_id": payload.get("query_id", ""),
                **dict(payload.get("metadata", {}) or {}),
                **hybrid_metadata,
                **memory_router_decision,
                **current_subject_metadata,
                **audit_anchor_metadata,
                **identifier_metadata,
                **depth_chain_metadata,
                **profile_focused_metadata,
                **topic_bucket_metadata,
                **temporal_runtime_metadata,
                **injection_planner_metadata,
                **facet_query_pack_metadata,
                **unit_coverage_metadata,
                **multi_unit_chain_slot_metadata,
                "profile_protected_reinserted_count": profile_protected_reinserted_count,
                "embedder_fusion_output_event_ids": list(embedder_fusion_output_event_ids),
                "embedder_fusion_output_reordered": bool(embedder_fusion_output_reordered),
            },
        )
        self._persist_graph()
        return result

    def export_dialog_graph(self, *, mode: str = "light") -> Dict[str, Any]:
        self._reload_graph()
        return self.graph.export_graph(
            snapshot_points=(1000, 5000, 10000, 20000, 50000, 100000, 200000, 300000, 500000),
            mode=mode,
        )

    def export_dialog_graph_mermaid(self) -> str:
        self._reload_graph()
        return self.graph.export_mermaid()

    def register_answer_support(self, *, answer_id: str, memory_ids: List[str], query_id: str = "", answer_text: str = "") -> None:
        self._reload_graph()
        self.graph.register_answer_support(answer_id=answer_id, memory_ids=memory_ids, query_id=query_id, answer_text=answer_text)
        self._persist_graph()

    def telemetry_snapshot(self) -> Dict[str, Any]:
        self._reload_graph()
        return self.graph.summary()

    def stats(self) -> Dict[str, Any]:
        self._reload_graph()
        storage = self._storage_breakdown()
        return _state_stats(
            storage_bytes=storage["storage_bytes"],
            retrieval_context_tokens=self._last_retrieval_context_tokens,
            total_state_tokens=storage["total_state_token_estimate"],
            core_storage_bytes=storage["core_storage_bytes"],
            audit_storage_bytes=storage["audit_storage_bytes"],
            core_state_token_estimate=storage["core_state_token_estimate"],
            audit_state_token_estimate=storage["audit_state_token_estimate"],
            lightweight_stats=bool(self.lightweight_stats),
            **self.graph.summary(),
        )

    def storage_bytes(self) -> int:
        self._reload_graph()
        return self._storage_breakdown()["storage_bytes"]

    def build_prompt_context(self, query: str, *, top_k: int = 8) -> Dict[str, Any]:
        retrieval = self.retrieve(query, top_k=top_k)
        retrieval_payload = retrieval.to_dict()
        stats = self.stats()
        context_summary = _graph_prompt_state_summary(self.graph, retrieval)
        prompt_context_payload = {
            "query": query,
            "retrieval": retrieval_payload,
            "context_summary": context_summary,
        }
        prompt_context_chars = len(json.dumps(prompt_context_payload, ensure_ascii=False))
        prompt_context_tokens_est = _estimate_tokens(json.dumps(prompt_context_payload, ensure_ascii=False))
        return {
            "mode": "graph_session_memory_v2",
            "query": query,
            "retrieval": retrieval_payload,
            "stats": stats,
            "state": context_summary,
            "context_summary": context_summary,
            "prompt_context_chars": int(prompt_context_chars),
            "prompt_context_tokens_est": int(prompt_context_tokens_est),
            "context_truncated": bool(context_summary.get("context_truncated", False)),
            "truncation_reason": _clean_text(context_summary.get("truncation_reason", "")),
        }


class SummaryWindowMemoryAdapter(MemoryAdapter):
    name = "summary_window_memory"

    def __init__(self, *, window_size: int = 24, auto_extract: bool = False) -> None:
        self.extractor = SessionMemoryExtractor()
        self.window_size = max(4, int(window_size))
        self.turn_index = 0
        self.active_slots: Dict[str, SessionMemoryRecordV2] = {}
        self.recent_turns: deque[Dict[str, Any]] = deque(maxlen=self.window_size)
        self.auto_extract = bool(auto_extract)
        self._last_retrieval_context_tokens = 0

    def reset(self) -> None:
        self.turn_index = 0
        self.active_slots = {}
        self.recent_turns = deque(maxlen=self.window_size)
        self._last_retrieval_context_tokens = 0

    def ingest_turn(
        self,
        user_text: str,
        assistant_text: str = "",
        *,
        answer_payload: Dict[str, Any] | None = None,
        extraction_result: Dict[str, Any] | None = None,
    ) -> None:
        self.turn_index += 1
        records = _build_turn_records(
            self.extractor,
            user_text=user_text,
            answer_payload=answer_payload,
            extraction_result=extraction_result,
            turn_index=self.turn_index,
            allow_auto_extract=self.auto_extract,
        )
        for record in records:
            previous = self.active_slots.get(record.slot_key)
            if previous:
                previous.state = "superseded"
                record.supersedes.append(previous.memory_id)
            record.state = "active"
            self.active_slots[record.slot_key] = record
        self.recent_turns.append(
            {
                "turn_index": self.turn_index,
                "text": _clean_text(user_text),
                "assistant": _clean_text(assistant_text),
            }
        )

    def retrieve(self, query: str, *, top_k: int = 6) -> MemoryRetrieval:
        start = time.perf_counter()
        query_tokens = set(_tokenize(query))
        hints = set(infer_category_hints(query))
        scored: List[tuple[float, SessionMemoryRecordV2]] = []
        for record in self.active_slots.values():
            token_set = record.token_set()
            overlap = len(query_tokens & token_set) if query_tokens and token_set else 0
            score = overlap / max(1, len(query_tokens | token_set)) if query_tokens and token_set else 0.0
            if hints and record.category in hints:
                score += 0.22
            if record.slot_key.lower() in _normalize(query):
                score += 0.12
            score += min(0.08, record.turn_index * 0.0004)
            score += 0.2
            if score > 0:
                scored.append((score, record))
        scored.sort(key=lambda item: (item[0], item[1].turn_index), reverse=True)
        selected_records = [record for _, record in scored[:top_k]]
        hits = [
            MemoryHit(
                memory_id=record.memory_id,
                category=record.category,
                value=record.value,
                relation=record.relation,
                anchors=list(record.anchor_concepts),
                score=float(score),
                source_kind=record.source_kind,
                slot_key=record.slot_key,
                state=record.state,
                turn_index=record.turn_index,
                metadata={"window_size": self.window_size},
            )
            for score, record in scored[:top_k]
        ]
        concepts = []
        relations = []
        for hit in hits:
            concepts.append({"concept": hit.value, "type": hit.category, "source_kind": hit.source_kind})
            for anchor in hit.anchors[:2]:
                concepts.append({"concept": anchor, "type": "context", "source_kind": hit.source_kind})
            relation = _relation_hit(hit, weight_bias=0.06)
            if relation:
                relations.append(relation)
        retrieval_context_tokens = _estimate_tokens_from_hits(hits)
        self._last_retrieval_context_tokens = retrieval_context_tokens
        return MemoryRetrieval(
            concepts=concepts,
            relations=relations,
            hits=hits,
            active_hits=list(hits),
            retrieval_seconds=time.perf_counter() - start,
            context_token_estimate=retrieval_context_tokens,
            retrieval_context_token_estimate=retrieval_context_tokens,
            metadata={
                "records": len(self.active_slots),
                "window_size": self.window_size,
                "recent_turns": len(self.recent_turns),
            },
        )

    def stats(self) -> Dict[str, Any]:
        payload = {
            "active_slots": {slot: record.to_dict() for slot, record in self.active_slots.items()},
            "recent_turns": list(self.recent_turns),
        }
        total_state_tokens = _estimate_tokens(json.dumps(payload, ensure_ascii=False))
        return _state_stats(
            storage_bytes=self.storage_bytes(),
            retrieval_context_tokens=self._last_retrieval_context_tokens,
            total_state_tokens=total_state_tokens,
            records=len(self.active_slots),
            active_slots=len(self.active_slots),
            recent_turns=len(self.recent_turns),
        )

    def storage_bytes(self) -> int:
        payload = {
            "active_slots": {slot: record.to_dict() for slot, record in self.active_slots.items()},
            "recent_turns": list(self.recent_turns),
        }
        return len(json.dumps(payload, ensure_ascii=False).encode("utf-8"))

    def build_prompt_context(self, query: str, *, top_k: int = 8) -> Dict[str, Any]:
        return {
            "mode": "summary_window_memory",
            "query": query,
            "retrieval": self.retrieve(query, top_k=top_k).to_dict(),
            "stats": self.stats(),
            "state": {
                "active_slots": {slot: record.to_dict() for slot, record in self.active_slots.items()},
                "recent_turns": list(self.recent_turns),
            },
        }


@dataclass(slots=True)
class _VectorRecord:
    memory_id: str
    category: str
    value: str
    relation: str
    anchors: List[str]
    tokens: List[str]
    turn_index: int
    slot_key: str = ""
    active: bool = True
    source_kind: str = "vector_memory"
    metadata: Dict[str, Any] = field(default_factory=dict)


class VectorRAGMemoryAdapter(MemoryAdapter):
    name = "vector_rag_memory"

    def __init__(self, *, auto_extract: bool = False) -> None:
        self.extractor = SessionMemoryExtractor()
        self.records: List[_VectorRecord] = []
        self.turn_index = 0
        self.auto_extract = bool(auto_extract)
        self._last_retrieval_context_tokens = 0

    def reset(self) -> None:
        self.records = []
        self.turn_index = 0
        self._last_retrieval_context_tokens = 0

    def ingest_turn(
        self,
        user_text: str,
        assistant_text: str = "",
        *,
        answer_payload: Dict[str, Any] | None = None,
        extraction_result: Dict[str, Any] | None = None,
    ) -> None:
        _ = assistant_text
        self.turn_index += 1
        records = _build_turn_records(
            self.extractor,
            user_text=user_text,
            answer_payload=answer_payload,
            extraction_result=extraction_result,
            turn_index=self.turn_index,
            allow_auto_extract=self.auto_extract,
        )
        for record in records:
            if record.slot_key:
                for previous in self.records:
                    if previous.slot_key == record.slot_key and previous.active:
                        previous.active = False
            self.records.append(
                _VectorRecord(
                    memory_id=record.memory_id,
                    category=record.category,
                    value=record.value,
                    relation=record.relation,
                    anchors=list(record.anchor_concepts),
                    tokens=list(record.token_set()),
                    turn_index=record.turn_index,
                    slot_key=record.slot_key,
                    active=record.state == "active",
                    source_kind=record.source_kind,
                    metadata=dict(record.metadata),
                )
            )

    def retrieve(self, query: str, *, top_k: int = 6) -> MemoryRetrieval:
        start = time.perf_counter()
        query_tokens = set(_tokenize(query))
        hints = set(infer_category_hints(query))
        scored: List[tuple[float, _VectorRecord]] = []
        for record in self.records:
            token_set = set(record.tokens)
            overlap = len(query_tokens & token_set) if query_tokens and token_set else 0
            score = overlap / max(1, len(query_tokens | token_set)) if query_tokens and token_set else 0.0
            if hints and record.category in hints:
                score += 0.18
            if record.slot_key and record.slot_key.lower() in _normalize(query):
                score += 0.1
            score += min(0.12, record.turn_index * 0.0004)
            score += 0.18 if record.active else -0.25
            if score > 0:
                scored.append((score, record))
        if not scored:
            for record in self.records[-top_k:]:
                scored.append((0.05 + (0.15 if record.active else 0.0), record))
        scored.sort(key=lambda item: (item[0], item[1].active, item[1].turn_index), reverse=True)
        hits = [
            MemoryHit(
                memory_id=record.memory_id,
                category=record.category,
                value=record.value,
                relation=record.relation,
                anchors=list(record.anchors),
                score=float(score),
                source_kind=record.source_kind,
                slot_key=record.slot_key,
                state="active" if record.active else "superseded",
                turn_index=record.turn_index,
                metadata=dict(record.metadata),
            )
            for score, record in scored[:top_k]
        ]
        concepts = []
        relations = []
        for hit in hits:
            concepts.append({"concept": hit.value, "type": hit.category, "source_kind": hit.source_kind})
            for anchor in hit.anchors[:2]:
                concepts.append({"concept": anchor, "type": "context", "source_kind": hit.source_kind})
            relation = _relation_hit(hit)
            if relation:
                relations.append(relation)
        retrieval_context_tokens = _estimate_tokens_from_hits(hits)
        self._last_retrieval_context_tokens = retrieval_context_tokens
        return MemoryRetrieval(
            concepts=concepts,
            relations=relations,
            hits=hits,
            active_hits=[hit for hit in hits if hit.state == "active"],
            history_hits=[hit for hit in hits if hit.state != "active"],
            retrieval_seconds=time.perf_counter() - start,
            context_token_estimate=retrieval_context_tokens,
            retrieval_context_token_estimate=retrieval_context_tokens,
            metadata={
                "records": len(self.records),
                "active_records": sum(1 for item in self.records if item.active),
            },
        )

    def stats(self) -> Dict[str, Any]:
        payload = [
            {
                "memory_id": record.memory_id,
                "category": record.category,
                "value": record.value,
                "relation": record.relation,
                "anchors": record.anchors,
                "slot_key": record.slot_key,
                "active": record.active,
            }
            for record in self.records
        ]
        total_state_tokens = _estimate_tokens(json.dumps(payload, ensure_ascii=False))
        return _state_stats(
            storage_bytes=self.storage_bytes(),
            retrieval_context_tokens=self._last_retrieval_context_tokens,
            total_state_tokens=total_state_tokens,
            records=len(self.records),
            active_records=sum(1 for item in self.records if item.active),
        )

    def storage_bytes(self) -> int:
        payload = [
            {
                "memory_id": record.memory_id,
                "category": record.category,
                "value": record.value,
                "relation": record.relation,
                "anchors": record.anchors,
                "slot_key": record.slot_key,
                "active": record.active,
            }
            for record in self.records
        ]
        return len(json.dumps(payload, ensure_ascii=False).encode("utf-8"))

    def build_prompt_context(self, query: str, *, top_k: int = 8) -> Dict[str, Any]:
        state = [
            {
                "memory_id": record.memory_id,
                "category": record.category,
                "value": record.value,
                "relation": record.relation,
                "anchors": list(record.anchors),
                "slot_key": record.slot_key,
                "active": bool(record.active),
                "turn_index": int(record.turn_index),
            }
            for record in self.records
        ]
        return {
            "mode": "vector_rag_memory",
            "query": query,
            "retrieval": self.retrieve(query, top_k=top_k).to_dict(),
            "stats": self.stats(),
            "state": state,
        }


class FullHistoryMemoryAdapter(MemoryAdapter):
    name = "full_history_memory"

    def __init__(self) -> None:
        self.turns: List[Dict[str, str]] = []
        self._last_retrieval_context_tokens = 0

    def reset(self) -> None:
        self.turns = []
        self._last_retrieval_context_tokens = 0

    def ingest_turn(
        self,
        user_text: str,
        assistant_text: str = "",
        *,
        answer_payload: Dict[str, Any] | None = None,
        extraction_result: Dict[str, Any] | None = None,
    ) -> None:
        _ = answer_payload, extraction_result
        self.turns.append({"user": _clean_text(user_text), "assistant": _clean_text(assistant_text)})

    def retrieve(self, query: str, *, top_k: int = 6) -> MemoryRetrieval:
        start = time.perf_counter()
        query_tokens = set(_tokenize(query))
        scored: List[tuple[float, Dict[str, str], int]] = []
        for index, turn in enumerate(self.turns):
            combined = f"{turn.get('user', '')} {turn.get('assistant', '')}"
            token_set = set(_tokenize(combined))
            if not token_set:
                continue
            overlap = len(query_tokens & token_set) if query_tokens else 0
            score = overlap / max(1, len(query_tokens | token_set)) if query_tokens else 0.0
            scored.append((score, turn, index))
        scored.sort(key=lambda item: (item[0], item[2]), reverse=True)
        hits = [
            MemoryHit(
                memory_id=f"turn:{index}",
                category="history_turn",
                value=turn.get("user", ""),
                relation="conversation_context",
                anchors=[turn.get("assistant", "")] if turn.get("assistant") else [],
                score=float(score),
                source_kind="full_history",
                slot_key=f"turn.{index}",
                state="active",
                turn_index=index + 1,
            )
            for score, turn, index in scored[:top_k]
            if turn.get("user", "")
        ]
        retrieval_context_tokens = _estimate_tokens(json.dumps(self.turns, ensure_ascii=False))
        self._last_retrieval_context_tokens = retrieval_context_tokens
        return MemoryRetrieval(
            hits=hits,
            active_hits=list(hits),
            retrieval_seconds=time.perf_counter() - start,
            context_token_estimate=retrieval_context_tokens,
            retrieval_context_token_estimate=retrieval_context_tokens,
            metadata={"records": len(self.turns)},
        )

    def stats(self) -> Dict[str, Any]:
        total_state_tokens = _estimate_tokens(json.dumps(self.turns, ensure_ascii=False))
        return _state_stats(
            storage_bytes=self.storage_bytes(),
            retrieval_context_tokens=self._last_retrieval_context_tokens,
            total_state_tokens=total_state_tokens,
            records=len(self.turns),
        )

    def storage_bytes(self) -> int:
        return len(json.dumps(self.turns, ensure_ascii=False).encode("utf-8"))

    def build_prompt_context(self, query: str, *, top_k: int = 8) -> Dict[str, Any]:
        return {
            "mode": "full_history_memory",
            "query": query,
            "retrieval": self.retrieve(query, top_k=top_k).to_dict(),
            "stats": self.stats(),
            "state": {
                "turns": list(self.turns),
            },
        }
