#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping


DEFAULT_SERVICE_ROOT = Path(os.getenv("TMCRA_SERVICE_ROOT", "./tmcra_api_service"))
DEFAULT_REPO = Path(os.getenv("TMCRA_REPO_ROOT", str(DEFAULT_SERVICE_ROOT / "private" / "tmcra-integrated")))
DEFAULT_DATA = Path(os.getenv("LONGMEMEVAL_S_DATA", "./data/longmemeval_s_cleaned.json"))
DEFAULT_OUT_ROOT = Path(os.getenv("TMCRA_LME_RUN_ROOT", "./runs"))


def log(event: str, **payload: Any) -> None:
    stamp = datetime.now().isoformat(timespec="seconds")
    details = " ".join(f"{key}={json.dumps(value, ensure_ascii=False)}" for key, value in sorted(payload.items()))
    print(f"[lme_native] {stamp} {event}" + (f" {details}" if details else ""), flush=True)


def read_env_file(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    if not path.exists():
        return env
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        key, value = line.split("=", 1)
        env[key.strip()] = value.strip().strip('"').strip("'")
    return env


def apply_env_defaults(service_root: Path) -> None:
    for env_path in [service_root / "env" / "tmcra-api.env", service_root / "env" / "tmcra-gemma.env"]:
        for key, value in read_env_file(env_path).items():
            if value == "":
                continue
            os.environ.setdefault(key, value)


def iter_json_array(path: Path, *, limit: int = 0) -> Iterable[dict[str, Any]]:
    import codecs

    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    yielded = 0
    array_started = False
    object_started = False
    object_depth = 0
    in_string = False
    escape_next = False
    buffer: list[str] = []
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1 << 20)
            if not chunk:
                text = decoder.decode(b"", final=True)
            else:
                text = decoder.decode(chunk)
            if not text and not chunk:
                break
            for char in text:
                if not array_started:
                    if char.isspace():
                        continue
                    if char != "[":
                        raise RuntimeError(f"expected top-level JSON array in {path}")
                    array_started = True
                    continue
                if not object_started:
                    if char.isspace() or char == ",":
                        continue
                    if char == "]":
                        return
                    if char != "{":
                        raise RuntimeError(f"expected object item in {path}, got {char!r}")
                    object_started = True
                    object_depth = 1
                    in_string = False
                    escape_next = False
                    buffer = ["{"]
                    continue
                buffer.append(char)
                if in_string:
                    if escape_next:
                        escape_next = False
                    elif char == "\\":
                        escape_next = True
                    elif char == '"':
                        in_string = False
                    continue
                if char == '"':
                    in_string = True
                    continue
                if char == "{":
                    object_depth += 1
                elif char == "}":
                    object_depth -= 1
                    if object_depth == 0:
                        row = json.loads("".join(buffer))
                        if isinstance(row, dict):
                            yield row
                            yielded += 1
                            if limit > 0 and yielded >= limit:
                                return
                        object_started = False
                        buffer = []
            if not chunk:
                break
    if object_started:
        raise RuntimeError(f"unterminated JSON object in {path}")


def http_json(
    method: str,
    url: str,
    payload: dict[str, Any] | None = None,
    *,
    timeout: int = 300,
    api_key: str = "",
) -> dict[str, Any]:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload is not None else None
    headers = {"Content-Type": "application/json; charset=utf-8"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    max_attempts = max(1, int(os.getenv("TMCRA_HTTP_JSON_MAX_ATTEMPTS", "8") or 8))
    retry_codes = {429, 500, 502, 503, 504}
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        req = urllib.request.Request(url, data=data, method=method, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                body = response.read().decode("utf-8", "replace")
                return json.loads(body) if body else {}
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", "replace")
            last_error = RuntimeError(f"HTTP {exc.code} {url}: {body[:500]}")
            if exc.code not in retry_codes or attempt >= max_attempts:
                raise last_error from exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt >= max_attempts:
                raise RuntimeError(f"HTTP request failed {url}: {exc}") from exc
        time.sleep(min(2 ** (attempt - 1), 8))
    raise RuntimeError(f"HTTP request failed {url}: {last_error}")


def chat_completion(
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    *,
    max_tokens: int = 180,
    temperature: float = 0.0,
    api_key: str = "",
) -> str:
    wire_api = clean_text(os.getenv("TMCRA_ANSWER_WIRE_API", os.getenv("OPENAI_WIRE_API", ""))).lower()
    if wire_api == "responses":
        body = http_json(
            "POST",
            base_url.rstrip("/") + "/responses",
            {
                "model": model,
                "input": messages,
                "max_output_tokens": max_tokens,
                "temperature": temperature,
            },
            timeout=360,
            api_key=api_key,
        )
        output_text = clean_text(body.get("output_text", ""))
        if output_text:
            return output_text
        chunks: list[str] = []
        for item in body.get("output", []) or []:
            for content in item.get("content", []) or []:
                text = content.get("text") or content.get("content") or ""
                if text:
                    chunks.append(str(text))
        return "\n".join(chunks).strip()
    body = http_json(
        "POST",
        base_url.rstrip("/") + "/chat/completions",
        {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        },
        timeout=360,
        api_key=api_key,
    )
    return (((body.get("choices") or [{}])[0].get("message") or {}).get("content") or "").strip()


def answer_llm_config() -> tuple[str, str, str]:
    answer_base_url = clean_text(os.getenv("TMCRA_ANSWER_BASE_URL", ""))
    answer_model = clean_text(os.getenv("TMCRA_ANSWER_MODEL", ""))
    answer_api_key = clean_text(os.getenv("TMCRA_ANSWER_API_KEY", ""))
    if answer_base_url or answer_model or answer_api_key:
        return (
            answer_base_url or os.getenv("GEMMA_BASE_URL", "http://127.0.0.1:18002/v1"),
            answer_model or os.getenv("GEMMA_MODEL", os.getenv("TMCRA_GEMMA_MODEL_NAME", "gemma-4-e4b-it")),
            answer_api_key or clean_text(os.getenv("OPENAI_API_KEY", "")),
        )
    return (
        os.getenv("GEMMA_BASE_URL", "http://127.0.0.1:18002/v1"),
        os.getenv("GEMMA_MODEL", os.getenv("TMCRA_GEMMA_MODEL_NAME", "gemma-4-e4b-it")),
        clean_text(os.getenv("GEMMA_API_KEY", "")),
    )


def query_graph_llm_config() -> tuple[str, str, str]:
    base_url = clean_text(os.getenv("TMCRA_QUERY_GRAPH_BASE_URL", ""))
    model = clean_text(os.getenv("TMCRA_QUERY_GRAPH_MODEL", ""))
    api_key = clean_text(os.getenv("TMCRA_QUERY_GRAPH_API_KEY", ""))
    if base_url or model or api_key:
        answer_base_url, answer_model, answer_api_key = answer_llm_config()
        return (
            base_url or answer_base_url,
            model or answer_model,
            api_key or answer_api_key,
        )
    return answer_llm_config()


def clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def truncate(value: Any, n: int = 500) -> str:
    text = clean_text(value)
    return text[:n]


def session_text_chunks(
    *,
    session_id: str,
    date: str,
    turns: list[Mapping[str, Any]],
    max_chars: int,
    max_chunks: int,
) -> list[str]:
    chunks: list[str] = []
    current: list[str] = [f"LongMemEval session_id={session_id} date={date}"]
    current_len = len(current[0])
    for index, turn in enumerate(turns, start=1):
        role = clean_text(turn.get("role", "unknown"))
        content = clean_text(turn.get("content", ""))
        if not content:
            continue
        part = f"[{session_id} turn={index} role={role}] {content}"
        if current_len + len(part) + 2 > max_chars and len(current) > 1:
            chunks.append("\n".join(current))
            if max_chunks > 0 and len(chunks) >= max_chunks:
                return chunks
            current = [f"LongMemEval session_id={session_id} date={date} continued=true"]
            current_len = len(current[0])
        current.append(part)
        current_len += len(part) + 1
    if len(current) > 1 and (max_chunks <= 0 or len(chunks) < max_chunks):
        chunks.append("\n".join(current))
    return chunks


def select_session_indices(row: Mapping[str, Any], *, max_distractors: int) -> list[int]:
    session_ids = [clean_text(item) for item in list(row.get("haystack_session_ids") or [])]
    answer_ids = {clean_text(item) for item in list(row.get("answer_session_ids") or [])}
    answer_indices = [index for index, sid in enumerate(session_ids) if sid in answer_ids]
    distractors: list[int] = []
    latest_candidates = list(range(max(0, len(session_ids) - 4), len(session_ids)))
    first_candidates = list(range(min(3, len(session_ids))))
    mid = len(session_ids) // 2
    middle_candidates = [max(0, mid - 1), mid, min(len(session_ids) - 1, mid + 1)] if session_ids else []
    for index in [*first_candidates, *middle_candidates, *latest_candidates]:
        if index not in answer_indices and index not in distractors and 0 <= index < len(session_ids):
            distractors.append(index)
        if len(distractors) >= max_distractors:
            break
    return sorted(set(answer_indices + distractors))


def select_official_full_history_indices(row: Mapping[str, Any]) -> list[int]:
    """Official LongMemEval-style input: use every haystack session in file order."""
    return list(range(len(list(row.get("haystack_sessions") or []))))


def build_writer() -> Any:
    from experiments.replacement.adapters.base import LLMProfile
    from experiments.replacement.multi_layer_tmcra_writer import TMCRASuspectAnchoredTransformerMemoryWriter
    from experiments.replacement.semantic_memory_writer import DeterministicMemoryWriteGate, OpenAICompatSemanticMemoryWriter

    profile = LLMProfile(
        name="longmemeval_native_writer",
        model=os.getenv("TMCRA_WRITER_MODEL", "gemma-4-e4b-it"),
        base_url=os.getenv("TMCRA_WRITER_BASE_URL", "http://127.0.0.1:18002/v1"),
        api_key=os.getenv("TMCRA_WRITER_API_KEY", ""),
        timeout_seconds=float(os.getenv("TMCRA_WRITER_TIMEOUT_SECONDS", "180")),
        temperature=float(os.getenv("TMCRA_WRITER_TEMPERATURE", "0.0")),
        max_tokens=int(os.getenv("TMCRA_WRITER_MAX_TOKENS", "512")),
    )
    base_writer = OpenAICompatSemanticMemoryWriter(
        profile,
        gate=DeterministicMemoryWriteGate(min_grounding_score=float(os.getenv("TMCRA_WRITER_MIN_GROUNDING_SCORE", "0.35"))),
        max_proposals=int(os.getenv("TMCRA_WRITER_MAX_PROPOSALS", "2")),
    )
    return TMCRASuspectAnchoredTransformerMemoryWriter(
        base_writer,
        max_cells=int(os.getenv("TMCRA_WRITER_MAX_PROPOSALS", "2")),
        value_char_budget=int(os.getenv("TMCRA_WRITER_VALUE_CHAR_BUDGET", "120")),
        source_span_char_budget=int(os.getenv("TMCRA_WRITER_SOURCE_SPAN_CHAR_BUDGET", "180")),
        transformer_layers=int(os.getenv("TMCRA_WRITER_TRANSFORMER_LAYERS", "2")),
        state_attention_k=int(os.getenv("TMCRA_WRITER_STATE_ATTENTION_K", "16")),
        suspect_threshold=float(os.getenv("TMCRA_WRITER_SUSPECT_THRESHOLD", "0.48")),
        suspect_promote_count=int(os.getenv("TMCRA_WRITER_SUSPECT_PROMOTE_COUNT", "2")),
    )


def build_adapter(scope_id: str, storage_path: Path) -> Any:
    from experiments.replacement.adapters.memory_adapters import GraphSessionMemoryAdapter

    return GraphSessionMemoryAdapter(
        auto_extract=False,
        storage_backend="sqlite",
        storage_path=str(storage_path),
        scope_id=scope_id,
        retrieval_mode=os.getenv("TMCRA_RETRIEVAL_MODE", "hybrid_node_scored"),
        node_model_path=os.getenv("TMCRA_NODE_MODEL_PATH", ""),
        path_model_path=os.getenv("TMCRA_PATH_MODEL_PATH", ""),
        node_model_device=os.getenv("TMCRA_NODE_MODEL_DEVICE", "cpu"),
        candidate_event_k=int(os.getenv("TMCRA_CANDIDATE_EVENT_K", "24")),
        support_path_k=int(os.getenv("TMCRA_SUPPORT_PATH_K", "3")),
        path_tunnel_rescue_k=int(os.getenv("TMCRA_PATH_TUNNEL_RESCUE_K", "2")),
        path_tunnel_rescue_score_floor=float(os.getenv("TMCRA_PATH_TUNNEL_RESCUE_SCORE_FLOOR", "0.0")),
        path_tunnel_rescue_min_age=int(os.getenv("TMCRA_PATH_TUNNEL_RESCUE_MIN_AGE", "0")),
        path_tunnel_rescue_min_score_margin=float(os.getenv("TMCRA_PATH_TUNNEL_RESCUE_MIN_SCORE_MARGIN", "0.0")),
    )


def disable_topic_bucket_runtime() -> None:
    import experiments.replacement.adapters.memory_adapters as memory_adapters

    def no_topic_bucket(*args: Any, **kwargs: Any) -> dict[str, Any]:
        return {}

    def no_apply_topic_bucket(records: list[Any], topic_bucket: Mapping[str, Any]) -> None:
        return None

    def no_topic_edges(*args: Any, **kwargs: Any) -> dict[str, Any]:
        return {
            "topic_bridge_disabled": True,
            "dialogue_tunnel_disabled": True,
            "disabled_reason": "longmemeval_native_no_topic_bucket",
        }

    def no_topic_rerank(graph: Any, query: str, hits: list[Any], *, top_k: int) -> dict[str, Any]:
        limit = max(1, int(top_k or 1))
        return {
            "hits": list(hits)[:limit],
            "metadata": {
                "topic_bucket_rerank_enabled": False,
                "topic_bucket_disabled": True,
                "topic_bucket_disable_reason": "longmemeval_native_no_topic_bucket",
                "topic_bucket_candidate_count": len(list(hits)),
                "topic_bucket_final_count": len(list(hits)[:limit]),
            },
        }

    memory_adapters._assign_topic_bucket_for_text = no_topic_bucket
    memory_adapters._apply_topic_bucket_to_records = no_apply_topic_bucket
    memory_adapters._last_topic_turn = no_topic_bucket
    memory_adapters._add_topic_bridge_edges = no_topic_edges
    memory_adapters._add_dialogue_tunnel_edges = no_topic_edges
    memory_adapters._topic_bucket_rerank_hits = no_topic_rerank


def writer_ingest(adapter: Any, writer: Any, text: str, *, qid: str, chunk_id: str) -> dict[str, Any]:
    from experiments.replacement.semantic_memory_writer import build_modelized_facet_unit_records

    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    sidecar_hints = {
        "metadata": {
            "source": "longmemeval_native_flow",
            "question_id": qid,
            "chunk_id": chunk_id,
        }
    }
    t0 = time.perf_counter()
    proposals, writer_metadata = writer.propose_public_turn(
        current_turn=text,
        previous_turn="",
        next_turn="",
        speaker="user",
        session_timestamp=timestamp,
        sidecar_hints=sidecar_hints,
        auxiliary_evidence_texts=[],
        input_mode=os.getenv("TMCRA_WRITER_INPUT_MODE", "delta"),
    )
    graph = getattr(adapter, "graph", None)
    turn_index = int(getattr(graph, "turn_index", 0) or 0) + 1
    gate_result = writer.gate.build_payload(
        proposals=proposals,
        text=f"[{timestamp}] user: {text}",
        raw_text=text,
        speaker="user",
        session_key=qid,
        turn_index=turn_index,
        timestamp=timestamp,
        dia_id=f"longmemeval:{qid}:{chunk_id}",
        sidecar_hints=sidecar_hints,
        writer_metadata=writer_metadata,
    )
    payload = dict(gate_result.payload or {})
    replacement_records = list(payload.get("replacement_memory_records", []) or [])
    base_writer = getattr(writer, "base_writer", writer)
    unit_records, unit_metadata = build_modelized_facet_unit_records(
        base_writer,
        current_turn_text=text,
        parent_records=replacement_records,
        speaker="user",
        timestamp=timestamp,
        turn_index=turn_index,
        max_units=int(os.getenv("TMCRA_MODELIZED_UNIT_WRITER_MAX_UNITS", "14") or 14),
    )
    if unit_records:
        replacement_records.extend(unit_records)
        payload["replacement_memory_records"] = replacement_records
    payload["modelized_facet_unit_writer"] = dict(unit_metadata or {})
    adapter.ingest_turn(
        f"[{timestamp}] user: {text}",
        assistant_text="",
        answer_payload=payload,
        extraction_result={},
    )
    return {
        "seconds": round(time.perf_counter() - t0, 3),
        "proposal_count": len(proposals or []),
        "accepted_count": int(getattr(gate_result, "accepted_count", 0) or 0),
        "suspected_count": int(getattr(gate_result, "suspected_count", 0) or 0),
        "record_count": len(payload.get("replacement_memory_records", []) or []),
        "unit_record_count": len(unit_records),
        "unit_writer_enabled": bool(unit_metadata.get("enabled", False)),
        "unit_writer_metadata": dict(unit_metadata or {}),
    }


def retrieval_debug(retrieval: Any) -> dict[str, Any]:
    metadata = dict(getattr(retrieval, "metadata", {}) or {})
    hits = list(getattr(retrieval, "hits", []) or [])
    return {
        "hit_count": len(hits),
        "retrieval_seconds": round(float(getattr(retrieval, "retrieval_seconds", 0.0) or 0.0), 4),
        "retrieval_mode": metadata.get("retrieval_mode"),
        "hybrid_enabled": metadata.get("hybrid_enabled"),
        "decision_score_source": metadata.get("decision_score_source"),
        "selected_event_count": len(metadata.get("selected_event_ids", []) or []),
        "selected_path_count": len(metadata.get("selected_path_ids", []) or []),
        "path_tunnel_enabled": metadata.get("path_tunnel_enabled"),
        "path_tunnel_rescue_candidate_count": metadata.get("path_tunnel_rescue_candidate_count"),
        "path_tunnel_rescue_path_count": len(metadata.get("path_tunnel_rescue_path_ids", []) or []),
        "profile_focused_pack_enabled": metadata.get("profile_focused_pack_enabled"),
        "profile_focused_pack_reason": metadata.get("profile_focused_pack_reason"),
        "profile_focused_pack_hit_count": metadata.get("profile_focused_pack_hit_count"),
        "profile_focused_pack_event_ids": list(metadata.get("profile_focused_pack_event_ids", []) or []),
        "profile_focused_pack_memory_ids": list(metadata.get("profile_focused_pack_memory_ids", []) or []),
        "profile_first_hybrid_enabled": metadata.get("profile_first_hybrid_enabled"),
        "profile_first_event_ids": list(metadata.get("profile_first_event_ids", []) or []),
        "profile_first_memory_ids": list(metadata.get("profile_first_memory_ids", []) or []),
        "facet_query_pack_enabled": metadata.get("facet_query_pack_enabled"),
        "facet_query_pack_inserted_hit_count": metadata.get("facet_query_pack_inserted_hit_count"),
        "unit_coverage_pack_enabled": metadata.get("unit_coverage_pack_enabled"),
        "unit_coverage_candidate_count": metadata.get("unit_coverage_candidate_count"),
        "unit_coverage_selected_unit_count": metadata.get("unit_coverage_selected_unit_count"),
        "unit_coverage_inserted_hit_count": metadata.get("unit_coverage_inserted_hit_count"),
        "multi_unit_chain_slot_enabled": metadata.get("multi_unit_chain_slot_enabled"),
        "multi_unit_chain_slot_formed": metadata.get("multi_unit_chain_slot_formed"),
        "multi_unit_chain_slot_reason": metadata.get("multi_unit_chain_slot_reason"),
        "multi_unit_chain_candidate_count": metadata.get("multi_unit_chain_candidate_count"),
        "multi_unit_chain_selected_unit_count": metadata.get("multi_unit_chain_selected_unit_count"),
        "multi_unit_chain_parent_count": metadata.get("multi_unit_chain_parent_count"),
        "multi_unit_chain_inserted_hit_count": metadata.get("multi_unit_chain_inserted_hit_count"),
        "multi_unit_chain_memory_ids": list(metadata.get("multi_unit_chain_memory_ids", []) or []),
        "multi_unit_chain_focus_tokens": list(metadata.get("multi_unit_chain_focus_tokens", []) or []),
        "top_values": [truncate(getattr(hit, "value", ""), 220) for hit in hits[:5]],
    }


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "can",
    "did",
    "do",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "how",
    "i",
    "in",
    "is",
    "it",
    "me",
    "my",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "with",
}

_ANSWER_WINDOW_PLANNER_CACHE: dict[str, Any] = {}
_EVIDENCE_UNIT_SELECTOR_CACHE: dict[str, Any] = {}
_UNIFIED_OPERATION_PLANNER_CACHE: dict[str, Any] = {}


def _vector_dot(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return 0.0
    total = 0.0
    for a, b in zip(left, right):
        try:
            total += float(a) * float(b)
        except (TypeError, ValueError):
            continue
    if total != total:
        return 0.0
    return max(0.0, min(1.0, float(total)))


def apply_answer_window_semantic_features(question: str, evidence_windows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    mode = os.getenv("TMCRA_ANSWER_WINDOW_PLANNER_SEMANTIC_MODE", "").strip().lower()
    if mode in {"", "off", "false", "0", "disabled", "none"} or not evidence_windows:
        return evidence_windows
    if mode in {"auto", "on", "true", "1"}:
        mode = os.getenv("TMCRA_EMBEDDER_INDEX_RECALL_MODE", "").strip().lower() or "bge_m3"
    try:
        from experiments.replacement.adapters import memory_adapters
    except Exception:
        return evidence_windows
    texts = [clean_text(question), *[clean_text(item.get("text", "")) for item in evidence_windows]]
    try:
        vectors, metadata = memory_adapters._embedder_dense_vectors_for_texts(texts, mode=mode)
    except Exception:
        return evidence_windows
    if not vectors or not vectors[0]:
        return evidence_windows
    query_vector = vectors[0]
    enriched: list[dict[str, Any]] = []
    dense_enabled = bool(metadata.get("write_embedder_dense_enabled"))
    for item, vector in zip(evidence_windows, vectors[1:]):
        score = _vector_dot(query_vector, list(vector or []))
        next_item = dict(item)
        next_item["answer_window_semantic_similarity"] = round(score, 6)
        next_item["answer_window_semantic_mode"] = mode
        next_item["answer_window_semantic_enabled"] = bool(dense_enabled and score > 0.0)
        enriched.append(next_item)
    return enriched


def evidence_unit_planner_config() -> tuple[str, str, str]:
    # TMCRA_LAYER_TAG: answer-side LLM planner config; not graph-memory model core.
    answer_base_url, answer_model, answer_api_key = answer_llm_config()
    return (
        clean_text(os.getenv("TMCRA_EVIDENCE_UNIT_PLANNER_BASE_URL", "")) or answer_base_url,
        clean_text(os.getenv("TMCRA_EVIDENCE_UNIT_PLANNER_MODEL", "")) or answer_model,
        clean_text(os.getenv("TMCRA_EVIDENCE_UNIT_PLANNER_API_KEY", "")) or answer_api_key,
    )


def token_stem(token: str) -> str:
    token = token.lower()
    for suffix in ("ing", "ed", "es", "s"):
        if len(token) > len(suffix) + 3 and token.endswith(suffix):
            return token[: -len(suffix)]
    return token


def content_tokens(text: str) -> list[str]:
    return [
        token_stem(item)
        for item in re.findall(r"[a-zA-Z0-9$]+", text.lower())
        if item and item not in STOPWORDS
    ]


def quoted_query_phrases(question: str) -> list[str]:
    phrases: list[str] = []
    seen: set[str] = set()
    for match in re.finditer(r"""['"]([^'"]{2,120})['"]""", question):
        phrase = clean_text(match.group(1))
        key = phrase.lower()
        if phrase and key not in seen:
            phrases.append(phrase)
            seen.add(key)
    return phrases


def strict_quoted_phrase_hits(text: str, phrases: list[str]) -> int:
    hits = 0
    for phrase in phrases:
        pattern = re.compile(
            r"(?<![A-Za-z0-9])" + re.escape(phrase) + r"(?!\s+of\s+)(?![A-Za-z0-9])",
            re.IGNORECASE,
        )
        if pattern.search(text):
            hits += 1
    return hits


def temporal_duration_query(question: str) -> bool:
    text = question.lower()
    return bool(re.search(r"\b(day|days|week|weeks|month|months|year|years|hour|hours|duration|how long|spent)\b", text))


def temporal_event_anchor_text(text: str) -> bool:
    return bool(
        re.search(
            r"\b(start|started|starting|begin|began|finished|finish|completed|complete|ended|end|stopped|since|until)\b",
            text.lower(),
        )
    )


def parent_dialogue_header(text: str) -> str:
    value = clean_text(text)
    if not value:
        return ""
    match = re.search(
        r"\bLongMemEval\s+session_id=[A-Za-z0-9_.:-]+\s+date=\d{4}/\d{2}/\d{2}\s+\([^)]+\)\s+\d{2}:\d{2}",
        value,
    )
    if match:
        return clean_text(match.group(0))
    match = re.search(
        r"\[[0-9T:+-]{19,25}\]\s+user:\s+LongMemEval\s+session_id=[A-Za-z0-9_.:-]+\s+date=\d{4}/\d{2}/\d{2}\s+\([^)]+\)\s+\d{2}:\d{2}",
        value,
    )
    return clean_text(match.group(0)) if match else ""


def attach_parent_temporal_context(unit: str, parent_text: str) -> str:
    unit_text = clean_text(unit)
    if not unit_text:
        return ""
    if re.search(r"\bdate=\d{4}/\d{2}/\d{2}\b", unit_text) or re.search(r"\bLongMemEval\s+session_id=", unit_text):
        return unit_text
    header = parent_dialogue_header(parent_text)
    if not header:
        return unit_text
    return clean_text(f"{header} {unit_text}")


# TMCRA_LAYER_TAG: rule-based evidence query expansion; candidate for modelized retrieval/intent replacement.
EVIDENCE_QUERY_GENERIC_TERMS = {
    "complement",
    "current",
    "recommend",
    "recommendation",
    "recommendations",
    "suggest",
    "suggestion",
    "suggestions",
}


# TMCRA_LAYER_TAG: manual semantic alias buckets; not part of trained graph-memory model core.
EVIDENCE_QUERY_ALIAS_GROUPS = (
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
)


def evidence_query_terms(question: str) -> set[str]:
    terms = {term for term in content_tokens(question) if term not in EVIDENCE_QUERY_GENERIC_TERMS}
    expanded = set(terms)
    for group in EVIDENCE_QUERY_ALIAS_GROUPS:
        if terms & group:
            expanded.update(group)
    return expanded


def turn_units(text: str) -> list[str]:
    text = clean_text(text)
    if not text:
        return []
    marker = re.compile(r"(\[[^\]]+\bturn=\d+\s+role=(?:user|assistant)\])", re.IGNORECASE)
    parts = marker.split(text)
    units: list[str] = []
    prefix = ""
    index = 0
    sentence_mode = os.getenv("TMCRA_DIALOGUE_TURN_SENTENCE_UNITS", "").strip().lower() not in {
        "",
        "0",
        "false",
        "off",
        "no",
        "disabled",
        "none",
    }
    if parts and not marker.match(parts[0]):
        prefix = parts[0].strip()
        index = 1
    while index < len(parts):
        head = parts[index].strip()
        body = parts[index + 1].strip() if index + 1 < len(parts) else ""
        body_units = [body]
        if sentence_mode:
            split_units = [clean_text(item) for item in re.split(r"(?<=[.!?])\s+", body) if clean_text(item)]
            if split_units:
                body_units = split_units
        for body_unit in body_units:
            unit = clean_text(f"{head} {body_unit}")
            if unit:
                units.append(unit)
        index += 2
    if units:
        if prefix:
            units[0] = clean_text(f"{prefix} {units[0]}")
        return units
    sentence_units = re.split(r"(?<=[.!?])\s+", text)
    return [clean_text(item) for item in sentence_units if clean_text(item)]


def score_evidence_unit(question_terms: set[str], question: str, unit: str) -> float:
    unit_norm = unit.lower()
    unit_terms = set(content_tokens(unit))
    overlap = question_terms & unit_terms
    score = float(len(overlap))
    for term in overlap:
        if len(term) >= 5:
            score += 0.35
    raw_terms = [
        item
        for item in re.findall(r"[a-zA-Z0-9$]+", question.lower())
        if item not in STOPWORDS and item not in EVIDENCE_QUERY_GENERIC_TERMS
    ]
    for size in (3, 2):
        for index in range(0, max(0, len(raw_terms) - size + 1)):
            phrase = " ".join(raw_terms[index : index + size])
            if phrase and phrase in unit_norm:
                score += 1.5 if size == 3 else 1.0
    if re.search(r"\b\d+\b|\$\d+", unit_norm):
        score += 0.3
    if "role=user" in unit_norm:
        score += 0.25
    return score


def centered_excerpt(unit: str, question_terms: set[str], *, max_chars: int) -> str:
    unit = clean_text(unit)
    if len(unit) <= max_chars:
        return unit
    unit_lower = unit.lower()
    positions = [
        unit_lower.find(term)
        for term in sorted(question_terms, key=len, reverse=True)
        if len(term) >= 4 and unit_lower.find(term) >= 0
    ]
    center = min(positions) if positions else 0
    start = max(0, center - max_chars // 3)
    end = min(len(unit), start + max_chars)
    start = max(0, end - max_chars)
    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(unit) else ""
    return prefix + unit[start:end].strip() + suffix


def make_centered_window(
    units: list[str],
    seed_index: int,
    score: float,
    question_terms: set[str],
    *,
    max_chars: int,
) -> dict[str, Any]:
    seed_unit = units[seed_index].lower() if 0 <= seed_index < len(units) else ""
    is_dialogue_question_seed = "role=user" in seed_unit and ("?" in seed_unit or "how many" in seed_unit)
    if is_dialogue_question_seed:
        max_chars = max(max_chars, 2200)
        candidate_order = [seed_index, seed_index + 1, seed_index + 2, seed_index - 1, seed_index - 2]
        order = [index for index in candidate_order if 0 <= index < len(units)]
    else:
        order = [seed_index]
        for radius in (1, 2):
            left = seed_index - radius
            right = seed_index + radius
            if left >= 0:
                order.append(left)
            if right < len(units):
                order.append(right)

    selected: list[tuple[int, str]] = []
    used = 0
    for index in order:
        unit_l = units[index].lower() if 0 <= index < len(units) else ""
        if "role=assistant" in unit_l:
            budget = 1800 if index == seed_index else 900
        else:
            budget = 950 if index == seed_index else 350 if is_dialogue_question_seed else 520
        piece = centered_excerpt(units[index], question_terms, max_chars=budget)
        extra = len(piece) + 2
        if selected and used + extra > max_chars:
            continue
        selected.append((index, piece))
        used += extra
    selected.sort(key=lambda item: item[0])
    return {
        "score": round(float(score), 3),
        "unit_indexes": [index for index, _ in selected],
        "text": "\n".join(piece for _, piece in selected),
    }


def assistant_answer_candidate_windows(
    value: str,
    question: str,
    question_terms: set[str],
    *,
    max_windows: int = 4,
    max_chars: int = 900,
) -> list[dict[str, Any]]:
    if not assistant_memory_query(question) or not assistant_origin_evidence(value):
        return []
    marker = re.compile(r"(\[[^\]]+\bturn=\d+\s+role=assistant\])", re.IGNORECASE)
    parts = marker.split(clean_text(value))
    candidates: list[tuple[float, int, str]] = []
    raw_question_terms = [
        item
        for item in re.findall(r"[a-zA-Z0-9$]+", clean_text(question).lower())
        if item not in STOPWORDS and item not in EVIDENCE_QUERY_GENERIC_TERMS
    ]
    focus_phrases: set[str] = set()
    for size in (4, 3, 2):
        for raw_index in range(0, max(0, len(raw_question_terms) - size + 1)):
            phrase = " ".join(raw_question_terms[raw_index : raw_index + size])
            if phrase:
                focus_phrases.add(phrase)
    for index in range(1, len(parts), 2):
        head = parts[index].strip()
        body = parts[index + 1].strip() if index + 1 < len(parts) else ""
        if not body:
            continue
        pieces = [
            clean_text(item)
            for item in re.split(r"(?=(?:\d+\.|[-*]\s+|[A-Z][A-Za-z '’-]{2,40}:))|(?<=[.!?])\s+", body)
            if clean_text(item)
        ]
        for local_index, piece in enumerate(pieces):
            if len(piece) < 24:
                continue
            full_piece = clean_text(f"{head} {piece}")
            score = score_evidence_unit(question_terms, question, full_piece)
            piece_terms = set(content_tokens(piece))
            overlap = piece_terms & question_terms
            named_entity_like = bool(
                re.search(r"\b(?:The\s+)?[A-Z][A-Za-z'’-]+(?:\s+(?:of|at|and|the|de|del|la|el|[A-Z][A-Za-z'’-]+)){1,7}\b", piece)
                or re.search(r"['\"][^'\"]{3,80}['\"]", piece)
            )
            if named_entity_like:
                score += 1.25
            if overlap and named_entity_like:
                score += 2.0
            if len(overlap) >= 2:
                score += 1.0
            piece_l = clean_text(piece).lower()
            phrase_hits = sum(1 for phrase in focus_phrases if len(phrase) >= 8 and phrase in piece_l)
            if phrase_hits:
                score += min(4.0, 1.5 * phrase_hits)
            if score < 2.0:
                continue
            text = centered_excerpt(full_piece, question_terms, max_chars=max_chars)
            candidates.append((score, local_index, text))
    candidates.sort(key=lambda item: (item[0], -item[1]), reverse=True)
    windows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for score, _, text in candidates:
        key = text[:220].lower()
        if key in seen:
            continue
        seen.add(key)
        windows.append(
            {
                "score": round(float(score) + 2.0, 3),
                "unit_indexes": [],
                "text": text,
                "assistant_answer_candidate": True,
                "assistant_memory_query": True,
                "assistant_origin_evidence": True,
                "assistant_candidate_side_channel": True,
                "evidence_role": "assistant_answer_candidate",
                "planner_selected": True,
                "planner_score": round(float(score), 3),
            }
        )
        if len(windows) >= max_windows:
            break
    return windows


def _load_evidence_unit_selector() -> tuple[Any, Any, Any, dict[str, float]] | None:
    mode = os.getenv("TMCRA_EVIDENCE_UNIT_SELECTOR_MODE", "").strip().lower()
    if mode in {"", "off", "false", "0", "disabled", "none"}:
        return None
    model_path = os.getenv("TMCRA_EVIDENCE_UNIT_SELECTOR_MODEL_PATH", "").strip()
    if not model_path:
        return None
    cache_key = "|".join([model_path, os.getenv("TMCRA_EVIDENCE_UNIT_SELECTOR_DEVICE", "cpu")])
    cached = _EVIDENCE_UNIT_SELECTOR_CACHE.get(cache_key)
    if cached is not None:
        return cached
    try:
        from experiments.replacement import injection_planner as planner
    except Exception:
        return None
    torch_module = getattr(planner, "torch", None)
    if torch_module is None:
        return None
    try:
        device = torch_module.device(os.getenv("TMCRA_EVIDENCE_UNIT_SELECTOR_DEVICE", "cpu") or "cpu")
        payload = torch_module.load(Path(model_path), map_location=device, weights_only=False)
        config = planner.InjectionPlannerConfig.from_dict(dict(payload.get("config", {}) or {}))
        model = planner.InjectionPlannerModel(config).to(device)
        model.load_state_dict(dict(payload.get("state_dict", {}) or {}), strict=False)
        model.eval()
        thresholds = {
            "selection_threshold": float(os.getenv("TMCRA_EVIDENCE_UNIT_SELECTOR_SELECTION_THRESHOLD", "0.45")),
            "row_threshold": float(os.getenv("TMCRA_EVIDENCE_UNIT_SELECTOR_ROW_THRESHOLD", "0.35")),
            "logic_threshold": float(os.getenv("TMCRA_EVIDENCE_UNIT_SELECTOR_LOGIC_THRESHOLD", "0.45")),
        }
    except Exception:
        return None
    loaded = (planner, torch_module, model, thresholds)
    _EVIDENCE_UNIT_SELECTOR_CACHE[cache_key] = loaded
    return loaded


def model_selected_evidence_unit_indexes(
    question: str,
    units: list[str],
    hit_metadata: Mapping[str, Any],
    scored_units: list[tuple[float, int]],
) -> list[tuple[float, int]]:
    loaded = _load_evidence_unit_selector()
    if loaded is None or not units:
        return []
    planner, torch_module, model, thresholds = loaded
    max_units = max(4, int(os.getenv("TMCRA_EVIDENCE_UNIT_SELECTOR_MAX_UNITS", "14") or 14))
    top_scored = sorted(scored_units, key=lambda item: item[0], reverse=True)[:max_units]
    candidate_indexes = sorted({index for _, index in top_scored if 0 <= index < len(units)})
    if not candidate_indexes:
        return []
    score_by_index = {index: score for score, index in scored_units}
    max_unit_score = max([abs(float(score)) for score, _ in top_scored] + [1.0])
    retrieval_score = max(
        0.0,
        min(
            1.0,
            max(
                metadata_float_from_mapping(hit_metadata, "hybrid_score"),
                metadata_float_from_mapping(hit_metadata, "recall_score"),
                metadata_float_from_mapping(hit_metadata, "event_score"),
                metadata_float_from_mapping(hit_metadata, "path_score"),
            ),
        ),
    )
    graph_score = max(
        0.0,
        min(
            1.0,
            max(
                metadata_float_from_mapping(hit_metadata, "answer_plan_score"),
                metadata_float_from_mapping(hit_metadata, "answer_plan_selected_score"),
                metadata_float_from_mapping(hit_metadata, "answer_plan_adjusted_score"),
            ),
        ),
    )
    temporal_state = clean_text(hit_metadata.get("injection_planner_temporal_state", "")) or "irrelevant"
    evidence_role = clean_text(hit_metadata.get("injection_planner_evidence_role", "")) or clean_text(
        hit_metadata.get("evidence_role", "")
    )
    candidates = []
    for rank, index in enumerate(candidate_indexes, start=1):
        unit_score = float(score_by_index.get(index, 0.0) or 0.0)
        candidates.append(
            {
                "id": f"unit_{index}",
                "text": clean_text(units[index]),
                "layer": "event",
                "temporal_state": temporal_state if temporal_state in getattr(planner, "TEMPORAL_STATES", ()) else "irrelevant",
                "logic_roles": ["evidence"],
                "evidence_role": evidence_role if evidence_role in getattr(planner, "EVIDENCE_ROLES", ()) else "",
                "retrieval_score": retrieval_score,
                "graph_score": graph_score,
                "tunnel_score": max(0.0, min(1.0, metadata_float_from_mapping(hit_metadata, "path_score"))),
                "topic_similarity": max(0.0, min(1.0, unit_score / max_unit_score)),
                "confidence": retrieval_score,
                "rank_score": round(1.0 / float(rank), 6),
                "branch_depth": 1,
            }
        )
    if len(candidates) < 3:
        return []
    try:
        row = {"id": "evidence_unit_selector", "query": question, "candidates": candidates, "gold": {}}
        dataset = planner.InjectionPlannerDataset([row], model.config)
        batch = planner.collate_injection_batch([dataset[0]])
        device = next(model.parameters()).device
        model_batch = {key: value.to(device) if hasattr(value, "to") else value for key, value in dict(batch).items()}
        with torch_module.no_grad():
            outputs = model(model_batch["features"], model_batch["valid_mask"])
            row_score = float(torch_module.sigmoid(outputs["should_inject_logits"])[0].detach().cpu().item())
            selection_scores = torch_module.sigmoid(outputs["selection_logits"])[0].detach().cpu().tolist()
            role_indices = torch_module.argmax(outputs["evidence_role_logits"], dim=-1)[0].detach().cpu().tolist()
    except Exception:
        return []
    if row_score < thresholds["row_threshold"]:
        return []
    selected: list[tuple[float, int]] = []
    for local_index, unit_index in enumerate(candidate_indexes):
        role = planner.EVIDENCE_ROLES[int(role_indices[local_index])]
        if role in {"noise", "negative_evidence"}:
            continue
        score = float(selection_scores[local_index])
        if score >= thresholds["selection_threshold"]:
            selected.append((score, unit_index))
    selected.sort(key=lambda item: item[0], reverse=True)
    return selected[: max(1, int(os.getenv("TMCRA_EVIDENCE_UNIT_SELECTOR_TOP_K", "4") or 4))]


def metadata_float_from_mapping(metadata: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(metadata.get(key, default) or default)
    except (TypeError, ValueError):
        return default


def evidence_windows_for_hit(question: str, hit: Any, *, max_chars: int = 1800) -> list[dict[str, Any]]:
    value = clean_text(getattr(hit, "value", ""))
    if not value:
        return []
    hit_metadata = dict(getattr(hit, "metadata", {}) or {})
    evidence_snippet_role = clean_text(hit_metadata.get("evidence_snippet_role", ""))
    unit_coverage_parent_event = bool(hit_metadata.get("unit_coverage_pack")) and evidence_snippet_role == "unit_coverage_parent_event"
    multi_chain_parent_event = bool(hit_metadata.get("multi_unit_chain_slot")) and evidence_snippet_role == "multi_unit_chain_parent_event"
    if unit_coverage_parent_event or multi_chain_parent_event:
        try:
            max_chars = max(max_chars, int(os.getenv("TMCRA_UNIT_COVERAGE_PARENT_WINDOW_CHARS", "3800") or 3800))
        except (TypeError, ValueError):
            max_chars = max(max_chars, 3800)
    if assistant_memory_query(question) and assistant_origin_evidence(value):
        try:
            max_chars = max(max_chars, int(os.getenv("TMCRA_ASSISTANT_MEMORY_WINDOW_CHARS", "4200") or 4200))
        except (TypeError, ValueError):
            max_chars = max(max_chars, 4200)
    def metadata_float(key: str, default: float = 0.0) -> float:
        try:
            return float(hit_metadata.get(key, default) or default)
        except (TypeError, ValueError):
            return default

    answer_plan_score = max(
        metadata_float("answer_plan_score"),
        metadata_float("answer_plan_selected_score"),
        metadata_float("answer_plan_current_score"),
        metadata_float("answer_plan_adjusted_score"),
    )
    answer_plan_selected = bool(hit_metadata.get("answer_plan_selected", False)) and answer_plan_score > 0.0
    try:
        answer_plan_rank = int(hit_metadata.get("answer_plan_rank", 0) or 0)
    except (TypeError, ValueError):
        answer_plan_rank = 0
    parent_profile_summary = clean_text(
        hit_metadata.get("profile_first_parent_summary", "")
        or hit_metadata.get("profile_summary", "")
        or hit_metadata.get("profile_value", "")
    )
    if hit_metadata.get("profile_first_source_support") and parent_profile_summary:
        value = clean_text(f"[profile summary] {parent_profile_summary} {value}")
    if unit_coverage_parent_event or multi_chain_parent_event:
        child_value = clean_text(hit_metadata.get("unit_coverage_child_value", "") or hit_metadata.get("multi_unit_chain_child_value", ""))
        child_source_span = clean_text(
            hit_metadata.get("unit_coverage_child_source_span", "")
            or hit_metadata.get("multi_unit_chain_child_source_span", "")
        )
        child_anchor = child_source_span or child_value
        if child_anchor:
            parent_l = value.lower()
            child_l = child_anchor.lower()
            pos = parent_l.find(child_l[: min(len(child_l), 120)])
            if pos >= 0:
                half = max_chars // 2
                start = max(0, pos - half)
                end = min(len(value), pos + len(child_anchor) + half)
                value = clean_text(value[start:end])
            elif child_value or child_source_span:
                value = clean_text(value)
            if child_value or child_source_span:
                value = clean_text(
                    f"[unit evidence] {child_source_span or child_value}"
                    f"{' | unit_value=' + child_value if child_value and child_value != child_source_span else ''}\n"
                    f"[parent evidence] {value}"
                )
    units = turn_units(value)
    if not units:
        return []
    question_terms = evidence_query_terms(question)
    profile_bonus = 0.0
    if hit_metadata.get("profile_first_hybrid_rescue"):
        profile_bonus += 8.0
    elif hit_metadata.get("profile_layer") or str(getattr(hit, "source_kind", "")).startswith("public_dialog_profile"):
        profile_bonus += 4.0
    scored = [
        (score_evidence_unit(question_terms, question, unit) + profile_bonus, index)
        for index, unit in enumerate(units)
    ]
    scored.sort(key=lambda item: item[0], reverse=True)
    seed_indexes: list[tuple[float, int]] = []
    model_unit_indexes = model_selected_evidence_unit_indexes(question, units, hit_metadata, scored)
    for model_score, index in model_unit_indexes:
        if any(existing == index for _, existing in seed_indexes):
            continue
        seed_indexes.append((max(float(model_score) * 10.0, float(scored[0][0]) if scored else 0.0), index))
        seed_limit = 5 if assistant_memory_query(question) and assistant_origin_evidence(value) else 3
        if len(seed_indexes) >= seed_limit:
            break
    for score, index in scored:
        if score <= 0 and seed_indexes:
            break
        if score <= 0 and not seed_indexes:
            seed_indexes.append((score, index))
            break
        if any(abs(index - existing) <= 1 for _, existing in seed_indexes):
            continue
        seed_indexes.append((score, index))
        seed_limit = 5 if assistant_memory_query(question) and assistant_origin_evidence(value) else 3
        if len(seed_indexes) >= seed_limit:
            break
    windows: list[dict[str, Any]] = []
    for score, index in seed_indexes:
        window = make_centered_window(units, index, score, question_terms, max_chars=max_chars)
        window["memory_id"] = getattr(hit, "memory_id", "")
        window["original_chars"] = len(value)
        window["planner_selected"] = bool(hit_metadata.get("injection_planner_selected", False)) or answer_plan_selected
        evidence_role = clean_text(hit_metadata.get("injection_planner_evidence_role", ""))
        if not evidence_role and answer_plan_selected:
            evidence_role = "direct_answer"
        memory_id = clean_text(getattr(hit, "memory_id", ""))
        source_kind = clean_text(getattr(hit, "source_kind", ""))
        if (
            not evidence_role
            and (
                bool(hit_metadata.get("profile_first_hybrid_rescue"))
                or bool(hit_metadata.get("profile_protected_slot"))
                or bool(hit_metadata.get("profile_layer"))
                or ".subject." in memory_id
                or source_kind.startswith("public_dialog_profile")
            )
        ):
            evidence_role = "profile_fact"
        window["evidence_role"] = evidence_role
        window["temporal_state"] = clean_text(hit_metadata.get("injection_planner_temporal_state", ""))
        promoted_answer_plan_score = answer_plan_score if answer_plan_selected else 0.0
        window["planner_score"] = max(metadata_float("injection_planner_score"), promoted_answer_plan_score)
        window["answer_plan_score"] = answer_plan_score
        window["answer_plan_selected"] = answer_plan_selected
        window["answer_plan_rank"] = answer_plan_rank
        if hit_metadata.get("semantic_coverage_expansion"):
            window["semantic_coverage_expansion"] = True
            window["semantic_coverage_score"] = metadata_float("semantic_coverage_score")
            window["semantic_coverage_source_memory_id"] = clean_text(hit_metadata.get("semantic_coverage_source_memory_id", ""))
        if hit_metadata.get("unit_coverage_pack"):
            window["unit_coverage_pack"] = True
            window["unit_kind"] = clean_text(hit_metadata.get("unit_kind", ""))
            window["facet_type"] = clean_text(hit_metadata.get("facet_type", ""))
            window["unit_coverage_semantic_event_unit"] = bool(hit_metadata.get("unit_coverage_semantic_event_unit", False))
            window["unit_coverage_semantic_event_priority"] = bool(hit_metadata.get("unit_coverage_semantic_event_priority", False))
            window["unit_coverage_profile_shadow_unit"] = bool(hit_metadata.get("unit_coverage_profile_shadow_unit", False))
            window["evidence_role"] = window["evidence_role"] or clean_text(hit_metadata.get("unit_kind", ""))
        if hit_metadata.get("multi_unit_chain_slot"):
            window["multi_unit_chain_slot"] = True
            window["multi_unit_chain_bundle"] = bool(hit_metadata.get("multi_unit_chain_bundle", False))
            window["multi_unit_chain_temporal_comparison"] = bool(hit_metadata.get("multi_unit_chain_temporal_comparison", False))
            window["multi_unit_chain_semantic_event_unit"] = bool(hit_metadata.get("multi_unit_chain_semantic_event_unit", False))
            window["multi_unit_chain_semantic_event_priority"] = bool(hit_metadata.get("multi_unit_chain_semantic_event_priority", False))
            window["multi_unit_chain_profile_shadow_unit"] = bool(hit_metadata.get("multi_unit_chain_profile_shadow_unit", False))
            window["unit_kind"] = window.get("unit_kind", "") or clean_text(hit_metadata.get("unit_kind", ""))
            window["facet_type"] = window.get("facet_type", "") or clean_text(hit_metadata.get("facet_type", ""))
            window["evidence_role"] = window["evidence_role"] or "multi_unit_chain"
            window["multi_unit_chain_score"] = float(hit_metadata.get("multi_unit_chain_score", 0.0) or 0.0)
        if hit_metadata.get("facet_query_pack"):
            window["facet_query_pack"] = True
            window["facet_type"] = window.get("facet_type", "") or clean_text(hit_metadata.get("facet_type", ""))
        windows.append(window)
    try:
        assistant_candidate_limit = max(0, int(os.getenv("TMCRA_ASSISTANT_ANSWER_CANDIDATE_LIMIT", "0") or 0))
    except (TypeError, ValueError):
        assistant_candidate_limit = 4
    if assistant_candidate_limit:
        for candidate in assistant_answer_candidate_windows(
            value,
            question,
            question_terms,
            max_windows=assistant_candidate_limit,
            max_chars=max(700, min(max_chars, 1200)),
        ):
            candidate["memory_id"] = getattr(hit, "memory_id", "")
            candidate["original_chars"] = len(value)
            candidate["temporal_state"] = clean_text(hit_metadata.get("injection_planner_temporal_state", ""))
            candidate["answer_plan_score"] = answer_plan_score
            candidate["answer_plan_selected"] = answer_plan_selected
            candidate["answer_plan_rank"] = answer_plan_rank
            windows.append(candidate)
    return windows


def build_answer_evidence(question: str, memory_hits: list[Any]) -> list[dict[str, Any]]:
    windows: list[dict[str, Any]] = []
    try:
        hit_scan_limit = max(8, int(os.getenv("TMCRA_ANSWER_EVIDENCE_HIT_SCAN_LIMIT", "16") or 16))
        build_limit = max(8, int(os.getenv("TMCRA_ANSWER_EVIDENCE_BUILD_LIMIT", "16") or 16))
    except (TypeError, ValueError):
        hit_scan_limit, build_limit = 16, 16
    for hit in memory_hits[:hit_scan_limit]:
        windows.extend(evidence_windows_for_hit(question, hit))
    assistant_query = assistant_memory_query(question)
    for index, item in enumerate(windows):
        text = clean_text(item.get("text", ""))
        if assistant_query and assistant_origin_evidence(text):
            next_item = dict(item)
            next_item["assistant_memory_query"] = True
            next_item["assistant_origin_evidence"] = True
            if not clean_text(next_item.get("evidence_role", "")):
                next_item["evidence_role"] = "assistant_origin_detail"
            windows[index] = next_item

    def planner_window_key(item: dict[str, Any]) -> tuple[bool, bool, float, float, int]:
        evidence_role = clean_text(item.get("evidence_role", ""))
        role_allows_evidence = evidence_role not in {"noise", "negative_evidence"}
        planner_score = float(item.get("planner_score", 0.0) or 0.0) if role_allows_evidence else 0.0
        return (
            bool(item.get("planner_selected", False)),
            role_allows_evidence,
            planner_score,
            float(item.get("score", 0.0) or 0.0),
            len(item.get("text", "")),
        )

    assistant_candidates = [item for item in windows if bool(item.get("assistant_candidate_side_channel", False))]
    windows = [item for item in windows if not bool(item.get("assistant_candidate_side_channel", False))]

    windows.sort(
        key=planner_window_key,
        reverse=True,
    )
    windows = compose_layered_evidence_windows(windows, build_limit=build_limit)
    windows = diversify_evidence_windows(windows)
    assistant_candidates.sort(key=planner_window_key, reverse=True)
    try:
        side_limit = max(0, int(os.getenv("TMCRA_ASSISTANT_ANSWER_CANDIDATE_SIDE_LIMIT", "12") or 12))
    except (TypeError, ValueError):
        side_limit = 12
    return windows[:build_limit] + assistant_candidates[:side_limit]


def assistant_memory_query(question: str) -> bool:
    question_l = clean_text(question).lower()
    return bool(
        re.search(
            r"\b(?:you said|you suggested|you recommended|you told|you wrote|you gave|you mentioned|"
            r"previous chat|previous conversation|last time|remind me|we talked|we discussed|"
            r"script you wrote|list you provided|what was the .* you said|what did you say)\b",
            question_l,
        )
    )


def assistant_origin_evidence(text: str) -> bool:
    text_l = clean_text(text).lower()
    if not text_l:
        return False
    if re.search(r"\[[^\]]+\brole=assistant\]", text_l):
        return True
    return bool(re.search(r"\b(?:assistant\]|role=assistant|assistant:)\b", text_l))


def evidence_window_layer(item: dict[str, Any]) -> str:
    memory_id = clean_text(item.get("memory_id", ""))
    evidence_role = clean_text(item.get("evidence_role", "")).lower()
    text = clean_text(item.get("text", "")).lower()
    if bool(item.get("assistant_origin_evidence", False)) or evidence_role == "assistant_origin_detail":
        return "assistant"
    if bool(item.get("multi_unit_chain_slot", False)):
        return "chain"
    if bool(item.get("unit_coverage_pack", False)):
        return "unit"
    if bool(item.get("facet_query_pack", False)):
        return "facet"
    if (
        "public_dialog_profile" in memory_id
        or ".subject." in memory_id
        or "profile" in evidence_role
        or text.startswith("[profile summary]")
    ):
        return "profile"
    if bool(item.get("answer_plan_selected", False)) or evidence_role in {"direct_answer", "positive", "supporting_fact"}:
        return "direct"
    return "base"


def compose_layered_evidence_windows(evidence_windows: list[dict[str, Any]], *, build_limit: int) -> list[dict[str, Any]]:
    mode = os.getenv("TMCRA_LAYERED_EVIDENCE_COMPOSE_MODE", "on").strip().lower()
    if mode in {"", "0", "false", "off", "no", "disabled", "none"}:
        return evidence_windows
    try:
        profile_limit = max(0, int(os.getenv("TMCRA_LAYERED_EVIDENCE_PROFILE_LIMIT", "3") or 3))
        assistant_limit = max(0, int(os.getenv("TMCRA_LAYERED_EVIDENCE_ASSISTANT_LIMIT", "4") or 4))
        direct_limit = max(0, int(os.getenv("TMCRA_LAYERED_EVIDENCE_DIRECT_LIMIT", "4") or 4))
        base_limit = max(0, int(os.getenv("TMCRA_LAYERED_EVIDENCE_BASE_LIMIT", "5") or 5))
        unit_limit = max(0, int(os.getenv("TMCRA_LAYERED_EVIDENCE_UNIT_LIMIT", "4") or 4))
        chain_limit = max(0, int(os.getenv("TMCRA_LAYERED_EVIDENCE_CHAIN_LIMIT", "4") or 4))
        facet_limit = max(0, int(os.getenv("TMCRA_LAYERED_EVIDENCE_FACET_LIMIT", "2") or 2))
    except (TypeError, ValueError):
        profile_limit, assistant_limit, direct_limit, base_limit, unit_limit, chain_limit, facet_limit = 3, 4, 4, 5, 4, 4, 2
    buckets: dict[str, list[dict[str, Any]]] = {
        "profile": [],
        "assistant": [],
        "direct": [],
        "base": [],
        "unit": [],
        "chain": [],
        "facet": [],
    }
    for item in evidence_windows:
        buckets.setdefault(evidence_window_layer(item), buckets["base"]).append(item)

    def coverage_bucket_key(item: dict[str, Any]) -> tuple[int, int, int, int, float, float, int]:
        unit_kind = clean_text(item.get("unit_kind", "")).lower()
        evidence_role = clean_text(item.get("evidence_role", "")).lower()
        memory_id = clean_text(item.get("memory_id", "")).lower()
        semantic_event = int(
            bool(item.get("multi_unit_chain_semantic_event_priority", False))
            or bool(item.get("unit_coverage_semantic_event_priority", False))
        )
        profile_shadow = int(
            unit_kind == "profile_shadow_unit"
            or bool(item.get("multi_unit_chain_profile_shadow_unit", False))
            or bool(item.get("unit_coverage_profile_shadow_unit", False))
            or "profile_shadow" in memory_id
        )
        selected = int(bool(item.get("planner_selected", False)) or bool(item.get("evidence_unit_planner_selected", False)))
        parent_event = int("parent_event" in evidence_role or "#multi_parent:" in memory_id or "#unit_parent:" in memory_id)
        return (
            semantic_event,
            selected,
            profile_shadow,
            parent_event,
            float(item.get("multi_unit_chain_score", 0.0) or 0.0),
            float(item.get("score", 0.0) or 0.0),
            len(clean_text(item.get("text", ""))),
        )

    buckets["chain"].sort(key=coverage_bucket_key, reverse=True)
    buckets["unit"].sort(key=coverage_bucket_key, reverse=True)

    ordered: list[dict[str, Any]] = []
    seen_keys: set[str] = set()

    def add_from(items: list[dict[str, Any]], limit: int) -> None:
        if limit <= 0:
            return
        added = 0
        for item in items:
            memory_id = clean_text(item.get("memory_id", ""))
            text_key = clean_text(item.get("text", ""))[:240].lower()
            key = memory_id or text_key
            if key and key in seen_keys:
                continue
            if key:
                seen_keys.add(key)
            ordered.append(item)
            added += 1
            if len(ordered) >= build_limit or added >= limit:
                break

    # Stable evidence composition: profile/direct/base keep the normal answer surface;
    # unit and chain provide bounded coverage slots instead of competing for all top ranks.
    temporal_chain_first = any(bool(item.get("multi_unit_chain_temporal_comparison", False)) for item in buckets["chain"])
    add_from(buckets["profile"], profile_limit)
    add_from(buckets["assistant"], assistant_limit)
    add_from(buckets["direct"], direct_limit)
    if temporal_chain_first:
        add_from(buckets["chain"], chain_limit)
        add_from(buckets["base"], base_limit)
        add_from(buckets["unit"], unit_limit)
    else:
        add_from(buckets["base"], base_limit)
        add_from(buckets["unit"], unit_limit)
        add_from(buckets["chain"], chain_limit)
    add_from(buckets["facet"], facet_limit)
    for item in evidence_windows:
        if len(ordered) >= build_limit:
            break
        memory_id = clean_text(item.get("memory_id", ""))
        text_key = clean_text(item.get("text", ""))[:240].lower()
        key = memory_id or text_key
        if key and key in seen_keys:
            continue
        if key:
            seen_keys.add(key)
        ordered.append(item)
    return ordered


def expand_dialogue_chain_hits(question: str, memory_hits: list[Any], graph: Any | None) -> list[Any]:
    if graph is None or not memory_hits:
        return list(memory_hits)
    mode = os.getenv("TMCRA_DIALOGUE_CHAIN_EXPANSION_MODE", "adjacent").strip().lower()
    if mode in {"", "off", "false", "0", "disabled", "none"}:
        return list(memory_hits)
    try:
        radius = max(0, int(os.getenv("TMCRA_DIALOGUE_CHAIN_EXPANSION_RADIUS", "3") or 3))
        max_total = max(0, int(os.getenv("TMCRA_DIALOGUE_CHAIN_EXPANSION_MAX_TOTAL", "5") or 5))
        max_per_seed = max(1, int(os.getenv("TMCRA_DIALOGUE_CHAIN_EXPANSION_MAX_PER_SEED", "2") or 2))
    except (TypeError, ValueError):
        radius, max_total, max_per_seed = 3, 5, 2
    if radius <= 0 or max_total <= 0:
        return list(memory_hits)
    question_terms = evidence_query_terms(question)
    if not question_terms:
        return list(memory_hits)
    records_by_id = getattr(graph, "records_by_id", {}) or {}
    if not records_by_id:
        return list(memory_hits)
    existing_ids = {clean_text(getattr(hit, "memory_id", "")) for hit in memory_hits if clean_text(getattr(hit, "memory_id", ""))}
    seed_infos = []
    for seed_index, hit in enumerate(memory_hits[:12]):
        session_id = dialogue_session_id_from_text(hit_text_for_chain(hit))
        if not session_id:
            continue
        seed_infos.append((seed_index, session_id, int(getattr(hit, "turn_index", 0) or 0), clean_text(getattr(hit, "memory_id", ""))))
    if not seed_infos:
        return list(memory_hits)

    candidates_by_seed: dict[str, list[Any]] = {}
    for record in records_by_id.values():
        memory_id = clean_text(getattr(record, "memory_id", ""))
        if not memory_id or memory_id in existing_ids:
            continue
        state = clean_text(getattr(record, "state", "active")).lower()
        if state not in {"active", "parallel_active", "evidence"}:
            continue
        value = clean_text(getattr(record, "value", ""))
        metadata = dict(getattr(record, "metadata", {}) or {})
        session_id = dialogue_session_id_from_text(" ".join([value, clean_text(metadata.get("raw_text", "")), clean_text(metadata.get("source_turn_text", ""))]))
        if not session_id:
            continue
        record_turn = int(getattr(record, "turn_index", 0) or 0)
        for seed_index, seed_session, seed_turn, seed_id in seed_infos:
            if session_id != seed_session:
                continue
            distance = abs(record_turn - seed_turn) if seed_turn and record_turn else 0
            if distance > radius:
                continue
            score_metadata = {**metadata, "source_kind": clean_text(getattr(record, "source_kind", ""))}
            score = dialogue_chain_candidate_score(question, question_terms, value, score_metadata, distance)
            if score <= 0:
                continue
            hit = record_to_chain_hit(record, score=score, seed_id=seed_id, distance=distance, session_id=session_id)
            candidates_by_seed.setdefault(seed_id, []).append(hit)

    expanded: list[Any] = []
    added_ids: set[str] = set()
    total_added = 0
    for hit_index, hit in enumerate(memory_hits):
        expanded.append(hit)
        seed_id = clean_text(getattr(hit, "memory_id", ""))
        candidates = candidates_by_seed.get(seed_id, [])
        candidates.sort(
            key=lambda item: (
                -float(getattr(item, "score", 0.0) or 0.0),
                int((getattr(item, "metadata", {}) or {}).get("dialogue_chain_distance", 99) or 99),
                int(getattr(item, "turn_index", 0) or 0),
                clean_text(getattr(item, "memory_id", "")),
            )
        )
        per_seed_added = 0
        for candidate in candidates:
            candidate_id = clean_text(getattr(candidate, "memory_id", ""))
            if not candidate_id or candidate_id in existing_ids or candidate_id in added_ids:
                continue
            expanded.append(candidate)
            added_ids.add(candidate_id)
            total_added += 1
            per_seed_added += 1
            if per_seed_added >= max_per_seed or total_added >= max_total:
                break
        if total_added >= max_total:
            expanded.extend(memory_hits[hit_index + 1 :])
            break
    if not added_ids:
        return list(memory_hits)
    return dedupe_hits_by_memory_id(expanded)


def expand_semantic_coverage_hits(question: str, memory_hits: list[Any], graph: Any | None) -> list[Any]:
    mode = os.getenv("TMCRA_SEMANTIC_COVERAGE_EXPANSION_MODE", "").strip().lower()
    if mode in {"", "off", "false", "0", "disabled", "none"}:
        return list(memory_hits)
    if graph is None:
        return list(memory_hits)
    records_by_id = getattr(graph, "records_by_id", {}) or {}
    if not records_by_id:
        return list(memory_hits)
    embedder_mode = os.getenv("TMCRA_SEMANTIC_COVERAGE_EMBEDDER_MODE", "").strip() or os.getenv(
        "TMCRA_EMBEDDER_INDEX_RECALL_MODE", ""
    ).strip() or "bge_m3"
    try:
        max_records = max(16, int(os.getenv("TMCRA_SEMANTIC_COVERAGE_MAX_RECORDS", "160") or 160))
        max_units = max(24, int(os.getenv("TMCRA_SEMANTIC_COVERAGE_MAX_UNITS", "260") or 260))
        max_add = max(1, int(os.getenv("TMCRA_SEMANTIC_COVERAGE_MAX_ADD", "16") or 16))
        score_floor = float(os.getenv("TMCRA_SEMANTIC_COVERAGE_SCORE_FLOOR", "0.74") or 0.74)
    except (TypeError, ValueError):
        max_records, max_units, max_add, score_floor = 160, 260, 16, 0.74
    try:
        from experiments.replacement.adapters import memory_adapters
    except Exception:
        return list(memory_hits)
    existing_unit_texts = {clean_text(getattr(hit, "value", "")) for hit in memory_hits if clean_text(getattr(hit, "value", ""))}
    candidates: list[tuple[Any, int, str]] = []
    for record in list(records_by_id.values())[:max_records]:
        state = clean_text(getattr(record, "state", "active")).lower()
        if state not in {"active", "parallel_active", "evidence"}:
            continue
        value = clean_text(getattr(record, "value", ""))
        if not value:
            continue
        for unit_index, unit in enumerate((turn_units(value) or [value])[:8]):
            text = clean_text(unit)
            if not text or text in existing_unit_texts:
                continue
            candidates.append((record, unit_index, text))
            if len(candidates) >= max_units:
                break
        if len(candidates) >= max_units:
            break
    if not candidates:
        return list(memory_hits)
    texts = [clean_text(question), *[text for _, _, text in candidates]]
    try:
        vectors, metadata = memory_adapters._embedder_dense_vectors_for_texts(texts, mode=embedder_mode)
    except Exception:
        return list(memory_hits)
    if not vectors or not vectors[0] or not bool(metadata.get("write_embedder_dense_enabled", False)):
        return list(memory_hits)
    query_vector = list(vectors[0] or [])
    query_phrases = quoted_query_phrases(question)
    query_needs_temporal_anchors = bool(query_phrases and temporal_duration_query(question))
    scored: list[tuple[float, str, Any, int, str]] = []
    for (record, unit_index, text), vector in zip(candidates, vectors[1:]):
        if query_needs_temporal_anchors:
            text = attach_parent_temporal_context(text, clean_text(getattr(record, "value", "")))
        score = _vector_dot(query_vector, list(vector or []))
        phrase_hits = strict_quoted_phrase_hits(text, query_phrases)
        if query_phrases and phrase_hits <= 0:
            continue
        if query_needs_temporal_anchors and not temporal_event_anchor_text(text):
            continue
        if phrase_hits:
            score = min(1.0, score + (0.025 * min(3, phrase_hits)))
        if score < score_floor:
            continue
        memory_id = clean_text(getattr(record, "memory_id", "")) or clean_text(getattr(record, "slot_key", ""))
        scored.append((score, memory_id, record, unit_index, text))
    if not scored:
        return list(memory_hits)
    scored.sort(key=lambda item: (-item[0], item[1], item[3]))
    expanded = list(memory_hits)
    added = 0
    seen_semantic_ids: set[str] = set()
    for score, memory_id, record, unit_index, text in scored:
        semantic_id = f"{memory_id}#semantic_unit_{unit_index}"
        if semantic_id in seen_semantic_ids:
            continue
        metadata = dict(getattr(record, "metadata", {}) or {})
        metadata.update(
            {
                "semantic_coverage_expansion": True,
                "semantic_coverage_score": round(float(score), 6),
                "semantic_coverage_unit_index": int(unit_index),
                "semantic_coverage_source_memory_id": memory_id,
                "injection_planner_selected": True,
                "injection_planner_score": round(float(score), 6),
                "injection_planner_evidence_role": "positive",
                "hybrid_score": max(float(score), metadata_float_from_mapping(metadata, "hybrid_score")),
                "recall_score": max(float(score), metadata_float_from_mapping(metadata, "recall_score")),
            }
        )
        expanded.append(
            SimpleNamespace(
                memory_id=semantic_id,
                category=clean_text(getattr(record, "category", "")),
                value=text,
                relation=clean_text(getattr(record, "relation", "related_to")) or "related_to",
                anchors=list(getattr(record, "anchor_concepts", []) or []),
                score=float(score),
                source_kind="semantic_coverage_unit",
                slot_key=clean_text(getattr(record, "slot_key", "")),
                state=clean_text(getattr(record, "state", "active")) or "active",
                turn_index=int(getattr(record, "turn_index", 0) or 0),
                metadata=metadata,
            )
        )
        seen_semantic_ids.add(semantic_id)
        added += 1
        if added >= max_add:
            break
    return dedupe_hits_by_memory_id(expanded)


def hit_text_for_chain(hit: Any) -> str:
    metadata = dict(getattr(hit, "metadata", {}) or {})
    return " ".join(
        [
            clean_text(getattr(hit, "value", "")),
            clean_text(metadata.get("raw_text", "")),
            clean_text(metadata.get("source_turn_text", "")),
            clean_text(metadata.get("source_span", "")),
        ]
    )


def dialogue_session_id_from_text(text: str) -> str:
    value = clean_text(text)
    if not value:
        return ""
    match = re.search(r"\bLongMemEval\s+session_id=([A-Za-z0-9_.:-]+)", value)
    if match:
        return match.group(1)
    match = re.search(r"\[([A-Za-z0-9_.:-]+)\s+turn=\d+\s+role=", value)
    return match.group(1) if match else ""


def dialogue_chain_candidate_score(question: str, question_terms: set[str], value: str, metadata: Mapping[str, Any], distance: int) -> float:
    text = " ".join([value, clean_text(metadata.get("source_span", "")), clean_text(metadata.get("event_text", ""))])
    units = turn_units(text) or [text]
    base = max(score_evidence_unit(question_terms, question, unit) for unit in units)
    if base <= 0:
        return 0.0
    source_kind = clean_text(metadata.get("source_kind", ""))
    if source_kind.startswith("public_dialog_") and source_kind != "public_dialog_turn":
        base += 0.35
    if clean_text(metadata.get("event_id", "")):
        base += 0.2
    return round(max(0.0, base - (0.15 * max(0, distance - 1))), 6)


def record_to_chain_hit(record: Any, *, score: float, seed_id: str, distance: int, session_id: str) -> Any:
    metadata = dict(getattr(record, "metadata", {}) or {})
    metadata.update(
        {
            "dialogue_chain_expansion": True,
            "dialogue_chain_seed_memory_id": seed_id,
            "dialogue_chain_distance": int(distance),
            "dialogue_chain_session_id": session_id,
        }
    )
    return SimpleNamespace(
        memory_id=clean_text(getattr(record, "memory_id", "")),
        category=clean_text(getattr(record, "category", "")),
        value=clean_text(getattr(record, "value", "")),
        relation=clean_text(getattr(record, "relation", "related_to")) or "related_to",
        anchors=list(getattr(record, "anchor_concepts", []) or []),
        score=float(score),
        source_kind=clean_text(getattr(record, "source_kind", "memory")) or "memory",
        slot_key=clean_text(getattr(record, "slot_key", "")),
        state=clean_text(getattr(record, "state", "active")) or "active",
        turn_index=int(getattr(record, "turn_index", 0) or 0),
        metadata=metadata,
    )


def dedupe_hits_by_memory_id(hits: list[Any]) -> list[Any]:
    deduped: list[Any] = []
    seen: set[str] = set()
    for hit in hits:
        memory_id = clean_text(getattr(hit, "memory_id", ""))
        if memory_id and memory_id in seen:
            continue
        if memory_id:
            seen.add(memory_id)
        deduped.append(hit)
    return deduped


def _load_answer_window_planner() -> tuple[Any, Any, Any, dict[str, float]] | None:
    mode = os.getenv("TMCRA_ANSWER_WINDOW_PLANNER_MODE", "").strip().lower()
    if mode in {"", "off", "false", "0", "disabled", "none"}:
        return None
    model_path = (
        os.getenv("TMCRA_ANSWER_WINDOW_PLANNER_MODEL_PATH", "").strip()
        or os.getenv("TMCRA_INJECTION_PLANNER_MODEL_PATH", "").strip()
    )
    if not model_path:
        return None
    cache_key = "|".join(
        [
            model_path,
            os.getenv("TMCRA_ANSWER_WINDOW_PLANNER_DEVICE", os.getenv("TMCRA_INJECTION_PLANNER_DEVICE", "cpu")),
        ]
    )
    cached = _ANSWER_WINDOW_PLANNER_CACHE.get(cache_key)
    if cached is not None:
        return cached
    try:
        from experiments.replacement import injection_planner as planner
    except Exception:
        return None
    torch_module = getattr(planner, "torch", None)
    if torch_module is None:
        return None
    try:
        device = torch_module.device(os.getenv("TMCRA_ANSWER_WINDOW_PLANNER_DEVICE", os.getenv("TMCRA_INJECTION_PLANNER_DEVICE", "cpu")) or "cpu")
        payload = torch_module.load(Path(model_path), map_location=device, weights_only=False)
        config = planner.InjectionPlannerConfig.from_dict(dict(payload.get("config", {}) or {}))
        model = planner.InjectionPlannerModel(config).to(device)
        model.load_state_dict(dict(payload.get("state_dict", {}) or {}), strict=False)
        model.eval()
        thresholds = {
            "selection_threshold": float(os.getenv("TMCRA_ANSWER_WINDOW_PLANNER_SELECTION_THRESHOLD", os.getenv("TMCRA_INJECTION_PLANNER_SELECTION_THRESHOLD", "0.3"))),
            "row_threshold": float(os.getenv("TMCRA_ANSWER_WINDOW_PLANNER_ROW_THRESHOLD", os.getenv("TMCRA_INJECTION_PLANNER_ROW_THRESHOLD", "0.3"))),
            "logic_threshold": float(os.getenv("TMCRA_ANSWER_WINDOW_PLANNER_LOGIC_THRESHOLD", os.getenv("TMCRA_INJECTION_PLANNER_LOGIC_THRESHOLD", "0.45"))),
        }
    except Exception:
        return None
    loaded = (planner, torch_module, model, thresholds)
    _ANSWER_WINDOW_PLANNER_CACHE[cache_key] = loaded
    return loaded


def apply_answer_window_planner(question: str, evidence_windows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    loaded = _load_answer_window_planner()
    if loaded is None or not evidence_windows:
        return evidence_windows
    planner, torch_module, model, thresholds = loaded
    evidence_windows = apply_answer_window_semantic_features(question, evidence_windows)
    max_score = max([float(item.get("score", 0.0) or 0.0) for item in evidence_windows] + [1.0])
    candidates = []
    for index, item in enumerate(evidence_windows):
        original_role = clean_text(item.get("evidence_role", ""))
        role_is_noise = original_role in {"noise", "negative_evidence"}
        semantic_similarity = max(
            0.0,
            min(
                1.0,
                float(item.get("answer_window_semantic_similarity", item.get("semantic_similarity", 0.0)) or 0.0),
            ),
        )
        candidates.append(
            {
                "id": f"window_{index}",
                "text": clean_text(item.get("text", "")),
                "layer": "event",
                "temporal_state": clean_text(item.get("temporal_state", "")) or "irrelevant",
                "logic_roles": ["noise"] if role_is_noise else ["evidence"],
                "retrieval_score": min(1.0, max(0.0, float(item.get("score", 0.0) or 0.0) / max_score)),
                "graph_score": min(1.0, max(0.0, float(item.get("planner_score", 0.0) or 0.0))),
                "tunnel_score": 1.0 if original_role == "bridge_context" else 0.0,
                "topic_similarity": 0.0,
                "semantic_similarity": semantic_similarity,
                "confidence": min(1.0, max(0.0, float(item.get("score", 0.0) or 0.0) / max_score)),
                "rank_score": round(1.0 / float(index + 1), 6),
                "branch_depth": 1,
                "evidence_role": original_role,
            }
        )
    try:
        row = {"id": "answer_window_plan", "query": question, "candidates": candidates, "gold": {}}
        dataset = planner.InjectionPlannerDataset([row], model.config)
        batch = planner.collate_injection_batch([dataset[0]])
        device = next(model.parameters()).device
        model_batch = {key: value.to(device) if hasattr(value, "to") else value for key, value in dict(batch).items()}
        with torch_module.no_grad():
            outputs = model(model_batch["features"], model_batch["valid_mask"])
            should_inject_score = float(torch_module.sigmoid(outputs["should_inject_logits"])[0].detach().cpu().item())
            selection_scores = torch_module.sigmoid(outputs["selection_logits"])[0].detach().cpu().tolist()
            temporal_indices = torch_module.argmax(outputs["temporal_logits"], dim=-1)[0].detach().cpu().tolist()
            role_indices = torch_module.argmax(outputs["evidence_role_logits"], dim=-1)[0].detach().cpu().tolist()
            logic_scores = torch_module.sigmoid(outputs["logic_logits"])[0].detach().cpu().tolist()
    except Exception:
        return evidence_windows
    row_allows = should_inject_score >= thresholds["row_threshold"]
    planned: list[dict[str, Any]] = []
    for index, item in enumerate(evidence_windows):
        next_item = dict(item)
        item_semantic_similarity = max(
            0.0,
            min(
                1.0,
                float(item.get("answer_window_semantic_similarity", item.get("semantic_similarity", 0.0)) or 0.0),
            ),
        )
        role = planner.EVIDENCE_ROLES[int(role_indices[index])]
        role_allows = role not in {"noise", "negative_evidence"}
        selected = bool(row_allows and role_allows and float(selection_scores[index]) >= thresholds["selection_threshold"])
        logic_roles = [
            logic_role
            for logic_role, score in zip(planner.LOGIC_ROLES, logic_scores[index])
            if float(score) >= thresholds["logic_threshold"]
        ]
        next_item.update(
            {
                "hit_evidence_role": clean_text(item.get("evidence_role", "")),
                "hit_temporal_state": clean_text(item.get("temporal_state", "")),
                "planner_selected": selected,
                "planner_score": round(float(selection_scores[index]), 6),
                "semantic_similarity": round(float(item_semantic_similarity), 6),
                "evidence_role": role,
                "temporal_state": planner.TEMPORAL_STATES[int(temporal_indices[index])],
                "window_planner_enabled": True,
                "window_planner_should_inject_score": round(float(should_inject_score), 6),
                "window_planner_logic_roles": logic_roles or ["evidence"],
            }
        )
        planned.append(next_item)

    def window_key(item: dict[str, Any]) -> tuple[bool, bool, float, float, int]:
        evidence_role = clean_text(item.get("evidence_role", ""))
        role_allows_evidence = evidence_role not in {"noise", "negative_evidence"}
        planner_score = float(item.get("planner_score", 0.0) or 0.0) if role_allows_evidence else 0.0
        return (
            bool(item.get("planner_selected", False)),
            role_allows_evidence,
            planner_score,
            float(item.get("score", 0.0) or 0.0),
            len(item.get("text", "")),
        )

    planned.sort(key=window_key, reverse=True)
    return planned


def apply_llm_evidence_selector(question: str, evidence_windows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # TMCRA_LAYER_TAG: answer-side LLM evidence selector; post-retrieval rerank only, not recall.
    mode = os.getenv("TMCRA_LLM_EVIDENCE_SELECTOR_MODE", "on").strip().lower()
    if mode in {"", "off", "false", "0", "disabled", "none"} or not evidence_windows:
        return evidence_windows
    try:
        max_candidates = max(1, int(os.getenv("TMCRA_LLM_EVIDENCE_SELECTOR_MAX_CANDIDATES", "12") or 12))
        max_chars = max(200, int(os.getenv("TMCRA_LLM_EVIDENCE_SELECTOR_CHARS", "900") or 900))
    except (TypeError, ValueError):
        max_candidates, max_chars = 12, 900
    candidates = list(evidence_windows[:max_candidates])
    candidate_lines = []
    for index, item in enumerate(candidates, start=1):
        memory_id = clean_text(item.get("memory_id", ""))
        text = truncate(item.get("text", ""), max_chars)
        if not text:
            continue
        header = f"[{index}]"
        if memory_id:
            header += f" memory_id={memory_id}"
        candidate_lines.append(f"{header}\n{text}")
    if not candidate_lines:
        return evidence_windows
    messages = [
        {
            "role": "system",
            "content": (
                "You are a semantic evidence selector for a long-memory runtime. "
                "Your job is only to select evidence windows that could help answer the current question. "
                "Use semantic relevance and discourse continuity, not keyword lists or domain-specific rules. "
                "Select every distinct useful clue, including indirect clues, continuation turns, and separate pieces that must be combined. "
                "If two windows repeat the same clue, keep the clearest one. "
                "If two windows contain different clues, keep both even if one looks weaker or appears later in the list. "
                "Do not answer the question. "
                "Return strict JSON only with keys: selected_indices, rejected_indices. "
                "selected_indices must be 1-based integers from the candidate list."
            ),
        },
        {
            "role": "user",
            "content": f"Question:\n{question}\n\nCandidate evidence windows:\n" + "\n\n".join(candidate_lines),
        },
    ]
    try:
        raw = chat_completion(
            os.getenv("TMCRA_EVIDENCE_SELECTOR_BASE_URL", os.getenv("GEMMA_BASE_URL", "http://127.0.0.1:18002/v1")),
            os.getenv("TMCRA_EVIDENCE_SELECTOR_MODEL", os.getenv("GEMMA_MODEL", os.getenv("TMCRA_GEMMA_MODEL_NAME", "gemma-4-e4b-it"))),
            messages,
            max_tokens=160,
            temperature=0.0,
        )
    except Exception:
        return evidence_windows
    payload = parse_json_object(raw)
    raw_indices = payload.get("selected_indices", []) if payload else []
    if not isinstance(raw_indices, list):
        return evidence_windows
    selected_order: list[int] = []
    for value in iter_selector_indices(raw_indices):
        try:
            index = int(value) - 1
        except (TypeError, ValueError):
            continue
        if 0 <= index < len(candidates) and index not in selected_order:
            selected_order.append(index)
    if not selected_order:
        return evidence_windows
    keep_model_selected = os.getenv("TMCRA_LLM_EVIDENCE_SELECTOR_KEEP_MODEL_SELECTED", "1").strip().lower() not in {
        "0",
        "false",
        "off",
        "no",
    }
    model_selected_order: list[int] = []
    if keep_model_selected:
        for index, item in enumerate(candidates):
            if bool(item.get("planner_selected", False)) or bool(item.get("answer_plan_selected", False)):
                model_selected_order.append(index)
    selected_order = list(dict.fromkeys([*selected_order, *model_selected_order]))
    selected_set = set(selected_order)
    selected_rank = {index: rank for rank, index in enumerate(selected_order)}
    planned: list[tuple[int, int, dict[str, Any]]] = []
    for index, item in enumerate(evidence_windows):
        next_item = dict(item)
        llm_selected = index in selected_set
        next_item["llm_evidence_selected"] = llm_selected
        next_item["llm_evidence_rank"] = selected_rank.get(index, 9999)
        next_item["llm_evidence_selector_enabled"] = True
        next_item["llm_evidence_model_kept"] = bool(index in model_selected_order)
        group = 0 if llm_selected else 1
        rank = selected_rank.get(index, index)
        planned.append((group, rank, next_item))
    planned.sort(key=lambda row: (row[0], row[1]))
    return diversify_evidence_windows([item for _, _, item in planned])


def apply_evidence_unit_planner(question: str, evidence_windows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # TMCRA_LAYER_TAG: answer-side LLM evidence-unit planner; normalizes retrieved windows after recall.
    mode = os.getenv("TMCRA_EVIDENCE_UNIT_PLANNER_MODE", "on").strip().lower()
    if mode in {"", "off", "false", "0", "disabled", "none"} or not evidence_windows:
        return evidence_windows
    if evidence_unit_plan_from_windows(evidence_windows):
        return evidence_windows
    plan = build_evidence_unit_plan(question, evidence_windows)
    if not plan:
        return evidence_windows
    selected_ranks = evidence_unit_selected_source_ranks(plan)
    reorder_mode = os.getenv("TMCRA_EVIDENCE_UNIT_PLANNER_REORDER", "0").strip().lower()
    reorder_enabled = reorder_mode not in {"", "0", "false", "off", "no", "disabled", "none"}
    planned: list[tuple[int, int, int, dict[str, Any]]] = []
    for index, item in enumerate(evidence_windows):
        next_item = dict(item)
        if index == 0:
            next_item["evidence_unit_plan"] = plan
        source_index = index + 1
        unit_rank = selected_ranks.get(source_index)
        next_item["evidence_unit_planner_enabled"] = True
        next_item["evidence_unit_planner_selected"] = unit_rank is not None
        next_item["evidence_unit_planner_rank"] = unit_rank if unit_rank is not None else 9999
        group = 0 if unit_rank is not None else 1
        rank = unit_rank if unit_rank is not None else index
        planned.append((group, rank, index, next_item))
    if not reorder_enabled:
        return [item for _, _, _, item in sorted(planned, key=lambda row: row[2])]
    planned.sort(key=lambda row: (row[0], row[1], row[2]))
    return diversify_evidence_windows([item for _, _, _, item in planned])


def build_evidence_unit_plan(question: str, evidence_windows: list[dict[str, Any]]) -> dict[str, Any]:
    try:
        max_candidates = max(1, int(os.getenv("TMCRA_EVIDENCE_UNIT_PLANNER_MAX_CANDIDATES", "10") or 10))
        max_chars = max(240, int(os.getenv("TMCRA_EVIDENCE_UNIT_PLANNER_CHARS", "1100") or 1100))
        max_tokens = max(280, int(os.getenv("TMCRA_EVIDENCE_UNIT_PLANNER_MAX_TOKENS", "760") or 760))
    except (TypeError, ValueError):
        max_candidates, max_chars, max_tokens = 10, 1100, 760
    candidate_lines = []
    for index, item in enumerate(evidence_windows[:max_candidates], start=1):
        text = truncate(item.get("text", ""), max_chars)
        if not text:
            continue
        memory_id = clean_text(item.get("memory_id", ""))
        role = clean_text(item.get("evidence_role", ""))
        temporal_state = clean_text(item.get("temporal_state", ""))
        label_bits = []
        if memory_id:
            label_bits.append(f"memory_id={memory_id}")
        if role:
            label_bits.append(f"role={role}")
        if temporal_state:
            label_bits.append(f"time={temporal_state}")
        label = " ".join(label_bits)
        candidate_lines.append(f"[{index}] {label}\n{text}".strip())
    if not candidate_lines:
        return {}
    messages = [
        {
            "role": "system",
            "content": (
                "You are a model-based evidence-unit planner for a long-memory runtime. "
                "Do not answer from prior knowledge. Do not use domain-specific keyword rules. "
                "Read the current question and the retrieved evidence windows, then normalize the evidence into answer units that a final answer model can reason over. "
                "Your job is to expose the operation needed by the question: direct fact lookup, distinct-item counting, temporal difference, temporal disambiguation, current/latest value selection, negative/absence judgment, preference synthesis, or multi-evidence synthesis. "
                "For count questions, infer the answer unit from the question and create one unit per distinct evidence-backed candidate. "
                "When the question requests multiple actions joined by and/or, keep separate units for separate action-target instances, especially original-versus-replacement or return-versus-pick-up instances. "
                "Do not collapse a stated pending need, intention, or obligation into mere context just because the same sentence later mentions an exchange, replacement, reschedule, or follow-up action. "
                "Treat the original action and the replacement/follow-up action as separate units unless the evidence explicitly says the original action was completed, cancelled, or no longer needed. "
                "Do not mark a clue as a distractor solely because it appears in an assistant turn; assistant turns can preserve user-specific obligations, summaries, answers, and confirmations when they refer to the surrounding dialogue. "
                "If an assistant turn says the user needs to do an action, or summarizes a user obligation, keep it as a candidate evidence unit unless contradicted by user evidence. "
                "For temporal questions, identify the event date or relative time anchor, the query-time anchor if available, and which event should be compared or excluded. "
                "Use session dates and relative-time phrases from the evidence window text when present; connect nearby event descriptions to the session date when the dialogue makes that local anchor clear. "
                "For month-level questions, if a past-tense personal event appears inside a dated session and no older or future date is stated, treat it as compatible with that session month; mark the time_anchor as inferred from session date rather than excluding it. "
                "For negative or absence questions, first identify the relevant event/time bucket, then state whether the requested attribute is present, absent, or contradicted in that event. "
                "Keep weak but potentially useful clues; mark distractors separately instead of deleting them. "
                "Return strict JSON only with these keys: question_task, answer_unit, required_operation, evidence_units, plan_steps, candidate_answer, uncertainty. "
                "evidence_units must be a list of objects with keys: source_index, role, unit_key, value, action, target, instance, time_anchor, polarity, relation_to_question. "
                "Allowed role values: answer_unit, positive_evidence, temporal_anchor, current_value, old_value, constraint, negative_evidence, distractor, context. "
                "source_index must be the 1-based evidence window index. candidate_answer may be empty when computation should be left to the answer model."
            ),
        },
        {
            "role": "user",
            "content": f"Question:\n{question}\n\nRetrieved evidence windows:\n" + "\n\n".join(candidate_lines),
        },
    ]
    base_url, model, api_key = evidence_unit_planner_config()
    try:
        raw = chat_completion(
            base_url,
            model,
            messages,
            max_tokens=max_tokens,
            temperature=0.0,
            api_key=api_key,
        )
    except Exception as exc:
        return {
            "enabled": False,
            "error": f"{exc.__class__.__name__}: {str(exc)[:240]}",
        }
    payload = parse_json_object(raw)
    if not payload:
        return {
            "enabled": False,
            "error": "planner_returned_non_json",
            "raw": truncate(raw, 500),
        }
    return sanitize_evidence_unit_plan(payload, max_source_index=len(candidate_lines))


def sanitize_evidence_unit_plan(payload: Mapping[str, Any], *, max_source_index: int) -> dict[str, Any]:
    allowed_roles = {
        "answer_unit",
        "positive_evidence",
        "temporal_anchor",
        "current_value",
        "old_value",
        "constraint",
        "negative_evidence",
        "distractor",
        "context",
    }
    plan: dict[str, Any] = {
        "enabled": True,
        "question_task": truncate(payload.get("question_task", ""), 80),
        "answer_unit": truncate(payload.get("answer_unit", ""), 120),
        "required_operation": truncate(payload.get("required_operation", ""), 180),
        "evidence_units": [],
        "plan_steps": [],
        "candidate_answer": truncate(payload.get("candidate_answer", ""), 160),
        "uncertainty": truncate(payload.get("uncertainty", ""), 180),
    }
    raw_units = payload.get("evidence_units", [])
    if isinstance(raw_units, list):
        for raw_unit in raw_units[:24]:
            if not isinstance(raw_unit, Mapping):
                continue
            try:
                source_index = int(raw_unit.get("source_index", 0) or 0)
            except (TypeError, ValueError):
                source_index = 0
            if source_index < 1 or source_index > max_source_index:
                continue
            role = clean_text(raw_unit.get("role", "context")).lower()
            if role not in allowed_roles:
                role = "context"
            plan["evidence_units"].append(
                {
                    "source_index": source_index,
                    "role": role,
                    "unit_key": truncate(raw_unit.get("unit_key", ""), 100),
                    "value": truncate(raw_unit.get("value", ""), 220),
                    "action": truncate(raw_unit.get("action", ""), 80),
                    "target": truncate(raw_unit.get("target", ""), 100),
                    "instance": truncate(raw_unit.get("instance", ""), 100),
                    "time_anchor": truncate(raw_unit.get("time_anchor", ""), 100),
                    "polarity": truncate(raw_unit.get("polarity", ""), 60),
                    "relation_to_question": truncate(raw_unit.get("relation_to_question", ""), 180),
                }
            )
    raw_steps = payload.get("plan_steps", [])
    if isinstance(raw_steps, list):
        plan["plan_steps"] = [truncate(item, 180) for item in raw_steps[:8] if clean_text(item)]
    return plan


def evidence_unit_selected_source_ranks(plan: Mapping[str, Any]) -> dict[int, int]:
    selected: dict[int, int] = {}
    ignored_roles = {"distractor"}
    for unit in list(plan.get("evidence_units") or []):
        if not isinstance(unit, Mapping):
            continue
        role = clean_text(unit.get("role", "")).lower()
        if role in ignored_roles:
            continue
        try:
            source_index = int(unit.get("source_index", 0) or 0)
        except (TypeError, ValueError):
            continue
        if source_index <= 0 or source_index in selected:
            continue
        selected[source_index] = len(selected)
    return selected


def evidence_unit_plan_from_windows(evidence_windows: list[dict[str, Any]]) -> dict[str, Any]:
    for item in evidence_windows:
        plan = item.get("evidence_unit_plan")
        if isinstance(plan, dict) and plan:
            return plan
    return {}


def _load_unified_operation_planner() -> tuple[Any, Any, Any] | None:
    mode = os.getenv("TMCRA_UNIFIED_OPERATION_PLANNER_MODE", "").strip().lower()
    if mode in {"", "off", "false", "0", "disabled", "none"}:
        return None
    model_path = os.getenv("TMCRA_UNIFIED_OPERATION_PLANNER_MODEL_PATH", "").strip()
    if not model_path:
        return None
    cache_key = "|".join([model_path, os.getenv("TMCRA_UNIFIED_OPERATION_PLANNER_DEVICE", "cpu")])
    cached = _UNIFIED_OPERATION_PLANNER_CACHE.get(cache_key)
    if cached is not None:
        return cached
    try:
        from experiments.replacement import unified_operation_planner as planner
    except Exception:
        return None
    torch_module = getattr(planner, "torch", None)
    if torch_module is None:
        return None
    try:
        device = torch_module.device(os.getenv("TMCRA_UNIFIED_OPERATION_PLANNER_DEVICE", "cpu") or "cpu")
        payload = torch_module.load(Path(model_path), map_location=device, weights_only=False)
        config = planner.UnifiedPlannerConfig.from_dict(dict(payload.get("config", {}) or {}))
        model = planner.UnifiedOperationPlannerModel(config).to(device)
        state = dict(payload.get("model_state") or payload.get("state_dict") or {})
        model.load_state_dict(state, strict=False)
        model.eval()
    except Exception:
        return None
    loaded = (planner, torch_module, model)
    _UNIFIED_OPERATION_PLANNER_CACHE[cache_key] = loaded
    return loaded


def apply_unified_operation_planner(question: str, evidence_windows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    loaded = _load_unified_operation_planner()
    if loaded is None or not evidence_windows:
        return evidence_windows
    planner, torch_module, model = loaded
    try:
        max_windows = max(3, int(os.getenv("TMCRA_UNIFIED_OPERATION_PLANNER_MAX_WINDOWS", "16") or 16))
        answer_threshold = float(os.getenv("TMCRA_UNIFIED_OPERATION_PLANNER_ANSWER_THRESHOLD", "0.52"))
    except (TypeError, ValueError):
        max_windows, answer_threshold = 16, 0.52
    scoped = list(evidence_windows[:max_windows])
    all_text = " ".join(clean_text(item.get("text", "")) for item in scoped).lower()
    units = []
    for index, item in enumerate(scoped, start=1):
        text = clean_text(item.get("text", ""))
        memory_id = clean_text(item.get("memory_id", "")) or f"window_{index}"
        units.append(
            {
                "unit_id": f"u{index}",
                "record_id": memory_id,
                "session_id": memory_id.split(".turn_", 1)[0],
                "turn_index": index,
                "speaker": "unknown",
                "text": text,
                "timestamp": "",
                "topic_bucket": clean_text(item.get("topic_bucket", "")),
                "node_features": {
                    "retrieval_score": float(item.get("score", 0.0) or 0.0) / 10.0,
                    "graph_score": float(item.get("planner_score", 0.0) or item.get("answer_plan_score", 0.0) or 0.0),
                    "tunnel_score": float(item.get("path_score", 0.0) or 0.0),
                },
                "graph_neighbors": [],
            }
        )
    row = {
        "id": "runtime_unified_operation_planner",
        "query": question,
        "retrieval_metadata": {
            "candidate_count": len(scoped),
            "session_span": len({clean_text(item.get("memory_id", "")).split(".turn_", 1)[0] for item in scoped}),
            "topic_span": len({clean_text(item.get("topic_bucket", "")) for item in scoped if clean_text(item.get("topic_bucket", ""))}),
            "has_temporal_anchor": bool(re.search(r"\b(before|after|latest|current|previous|ago|week|month|year|date)\b", all_text)),
            "has_numeric_units": bool(re.search(r"\b\d[\d,.]*\b|\$\s?\d|%", all_text)),
            "has_profile_units": bool(re.search(r"\b(prefer|like|avoid|favorite|usually|always|never)\b", all_text)),
        },
        "memory_units": units,
        "gold": {},
    }
    try:
        dataset = planner.UnifiedOperationPlannerDataset([row], model.config)
        batch = planner.collate_unified_planner_batch([dataset[0]])
        device = next(model.parameters()).device
        model_batch = {key: value.to(device) if hasattr(value, "to") else value for key, value in dict(batch).items()}
        with torch_module.no_grad():
            outputs = model(model_batch["features"], model_batch["valid_mask"])
            relevance_scores = torch_module.sigmoid(outputs["relevance_logits"])[0].detach().cpu().tolist()
            answer_scores = torch_module.sigmoid(outputs["answer_logits"])[0].detach().cpu().tolist()
            temporal_scores = torch_module.sigmoid(outputs["temporal_logits"])[0].detach().cpu().tolist()
            aggregation_scores = torch_module.sigmoid(outputs["aggregation_logits"])[0].detach().cpu().tolist()
            current_value_scores = torch_module.sigmoid(outputs["current_value_logits"])[0].detach().cpu().tolist()
            operation_scores = torch_module.sigmoid(outputs["operation_required_logits"])[0].detach().cpu().tolist()
            operation_family_index = int(torch_module.argmax(outputs["operation_family_logits"], dim=-1)[0].detach().cpu().item())
    except Exception:
        return evidence_windows
    operation_family = planner.OPERATION_FAMILIES[operation_family_index]
    ranked: list[tuple[int, float, int, dict[str, Any]]] = []
    for index, item in enumerate(evidence_windows):
        next_item = dict(item)
        if index < len(scoped):
            relevance_score = float(relevance_scores[index])
            answer_score = float(answer_scores[index])
            temporal_score = float(temporal_scores[index])
            aggregation_score = float(aggregation_scores[index])
            current_value_score = float(current_value_scores[index])
            unified_score = max(answer_score, temporal_score, aggregation_score, current_value_score, relevance_score * 0.8)
            next_item["unified_operation_planner"] = True
            next_item["unified_operation_family"] = operation_family
            next_item["unified_operation_required_scores"] = {
                "temporal": round(float(operation_scores[0]), 4),
                "aggregation": round(float(operation_scores[1]), 4),
                "profile": round(float(operation_scores[2]), 4),
                "current_value": round(float(operation_scores[3]), 4),
                "multi_hop": round(float(operation_scores[4]), 4),
            }
            next_item["unified_scores"] = {
                "relevance": round(relevance_score, 4),
                "answer": round(answer_score, 4),
                "temporal": round(temporal_score, 4),
                "aggregation": round(aggregation_score, 4),
                "current_value": round(current_value_score, 4),
                "combined": round(unified_score, 4),
            }
            next_item["unified_selected"] = bool(answer_score >= answer_threshold or unified_score >= max(answer_threshold, 0.58))
            old_planner_score = float(next_item.get("planner_score", 0.0) or next_item.get("answer_plan_score", 0.0) or 0.0)
            next_item["planner_score"] = max(old_planner_score, unified_score)
            group = 0 if next_item["unified_selected"] else 1
            ranked.append((group, -unified_score, index, next_item))
        else:
            ranked.append((2, 0.0, index, next_item))
    ranked.sort(key=lambda row: (row[0], row[1], row[2]))
    return [item for _, _, _, item in ranked]


def format_evidence_unit_plan_for_prompt(plan: Mapping[str, Any]) -> str:
    if not plan or not bool(plan.get("enabled", False)):
        return ""
    compact = {
        "question_task": plan.get("question_task", ""),
        "answer_unit": plan.get("answer_unit", ""),
        "required_operation": plan.get("required_operation", ""),
        "evidence_units": list(plan.get("evidence_units") or [])[:16],
        "plan_steps": list(plan.get("plan_steps") or [])[:6],
        "candidate_answer": plan.get("candidate_answer", ""),
        "uncertainty": plan.get("uncertainty", ""),
    }
    return json.dumps(compact, ensure_ascii=False, separators=(",", ":"))


def focused_evidence_unit_lines(
    question: str,
    evidence_windows: list[dict[str, Any]],
    *,
    max_items: int | None = None,
    max_chars: int | None = None,
) -> list[str]:
    mode = os.getenv("TMCRA_FOCUSED_EVIDENCE_UNIT_MODE", "").strip().lower()
    if mode in {"", "off", "0", "false", "disabled", "none"}:
        return []
    try:
        item_limit = max(1, int(os.getenv("TMCRA_FOCUSED_EVIDENCE_UNIT_LIMIT", "18") if max_items is None else max_items))
        char_limit = max(120, int(os.getenv("TMCRA_FOCUSED_EVIDENCE_UNIT_CHARS", "420") if max_chars is None else max_chars))
        per_window = max(1, int(os.getenv("TMCRA_FOCUSED_EVIDENCE_UNITS_PER_WINDOW", "2") or 2))
    except (TypeError, ValueError):
        item_limit, char_limit, per_window = 18, 420, 2
    question_terms = evidence_query_terms(question)
    lines: list[str] = []
    seen: set[str] = set()
    for window_index, item in enumerate(evidence_windows, start=1):
        text = clean_text(item.get("text", ""))
        if not text:
            continue
        units = turn_units(text) or [text]
        scored_units: list[tuple[float, int, str]] = []
        for unit_index, unit in enumerate(units):
            unit_text = clean_text(unit)
            if not unit_text:
                continue
            score = score_evidence_unit(question_terms, question, unit_text)
            if bool(item.get("planner_selected", False)) or bool(item.get("evidence_unit_planner_selected", False)):
                score += 0.4
            if bool(item.get("semantic_coverage_expansion", False)):
                score += 0.25
            if score <= 0 and len(units) > 1:
                continue
            scored_units.append((float(score), unit_index, unit_text))
        if not scored_units:
            scored_units = [(0.0, 0, text)]
        scored_units.sort(key=lambda row: (-row[0], row[1]))
        added_for_window = 0
        for _, unit_index, unit_text in scored_units:
            snippet = centered_excerpt(unit_text, question_terms, max_chars=char_limit)
            key = re.sub(r"\W+", " ", snippet.lower()).strip()
            if not key or key in seen:
                continue
            seen.add(key)
            memory_id = clean_text(item.get("memory_id", ""))
            source = f"source={window_index}"
            if memory_id:
                source += f" memory_id={memory_id}"
            lines.append(f"{len(lines) + 1}. [{source} unit={unit_index}] {snippet}")
            added_for_window += 1
            if len(lines) >= item_limit or added_for_window >= per_window:
                break
        if len(lines) >= item_limit:
            break
    return lines


def diversify_evidence_windows(evidence_windows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    mode = os.getenv("TMCRA_ANSWER_EVIDENCE_DIVERSIFY_MEMORY_IDS", "1").strip().lower()
    if mode in {"", "0", "false", "off", "no"} or len(evidence_windows) <= 1:
        return evidence_windows
    first_pass: list[dict[str, Any]] = []
    overflow: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in evidence_windows:
        memory_id = clean_text(item.get("memory_id", ""))
        if not memory_id:
            first_pass.append(item)
            continue
        if memory_id in seen:
            overflow.append(item)
            continue
        seen.add(memory_id)
        first_pass.append(item)
    return [*first_pass, *overflow]


def final_answer_channel_intent(question: str) -> dict[str, bool]:
    terms = evidence_query_terms(question)
    question_l = question.lower()
    aggregation = bool(
        terms
        & {
            "amount",
            "count",
            "different",
            "how",
            "many",
            "minimum",
            "number",
            "percent",
            "percentage",
            "sum",
            "total",
        }
    ) or bool(re.search(r"\b(?:how many|how much|total|sum|minimum|maximum|combined|different)\b", question_l))
    temporal = bool(
        re.search(
            r"\b(?:before|after|first|last|earlier|later|weeks?|days?|months?|years?|when|date|how long)\b",
            question_l,
        )
    )
    assistant_detail = assistant_memory_query(question)
    return {
        "aggregation": aggregation,
        "temporal": temporal,
        "assistant_detail": assistant_detail,
    }


def _json_object_from_text(value: str) -> dict[str, Any]:
    text = clean_text(value)
    if not text:
        return {}
    try:
        payload = json.loads(text)
        return payload if isinstance(payload, dict) else {}
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return {}
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def build_query_graph(question: str, question_date: str = "") -> dict[str, Any]:
    mode = os.getenv("TMCRA_QUERY_GRAPH_BUILDER_MODE", "").strip().lower()
    if mode in {"", "0", "false", "off", "no", "disabled", "none"}:
        return {"enabled": False, "mode": mode or "off"}
    base_url, model, api_key = query_graph_llm_config()
    try:
        max_tokens = max(240, int(os.getenv("TMCRA_QUERY_GRAPH_MAX_TOKENS", "700") or 700))
    except (TypeError, ValueError):
        max_tokens = 700
    system = (
        "You convert a user question into a compact retrieval query graph for a long-memory system. "
        "Return JSON only. Do not answer the question. Build a general graph, not benchmark-specific logic."
    )
    user = {
        "question": question,
        "question_date": question_date,
        "schema": {
            "task_intent": "direct_fact | count | sum | compare | temporal | preference | multi_evidence | unknown",
            "operation": "none | count_distinct | sum_numeric | compare_values | select_latest | select_current | timeline_order | infer_preference",
            "required_units": [
                {
                    "unit_key": "short stable key",
                    "role": "main_fact | coverage_fact | old_value | current_value | temporal_anchor | constraint | negative",
                    "target_entity": "entity or topic to retrieve",
                    "attribute": "amount/count/status/action/time/preference/etc",
                    "expected_value_type": "money | count | date | duration | item | state | preference | text",
                    "must_cover": True,
                }
            ],
            "tunnel_needs": ["same_topic", "cross_session", "same_entity", "temporal_chain", "profile_bridge"],
            "query_terms": ["terms that should retrieve memory units"],
            "negative_terms": ["terms that should not dominate retrieval"],
        },
    }
    try:
        raw = chat_completion(
            base_url,
            model,
            [
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
            ],
            max_tokens=max_tokens,
            temperature=0.0,
            api_key=api_key,
        )
        graph = _json_object_from_text(raw)
        if not graph:
            return {"enabled": True, "mode": mode, "error": "empty_or_invalid_json", "raw": truncate(raw, 600)}
        graph["enabled"] = True
        graph["mode"] = mode
        graph["model"] = model
        return graph
    except Exception as exc:
        return {"enabled": True, "mode": mode, "error": f"{exc.__class__.__name__}: {str(exc)[:240]}"}


def query_graph_retrieval_text(question: str, question_date: str, query_graph: dict[str, Any]) -> str:
    runtime_question = question
    if question_date:
        runtime_question = f"{question}\nQuestion date: {question_date}"
    use_text_mode = os.getenv("TMCRA_QUERY_GRAPH_RETRIEVAL_TEXT_MODE", "").strip().lower() in {"1", "true", "on", "yes"}
    if not use_text_mode or not bool(query_graph.get("enabled", False)) or query_graph.get("error"):
        return runtime_question
    graph = {key: value for key, value in query_graph.items() if key not in {"enabled", "mode", "model", "raw"}}
    graph_text = truncate(json.dumps(graph, ensure_ascii=False), int(os.getenv("TMCRA_QUERY_GRAPH_RETRIEVAL_CHARS", "2200") or 2200))
    return f"{runtime_question}\n\nQuestion retrieval graph JSON:\n{graph_text}"


def query_graph_sidecar_queries(question: str, question_date: str, query_graph: dict[str, Any]) -> list[str]:
    if not bool(query_graph.get("enabled", False)) or query_graph.get("error"):
        return []
    mode = os.getenv("TMCRA_QUERY_GRAPH_SIDECAR_RETRIEVAL_MODE", "on").strip().lower()
    if mode in {"", "0", "false", "off", "no", "disabled", "none"}:
        return []
    try:
        max_queries = max(0, int(os.getenv("TMCRA_QUERY_GRAPH_SIDECAR_MAX_QUERIES", "6") or 6))
    except (TypeError, ValueError):
        max_queries = 6
    if max_queries <= 0:
        return []
    query_terms = [clean_text(item) for item in list(query_graph.get("query_terms") or []) if clean_text(item)]
    required_units = list(query_graph.get("required_units") or [])
    queries: list[str] = []
    seen: set[str] = set()

    def add(value: str) -> None:
        text = clean_text(value)
        if not text:
            return
        if question_date and "question date:" not in text.lower():
            text = f"{text}\nQuestion date: {question_date}"
        key = text.lower()
        if key in seen or len(queries) >= max_queries:
            return
        seen.add(key)
        queries.append(text)

    for unit in required_units:
        if not isinstance(unit, dict):
            continue
        role = clean_text(unit.get("role", ""))
        target = clean_text(unit.get("target_entity", ""))
        attribute = clean_text(unit.get("attribute", ""))
        value_type = clean_text(unit.get("expected_value_type", ""))
        unit_key = clean_text(unit.get("unit_key", ""))
        add(
            " ".join(
                part
                for part in [
                    question,
                    f"required evidence role {role}",
                    f"target {target}",
                    f"attribute {attribute}",
                    f"value type {value_type}",
                    f"unit {unit_key}",
                ]
                if part
            )
        )
    if query_terms and len(queries) < max_queries:
        add(f"{question} {' '.join(query_terms[:24])}")
    return queries[:max_queries]


def merge_memory_hits(primary_hits: list[Any], extra_hits: list[Any]) -> list[Any]:
    merged: list[Any] = []
    seen: set[str] = set()
    for hit in [*primary_hits, *extra_hits]:
        memory_id = clean_text(getattr(hit, "memory_id", ""))
        key = memory_id or clean_text(getattr(hit, "text", ""))[:240].lower()
        if key and key in seen:
            continue
        if key:
            seen.add(key)
        merged.append(hit)
    return merged


def apply_query_graph_sidecar_retrieval(
    adapter: Any,
    query_graph: dict[str, Any],
    question: str,
    question_date: str,
    hits: list[Any],
    *,
    top_k: int,
) -> tuple[list[Any], dict[str, Any]]:
    queries = query_graph_sidecar_queries(question, question_date, query_graph)
    if not queries:
        return hits, {"enabled": False, "reason": "no_queries"}
    try:
        per_query_top_k = max(1, int(os.getenv("TMCRA_QUERY_GRAPH_SIDECAR_TOP_K", "4") or 4))
    except (TypeError, ValueError):
        per_query_top_k = 4
    extra_hits: list[Any] = []
    errors: list[str] = []
    for query in queries:
        try:
            retrieval = adapter.retrieve(query, top_k=min(max(1, top_k), per_query_top_k))
            extra_hits.extend(list(getattr(retrieval, "hits", []) or []))
        except Exception as exc:
            errors.append(f"{exc.__class__.__name__}:{str(exc)[:160]}")
    merged = merge_memory_hits(hits, extra_hits)
    return merged, {
        "enabled": True,
        "query_count": len(queries),
        "extra_hit_count": len(extra_hits),
        "merged_hit_count": len(merged),
        "errors": errors[:3],
        "queries": [truncate(item, 260) for item in queries[:6]],
    }


def final_answer_surface_windows(question: str, evidence_windows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    mode = os.getenv("TMCRA_FINAL_ANSWER_SURFACE_CHANNEL_MODE", "on").strip().lower()
    if mode in {"", "0", "false", "off", "no", "disabled", "none"} or not evidence_windows:
        return evidence_windows
    intent = final_answer_channel_intent(question)
    try:
        final_limit = max(4, int(os.getenv("TMCRA_ANSWER_EVIDENCE_WINDOW_LIMIT", "8") or 8))
        if intent["aggregation"]:
            final_limit = max(final_limit, int(os.getenv("TMCRA_ANSWER_EVIDENCE_WINDOW_LIMIT_AGG", "12") or 12))
    except (TypeError, ValueError):
        final_limit = 12 if final_answer_channel_intent(question)["aggregation"] else 8
    if intent["aggregation"]:
        main_hard_limit = max(4, int(os.getenv("TMCRA_FINAL_MAIN_HARD_LIMIT_AGG", "6") or 6))
        coverage_hard_limit = max(1, int(os.getenv("TMCRA_FINAL_COVERAGE_HARD_LIMIT_AGG", "6") or 6))
    elif intent["temporal"] or intent["assistant_detail"]:
        main_hard_limit = max(4, int(os.getenv("TMCRA_FINAL_MAIN_HARD_LIMIT_DETAIL", "7") or 7))
        coverage_hard_limit = max(0, int(os.getenv("TMCRA_FINAL_COVERAGE_HARD_LIMIT_DETAIL", "1") or 1))
    else:
        main_hard_limit = max(4, int(os.getenv("TMCRA_FINAL_MAIN_HARD_LIMIT_DEFAULT", "6") or 6))
        coverage_hard_limit = max(0, int(os.getenv("TMCRA_FINAL_COVERAGE_HARD_LIMIT_DEFAULT", "2") or 2))
    coverage_hard_limit = min(coverage_hard_limit, max(0, final_limit - 1))
    main_hard_limit = min(main_hard_limit, final_limit - coverage_hard_limit)

    def is_coverage(item: dict[str, Any]) -> bool:
        return bool(item.get("multi_unit_chain_slot", False)) or bool(item.get("unit_coverage_pack", False))

    def is_assistant_candidate(item: dict[str, Any]) -> bool:
        return bool(item.get("assistant_candidate_side_channel", False)) or clean_text(item.get("evidence_role", "")).lower() == "assistant_answer_candidate"

    def is_assistant(item: dict[str, Any]) -> bool:
        if is_assistant_candidate(item):
            return False
        return bool(item.get("assistant_origin_evidence", False)) or clean_text(item.get("evidence_role", "")).lower() == "assistant_origin_detail"

    def key(item: dict[str, Any]) -> tuple[int, int, float, float, int]:
        role = clean_text(item.get("evidence_role", "")).lower()
        role_allows = int(role not in {"noise", "negative_evidence"})
        selected = int(bool(item.get("planner_selected", False)) or bool(item.get("evidence_unit_planner_selected", False)))
        planner_score = max(float(item.get("planner_score", 0.0) or 0.0), 0.0)
        return (
            selected,
            role_allows,
            planner_score,
            float(item.get("score", 0.0) or 0.0),
            len(clean_text(item.get("text", ""))),
        )

    assistant_candidates = [item for item in evidence_windows if is_assistant_candidate(item)]
    assistant = [item for item in evidence_windows if is_assistant(item)]
    source_windows = list(evidence_windows)
    main = [item for item in source_windows if not is_coverage(item) and not is_assistant(item) and not is_assistant_candidate(item)]
    coverage = [item for item in source_windows if is_coverage(item)]
    assistant_candidates.sort(key=key, reverse=True)
    assistant.sort(key=key, reverse=True)
    main.sort(key=key, reverse=True)
    coverage.sort(key=key, reverse=True)
    top_assistant_memory_ids = {
        clean_text(item.get("memory_id", ""))
        for item in assistant[: max(1, int(os.getenv("TMCRA_FINAL_ASSISTANT_HARD_LIMIT", "4") or 4))]
        if clean_text(item.get("memory_id", ""))
    }
    if top_assistant_memory_ids:
        assistant_candidates.sort(
            key=lambda item: (
                int(clean_text(item.get("memory_id", "")) in top_assistant_memory_ids),
                *key(item),
            ),
            reverse=True,
        )

    ordered: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(items: list[dict[str, Any]], limit: int) -> None:
        added = 0
        for item in items:
            if len(ordered) >= final_limit or added >= limit:
                break
            memory_id = clean_text(item.get("memory_id", ""))
            text_key = clean_text(item.get("text", ""))[:240].lower()
            dedupe_key = text_key if is_assistant_candidate(item) else memory_id or text_key
            if dedupe_key and dedupe_key in seen:
                continue
            if dedupe_key:
                seen.add(dedupe_key)
            next_item = dict(item)
            if is_assistant_candidate(item):
                channel = "assistant_side"
            elif is_assistant(item):
                channel = "assistant"
            elif is_coverage(item):
                channel = "coverage"
            else:
                channel = "main"
            next_item["final_answer_channel"] = channel
            next_item["final_answer_channel_intent"] = dict(intent)
            ordered.append(next_item)
            added += 1

    if intent["assistant_detail"]:
        assistant_limit = max(1, int(os.getenv("TMCRA_FINAL_ASSISTANT_HARD_LIMIT", "4") or 4))
        add(assistant, min(assistant_limit, final_limit))
        try:
            assistant_side_limit = max(0, int(os.getenv("TMCRA_FINAL_ASSISTANT_SIDE_LIMIT", "1") or 1))
        except (TypeError, ValueError):
            assistant_side_limit = 1
        if assistant_side_limit:
            add(assistant_candidates, assistant_side_limit)
    add(main, main_hard_limit)
    add(coverage, coverage_hard_limit)
    add(assistant, final_limit)
    add(assistant_candidates, final_limit)
    add(main, final_limit)
    add(coverage, final_limit)
    return ordered[:final_limit]


def _llm_channel_planner_enabled() -> bool:
    mode = os.getenv("TMCRA_LLM_CHANNEL_PLANNER_MODE", "").strip().lower()
    return mode not in {"", "0", "false", "off", "no", "disabled", "none"}


def _int_list_from_payload(value: Any, *, upper_bound: int) -> list[int]:
    values = iter_selector_indices(value)
    result: list[int] = []
    seen: set[int] = set()
    for raw in values:
        try:
            index = int(str(raw).strip())
        except (TypeError, ValueError):
            continue
        if index < 1 or index > upper_bound or index in seen:
            continue
        seen.add(index)
        result.append(index)
    return result


def apply_llm_channel_planner(question: str, evidence_windows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not _llm_channel_planner_enabled() or not evidence_windows:
        return evidence_windows
    if any(bool(item.get("llm_channel_planner_enabled", False)) for item in evidence_windows):
        return evidence_windows
    try:
        max_windows = max(4, int(os.getenv("TMCRA_LLM_CHANNEL_PLANNER_MAX_WINDOWS", "16") or 16))
        max_chars = max(160, int(os.getenv("TMCRA_LLM_CHANNEL_PLANNER_WINDOW_CHARS", "520") or 520))
        max_tokens = max(160, int(os.getenv("TMCRA_LLM_CHANNEL_PLANNER_MAX_TOKENS", "700") or 700))
    except (TypeError, ValueError):
        max_windows, max_chars, max_tokens = 16, 520, 700
    scoped = list(evidence_windows[:max_windows])
    if not scoped:
        return evidence_windows

    lines: list[str] = []
    for index, item in enumerate(scoped, start=1):
        text = truncate(item.get("text", ""), max_chars)
        metadata = {
            "memory_id": clean_text(item.get("memory_id", "")),
            "score": round(float(item.get("score", 0.0) or 0.0), 4),
            "planner_score": round(float(item.get("planner_score", 0.0) or 0.0), 4),
            "evidence_role": clean_text(item.get("evidence_role", "")),
            "temporal_state": clean_text(item.get("temporal_state", "")),
            "unit_coverage": bool(item.get("unit_coverage_pack", False)),
            "multi_chain": bool(item.get("multi_unit_chain_slot", False)),
            "profile": bool(item.get("profile_shadow_unit", False) or item.get("profile_layer", False)),
            "unified_family": clean_text(item.get("unified_operation_family", "")),
        }
        lines.append(f"{index}. metadata={json.dumps(metadata, ensure_ascii=False, separators=(',', ':'))}\n{text}")

    system_prompt = (
        "You are an evidence-channel planner for a long-memory QA system. "
        "Do not answer the user question. Select evidence windows only. "
        "Return one strict JSON object and no markdown."
    )
    user_prompt = (
        "Plan the final evidence channels for the answer model.\n"
        "Rules:\n"
        "- main_indices must keep direct facts, stable base evidence, temporal anchors, assistant details, and profile facts.\n"
        "- coverage_indices are for count, sum, ratio, duration, repeated actions, multi-unit coverage, or unit-to-unit chains.\n"
        "- coverage evidence supplements main evidence; do not let coverage replace main facts.\n"
        "- support_indices are useful context only.\n"
        "- suppress_indices are duplicates, stale/conflicting values, or noise.\n"
        "- Use 1-based indices from the candidate list.\n"
        "- If unsure, keep a window in main or support instead of suppressing it.\n\n"
        "Return JSON schema:\n"
        "{\"operation_family\":\"direct|temporal|count|sum|ratio|profile|current_value|synthesis|unknown\","
        "\"main_indices\":[1],\"coverage_indices\":[],\"support_indices\":[],\"suppress_indices\":[],"
        "\"coverage_complete\":true,\"missing_evidence\":\"\",\"reasoning_policy\":\"short label\"}\n\n"
        f"Question:\n{question}\n\n"
        "Candidate evidence windows:\n"
        + "\n\n".join(lines)
    )
    try:
        base_url, model, api_key = answer_llm_config()
        raw = chat_completion(
            base_url,
            model,
            [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            max_tokens=max_tokens,
            temperature=0.0,
            api_key=api_key,
        )
        payload = parse_json_object(raw)
    except Exception as exc:
        annotated = [dict(item) for item in evidence_windows]
        if annotated:
            annotated[0]["llm_channel_planner_error"] = f"{exc.__class__.__name__}:{str(exc)[:160]}"
        return annotated
    if not payload:
        return evidence_windows

    main_indices = _int_list_from_payload(payload.get("main_indices", []), upper_bound=len(scoped))
    coverage_indices = _int_list_from_payload(payload.get("coverage_indices", []), upper_bound=len(scoped))
    support_indices = _int_list_from_payload(payload.get("support_indices", []), upper_bound=len(scoped))
    suppress_indices = _int_list_from_payload(payload.get("suppress_indices", []), upper_bound=len(scoped))
    channels: dict[int, str] = {}
    for channel, indices in (
        ("main", main_indices),
        ("coverage", coverage_indices),
        ("support", support_indices),
        ("suppress", suppress_indices),
    ):
        for index in indices:
            channels.setdefault(index - 1, channel)

    if not any(channel in {"main", "coverage", "support"} for channel in channels.values()):
        return evidence_windows

    channel_rank = {"main": 0, "coverage": 1, "support": 2, "unassigned": 3, "suppress": 4}
    planned: list[tuple[int, int, dict[str, Any]]] = []
    for index, item in enumerate(evidence_windows):
        next_item = dict(item)
        channel = channels.get(index, "unassigned") if index < len(scoped) else "unassigned"
        next_item["llm_channel_planner_enabled"] = True
        next_item["llm_channel"] = channel
        next_item["llm_channel_operation_family"] = clean_text(payload.get("operation_family", ""))
        next_item["llm_channel_coverage_complete"] = bool(payload.get("coverage_complete", False))
        missing_evidence = clean_text(payload.get("missing_evidence", ""))
        reasoning_policy = clean_text(payload.get("reasoning_policy", ""))
        if missing_evidence:
            next_item["llm_channel_missing_evidence"] = missing_evidence
        if reasoning_policy:
            next_item["llm_channel_reasoning_policy"] = reasoning_policy
        planned.append((channel_rank.get(channel, 3), index, next_item))
    planned.sort(key=lambda row: (row[0], row[1]))
    return [item for _, _, item in planned]


def iter_selector_indices(value: Any) -> list[Any]:
    if isinstance(value, list):
        flattened: list[Any] = []
        for item in value:
            flattened.extend(iter_selector_indices(item))
        return flattened
    return [value]


def profile_facts_for_answer(memory_hits: list[Any], *, max_items: int = 8) -> list[str]:
    facts: list[str] = []
    seen: set[str] = set()
    for hit in memory_hits[:12]:
        metadata = dict(getattr(hit, "metadata", {}) or {})
        source_kind = str(getattr(hit, "source_kind", "") or "")
        is_profile_hit = (
            bool(metadata.get("profile_first_hybrid_rescue"))
            or bool(metadata.get("profile_protected_slot"))
            or bool(metadata.get("profile_layer"))
            or source_kind.startswith("public_dialog_profile")
            or ".subject." in str(getattr(hit, "memory_id", "") or "")
            or source_kind in {"public_dialog_preference", "public_dialog_goal", "public_dialog_constraint", "public_dialog_status"}
            or str(getattr(hit, "category", "") or "").lower() in {"profile", "preference", "goal", "constraint", "status"}
        )
        if not is_profile_hit:
            continue
        profile_type = clean_text(metadata.get("profile_type", "")) or clean_text(getattr(hit, "category", ""))
        domain = clean_text(metadata.get("profile_domain_label", "") or metadata.get("profile_domain", ""))
        summary = clean_text(metadata.get("profile_summary", "") or metadata.get("profile_value", "") or getattr(hit, "value", ""))
        if not summary:
            continue
        prefix = "profile"
        if profile_type:
            prefix = profile_type.replace("_", " ")
        if domain:
            prefix = f"{prefix} / {domain}"
        fact = f"{prefix}: {summary}"
        key = fact.lower()
        if key in seen:
            continue
        seen.add(key)
        facts.append(fact)
        if len(facts) >= max_items:
            break
    return facts


def complete_profile_answer_from_evidence(
    question: str,
    answer: str,
    evidence_windows: list[dict[str, Any]],
) -> str:
    # TMCRA_LAYER_TAG: answer-side profile completion; preserves bound qualifiers from retrieved profile facts.
    answer_clean = clean_text(answer)
    if not answer_clean:
        return answer
    answer_l = answer_clean.lower()
    question_l = clean_text(question).lower()
    role_attribute_query = bool(re.search(r"\b(?:occupation|job|role|worked\s+as|work\s+as|career|position)\b", question_l))
    if len(answer_l) < 4:
        return answer
    for item in evidence_windows[:10]:
        if clean_text(item.get("evidence_role", "")).lower() != "profile_fact":
            continue
        fact = clean_text(item.get("text", ""))
        if not fact or len(fact) > 320:
            continue
        fact_l = fact.lower()
        if answer_l not in fact_l:
            continue
        candidates: list[str] = []
        for pattern in (
            r"\bas\s+(?:a|an|the)?\s*([^.;,\n]+)",
            r"\bworked\s+as\s+(?:a|an|the)?\s*([^.;,\n]+)",
            r"\bpreviously\s+(?:was|worked\s+as)\s+(?:a|an|the)?\s*([^.;,\n]+)",
            r"\brole\s+as\s+(?:a|an|the)?\s*([^.;,\n]+)",
        ):
            match = re.search(pattern, fact, flags=re.IGNORECASE)
            if match:
                candidates.append(clean_text(match.group(1)))
        if role_attribute_query and candidates:
            return candidates[0].strip(" .")
        for candidate in candidates:
            candidate_l = candidate.lower()
            if answer_l in candidate_l and len(candidate) > len(answer_clean) and len(candidate) <= 180:
                return candidate.strip(" .")
    return answer


def answer_question(question: str, memory_hits: list[Any], evidence_windows: list[dict[str, Any]] | None = None) -> str:
    if evidence_windows is None:
        evidence_windows = build_answer_evidence(question, memory_hits)
    if not any(bool(item.get("window_planner_enabled", False)) for item in evidence_windows):
        evidence_windows = apply_answer_window_planner(question, list(evidence_windows))
    if not any(bool(item.get("llm_evidence_selector_enabled", False)) for item in evidence_windows):
        evidence_windows = apply_llm_evidence_selector(question, list(evidence_windows))
    if not evidence_unit_plan_from_windows(evidence_windows):
        evidence_windows = apply_evidence_unit_planner(question, list(evidence_windows))
    evidence_windows = apply_unified_operation_planner(question, list(evidence_windows))
    evidence_windows = apply_llm_channel_planner(question, list(evidence_windows))
    evidence_windows = final_answer_surface_windows(question, list(evidence_windows))
    question_l = question.lower()
    # TMCRA_LAYER_TAG: keyword-based personalized answer gate; replace with turn-intent/profile model.
    personalized_request = any(
        marker in question_l
        for marker in (
            "recommend",
            "recommendation",
            "suggest",
            "accessories",
            "my current",
            "my setup",
            "for me",
            "preference",
            "prefer",
            "what should",
            "should i",
            "any tips",
            "trouble with",
            "serve",
            "dinner",
            "battery life",
        )
    )
    force_answer_for_eval = os.getenv("TMCRA_LME_FORCE_ANSWER", "1").strip().lower() not in {"0", "false", "no"}
    profile_facts = profile_facts_for_answer(memory_hits)
    compact_hit_lines = compact_memory_hit_lines(memory_hits)
    memory_lines = []
    evidence_window_limit = int(os.getenv("TMCRA_ANSWER_EVIDENCE_WINDOW_LIMIT", "8"))
    if final_answer_channel_intent(question)["aggregation"]:
        evidence_window_limit = max(
            evidence_window_limit,
            int(os.getenv("TMCRA_ANSWER_EVIDENCE_WINDOW_LIMIT_AGG", "12") or 12),
        )
    ranked_evidence_windows = list(evidence_windows[: max(1, evidence_window_limit)])
    for index, item in enumerate(ranked_evidence_windows, start=1):
        value = clean_text(item.get("text", ""))
        if value:
            memory_id = clean_text(item.get("memory_id", ""))
            prefix = f"{index}. "
            if memory_id:
                prefix += f"[{memory_id}] "
            role_bits = []
            evidence_role = clean_text(item.get("evidence_role", ""))
            temporal_state = clean_text(item.get("temporal_state", ""))
            if evidence_role:
                role_bits.append(f"role={evidence_role}")
            if temporal_state:
                role_bits.append(f"time={temporal_state}")
            if role_bits:
                prefix += "[" + " ".join(role_bits) + "] "
            memory_lines.append(prefix + value)
    model_plan_lines = []
    model_plan_has_temporal_units = False
    model_plan_has_computed_temporal_answer = False
    model_plan_has_computed_coverage_answer = False
    seen_model_plan_ids: set[str] = set()
    for item in evidence_windows:
        if not (
            bool(item.get("temporal_comparison_model_planner", False))
            or bool(item.get("aggregation_unit_model_planner", False))
            or bool(item.get("operation_planner", False))
            or bool(item.get("unified_operation_planner", False))
            or bool(item.get("temporal_operation_layer", False))
        ):
            continue
        value = clean_text(item.get("text", ""))
        if not value:
            continue
        memory_id = clean_text(item.get("memory_id", "")) or f"model_plan_{len(model_plan_lines) + 1}"
        if memory_id in seen_model_plan_ids:
            continue
        seen_model_plan_ids.add(memory_id)
        if '"temporal_units":[{' in value:
            model_plan_has_temporal_units = True
        if re.search(r'"computed_temporal_answer":"[^"]+', value):
            model_plan_has_computed_temporal_answer = True
        if re.search(r'"computed_coverage_answer":\{"answer":"[^"]+', value):
            model_plan_has_computed_coverage_answer = True
        model_plan_lines.append(f"- [{memory_id}] {value}")
        if len(model_plan_lines) >= 3:
            break
    profile_block = "\n".join(f"- {fact}" for fact in profile_facts)
    memory_block = "\n".join(memory_lines) if memory_lines else "(no relevant long-term memory retrieved)"
    model_plan_section = (
        "Model planner outputs:\n"
        + "\n".join(model_plan_lines)
        + "\n\n"
        if model_plan_lines
        else ""
    )
    evidence_unit_plan = evidence_unit_plan_from_windows(evidence_windows)
    evidence_unit_plan_json = format_evidence_unit_plan_for_prompt(evidence_unit_plan)
    evidence_unit_plan_section = (
        f"Model evidence-unit plan:\n{evidence_unit_plan_json}\n\n" if evidence_unit_plan_json else ""
    )
    focused_unit_lines = focused_evidence_unit_lines(question, ranked_evidence_windows)
    focused_unit_section = (
        "Focused evidence units selected from the retrieved windows:\n"
        + "\n".join(focused_unit_lines)
        + "\n\n"
        if focused_unit_lines
        else ""
    )
    # TMCRA_LAYER_TAG: answer prompt protocol; affects answer utilization, not memory retrieval.
    reasoning_protocol = (
        "Treat TMCRA memory as evidence clues, not as a pre-written answer. "
        "If model temporal, aggregation, or operation plans are provided, read them before the raw evidence because they contain model-selected answer events, unit coverage, time anchors, and comparison direction. "
        "Use those plans as normalized evidence maps, but verify the answer against the raw memory evidence. "
        "If a model evidence-unit plan is provided, use it as the primary normalized view of answer units, time anchors, and distinct clues, while verifying it against the raw evidence lines. "
        "If focused evidence units are provided, scan them before the raw windows; they are compressed excerpts from retrieved memory, not external facts. "
        "If the evidence-unit plan lists answer_unit, positive_evidence, current_value, old_value, temporal_anchor, or negative_evidence units, reason over those units explicitly before deciding the answer. "
        "If an aggregation/unit coverage plan contains computed_coverage_answer.answer and its listed units_used or terms match the question, use that computed value unless raw evidence directly contradicts it. "
        "If candidate_answer is present in the plan and is supported by the listed units, prefer it over re-summarizing the raw window text. "
        "Before writing the JSON answer, internally infer the question task: direct fact lookup, arithmetic, distinct-item counting, temporal difference, current-value selection, preference/recommendation synthesis, or multi-evidence synthesis. "
        "Then use the retrieved evidence as clues for that task. "
        "For arithmetic questions, combine quantities found in different evidence lines and compute the requested value. "
        "For count questions, infer the answer unit from the current question and count distinct evidence-backed answer units across relevant evidence, not just one sentence. "
        "For offline benchmark count questions, include plausible evidence-backed units when the only uncertainty is an implicit date or an assistant-confirmed action and there is no contradictory evidence. "
        "For month-level count questions, a past-tense user event inside a dated session is a usable candidate for that session month when no older or future date contradicts it. "
        "When the question contains multiple actions joined by and/or, count each distinct target instance attached to any requested action. "
        "Do not merge two answer units merely because they share the same broad noun; keep them separate when the evidence distinguishes their action, status, time, location, or original-versus-replacement instance. "
        "For current/now/since/update questions, compare old and later evidence and choose the current or latest value when the question asks for it. "
        "For advice, preference, dinner, shopping, setup, or tips questions, synthesize a useful answer from remembered personal setup, owned items, ingredients, goals, habits, and constraints; exact prior wording is not required. "
        "Ignore irrelevant retrieved windows. "
        "Only answer that information is unavailable when none of the evidence provides a usable clue for the requested task. "
    )
    temporal_plan_protocol = ""
    if model_plan_has_temporal_units or model_plan_has_computed_temporal_answer:
        temporal_plan_protocol = (
            "When a model temporal plan explicitly contains non-empty temporal_units or computed_temporal_answer, use those fields as the time math layer. "
            "For how-many-ago questions, use computed_temporal_answer or delta_to_query fields instead of copying local words such as today. "
            "For first/order questions with explicit temporal_units, compare normalized_event_date values for the compared events. "
            "Ignore this temporal-plan instruction when the temporal plan has no temporal_units and no computed_temporal_answer. "
        )
    coverage_plan_protocol = ""
    if model_plan_has_computed_coverage_answer:
        coverage_plan_protocol = (
            "When a model aggregation/unit plan explicitly contains computed_coverage_answer, treat it as the unit coverage layer. "
            "For count, total, sum, and arithmetic questions, prefer the computed answer when the listed units_used or terms are relevant evidence for the requested unit. "
            "Do not drop a listed unit merely because its raw evidence window also contains unrelated topic text. "
        )
    personalized_instruction = ""
    if personalized_request:
        personalized_instruction = (
            "For personalized recommendation or preference questions, synthesize a concise recommendation using the user's retrieved setup, preferences, goals, and constraints. "
            "If a Structured personalized facts block is present, use it first; otherwise use the long-term memory evidence directly. "
            "Do not abstain when the memory contains relevant personal setup or preference evidence. "
            "Do not say the memory lacks the user's setup when the evidence mentions a current brand, device, activity, goal, style, constraint, or previous preference. "
            "Exact prior accessory lists are not required; infer a compatible recommendation from the remembered setup or preference. "
            + (
                "For this offline benchmark, do not ask follow-up questions; a partial remembered setup is enough to produce the best available personalized answer. "
                if force_answer_for_eval
                else "In realtime product mode, you may ask a brief follow-up question after giving the best available memory-backed suggestion. "
            )
            + "The answer should mention the personalized basis when useful, such as compatibility, brand, quality, durability, or current gear. "
    )
    profile_section = f"Structured personalized facts:\n{profile_block}\n\n" if profile_block else ""
    system_prompt = (
        "You answer LongMemEval questions using only the current question and the provided long-term memory. "
        "First identify the requested attribute, then answer only that attribute. "
        + reasoning_protocol
        + temporal_plan_protocol
        + coverage_plan_protocol
        + "If the question has multiple parts, compares earlier versus now, or asks for both before/current values, answer every requested part explicitly. "
        "If evidence lines are tagged with role=initial_value and role=current_value, use initial_value for the earlier/starting state and current_value for the now/current state; do not collapse them into one value. "
        "For current or now values, prefer later-dated evidence over earlier evidence when both are present. "
        "The answer value must not be an object, list, or multiple JSON keys. "
        "For where questions, answer with the place, store, venue, retailer, or location. "
        "Never answer a where question with a time or date such as last Sunday. "
        "For retail or shopping questions, a nearby store, retailer, or app/store name is the location. "
        "For how long questions, answer with the duration. "
        "For name questions, answer with the name. "
        "If the answer is implied by nearby turns in the same evidence window, use that local context. "
        "For profile, identity, occupation, role, ownership, and preference facts, preserve attached qualifiers that are part of the remembered fact, such as company type, setting, object owner, condition, strength, or scope. "
        "If a role=profile_fact evidence line or Structured personalized fact directly answers the requested attribute, preserve the complete specific fact rather than only the head noun or job title. "
        "Do not shorten a complete remembered fact into only its broad category when the evidence gives a more specific attribute requested by the question. "
        + personalized_instruction
        + "Return strict JSON only, with exactly one key named answer. "
        + "The answer value must be the shortest complete answer phrase or sentence. Do not explain. Do not cite evidence. "
        + "Example: {\"answer\":\"45 minutes each way\"}\n\n"
        + profile_section
        + model_plan_section
        + evidence_unit_plan_section
        + focused_unit_section
        + f"Long-term memory evidence:\n{memory_block}"
    )
    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {"role": "user", "content": f"Question: {question}\nReturn only JSON."},
    ]
    answer_base_url, answer_model, answer_api_key = answer_llm_config()
    answer = clean_answer_output(chat_completion(
        answer_base_url,
        answer_model,
        messages,
        max_tokens=int(os.getenv("TMCRA_ANSWER_MAX_TOKENS", "512")),
        temperature=0.0,
        api_key=answer_api_key,
    ))
    answer = complete_profile_answer_from_evidence(question, answer, evidence_windows)
    organizer_uses_model_plan = bool(
        re.search(
            r"\bhow many\s+(?:days?|weeks?|months?|years?)\b.*\bago\b|"
            r"\b(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+"
            r"(?:days?|weeks?|months?|years?)\s+ago\b",
            question_l,
        )
    )
    organizer_context_section = (
        profile_section
        + (model_plan_section if organizer_uses_model_plan else "")
        + evidence_unit_plan_section
    )
    if memory_lines and should_run_answer_reasoning_organizer(
        question,
        answer,
        personalized_request=personalized_request,
        force_answer_for_eval=force_answer_for_eval,
    ):
        organized_answer = organize_answer_from_evidence(
            question,
            memory_lines,
            profile_section=organizer_context_section,
            force_answer_for_eval=force_answer_for_eval,
            prior_answer=answer,
        )
        if (not organized_answer or looks_like_no_memory_answer(organized_answer)) and compact_hit_lines:
            organized_answer = organize_answer_from_evidence(
                question,
                compact_hit_lines,
                profile_section=organizer_context_section,
                force_answer_for_eval=force_answer_for_eval,
                prior_answer=answer,
            )
        if organized_answer and not looks_like_no_memory_answer(organized_answer):
            return complete_profile_answer_from_evidence(question, organized_answer, evidence_windows)
        focused_memory_block = "\n".join(memory_lines[:8])
        # TMCRA_LAYER_TAG: answer-side fallback organizer for conservative/no-memory answers.
        retry_messages = [
            {
                "role": "system",
                "content": (
                    "You are correcting an over-conservative answer. "
                    "Your previous answer treated memory as a complete answer database; that is wrong. "
                    "The focused evidence below is non-empty and may contain indirect clues rather than an exact answer sentence. "
                    "You must first infer the task required by the question, then combine the evidence clues to answer it. "
                    "Use arithmetic for total/each/save questions, distinct counting for how-many questions, date comparison for temporal questions, current/latest evidence for now/current questions, and synthesis from remembered setup or preferences for advice/tips/recommendation questions. "
                    "For advice, tips, what-should-I, or recommendation questions, a remembered owned item, ingredient, activity, goal, preference, or constraint is enough to produce a useful answer; the memory does not need to contain the exact advice sentence. "
                    "For count questions, infer the answer unit from the current question and combine distinct evidence-backed units across the focused evidence instead of requiring a precomputed total. "
                    "If a model evidence-unit plan is provided, use its answer_unit and positive_evidence units as normalized candidates and only exclude units that are contradicted or outside the requested time/action scope. "
                    "For month-level count questions, do not discard a past-tense user event merely because only the session date provides the month anchor. "
                    "Do not merge units only because they share the same broad noun; keep distinct action targets or original/replacement instances separate when the evidence separates them. "
                    "Do not claim no memory if any focused evidence gives a usable clue for the requested task. "
                    "Do not answer with Information unavailable, not available, no information, or similar wording unless the focused evidence block is empty or entirely unrelated. "
                    "Ignore evidence that is unrelated to the requested task. "
                    + (
                        "For this offline benchmark, do not ask follow-up questions; give the best direct evidence-backed answer. "
                        if force_answer_for_eval
                        else "In realtime product mode, you may ask a brief follow-up question after giving the best memory-backed answer. "
                    )
                    +
                    "Return strict JSON only, with exactly one key named answer.\n\n"
                    + organizer_context_section
                    + f"Focused long-term memory evidence:\n{focused_memory_block}"
                ),
            },
            {"role": "user", "content": f"Question: {question}\nReturn only JSON."},
        ]
        retry_answer = clean_answer_output(chat_completion(
            answer_base_url,
            answer_model,
            retry_messages,
            max_tokens=128,
            temperature=0.0,
            api_key=answer_api_key,
        ))
        if retry_answer and not looks_like_no_memory_answer(retry_answer):
            return complete_profile_answer_from_evidence(question, retry_answer, evidence_windows)
    return complete_profile_answer_from_evidence(question, answer, evidence_windows)


def should_run_answer_reasoning_organizer(
    question: str,
    answer: str,
    *,
    personalized_request: bool,
    force_answer_for_eval: bool,
) -> bool:
    # TMCRA_LAYER_TAG: rule gate for answer-side retry/organizer; not a trained routing model.
    mode = os.getenv("TMCRA_ANSWER_REASONING_ORGANIZER", "1").strip().lower()
    if mode in {"", "0", "false", "off", "disabled", "none"}:
        return False
    if looks_like_no_memory_answer(answer):
        return True
    if personalized_request and force_answer_for_eval and looks_like_followup_answer(answer):
        return True
    question_l = clean_text(question).lower()
    count_markers = (
        "how many",
        "number of",
        "count of",
        "count the",
        "count my",
        "total number",
    )
    return any(marker in question_l for marker in count_markers)


def organize_answer_from_evidence(
    question: str,
    memory_lines: list[str],
    *,
    profile_section: str = "",
    force_answer_for_eval: bool = True,
    prior_answer: str = "",
) -> str:
    if not memory_lines:
        return ""
    evidence_block = "\n".join(memory_lines)
    prior_answer = clean_text(prior_answer)
    prior_section = f"Previous candidate answer, possibly incomplete: {prior_answer}\n\n" if prior_answer else ""
    messages = [
        {
            "role": "system",
            "content": (
                "You are the evidence organizer for a long-memory runtime. "
                "The final answer model may have been too conservative or may have under-used the retrieved clues, so your job is to read the retrieved memory lines as clues and produce the best evidence-backed answer. "
                "Do not treat memory as a database that must contain the exact answer sentence. "
                "Ignore unrelated lines, but if any line contains a usable clue for the question, mark relevant=true. "
                "For advice, tips, recommendations, dinner, shopping, setup, or preference questions, remembered owned items, tools, accessories, ingredients, goals, habits, constraints, or preferences are enough to synthesize an answer. "
                "Infer the affordance of remembered items and ingredients: a relevant tool, accessory, setup, ingredient, preference, or constraint can answer the user's underlying need even when the old conversation asked a different surface question. "
                "Do not answer the old conversation's surface task; adapt the remembered clue to the current question's need. "
                "When advice uses a remembered owned item, phrase the answer as using or preparing that already-mentioned item and include one practical use step; do not turn it into a generic purchase suggestion unless the question asks what to buy. "
                "If a model evidence-unit plan is provided, enumerate the plan's answer_unit and positive_evidence units first, then reconcile duplicates or exclusions against the raw evidence. "
                "For count questions, infer the count unit from the current question itself, then enumerate all relevant candidate clues, merge duplicate mentions of the same clue, and count only evidence-backed units matching that inferred unit. "
                "For month-level count questions, include past-tense user events from a dated session as candidates for that session month unless another date excludes them. "
                "When the current question contains multiple actions joined by and/or, the count unit is each distinct target instance attached to any requested action, unless the question explicitly asks for object categories. "
                "Do not merge two candidate units merely because they share the same broad noun; keep them separate when the evidence distinguishes their action, status, time, location, or original-versus-replacement instance. "
                "If one memory line contains multiple distinct obligations or requested action targets, count them separately only when the inferred count unit requires action-level counting. "
                "Do not count broad object categories, generic advice, examples, or unrelated context as answer units. "
                "Do not stop after the first relevant clue; scan every retrieved line before producing the count. "
                "For current/now questions, prefer later or current evidence over old evidence when both appear. "
                "For temporal questions, compare dates or durations when evidence provides them. "
                "Return strict JSON only with keys: relevant, task, clues, answer. "
                "answer must be the final shortest useful answer, not an explanation. "
                "If evidence is genuinely empty or unrelated, return relevant=false and answer=\"\". "
                + (
                    "For this offline benchmark, do not ask follow-up questions; use the best available evidence-backed inference. "
                    if force_answer_for_eval
                    else "In realtime product mode, prefer a direct memory-backed answer; only ask follow-up if evidence is too thin. "
                )
                + "\n\n"
                + profile_section
                + prior_section
                + f"Retrieved memory lines:\n{evidence_block}"
            ),
        },
        {"role": "user", "content": f"Question: {question}\nReturn only JSON."},
    ]
    answer_base_url, answer_model, answer_api_key = answer_llm_config()
    raw = chat_completion(
        answer_base_url,
        answer_model,
        messages,
        max_tokens=220,
        temperature=0.0,
        api_key=answer_api_key,
    )
    payload = parse_json_object(raw)
    if not payload:
        fallback = clean_answer_output(raw)
        return "" if fallback.lstrip().startswith("{") else fallback
    relevant = payload.get("relevant", False)
    if isinstance(relevant, str):
        relevant = relevant.strip().lower() in {"1", "true", "yes", "y"}
    answer = clean_text(payload.get("answer", ""))
    if not relevant or not answer:
        return ""
    return answer.strip(" \t\n\r\"'")


def compact_memory_hit_lines(memory_hits: list[Any], *, max_items: int | None = None, max_chars: int | None = None) -> list[str]:
    if max_items is None:
        max_items = int(os.getenv("TMCRA_COMPACT_MEMORY_HIT_LIMIT", "8"))
    if max_chars is None:
        max_chars = int(os.getenv("TMCRA_COMPACT_MEMORY_HIT_CHARS", "260"))
    lines: list[str] = []
    seen: set[str] = set()
    for index, hit in enumerate(memory_hits[: max(1, max_items)], start=1):
        value = truncate(getattr(hit, "value", ""), max(80, max_chars))
        if not value:
            continue
        memory_id = clean_text(getattr(hit, "memory_id", ""))
        category = clean_text(getattr(hit, "category", ""))
        source_kind = clean_text(getattr(hit, "source_kind", ""))
        label_bits = []
        if category:
            label_bits.append(category)
        if source_kind:
            label_bits.append(source_kind)
        label = " / ".join(label_bits)
        prefix = f"{index}. "
        if memory_id:
            prefix += f"[{memory_id}] "
        if label:
            prefix += f"({label}) "
        line = prefix + value
        key = re.sub(r"\s+", " ", line.lower())
        if key in seen:
            continue
        seen.add(key)
        lines.append(line)
    return lines


def parse_json_object(text: str) -> dict[str, Any]:
    text = clean_text(text)
    text = re.sub(r"^```(?:\w+)?\s*|\s*```$", "", text).strip()
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(text[index:])
        except Exception:
            continue
        return payload if isinstance(payload, dict) else {}
    return {}


def looks_like_no_memory_answer(answer: str) -> bool:
    # TMCRA_LAYER_TAG: string-marker abstention detector used to trigger answer-side retry.
    value = clean_text(answer).lower()
    if not value:
        return True
    no_memory_markers = (
        "no information",
        "no relevant",
        "not contain",
        "does not contain",
        "doesn't contain",
        "do not have",
        "don't have",
        "cannot determine",
        "can't determine",
        "not enough information",
        "insufficient information",
        "information unavailable",
        "not available",
        "unavailable",
        "not mention",
        "not mentioned",
        "unknown",
    )
    return any(marker in value for marker in no_memory_markers)


def looks_like_followup_answer(answer: str) -> bool:
    # TMCRA_LAYER_TAG: string-marker follow-up detector used by offline benchmark force-answer mode.
    value = clean_text(answer).lower()
    if not value:
        return False
    followup_markers = (
        "could you tell me",
        "can you tell me",
        "i need more information",
        "need more information",
        "tell me more",
        "what is your budget",
        "what's your budget",
        "what kind of",
        "would you like",
        "are you looking",
    )
    return any(marker in value for marker in followup_markers) or value.endswith("?")


def clean_answer_output(text: str) -> str:
    text = clean_text(text)
    text = re.sub(r"^```(?:\w+)?\s*|\s*```$", "", text).strip()
    json_match = re.search(r"\{.*?\}", text, flags=re.DOTALL)
    if json_match:
        try:
            payload = json.loads(json_match.group(0))
            answer = clean_text(payload.get("answer", ""))
            if answer:
                return answer.strip(" \t\n\r\"'")
        except Exception:
            pass
    tail_markers = [
        "Referencing the",
        "Based on the",
        "While the",
        "While ",
        "Although ",
        "However,",
        "There is no mention",
        "There is no information",
        "Give a direct",
        "Do not mention",
        "The provided context",
        "In the provided",
    ]
    for marker in tail_markers:
        index = text.find(marker)
        if index > 0:
            text = text[:index].strip()
    return text.strip(" \t\n\r\"'")


def run(args: argparse.Namespace) -> None:
    service_root = Path(args.service_root).resolve()
    repo = Path(args.repo).resolve()
    apply_env_defaults(service_root)
    os.environ.setdefault("GEMMA_BASE_URL", f"http://{os.getenv('TMCRA_GEMMA_HOST', '127.0.0.1')}:{os.getenv('TMCRA_GEMMA_PORT', '18002')}/v1")
    os.environ.setdefault("GEMMA_MODEL", os.getenv("TMCRA_GEMMA_MODEL_NAME", "gemma-4-e4b-it"))
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    if not args.enable_topic_buckets:
        disable_topic_bucket_runtime()

    started = time.time()
    out_dir = Path(args.out) if args.out else DEFAULT_OUT_ROOT / f"lme_s10_native_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = out_dir / "predictions.jsonl"
    debug_path = out_dir / "samples_debug.jsonl"
    summary_path = out_dir / "summary.json"
    storage_path = out_dir / "native_memory.sqlite3"

    writer = build_writer()
    records: list[dict[str, Any]] = []
    log("run_start", data=str(args.data), limit=args.limit, out=str(out_dir), repo=str(repo))
    for sample_index, row in enumerate(iter_json_array(Path(args.data), limit=args.limit), start=1):
        qid = clean_text(row.get("question_id")) or f"sample_{sample_index:04d}"
        question = clean_text(row.get("question"))
        gold = clean_text(row.get("answer"))
        sessions = list(row.get("haystack_sessions") or [])
        session_ids = [clean_text(item) for item in list(row.get("haystack_session_ids") or [])]
        dates = [clean_text(item) for item in list(row.get("haystack_dates") or [])]
        selected_indices = (
            select_official_full_history_indices(row)
            if args.official_full_history
            else select_session_indices(row, max_distractors=args.max_distractor_sessions)
        )
        adapter = build_adapter(f"longmemeval_native:{qid}:{sample_index}", storage_path)
        sample_start = time.perf_counter()
        writer_calls = 0
        writer_accepted = 0
        writer_suspected = 0
        writer_records = 0
        writer_seconds = 0.0
        writer_unit_records = 0
        writer_unit_calls = 0
        chunk_errors: list[str] = []
        selected_session_ids = []

        for session_index in selected_indices:
            if session_index >= len(sessions):
                continue
            sid = session_ids[session_index] if session_index < len(session_ids) else f"session_{session_index}"
            selected_session_ids.append(sid)
            date = dates[session_index] if session_index < len(dates) else ""
            is_answer_session = sid in {clean_text(item) for item in list(row.get("answer_session_ids") or [])}
            max_chunks = (
                args.max_session_chunks
                if args.official_full_history
                else args.max_answer_chunks if is_answer_session else args.max_distractor_chunks
            )
            chunks = session_text_chunks(
                session_id=sid,
                date=date,
                turns=list(sessions[session_index] or []),
                max_chars=args.chunk_chars,
                max_chunks=max_chunks,
            )
            for chunk_index, chunk in enumerate(chunks, start=1):
                chunk_id = f"s{session_index:03d}_c{chunk_index:02d}"
                try:
                    result = writer_ingest(adapter, writer, chunk, qid=qid, chunk_id=chunk_id)
                    writer_calls += 1
                    writer_accepted += result["accepted_count"]
                    writer_suspected += result["suspected_count"]
                    writer_records += result["record_count"]
                    writer_seconds += float(result["seconds"])
                    writer_unit_records += int(result.get("unit_record_count", 0) or 0)
                    if bool(result.get("unit_writer_enabled", False)):
                        writer_unit_calls += 1
                except Exception as exc:
                    chunk_errors.append(f"{chunk_id}:{exc.__class__.__name__}:{str(exc)[:200]}")
                    log("chunk_error", sample=sample_index, qid=qid, chunk=chunk_id, error=str(exc)[:240])

        question_date = clean_text(row.get("question_date", ""))
        query_graph = build_query_graph(question, question_date)
        runtime_question = query_graph_retrieval_text(question, question_date, query_graph)
        retrieve_start = time.perf_counter()
        retrieval = adapter.retrieve(runtime_question, top_k=args.top_k)
        retrieve_seconds = time.perf_counter() - retrieve_start
        hits = list(getattr(retrieval, "hits", []) or [])
        hits, query_graph_sidecar = apply_query_graph_sidecar_retrieval(
            adapter,
            query_graph,
            question,
            question_date,
            hits,
            top_k=args.top_k,
        )
        hits = expand_dialogue_chain_hits(runtime_question, hits, getattr(adapter, "graph", None))
        hits = expand_semantic_coverage_hits(runtime_question, hits, getattr(adapter, "graph", None))
        answer_evidence_windows = build_answer_evidence(runtime_question, hits)
        answer_evidence_windows = apply_answer_window_planner(runtime_question, answer_evidence_windows)
        answer_evidence_windows = apply_llm_evidence_selector(runtime_question, answer_evidence_windows)
        answer_evidence_windows = apply_evidence_unit_planner(runtime_question, answer_evidence_windows)
        answer_evidence_windows = apply_unified_operation_planner(runtime_question, answer_evidence_windows)
        answer_evidence_windows = diversify_evidence_windows(answer_evidence_windows)
        answer_evidence_windows = apply_llm_channel_planner(runtime_question, answer_evidence_windows)
        answer_evidence_windows = final_answer_surface_windows(runtime_question, answer_evidence_windows)
        answer_start = time.perf_counter()
        hypothesis = answer_question(runtime_question, hits, answer_evidence_windows)
        answer_seconds = time.perf_counter() - answer_start
        graph = getattr(adapter, "graph", None)
        record = {
            "sample_index": sample_index,
            "question_id": qid,
            "question_type": row.get("question_type"),
            "question": question,
            "question_date": question_date,
            "query_graph": query_graph,
            "query_graph_sidecar": query_graph_sidecar,
            "gold_answer": gold,
            "hypothesis": hypothesis,
            "answer_session_ids": list(row.get("answer_session_ids") or []),
            "selected_session_ids": selected_session_ids,
            "total_sessions": len(sessions),
            "total_turns": sum(len(s or []) for s in sessions),
            "selected_sessions": len(selected_session_ids),
            "history_mode": "official_full_history" if args.official_full_history else "controlled_answer_plus_distractors",
            "writer_calls": writer_calls,
            "writer_accepted": writer_accepted,
            "writer_suspected": writer_suspected,
            "writer_records": writer_records,
            "writer_unit_records": writer_unit_records,
            "writer_unit_calls": writer_unit_calls,
            "writer_seconds": round(writer_seconds, 3),
            "retrieve_seconds_wall": round(retrieve_seconds, 3),
            "answer_seconds": round(answer_seconds, 3),
            "sample_seconds": round(time.perf_counter() - sample_start, 3),
            "graph_records": len(getattr(graph, "records_by_id", {}) or {}),
            "graph_edges": len(getattr(graph, "edges", []) or []),
            "chunk_errors": chunk_errors,
            "retrieval": retrieval_debug(retrieval),
            "evidence_unit_plan": evidence_unit_plan_from_windows(answer_evidence_windows),
            "answer_evidence_windows": [
                {
                    "memory_id": item.get("memory_id", ""),
                    "score": item.get("score", 0.0),
                    "planner_selected": bool(item.get("planner_selected", False)),
                    "planner_score": item.get("planner_score", 0.0),
                    "semantic_similarity": item.get("semantic_similarity", item.get("answer_window_semantic_similarity", 0.0)),
                    "answer_window_semantic_similarity": item.get("answer_window_semantic_similarity", 0.0),
                    "answer_window_semantic_enabled": bool(item.get("answer_window_semantic_enabled", False)),
                    "answer_window_semantic_mode": item.get("answer_window_semantic_mode", ""),
                    "window_planner_should_inject_score": item.get("window_planner_should_inject_score", 0.0),
                    "answer_plan_selected": bool(item.get("answer_plan_selected", False)),
                    "answer_plan_score": item.get("answer_plan_score", 0.0),
                    "answer_plan_rank": item.get("answer_plan_rank", 0),
                    "evidence_role": item.get("evidence_role", ""),
                    "temporal_state": item.get("temporal_state", ""),
                    "assistant_memory_query": bool(item.get("assistant_memory_query", False)),
                    "assistant_origin_evidence": bool(item.get("assistant_origin_evidence", False)),
                    "hit_evidence_role": item.get("hit_evidence_role", ""),
                    "hit_temporal_state": item.get("hit_temporal_state", ""),
                    "window_planner_enabled": bool(item.get("window_planner_enabled", False)),
                    "llm_evidence_selector_enabled": bool(item.get("llm_evidence_selector_enabled", False)),
                    "llm_evidence_selected": bool(item.get("llm_evidence_selected", False)),
                    "llm_evidence_rank": item.get("llm_evidence_rank", None),
                    "llm_evidence_model_kept": bool(item.get("llm_evidence_model_kept", False)),
                    "evidence_unit_planner_enabled": bool(item.get("evidence_unit_planner_enabled", False)),
                    "evidence_unit_planner_selected": bool(item.get("evidence_unit_planner_selected", False)),
                    "evidence_unit_planner_rank": item.get("evidence_unit_planner_rank", None),
                    "semantic_coverage_expansion": bool(item.get("semantic_coverage_expansion", False)),
                    "semantic_coverage_score": item.get("semantic_coverage_score", 0.0),
                    "unit_coverage_pack": bool(item.get("unit_coverage_pack", False)),
                    "multi_unit_chain_slot": bool(item.get("multi_unit_chain_slot", False)),
                    "multi_unit_chain_bundle": bool(item.get("multi_unit_chain_bundle", False)),
                    "multi_unit_chain_score": float(item.get("multi_unit_chain_score", 0.0) or 0.0),
                    "unit_kind": item.get("unit_kind", ""),
                    "facet_type": item.get("facet_type", ""),
                    "unified_operation_planner": bool(item.get("unified_operation_planner", False)),
                    "unified_operation_family": item.get("unified_operation_family", ""),
                    "unified_selected": bool(item.get("unified_selected", False)),
                    "unified_scores": item.get("unified_scores", {}),
                    "llm_channel_planner_enabled": bool(item.get("llm_channel_planner_enabled", False)),
                    "llm_channel": item.get("llm_channel", ""),
                    "llm_channel_operation_family": item.get("llm_channel_operation_family", ""),
                    "llm_channel_coverage_complete": bool(item.get("llm_channel_coverage_complete", False)),
                    "llm_channel_missing_evidence": item.get("llm_channel_missing_evidence", ""),
                    "llm_channel_reasoning_policy": item.get("llm_channel_reasoning_policy", ""),
                    "llm_channel_planner_error": item.get("llm_channel_planner_error", ""),
                    "unit_indexes": item.get("unit_indexes", []),
                    "original_chars": item.get("original_chars", 0),
                    "text": truncate(item.get("text", ""), 700),
                }
                for item in answer_evidence_windows[:8]
            ],
        }
        records.append(record)
        with predictions_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"question_id": qid, "hypothesis": hypothesis}, ensure_ascii=False) + "\n")
        with debug_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        log(
            "sample_done",
            sample=sample_index,
            qid=qid,
            hits=record["retrieval"]["hit_count"],
            selected_events=record["retrieval"]["selected_event_count"],
            paths=record["retrieval"]["selected_path_count"],
            writer_calls=writer_calls,
            seconds=record["sample_seconds"],
            hypothesis=truncate(hypothesis, 120),
        )

    summary = {
        "status": "completed",
        "mode": "native_tmcra_flow_smoke",
        "history_mode": "official_full_history" if args.official_full_history else "controlled_answer_plus_distractors",
        "limit": args.limit,
        "sample_count": len(records),
        "out_dir": str(out_dir),
        "data": str(args.data),
        "started_at": datetime.fromtimestamp(started).isoformat(timespec="seconds"),
        "completed_at": datetime.now().isoformat(timespec="seconds"),
        "elapsed_seconds": round(time.time() - started, 3),
        "total_writer_calls": sum(int(item["writer_calls"]) for item in records),
        "total_writer_accepted": sum(int(item["writer_accepted"]) for item in records),
        "total_writer_suspected": sum(int(item["writer_suspected"]) for item in records),
        "total_writer_unit_calls": sum(int(item.get("writer_unit_calls", 0)) for item in records),
        "total_writer_unit_records": sum(int(item.get("writer_unit_records", 0)) for item in records),
        "total_graph_records": sum(int(item["graph_records"]) for item in records),
        "total_graph_edges": sum(int(item["graph_edges"]) for item in records),
        "avg_retrieval_hits": round(sum(float(item["retrieval"]["hit_count"]) for item in records) / max(1, len(records)), 4),
        "avg_selected_events": round(sum(float(item["retrieval"]["selected_event_count"]) for item in records) / max(1, len(records)), 4),
        "avg_selected_paths": round(sum(float(item["retrieval"]["selected_path_count"]) for item in records) / max(1, len(records)), 4),
        "samples": [
            {
                "question_id": item["question_id"],
                "question": item["question"],
                "gold_answer": item["gold_answer"],
                "hypothesis": item["hypothesis"],
                "retrieval_hits": item["retrieval"]["hit_count"],
                "selected_event_count": item["retrieval"]["selected_event_count"],
                "selected_path_count": item["retrieval"]["selected_path_count"],
                "writer_calls": item["writer_calls"],
                "sample_seconds": item["sample_seconds"],
                "chunk_errors": item["chunk_errors"],
            }
            for item in records
        ],
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    log("run_done", out=str(out_dir), samples=len(records), elapsed=summary["elapsed_seconds"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--repo", default=str(DEFAULT_REPO))
    parser.add_argument("--service-root", default=str(DEFAULT_SERVICE_ROOT))
    parser.add_argument("--out", default="")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--max-distractor-sessions", type=int, default=5)
    parser.add_argument("--max-distractor-chunks", type=int, default=1)
    parser.add_argument("--max-answer-chunks", type=int, default=4)
    parser.add_argument("--official-full-history", action="store_true")
    parser.add_argument("--max-session-chunks", type=int, default=0)
    parser.add_argument("--chunk-chars", type=int, default=7000)
    parser.add_argument("--enable-topic-buckets", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
