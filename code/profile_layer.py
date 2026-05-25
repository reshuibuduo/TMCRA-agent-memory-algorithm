from __future__ import annotations

import re
from typing import Any, Iterable, Mapping, Sequence


PROFILE_CATEGORIES = {"profile", "preference", "goal", "constraint", "stage_state", "status"}
PROFILE_TYPES = {"setup", "preference", "constraint", "goal", "avoid", "usage_context"}
PROFILE_SEMANTIC_SLOTS = {"identity", "research_topic", "education", "occupation"}
PROFILE_AGGREGATE_SOURCE_KIND = "public_dialog_profile"
PROFILE_AGGREGATE_CATEGORY = "profile"
PROFILE_CLUSTER_SOURCE_KIND = "public_dialog_profile_cluster"
PROFILE_CLUSTER_CATEGORY = "profile"
PROFILE_CONSOLIDATOR_VERSION = "profile_consolidator_v1_structured_summary"

_PROFILE_QUERY_MARKERS = (
    "preference",
    "prefer",
    "like",
    "dislike",
    "recommend",
    "suggest",
    "advice",
    "advise",
    "any advice",
    "any suggestions",
    "any tips",
    "tips",
    "trouble with",
    "struggling with",
    "what do you think",
    "learn more",
    "resources",
    "suited",
    "fit me",
    "for me",
    "based on me",
    "based on my",
    "my setup",
    "my profile",
    "my goal",
    "my constraint",
    "my occupation",
    "my previous occupation",
    "my role",
    "my previous role",
    "my job",
    "my previous job",
    "what was my",
    "where did i work",
    "worked as",
    "occupation",
    "previous occupation",
    "role",
    "previous role",
    "job",
    "career",
    "background",
    "identity",
    "experience",
    "avoid",
    "what should i",
    "should i",
    "what should",
    "serve",
    "dinner",
    "homegrown",
    "ingredients",
    "battery life",
    "getting around",
    "偏好",
    "喜欢",
    "不喜欢",
    "推荐",
    "建议",
    "适合我",
    "根据我",
    "我的情况",
    "我的配置",
    "我的目标",
    "我的约束",
    "画像",
    "避免",
)


def _clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def _normalize(value: Any) -> str:
    return _clean_text(value).lower()


def _tokens(value: Any) -> list[str]:
    text = _normalize(value)
    english = re.findall(r"[a-z0-9_]+", text)
    cjk = [char for char in text if "\u4e00" <= char <= "\u9fff"]
    return _dedupe([*english, *cjk])


def _slug(value: Any, *, fallback: str = "general") -> str:
    text = _normalize(value)
    parts = re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", text)
    slug = "_".join(parts[:10]).strip("_")
    return slug or fallback


def _dedupe(items: Iterable[Any], *, max_items: int | None = None) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()
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


def _first_nonempty(*values: Any) -> str:
    for value in values:
        text = _clean_text(value)
        if text:
            return text
    return ""


def _bounded_union(*groups: Iterable[Any], max_items: int) -> list[str]:
    return _dedupe([item for group in groups for item in group], max_items=max_items)


def _bounded_int_union(*groups: Iterable[Any], max_items: int) -> list[int]:
    values: list[int] = []
    seen: set[int] = set()
    for group in groups:
        for item in group:
            try:
                value = int(item)
            except Exception:
                continue
            if value in seen:
                continue
            seen.add(value)
            values.append(value)
            if len(values) >= max_items:
                return sorted(values)
    return sorted(values)


def is_profile_layer_record(*, category: Any, source_kind: Any = "", semantic_slot: Any = "", metadata: Mapping[str, Any] | None = None) -> bool:
    data = dict(metadata or {})
    if bool(data.get("profile_layer")):
        return True
    category_text = _normalize(category)
    slot_text = _normalize(semantic_slot or data.get("semantic_slot", ""))
    source_text = _normalize(source_kind)
    if category_text in PROFILE_CATEGORIES:
        return True
    if slot_text in PROFILE_SEMANTIC_SLOTS or slot_text.startswith("profile_"):
        return True
    return source_text in {
        "public_dialog_profile",
        "public_dialog_profile_cluster",
        "public_dialog_preference",
        "public_dialog_goal",
        "public_dialog_constraint",
    }


def infer_profile_type(*, category: Any, semantic_slot: Any = "", relation: Any = "", value: Any = "", metadata: Mapping[str, Any] | None = None) -> str:
    data = dict(metadata or {})
    explicit = _normalize(data.get("profile_type", ""))
    if explicit in PROFILE_TYPES:
        return explicit
    category_text = _normalize(category)
    combined = _normalize(f"{semantic_slot} {relation} {value}")
    if category_text == "preference" or any(marker in combined for marker in ("prefer", "like", "default", "偏好", "喜欢", "默认")):
        return "preference"
    if category_text == "constraint" or any(marker in combined for marker in ("must", "cannot", "forbid", "constraint", "必须", "不能", "约束")):
        return "constraint"
    if category_text == "goal" or any(marker in combined for marker in ("goal", "target", "objective", "目标")):
        return "goal"
    if any(marker in combined for marker in ("avoid", "dislike", "do not", "don't", "避免", "不喜欢", "不要")):
        return "avoid"
    if any(marker in combined for marker in ("setup", "environment", "workflow", "current", "配置", "环境", "流程", "当前")):
        return "setup"
    return "usage_context"


def infer_profile_domain(
    *,
    category: Any,
    semantic_slot: Any = "",
    slot_key: Any = "",
    anchors: Sequence[Any] = (),
    value: Any = "",
    metadata: Mapping[str, Any] | None = None,
) -> tuple[str, str]:
    data = dict(metadata or {})
    explicit = _first_nonempty(data.get("profile_domain", ""), data.get("profile_domain_label", ""), data.get("domain", ""))
    if explicit:
        return _slug(explicit), explicit
    subject = _first_nonempty(data.get("subject", ""), data.get("extracted_subject", ""))
    if subject:
        return _slug(subject), subject
    slot = _normalize(semantic_slot)
    if slot and slot not in {"profile", "preference", "goal", "constraint", "event", "fact", "status"}:
        return _slug(slot), slot.replace("_", " ")
    for anchor in anchors:
        anchor_text = _clean_text(anchor)
        if not anchor_text:
            continue
        if re.search(r"^\d{4}|\d{1,2}\s+[A-Za-z]+|^[A-Z][a-z]+$", anchor_text):
            continue
        return _slug(anchor_text), anchor_text
    slot_key_text = _clean_text(slot_key)
    if slot_key_text:
        tail = slot_key_text.split(".")[-1].replace("_", " ")
        if tail:
            return _slug(tail), tail
    value_tokens = _tokens(value)
    if value_tokens:
        label = " ".join(value_tokens[:4])
        return _slug(label), label
    return _slug(category), _normalize(category) or "general"


def profile_candidate_metadata(
    *,
    category: Any,
    semantic_slot: Any = "",
    relation: Any = "",
    value: Any = "",
    source_span: Any = "",
    slot_key: Any = "",
    anchors: Sequence[Any] = (),
    subject: Any = "",
    subject_signature: Any = "",
    proposal: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    proposal_data = dict(proposal or {})
    base_metadata = {
        "profile_type": proposal_data.get("profile_type", ""),
        "profile_domain": proposal_data.get("profile_domain", ""),
        "profile_domain_label": proposal_data.get("profile_domain_label", ""),
        "subject": subject,
        "extracted_subject": proposal_data.get("extracted_subject", ""),
    }
    if not is_profile_layer_record(category=category, semantic_slot=semantic_slot, metadata=base_metadata):
        return {}
    profile_type = infer_profile_type(
        category=category,
        semantic_slot=semantic_slot,
        relation=relation,
        value=value or source_span,
        metadata=proposal_data,
    )
    domain, domain_label = infer_profile_domain(
        category=category,
        semantic_slot=semantic_slot,
        slot_key=slot_key,
        anchors=anchors,
        value=value or source_span,
        metadata={**proposal_data, "subject": subject},
    )
    normalized_subject_signature = _slug(subject_signature or subject or domain, fallback=domain)
    route_terms = _dedupe(
        [
            profile_type,
            domain_label,
            semantic_slot,
            subject,
            *list(anchors or []),
        ],
        max_items=12,
    )
    return {
        "profile_layer": True,
        "profile_candidate_status": _clean_text(proposal_data.get("profile_candidate_status", "")) or "writer_candidate",
        "profile_consolidation_stage": _clean_text(proposal_data.get("profile_consolidation_stage", "")) or "pre_consolidation",
        "profile_type": profile_type,
        "profile_domain": domain,
        "profile_domain_label": domain_label,
        "profile_subject_signature": normalized_subject_signature,
        "profile_support_key": f"{profile_type}:{domain}:{normalized_subject_signature}",
        "profile_route_terms": route_terms,
    }


def profile_aggregate_slot_key(metadata: Mapping[str, Any]) -> str:
    data = dict(metadata or {})
    support_key = _clean_text(data.get("profile_support_key", ""))
    if support_key:
        return f"tmcra.profile.aggregate.{_slug(support_key)}"
    profile_type = _normalize(data.get("profile_type", "")) or "usage_context"
    domain = _normalize(data.get("profile_domain", "")) or _slug(data.get("profile_domain_label", "general"))
    subject = _normalize(data.get("profile_subject_signature", "")) or domain
    return f"tmcra.profile.aggregate.{_slug(f'{profile_type}:{domain}:{subject}')}"


def profile_aggregate_value(
    *,
    profile_type: Any,
    domain_label: Any,
    support_values: Sequence[Any],
) -> str:
    typed = _clean_text(profile_type) or "usage_context"
    domain = _clean_text(domain_label) or "general"
    values = _dedupe(support_values, max_items=5)
    if not values:
        return f"User {typed} profile for {domain}."
    return f"User {typed} profile for {domain}: " + "; ".join(values)


def _profile_output_kind(profile_types: Sequence[Any]) -> str:
    normalized = {_normalize(item) for item in profile_types if _clean_text(item)}
    if "constraint" in normalized:
        return "constraint_profile"
    if "goal" in normalized:
        return "goal_profile"
    if normalized.intersection({"preference", "avoid"}):
        return "preference_profile"
    if "setup" in normalized:
        return "setup_profile"
    return "usage_context_profile"


def _profile_update_policy(profile_types: Sequence[Any]) -> str:
    normalized = {_normalize(item) for item in profile_types if _clean_text(item)}
    if "constraint" in normalized:
        return "preserve_until_explicitly_changed"
    if normalized.intersection({"preference", "avoid", "goal", "setup"}):
        return "update_on_newer_user_evidence"
    return "background_context"


def _profile_memory_type(profile_types: Sequence[Any]) -> str:
    normalized = {_normalize(item) for item in profile_types if _clean_text(item)}
    if "constraint" in normalized:
        return "hard_constraint"
    if normalized.intersection({"preference", "avoid"}):
        return "durable_preference"
    return "profile_context"


def profile_summary(
    *,
    profile_types: Sequence[Any],
    domain_label: Any,
    support_values: Sequence[Any],
    stage: str,
) -> str:
    kind = _profile_output_kind(profile_types)
    domain = _clean_text(domain_label) or "general"
    values = _dedupe(support_values, max_items=6 if stage == "cluster" else 4)
    prefix = {
        "constraint_profile": "User constraint profile",
        "goal_profile": "User goal profile",
        "preference_profile": "User preference profile",
        "setup_profile": "User setup profile",
        "usage_context_profile": "User usage-context profile",
    }.get(kind, "User profile")
    if not values:
        return f"{prefix} for {domain}."
    return f"{prefix} for {domain}: " + "; ".join(values)


def build_profile_aggregate_metadata(
    *,
    support_record_id: Any,
    support_turn_index: int,
    support_value: Any,
    support_anchors: Sequence[Any],
    support_metadata: Mapping[str, Any],
    existing_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    support = dict(support_metadata or {})
    existing = dict(existing_metadata or {})
    profile_type = _normalize(support.get("profile_type", "")) or "usage_context"
    domain = _normalize(support.get("profile_domain", "")) or _slug(support.get("profile_domain_label", "general"))
    domain_label = _clean_text(support.get("profile_domain_label", "")) or domain.replace("_", " ")
    subject_signature = _clean_text(support.get("profile_subject_signature", "")) or domain
    support_ids = _bounded_union(existing.get("profile_support_ids", []) or [], [support_record_id], max_items=64)
    support_turns = _bounded_int_union(existing.get("profile_support_turns", []) or [], [support_turn_index], max_items=64)
    support_values = _bounded_union(existing.get("profile_support_values", []) or [], [support_value], max_items=12)
    support_route_terms = _bounded_union(
        existing.get("profile_route_terms", []) or [],
        support.get("profile_route_terms", []) or [],
        support_anchors,
        max_items=24,
    )
    profile_types = [profile_type]
    output_kind = _profile_output_kind(profile_types)
    memory_type = _profile_memory_type(profile_types)
    summary = profile_summary(
        profile_types=profile_types,
        domain_label=domain_label,
        support_values=support_values,
        stage="aggregate",
    )
    value = profile_aggregate_value(
        profile_type=profile_type,
        domain_label=domain_label,
        support_values=support_values,
    )
    return {
        **existing,
        "profile_layer": True,
        "profile_candidate_status": "consolidated",
        "profile_consolidation_stage": "aggregate",
        "profile_consolidator_version": PROFILE_CONSOLIDATOR_VERSION,
        "profile_aggregate_node": True,
        "profile_type": profile_type,
        "profile_domain": domain,
        "profile_domain_label": domain_label,
        "profile_subject_signature": subject_signature,
        "profile_support_key": f"{profile_type}:{domain}:{subject_signature}",
        "profile_support_ids": support_ids,
        "profile_support_turns": support_turns,
        "profile_support_values": support_values,
        "profile_support_count": len(support_ids),
        "profile_route_terms": support_route_terms,
        "profile_value": value,
        "profile_summary": summary,
        "profile_output_kind": output_kind,
        "profile_update_policy": _profile_update_policy(profile_types),
        "profile_conflict_policy": "latest_active_support_only",
        "profile_evidence_count": len(support_ids),
        "memory_type": memory_type,
        "durable_memory_type": memory_type,
        "memory_chain_depth_layer": "profile",
        "depth_layer": "profile",
    }


_PROFILE_CLUSTER_STOPWORDS = {
    "user",
    "profile",
    "preference",
    "preferences",
    "constraint",
    "constraints",
    "goal",
    "goals",
    "avoid",
    "usage",
    "context",
    "general",
    "default",
    "should",
    "would",
    "could",
    "want",
    "wants",
    "need",
    "needs",
    "when",
    "then",
    "than",
    "with",
    "that",
    "this",
    "from",
    "into",
    "instead",
}


def profile_cluster_tokens(*values: Any) -> list[str]:
    tokens: list[str] = []
    for value in values:
        if isinstance(value, (list, tuple, set)):
            for item in value:
                tokens.extend(_tokens(item))
        else:
            tokens.extend(_tokens(value))
    filtered = [
        token
        for token in tokens
        if token
        and token not in _PROFILE_CLUSTER_STOPWORDS
        and not token.isdigit()
        and (len(token) > 2 or any("\u4e00" <= char <= "\u9fff" for char in token))
    ]
    return _dedupe(filtered, max_items=32)


def profile_cluster_similarity(source_metadata: Mapping[str, Any], target_metadata: Mapping[str, Any]) -> float:
    source = dict(source_metadata or {})
    target = dict(target_metadata or {})
    source_tokens = set(
        profile_cluster_tokens(
            source.get("profile_domain_label", ""),
            source.get("profile_domain", ""),
            source.get("profile_route_terms", []) or [],
            source.get("profile_support_values", []) or [],
            source.get("profile_value", ""),
        )
    )
    target_tokens = set(
        profile_cluster_tokens(
            target.get("profile_domain_label", ""),
            target.get("profile_domain", ""),
            target.get("profile_route_terms", []) or [],
            target.get("profile_support_values", []) or [],
            target.get("profile_value", ""),
        )
    )
    if not source_tokens or not target_tokens:
        return 0.0
    overlap = len(source_tokens & target_tokens) / max(1, len(source_tokens | target_tokens))
    containment = len(source_tokens & target_tokens) / max(1, min(len(source_tokens), len(target_tokens)))
    type_bonus = 0.06 if _normalize(source.get("profile_type", "")) == _normalize(target.get("profile_type", "")) else 0.0
    return round(min(1.0, (0.62 * overlap) + (0.38 * containment) + type_bonus), 6)


def profile_cluster_slot_key(metadata: Mapping[str, Any]) -> str:
    data = dict(metadata or {})
    tokens = profile_cluster_tokens(
        data.get("profile_domain_label", ""),
        data.get("profile_domain", ""),
        data.get("profile_route_terms", []) or [],
        data.get("profile_support_values", []) or [],
        data.get("profile_value", ""),
    )
    if tokens:
        return f"tmcra.profile.cluster.{_slug('_'.join(tokens[:5]))}"
    support_profiles = data.get("profile_support_profile_ids", []) or []
    support_seed = _clean_text(support_profiles[0] if support_profiles else "")
    seed = support_seed or _clean_text(data.get("profile_support_key", "general"))
    return f"tmcra.profile.cluster.{_slug(seed)}"


def profile_cluster_value(*, support_values: Sequence[Any]) -> str:
    values = _dedupe(support_values, max_items=8)
    if not values:
        return "User profile cluster."
    return "User profile cluster: " + "; ".join(values)


def build_profile_cluster_metadata(
    *,
    support_profile_id: Any,
    support_metadata: Mapping[str, Any],
    existing_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    support = dict(support_metadata or {})
    existing = dict(existing_metadata or {})
    support_profile_ids = _bounded_union(
        existing.get("profile_support_profile_ids", []) or [],
        [support_profile_id],
        max_items=32,
    )
    support_ids = _bounded_union(
        existing.get("profile_support_ids", []) or [],
        support.get("profile_support_ids", []) or [],
        max_items=96,
    )
    support_turns = _bounded_int_union(
        existing.get("profile_support_turns", []) or [],
        support.get("profile_support_turns", []) or [],
        max_items=96,
    )
    raw_support_values = support.get("profile_support_values", []) or []
    support_values = _bounded_union(
        existing.get("profile_support_values", []) or [],
        raw_support_values,
        [] if raw_support_values else [support.get("profile_value", "")],
        max_items=16,
    )
    route_terms = _bounded_union(
        existing.get("profile_route_terms", []) or [],
        existing.get("profile_cluster_route_terms", []) or [],
        support.get("profile_route_terms", []) or [],
        profile_cluster_tokens(
            support.get("profile_domain_label", ""),
            support.get("profile_domain", ""),
            support.get("profile_value", ""),
            support.get("profile_support_values", []) or [],
        ),
        max_items=32,
    )
    profile_types = _bounded_union(
        existing.get("profile_cluster_types", []) or [],
        [support.get("profile_type", "")],
        max_items=8,
    )
    domains = _bounded_union(
        existing.get("profile_cluster_domains", []) or [],
        [support.get("profile_domain", ""), support.get("profile_domain_label", "")],
        max_items=16,
    )
    primary_type = _clean_text(profile_types[0] if profile_types else support.get("profile_type", "")) or "usage_context"
    primary_domain = _clean_text(domains[0] if domains else support.get("profile_domain", "")) or "general"
    output_kind = _profile_output_kind(profile_types)
    memory_type = _profile_memory_type(profile_types)
    summary = profile_summary(
        profile_types=profile_types,
        domain_label=primary_domain,
        support_values=support_values,
        stage="cluster",
    )
    value = profile_cluster_value(support_values=support_values)
    return {
        **existing,
        "profile_layer": True,
        "profile_candidate_status": "consolidated",
        "profile_consolidation_stage": "cluster",
        "profile_consolidator_version": PROFILE_CONSOLIDATOR_VERSION,
        "profile_cluster_node": True,
        "profile_type": primary_type,
        "profile_domain": _slug(primary_domain),
        "profile_domain_label": primary_domain.replace("_", " "),
        "profile_subject_signature": _clean_text(existing.get("profile_subject_signature", "")) or _slug(primary_domain),
        "profile_support_key": _clean_text(existing.get("profile_support_key", "")) or f"cluster:{_slug(primary_domain)}",
        "profile_support_profile_ids": support_profile_ids,
        "profile_support_ids": support_ids,
        "profile_support_turns": support_turns,
        "profile_support_values": support_values,
        "profile_support_count": len(support_ids),
        "profile_cluster_profile_count": len(support_profile_ids),
        "profile_cluster_types": profile_types,
        "profile_cluster_domains": domains,
        "profile_cluster_route_terms": route_terms,
        "profile_route_terms": route_terms,
        "profile_value": value,
        "profile_summary": summary,
        "profile_output_kind": output_kind,
        "profile_update_policy": _profile_update_policy(profile_types),
        "profile_conflict_policy": "latest_active_support_only",
        "profile_evidence_count": len(support_ids),
        "memory_type": memory_type,
        "durable_memory_type": memory_type,
        "memory_chain_depth_layer": "profile",
        "depth_layer": "profile",
    }


def infer_profile_query_intent(query: Any) -> dict[str, Any]:
    query_text = _clean_text(query)
    lowered = _normalize(query_text)
    enabled = any(marker in lowered for marker in _PROFILE_QUERY_MARKERS)
    types: list[str] = []
    if any(marker in lowered for marker in ("prefer", "preference", "like", "dislike", "偏好", "喜欢", "不喜欢")):
        types.append("preference")
    if any(marker in lowered for marker in ("constraint", "must", "cannot", "policy", "约束", "限制", "必须", "不能")):
        types.append("constraint")
    if any(marker in lowered for marker in ("goal", "target", "objective", "目标")):
        types.append("goal")
    if any(marker in lowered for marker in ("avoid", "dislike", "避免", "不喜欢")):
        types.append("avoid")
    if any(marker in lowered for marker in ("setup", "profile", "current", "occupation", "role", "job", "career", "background", "identity", "experience", "配置", "画像", "当前")):
        types.append("setup")
    if enabled and any(
        marker in lowered
        for marker in (
            "recommend",
            "suggest",
            "advice",
            "tips",
            "trouble with",
            "struggling with",
            "what do you think",
            "should i",
            "what should",
            "serve",
            "dinner",
            "battery life",
            "getting around",
            "resources",
            "learn more",
            "推荐",
            "建议",
        )
    ):
        types.append("usage_context")
    if not types and enabled:
        types.append("usage_context")
    return {
        "enabled": enabled,
        "types": _dedupe(types),
        "tokens": _tokens(query_text),
    }


def profile_query_score_delta(
    *,
    query: Any,
    query_tokens: set[str],
    category: Any,
    source_kind: Any,
    semantic_slot: Any,
    value: Any,
    anchors: Sequence[Any],
    metadata: Mapping[str, Any] | None = None,
) -> tuple[float, str]:
    data = dict(metadata or {})
    intent = infer_profile_query_intent(query)
    if not intent.get("enabled"):
        return 0.0, ""
    if not is_profile_layer_record(category=category, source_kind=source_kind, semantic_slot=semantic_slot, metadata=data):
        return 0.0, ""
    profile_type = _normalize(data.get("profile_type", "")) or infer_profile_type(
        category=category,
        semantic_slot=semantic_slot,
        value=value,
        metadata=data,
    )
    domain = _normalize(data.get("profile_domain_label", "") or data.get("profile_domain", ""))
    route_terms = " ".join(str(item) for item in data.get("profile_route_terms", []) or [])
    record_tokens = set(_tokens(f"{profile_type} {domain} {semantic_slot} {value} {' '.join(str(item) for item in anchors)} {route_terms}"))
    overlap = len(set(query_tokens) & record_tokens) / max(1, len(set(query_tokens) | record_tokens)) if query_tokens or record_tokens else 0.0
    type_match = profile_type in set(intent.get("types", []) or [])
    delta = 0.20 + (0.16 if type_match else 0.0) + (0.18 * overlap)
    if data.get("profile_candidate_status") == "consolidated":
        delta += 0.06
    if data.get("profile_cluster_node") or data.get("profile_consolidation_stage") == "cluster":
        delta += 0.08
    return round(min(delta, 0.54), 6), "profile_route"


def profile_edge_score(source_metadata: Mapping[str, Any], target_metadata: Mapping[str, Any], *, source_value: Any = "", target_value: Any = "") -> tuple[float, str]:
    source = dict(source_metadata or {})
    target = dict(target_metadata or {})
    if not source.get("profile_layer") or not target.get("profile_layer"):
        return 0.0, ""
    source_key = _normalize(source.get("profile_support_key", ""))
    target_key = _normalize(target.get("profile_support_key", ""))
    source_domain = _normalize(source.get("profile_domain", ""))
    target_domain = _normalize(target.get("profile_domain", ""))
    source_type = _normalize(source.get("profile_type", ""))
    target_type = _normalize(target.get("profile_type", ""))
    source_tokens = set(_tokens(f"{source_value} {source.get('profile_domain_label', '')} {' '.join(source.get('profile_route_terms', []) or [])}"))
    target_tokens = set(_tokens(f"{target_value} {target.get('profile_domain_label', '')} {' '.join(target.get('profile_route_terms', []) or [])}"))
    overlap = len(source_tokens & target_tokens) / max(1, len(source_tokens | target_tokens)) if source_tokens or target_tokens else 0.0
    if source_key and source_key == target_key:
        return round(0.72 + (0.14 * overlap), 6), "profile_support"
    if source_domain and source_domain == target_domain:
        return round(0.58 + (0.12 if source_type == target_type else 0.04) + (0.12 * overlap), 6), "profile_tunnel"
    if source_type and source_type == target_type and overlap >= 0.22:
        return round(0.44 + (0.18 * overlap), 6), "profile_soft_tunnel"
    return 0.0, ""
