# -*- coding: utf-8 -*-
"""
kmds_parallel.py — Parallel KMDS extraction (5 focused section calls + 1 translation).

Replaces the single monolithic ~16K-token Claude call in extract_paper.py with
5 concurrent, focused section calls (asyncio.gather) merged in Python, followed
by one English→Japanese translation pass. Target: ~5 min → under ~90 sec.

Design (approved):
  - asyncio + anthropic.AsyncAnthropic
  - Sonnet 4.6 for ALL calls (section + translation)
  - PDF document block carries cache_control ephemeral, 1-hour TTL
  - Pure 5-way gather (no cache pre-warming)
  - Universal ground rules are copied VERBATIM from extraction_prompt.md at the
    top of every section prompt (extracted from the file at runtime — no drift)
  - figures/graphs/tables merge under metadata.publication.figures[]/.graphs[]/.tables[]
  - If any section fails: save partials + continue; never crash the whole run

Public entry point:
  await extract_kmds_parallel(pdf_path, output_dir, base_name=None,
                              prompt_path="extraction_prompt.md")
Returns a summary dict (same en_path/ja_path/input_tokens/output_tokens/elapsed_sec
shape extract_paper.py already expects, plus per-section detail).
"""

import os
import re
import json
import time
import base64
import asyncio
from pathlib import Path
from typing import Optional, Dict, Any, List

try:
    import anthropic
    from anthropic import AsyncAnthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    anthropic = None
    AsyncAnthropic = None
    ANTHROPIC_AVAILABLE = False


MODEL = "claude-sonnet-4-6"          # user-specified: Sonnet 4.6 for all calls
SECTION_MAX_TOKENS = 12000           # headroom for publication + many-figure papers
TRANSLATION_MAX_TOKENS = 16000       # JA output ≈ EN size


# ===================================================================
# Verbatim rule extraction from extraction_prompt.md (no paraphrase)
# ===================================================================

def _slice(text: str, start_marker: str, end_marker: str) -> str:
    """Return text from start_marker up to (not including) end_marker, verbatim."""
    i = text.find(start_marker)
    if i == -1:
        return ""
    j = text.find(end_marker, i + len(start_marker))
    if j == -1:
        j = len(text)
    return text[i:j].strip()


def load_prompt_blocks(prompt_path: str) -> Dict[str, str]:
    """Pull the verbatim rule/overview blocks out of extraction_prompt.md.

    These are sliced (not retyped) so the section prompts can never drift from
    the source specification.
    """
    txt = open(prompt_path, "r", encoding="utf-8").read()
    return {
        # Universal ground rules — prepended verbatim to EVERY section call.
        "ground_rules": _slice(txt, "### Project intent & ground rules",
                               "### What counts as a material vs a sample"),
        # Material-vs-sample distinction — for publication + materials calls.
        "material_vs_sample": _slice(txt, "### What counts as a material vs a sample",
                                     "### Populate structured conditions"),
        # "Fill structured conditions" rules — for process/property calls.
        "conditions": _slice(txt, "### Populate structured conditions",
                             "### Schema structure overview"),
        # Schema navigation map — inlined into every section call.
        "overview": _slice(txt, "### Schema structure overview",
                           "### Output format"),
        # Translation rules — for the JA pass.
        "translation": _slice(txt, "### Output format (two files: English + Japanese)",
                              "### Schema extension candidates"),
    }


# ===================================================================
# Section-specific instructions + explicit output skeletons
# ===================================================================

def _skel(obj: Any) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False)


# Section instructions (what to extract). The authoritative STRUCTURE comes from
# the per-section JSON Schema fragment appended at call time (build_section_schemas)
# — so there are no hand-written skeletons here to drift from the real schema.
SUB_PROMPTS: Dict[str, str] = {
    # 1) bibliography + scope + samples + system
    "publication": (
        "## For THIS call only\n"
        "Extract ONLY `metadata` (including `metadata.publication`: bibliography, "
        "`scope`, and `samples[]`) and the top-level `system` object. Do NOT extract "
        "`materials[]`, and do NOT extract figures/graphs/tables — separate calls handle "
        "those; omit those keys entirely from your output."
    ),
    # 2) composition + doping + purity + structure per material
    "materials_chem": (
        "## For THIS call only\n"
        "Extract ONLY each material's `chemical information` and `structure`. Assign "
        "stable ids `material_01`, `material_02`, … in the order materials appear (one "
        "entry per distinct material system the original samples embody). Do NOT extract "
        "`process` or `property` here — omit those keys. Return a top-level `materials` "
        "array."
    ),
    # 3) process[] per material
    "materials_process": (
        "## For THIS call only\n"
        "Extract ONLY each material's `process` (synthesis/processing steps such as ball "
        "milling, arc melting, spark plasma sintering, encapsulated melting). For every "
        "process type you select, FILL its structured condition sub-fields from Methods — "
        "do not leave it empty. Use the SAME `id`s as the chemistry call. Return a "
        "top-level `materials` array, each item with `id` and `process` only."
    ),
    # 4) measured properties + measurement conditions
    "materials_property": (
        "## For THIS call only\n"
        "Extract ONLY each material's measured `property` values (with the measurement "
        "conditions the schema nests inside it). Record only numbers stated EXPLICITLY in "
        "text/tables/captions — do NOT digitize plotted curves. Use the SAME `id`s. Return "
        "a top-level `materials` array, each item with `id` and `property` only."
    ),
    # 5) figures + graphs + tables structure/description
    "figures": (
        "## For THIS call only\n"
        "Extract the figure / graph / table entries. Return top-level `figures`, `graphs` "
        "and `tables` arrays (they will be merged under metadata.publication). For each, "
        "fill `structure` (brief) and `description` (objective detail; mention series "
        "values in prose, not as data) and leave `digitization` as \"\". Give every entry "
        "its required local id. Do NOT digitize plotted curves."
    ),
}

# Extra rule blocks each section should also receive (beyond ground rules + overview).
_SECTION_EXTRA = {
    "publication": ("material_vs_sample",),
    "materials_chem": ("material_vs_sample",),
    "materials_process": ("conditions",),
    "materials_property": ("conditions",),
    "figures": (),
}


# ===================================================================
# Per-section JSON Schema slicing (so each call conforms to the real schema)
# ===================================================================

_REF_RE = re.compile(r'"\$ref":\s*"#/\$defs/([^"]+)"')


def _refs_in(obj) -> set:
    return set(_REF_RE.findall(json.dumps(obj)))


def _def_closure(defs: Dict[str, Any], roots) -> set:
    """All $defs transitively referenced from `roots` (a set/iterable of names)."""
    seen, stack = set(), list(roots)
    while stack:
        d = stack.pop()
        if d in seen or d not in defs:
            continue
        seen.add(d)
        stack.extend(_refs_in(defs[d]))
    return seen


def build_section_schemas(schema: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Slice the full KMDS schema into a focused sub-schema per section call, so the
    model conforms to the REAL field names / value types / enums instead of a
    hand-written skeleton. Each fragment carries only the relevant top-level
    properties plus the $defs they transitively reference.
    """
    defs = schema.get("$defs", {})
    metaprops = schema["properties"]["metadata"]["properties"]
    sysprop = schema["properties"].get("system")
    mitems = schema["properties"]["materials"]["items"]
    mprops = mitems.get("properties", {})

    # `process` (~53k tok) and `measurement` (~50k tok) are the huge taxonomies.
    # The dense cross-references would pull them into every closure, so keep each
    # only in the section that actually owns it (the others get a dangling $ref,
    # which is fine — the fragment is guidance for the model, not API-validated).
    HEAVY = {"process", "measurement"}

    def _pick_defs(names, owned_heavy=()):
        cl = _def_closure(defs, names)
        cl = {d for d in cl if d not in HEAVY or d in owned_heavy}
        return {k: defs[k] for k in sorted(cl) if k in defs}

    def _materials_fragment(keys, owned_heavy=()):
        sel = {k: mprops[k] for k in keys if k in mprops}
        return {
            "$defs": _pick_defs(_refs_in(sel), owned_heavy=owned_heavy),
            "properties": {
                "materials": {"type": "array", "items": {
                    "type": "object", "properties": sel}},
            },
        }

    out: Dict[str, Dict[str, Any]] = {}

    # publication: metadata (incl. publication ref) + system. Omit process /
    # measurement / figure / graph / table / axis defs (other sections own them).
    pub_props = {"metadata": {"type": "object",
                              "properties": {k: metaprops[k] for k in metaprops}}}
    if sysprop is not None:
        pub_props["system"] = sysprop
    pub_defs = _pick_defs(_refs_in(pub_props))
    for drop in ("figure", "graph", "table", "axis"):
        pub_defs.pop(drop, None)
    out["publication"] = {"$defs": pub_defs, "properties": pub_props}

    out["materials_chem"] = _materials_fragment(["id", "name", "chemical information", "structure"])
    out["materials_process"] = _materials_fragment(["id", "process"], owned_heavy=("process",))
    out["materials_property"] = _materials_fragment(["id", "property"], owned_heavy=("measurement",))

    # figures: top-level figures/graphs/tables arrays (merged under publication later)
    fig_props = {
        "figures": {"type": "array", "items": {"$ref": "#/$defs/figure"}},
        "graphs": {"type": "array", "items": {"$ref": "#/$defs/graph"}},
        "tables": {"type": "array", "items": {"$ref": "#/$defs/table"}},
    }
    out["figures"] = {"$defs": _pick_defs({"figure", "graph", "table"}), "properties": fig_props}
    return out


def validate_record(record: Dict[str, Any], schema: Dict[str, Any]) -> List[str]:
    """Validate a merged KMDS record against the full schema. Returns a list of
    human-readable violation strings (empty = valid). No-op if jsonschema is
    unavailable."""
    try:
        from jsonschema import Draft202012Validator
    except Exception:
        return []
    try:
        v = Draft202012Validator(schema)
        msgs = []
        for e in sorted(v.iter_errors(record), key=lambda e: list(e.path)):
            path = "/".join(str(p) for p in e.path) or "(root)"
            msgs.append(f"[{path}] {e.message}")
        return msgs
    except Exception as e:
        return [f"(validator error: {type(e).__name__}: {e})"]


# ===================================================================
# Helpers
# ===================================================================

def _encode_pdf_b64(pdf_path: str) -> str:
    with open(pdf_path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8")


def _parse_json_block(text: str) -> Optional[Any]:
    """Pull the first JSON object out of a model response (fenced or bare)."""
    fence = re.search(r"```(?:json[a-z_]*)?\s*\n([\s\S]*?)```", text)
    blob = fence.group(1) if fence else text
    i = blob.find("{")
    j = blob.rfind("}")
    if i == -1 or j == -1 or j <= i:
        return None
    try:
        return json.loads(blob[i:j + 1])
    except json.JSONDecodeError:
        return None


def _usage(resp) -> Dict[str, int]:
    u = resp.usage
    return {
        "input_tokens": u.input_tokens,
        "output_tokens": u.output_tokens,
        "cache_read_tokens": getattr(u, "cache_read_input_tokens", 0) or 0,
        "cache_creation_tokens": getattr(u, "cache_creation_input_tokens", 0) or 0,
    }


# ===================================================================
# One focused section call
# ===================================================================

async def extract_one_section(pdf_b64: str, section_key: str, client,
                              blocks: Dict[str, str],
                              section_schema: Optional[Dict[str, Any]] = None,
                              model: str = MODEL) -> Dict[str, Any]:
    """One focused Claude call for a single KMDS section. Never raises."""
    t0 = time.time()

    extras = "\n\n".join(blocks[name] for name in _SECTION_EXTRA[section_key] if blocks.get(name))
    instruction = (
        blocks["ground_rules"]                       # verbatim universal rules, at the top
        + "\n\n" + blocks["overview"]                 # verbatim schema overview
        + (("\n\n" + extras) if extras else "")       # section-relevant rule blocks
        + "\n\n" + SUB_PROMPTS[section_key]           # section-specific instruction
    )

    if section_schema is not None:
        schema_json = json.dumps(section_schema, ensure_ascii=False)
        instruction += (
            "\n\n## OUTPUT SCHEMA — authoritative\n"
            "Your JSON for THIS section MUST conform EXACTLY to the JSON Schema below.\n"
            "- Use its EXACT field names, value TYPES, and enum/pattern values. A field "
            "typed `number` takes a bare number (e.g. 350), NOT a `{value, unit}` object — "
            "only use an object where the schema defines an object.\n"
            "- `additionalProperties` is false everywhere — NEVER add a key that is not in "
            "the schema. If the paper states something the schema cannot hold, put it in the "
            "nearest `notes`/`comments` (if the schema has one) — never invent a new key.\n"
            "- Fill what the paper states; use null (or omit optional keys) when it is "
            "silent. Output ONLY the keys this section is responsible for.\n"
            "```json\n" + schema_json + "\n```\n"
            "Output the JSON in a single ```json code block and nothing else."
        )
    else:
        instruction += "\nOutput the JSON in a single ```json code block and nothing else."

    content = [
        {"type": "document",
         "source": {"type": "base64", "media_type": "application/pdf", "data": pdf_b64},
         "cache_control": {"type": "ephemeral", "ttl": "1h"}},
        {"type": "text", "text": instruction,
         "cache_control": {"type": "ephemeral", "ttl": "1h"}},
    ]

    base = {"key": section_key, "fragment": None, "raw": None,
            "input_tokens": 0, "output_tokens": 0,
            "cache_read_tokens": 0, "cache_creation_tokens": 0}
    try:
        resp = await client.messages.create(
            model=model,
            max_tokens=SECTION_MAX_TOKENS,
            messages=[{"role": "user", "content": content}],
        )
    except Exception as e:  # noqa: BLE001 — never crash the whole run
        base.update({"ok": False, "error": f"{type(e).__name__}: {e}",
                     "elapsed_sec": time.time() - t0})
        return base

    raw = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
    frag = _parse_json_block(raw)
    base.update(_usage(resp))
    base.update({
        "ok": frag is not None,
        "fragment": frag,
        "raw": raw,
        "error": None if frag is not None else "no JSON block parsed from response",
        "elapsed_sec": time.time() - t0,
    })
    return base


# ===================================================================
# Merge section fragments into a single KMDS dict
# ===================================================================

_EMPTY = (None, "", [], {})


def merge_sections(results: List[Dict[str, Any]]) -> (Dict[str, Any], List[str]):
    """Combine section fragments into one KMDS root. First non-null wins on conflict."""
    by_key = {r["key"]: r for r in results if r}
    warnings: List[str] = []
    merged: Dict[str, Any] = {}

    # metadata (+ publication) and system come from the publication call
    pub_frag = (by_key.get("publication") or {}).get("fragment") or {}
    merged["metadata"] = pub_frag.get("metadata") if isinstance(pub_frag.get("metadata"), dict) else {}
    if "system" in pub_frag:
        merged["system"] = pub_frag.get("system")

    # materials: union by id across the three materials_* calls
    by_id: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []
    for key in ("materials_chem", "materials_process", "materials_property"):
        frag = (by_key.get(key) or {}).get("fragment") or {}
        for m in (frag.get("materials") or []):
            if not isinstance(m, dict):
                continue
            mid = m.get("id")
            if not mid:
                continue
            if mid not in by_id:
                by_id[mid] = {"id": mid}
                order.append(mid)
            tgt = by_id[mid]
            for field, val in m.items():
                if field == "id":
                    continue
                cur = tgt.get(field)
                if field not in tgt or cur in _EMPTY:
                    tgt[field] = val
                elif cur != val and val not in _EMPTY:
                    warnings.append(
                        f"material {mid}.{field}: sections disagree — keeping first "
                        f"non-null value (from an earlier section)")
    merged["materials"] = [by_id[mid] for mid in order]

    # figures/graphs/tables → metadata.publication.{figures,graphs,tables}
    fig_frag = (by_key.get("figures") or {}).get("fragment") or {}
    if fig_frag:
        if not isinstance(merged.get("metadata"), dict):
            merged["metadata"] = {}
        pub_obj = merged["metadata"].setdefault("publication", {})
        if isinstance(pub_obj, dict):
            for k in ("figures", "graphs", "tables"):
                if fig_frag.get(k):
                    pub_obj[k] = fig_frag[k]

    return merged, warnings


# ===================================================================
# Translation pass (EN → JA)
# ===================================================================

async def translate_kmds(en_dict: Dict[str, Any], client,
                         translation_rules: str) -> Dict[str, Any]:
    """One Claude call: translate natural-language values to Japanese. Never raises."""
    t0 = time.time()
    en_json = json.dumps(en_dict, ensure_ascii=False, indent=2)
    instruction = (
        "You are translating a COMPLETED KMDS JSON record from English to Japanese. "
        "This is a translation, not a re-extraction — the two files MUST be structurally "
        "identical. Apply these rules from the extraction specification VERBATIM:\n\n"
        + translation_rules
        + "\n\nReturn ONLY the Japanese JSON in a single ```json code block — same keys, "
          "enums, numbers, units, sample/material ids, formulas, and identifiers "
          "byte-for-byte; translate only the natural-language free-text values listed "
          "above; a field null in English stays null.\n\nEnglish KMDS JSON:\n```json\n"
        + en_json + "\n```"
    )

    out = {"ja": None, "raw": None, "input_tokens": 0, "output_tokens": 0,
           "cache_read_tokens": 0, "cache_creation_tokens": 0}
    try:
        resp = await client.messages.create(
            model=MODEL,
            max_tokens=TRANSLATION_MAX_TOKENS,
            messages=[{"role": "user", "content": [{"type": "text", "text": instruction}]}],
        )
    except Exception as e:  # noqa: BLE001
        out.update({"ok": False, "error": f"{type(e).__name__}: {e}",
                    "elapsed_sec": time.time() - t0})
        return out

    raw = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
    ja = _parse_json_block(raw)
    out.update(_usage(resp))
    out.update({
        "ok": ja is not None,
        "ja": ja,
        "raw": raw,
        "error": None if ja is not None else "no JSON block parsed from translation",
        "elapsed_sec": time.time() - t0,
    })
    return out


# ===================================================================
# Top-level orchestrator
# ===================================================================

async def extract_kmds_parallel(pdf_path: str, output_dir: str,
                                base_name: Optional[str] = None,
                                prompt_path: str = "extraction_prompt.md",
                                model: str = MODEL,
                                schema_path: Optional[str] = None) -> Dict[str, Any]:
    """Run the parallel KMDS extraction + translation. Returns a summary dict."""
    if not ANTHROPIC_AVAILABLE:
        return {"_error": "anthropic SDK not installed. pip install anthropic"}
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return {"_error": "ANTHROPIC_API_KEY not set"}
    if not os.path.exists(prompt_path):
        return {"_error": f"Prompt file not found: {prompt_path}"}
    if base_name is None:
        base_name = Path(pdf_path).stem

    blocks = load_prompt_blocks(prompt_path)
    if not blocks["ground_rules"]:
        return {"_error": f"Could not extract ground-rules block from {prompt_path}"}
    pdf_b64 = _encode_pdf_b64(pdf_path)

    # Load the real KMDS schema and slice a focused sub-schema per section so the
    # model conforms to actual field names / types / enums (not a skeleton).
    if schema_path is None:
        here = os.path.dirname(os.path.abspath(__file__))
        schema_path = os.path.join(here, "kmds_v15.2.4_nullable.json")
    full_schema = None
    section_schemas: Dict[str, Any] = {}
    try:
        with open(schema_path, "r", encoding="utf-8") as f:
            full_schema = json.load(f)
        section_schemas = build_section_schemas(full_schema)
        print(f"⤷ Using KMDS schema: {os.path.basename(schema_path)} "
              f"({len(section_schemas)} section sub-schemas)")
    except Exception as e:
        print(f"⚠ schema not loaded ({e}); falling back to skeleton-free prompts.")

    print(f"⤷ Parallel KMDS: firing {len(SUB_PROMPTS)} focused section calls to "
          f"{model} (concurrent)...")
    t0 = time.time()

    async with AsyncAnthropic() as client:
        # 5-way gather — pure parallel, no cache pre-warming
        results = await asyncio.gather(
            *(extract_one_section(pdf_b64, key, client, blocks,
                                  section_schema=section_schemas.get(key), model=model)
              for key in SUB_PROMPTS)
        )
        sec_wall = time.time() - t0
        for r in results:
            mark = "✓" if r["ok"] else "✗"
            extra = "" if r["ok"] else f" — {r['error']}"
            print(f"   {mark} {r['key']:<20} out={r['output_tokens']:>5} tok  "
                  f"cache_read={r['cache_read_tokens']:>6}  ({r['elapsed_sec']:.1f}s){extra}")

        merged, warnings = merge_sections(results)
        for w in warnings:
            print(f"   ⚠ merge: {w}")

        en_path = os.path.join(output_dir, f"{base_name}.json")
        with open(en_path, "w", encoding="utf-8") as f:
            json.dump(merged, f, indent=2, ensure_ascii=False)
        print(f"   ✅ Saved EN: {en_path}")

        # Validate against the full schema and save a conformance report.
        n_violations = None
        if full_schema is not None:
            viol = validate_record(merged, full_schema)
            n_violations = len(viol)
            val_path = os.path.join(output_dir, f"{base_name}_validation.txt")
            with open(val_path, "w", encoding="utf-8") as f:
                f.write(f"KMDS schema validation — {base_name}\n")
                f.write(f"schema: {os.path.basename(schema_path)}\n")
                f.write(f"violations: {n_violations}\n\n")
                f.write("\n".join(viol) if viol else "VALID — conforms to the schema.")
            mark = "✓ VALID" if n_violations == 0 else f"⚠ {n_violations} violation(s)"
            print(f"   schema validation: {mark}  -> {val_path}")

        # per-section raw dump for debugging
        dbg_path = os.path.join(output_dir, "_parallel_sections_raw.json")
        with open(dbg_path, "w", encoding="utf-8") as f:
            json.dump({r["key"]: {"ok": r["ok"], "error": r["error"],
                                  "fragment": r["fragment"]} for r in results},
                      f, indent=2, ensure_ascii=False)

        # translation pass (sequential, after gather)
        print("⤷ Translating EN → JA (1 call)...")
        tr = await translate_kmds(merged, client, blocks["translation"])

    ja_path = None
    if tr["ok"]:
        ja_path = os.path.join(output_dir, f"{base_name}_ja.json")
        with open(ja_path, "w", encoding="utf-8") as f:
            json.dump(tr["ja"], f, indent=2, ensure_ascii=False)
        print(f"   ✅ Saved JA: {ja_path}")
    else:
        print(f"   ⚠ JA translation failed: {tr['error']} (EN saved, continuing)")

    wall = time.time() - t0
    total_in = sum(r["input_tokens"] for r in results) + tr["input_tokens"]
    total_out = sum(r["output_tokens"] for r in results) + tr["output_tokens"]
    total_cr = sum(r["cache_read_tokens"] for r in results) + tr.get("cache_read_tokens", 0)
    total_cc = sum(r["cache_creation_tokens"] for r in results) + tr.get("cache_creation_tokens", 0)

    summary = {
        "mode": "parallel",
        "model": model,
        "en_path": en_path,
        "ja_path": ja_path,
        "n_sections_ok": sum(1 for r in results if r["ok"]),
        "n_sections": len(results),
        "n_schema_violations": n_violations,
        "sections": {r["key"]: {
            "ok": r["ok"], "error": r["error"],
            "input_tokens": r["input_tokens"], "output_tokens": r["output_tokens"],
            "cache_read_tokens": r["cache_read_tokens"],
            "cache_creation_tokens": r["cache_creation_tokens"],
            "elapsed_sec": round(r["elapsed_sec"], 2),
        } for r in results},
        "translation": {"ok": tr["ok"], "error": tr.get("error"),
                        "input_tokens": tr["input_tokens"], "output_tokens": tr["output_tokens"]},
        "section_wall_sec": round(sec_wall, 2),
        "elapsed_sec": round(wall, 2),
        "input_tokens": total_in,
        "output_tokens": total_out,
        "cache_read_tokens": total_cr,
        "cache_creation_tokens": total_cc,
        "merge_warnings": warnings,
    }
    print(f"   KMDS parallel done in {wall:.1f}s "
          f"(input {total_in:,} + output {total_out:,} tokens; "
          f"cache_read {total_cr:,}, cache_write {total_cc:,})")
    return summary
