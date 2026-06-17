"""
Stage Small Describe - Per-item description for surface small objects
======================================================================

Optional pipeline stage that runs AFTER Stage7_small_objects (7) and BEFORE
Stage9_small_geometry (9). For every small item placed by Stage 7 (bbox
only), it asks a vision-language model to produce a tighter, geometry-ready
description: canonical `object_type`, appearance summary, material, color,
and a coarse `part_hierarchy_hint`.

Memory stage tag: `stage8_small_describe`.
Output dir:       `pipeline_output/<run>/stage8_small_describe/`
Output JSON:      `small_describe_output.json`

Usage (standalone):
    cd agent_utils
    python stage8_small_describe.py \
        --image ../agent_input/room.png \
        --small-objects-json pipeline_output/<run>/stage7_small_objects/small_objects.json \
        --output-dir         pipeline_output/<run>/stage8_small_describe
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# Path setup identical to sibling stage modules.
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.join(current_dir, "stage3"))

from stage3.core import LLMClient  # noqa: E402
from memory import Memory  # noqa: E402
from langchain_core.messages import HumanMessage, SystemMessage  # noqa: E402


# ==============================================================================
# Configuration
# ==============================================================================

# Where the prompt template lives. The template uses {ROOM_STYLE} and {ITEMS}
# placeholders that this module fills in at runtime.
PROMPT_PATH = os.path.join(
    current_dir, "..", "agent_prompt", "Stage8_small_describe_task"
)

# Default batch size = items per LLM call. We bias slightly low so the LLM
# can return a complete JSON without truncation; 50+ items per call is too
# fragile under a 4-8K JSON output cap.
DEFAULT_BATCH_SIZE = 8

# Default parallel workers. Matches Stage7_small_objects' typical concurrency.
DEFAULT_PARALLEL = 4

# Hard cap on LLM retries per batch.
MAX_BATCH_ATTEMPTS = 3


# ==============================================================================
# Data structures
# ==============================================================================
@dataclass
class SmallItemInput:
    """The fields we keep from Stage 7's `small_objects.json`."""
    name: str
    item_type: str
    parent_name: str
    plane_id: str
    plane_type: str
    shape: str
    size: Tuple[float, float, float]
    world_location: Tuple[float, float, float]
    rotation_z: float
    color_hint: str
    description: str
    stack_index: int = 0

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SmallItemInput":
        return cls(
            name=str(d.get("name", "")),
            item_type=str(d.get("item_type", "")),
            parent_name=str(d.get("parent_name", "")),
            plane_id=str(d.get("plane_id", "")),
            plane_type=str(d.get("plane_type", "")),
            shape=str(d.get("shape", "box")),
            size=tuple(float(x) for x in d.get("size", [0.0, 0.0, 0.0]))[:3],
            world_location=tuple(
                float(x) for x in d.get("world_location", [0.0, 0.0, 0.0])
            )[:3],
            rotation_z=float(d.get("rotation_z", 0.0)),
            color_hint=str(d.get("color_hint", "")),
            description=str(d.get("description", "")),
            stack_index=int(d.get("stack_index", 0)),
        )


@dataclass
class SmallItemDescribed:
    """LLM-augmented record. Mirrors Stage 7's per-object describe schema
    while keeping Stage 7 placement fields intact so Stage 9 (geometry)
    can render the items in place without re-deriving locations."""
    # Identity & placement (from Stage 7, never modified here)
    name: str
    parent_name: str
    plane_id: str
    plane_type: str
    shape: str
    size: Tuple[float, float, float]
    world_location: Tuple[float, float, float]
    rotation_z: float

    # Stage 7 hints kept for fallback / debugging
    source_item_type: str = ""
    source_color_hint: str = ""
    source_description: str = ""

    # LLM-generated description fields
    object_type: str = ""
    appearance: str = ""
    material_description: str = ""
    color_description: str = ""
    description: str = ""
    part_hierarchy_hint: str = ""

    # Whether the geometry stage should generate detailed primitives for
    # this item. Currently always True (strategy = all); the field is kept
    # for forward-compat with whitelist / topk strategies.
    should_detail: bool = True

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["size"] = list(self.size)
        d["world_location"] = list(self.world_location)
        return d


# ==============================================================================
# Stage Runner
# ==============================================================================
class StageSmallDescribeRunner:
    """Run LLM-backed description for every Stage-7 small item."""

    def __init__(
        self,
        image_path: Optional[str] = None,
        small_objects_json_path: Optional[str] = None,
        output_dir: Optional[str] = None,
        use_memory: bool = True,
        verbose: bool = True,
        memory_file: str = "agent_memory.jsonl",
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        parallel: int = DEFAULT_PARALLEL,
    ):
        self.image_path = image_path
        self.small_objects_json_path = small_objects_json_path
        self.output_dir = output_dir or os.path.join(
            current_dir, "pipeline_output", "stage8_small_describe"
        )
        self.use_memory = use_memory
        self.verbose = verbose
        self.batch_size = max(1, int(batch_size))
        self.parallel = max(1, int(parallel))

        self.memory = (
            Memory(workspace_dir=current_dir, memory_file=memory_file)
            if use_memory
            else None
        )
        self.llm = LLMClient(model=model, base_url=base_url, api_key=api_key)
        self.scene_type_info = self._load_scene_type_info()

        # Loaded data
        self.system_prompt: Optional[str] = None
        self.items: List[SmallItemInput] = []
        self.parent_type_map: Dict[str, str] = {}
        self.room_style: Dict[str, Any] = {}

        # Output
        self.described: List[SmallItemDescribed] = []
        self._save_lock = threading.Lock()

    # ---------------------------------------------------------------- scene type
    def _load_scene_type_info(self) -> Dict[str, Any]:
        fallback = {
            "scene_type": "other",
            "confidence": 0.0,
            "reasoning": "no scene_type in memory",
            "lab_subtype": None,
            "industrial_subtype": None,
            "source": "fallback",
        }
        if not self.memory:
            return fallback
        try:
            from scene_classifier import read_scene_type  # type: ignore
            return read_scene_type(self.memory)
        except Exception as exc:
            if self.verbose:
                print(f"Stage8: cannot read scene_type ({exc}); using generic small-describe prompt")
            return fallback

    def _refresh_scene_type_from_payload(self, data: Dict[str, Any]) -> None:
        summary = data.get("summary") if isinstance(data, dict) else None
        info = None
        if isinstance(summary, dict):
            info = summary.get("scene_type_info")
        if not info and isinstance(data, dict):
            info = data.get("scene_type_info")
        if isinstance(info, dict) and info.get("scene_type"):
            self.scene_type_info = info

    def _is_industrial_scene(self) -> bool:
        return (
            (self.scene_type_info or {}).get("scene_type") == "industrial"
            and float((self.scene_type_info or {}).get("confidence", 0.0) or 0.0) >= 0.5
        )

    def _industrial_prompt_addendum(self) -> str:
        if not self._is_industrial_scene():
            return ""
        subtype = (self.scene_type_info or {}).get("industrial_subtype") or "general"
        return f"""

===============================================================================
INDUSTRIAL / FACTORY SMALL-OBJECT GUIDANCE
===============================================================================
Scene subtype: {subtype}
If an item is on a workbench, inspection table, packing table, conveyor, pallet,
or material rack, interpret it as a functional production object, not decor.

Good industrial object_type examples:
  "wrench", "screwdriver", "caliper", "dial gauge", "fixture", "jig",
  "parts tray", "fastener cup", "small parts bin", "tote box", "carton",
  "workpiece", "machined part", "label roll", "tape roll", "clipboard",
  "tablet", "barcode scanner", "warning tag", "sample part".

Industrial material/color bias:
  powder-coated metal, stainless steel, anodized aluminum, black rubber,
  polypropylene plastic, cardboard, safety yellow, emergency red, matte grey,
  stainless silver, translucent plastic.

Do not reinterpret factory items as residential decor: avoid vases, candles,
plants, throw pillows, decorative bowls, framed photos, ornaments, or coffee-table objects.
"""

    def _apply_scene_prompt_addendum(self) -> None:
        if not self.system_prompt:
            return
        marker = "INDUSTRIAL / FACTORY SMALL-OBJECT GUIDANCE"
        if self._is_industrial_scene() and marker not in self.system_prompt:
            self.system_prompt += self._industrial_prompt_addendum()

    # ------------------------------------------------------------------ logging
    def _log(self, msg: str, level: str = "info"):
        if not self.verbose:
            return
        prefix = {
            "info": "ℹ️",
            "success": "✅",
            "warning": "⚠️",
            "error": "❌",
            "step": "📋",
            "batch": "📦",
        }.get(level, "")
        print(f"{prefix} {msg}")

    # --------------------------------------------------------------- I/O helpers
    def _encode_image(self, path: str) -> Tuple[str, str]:
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        ext = os.path.splitext(path)[1].lower()
        mime = {".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                ".png": "image/png"}.get(ext, "image/png")
        return b64, mime

    def _load_prompt(self) -> bool:
        if not os.path.exists(PROMPT_PATH):
            self._log(f"Prompt template missing: {PROMPT_PATH}", "error")
            return False
        with open(PROMPT_PATH, "r", encoding="utf-8") as f:
            self.system_prompt = f.read()
        return True

    def _load_small_objects_json(self) -> bool:
        """Load Stage 7 small_objects.json from explicit path, Memory, or
        the sibling stage directory of the current run."""
        data: Optional[Dict[str, Any]] = None

        # 1) Explicit path
        if self.small_objects_json_path and os.path.exists(
            self.small_objects_json_path
        ):
            with open(self.small_objects_json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._log(
                f"small_objects.json: {self.small_objects_json_path}",
                "success",
            )

        # 2) Memory metadata pointer
        if data is None and self.use_memory:
            entry = self.memory.get_latest(
                stage="stage7_small_objects", type="result"
            )
            if entry:
                meta_path = entry.metadata.get("small_objects_json")
                if meta_path and os.path.exists(meta_path):
                    with open(meta_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    self._log(
                        f"small_objects.json: from Memory metadata "
                        f"({meta_path})",
                        "success",
                    )
                elif entry.content:
                    try:
                        data = json.loads(entry.content)
                        self._log(
                            "small_objects.json: from Memory content",
                            "success",
                        )
                    except json.JSONDecodeError:
                        pass

        # 3) Sibling file under the same run directory
        if data is None:
            run_dir = os.path.dirname(self.output_dir) if self.output_dir else None
            if run_dir:
                candidate = os.path.join(
                    run_dir, "stage7_small_objects", "small_objects.json"
                )
                if os.path.exists(candidate):
                    with open(candidate, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    self._log(
                        f"small_objects.json: sibling ({candidate})",
                        "success",
                    )

        if data is None:
            self._log(
                "Cannot find Stage 7 small_objects.json. Run "
                "stage7_small_objects first or pass --small-objects-json.",
                "error",
            )
            return False

        self._refresh_scene_type_from_payload(data if isinstance(data, dict) else {})
        raw_items = data.get("items", []) if isinstance(data, dict) else []
        self.items = [SmallItemInput.from_dict(it) for it in raw_items]
        self._log(f"Loaded {len(self.items)} small items", "success")
        return len(self.items) > 0

    def _load_parent_type_map(self) -> None:
        """Pull each parent furniture's `object_type` from Stage 7 / Stage 8
        outputs, so the prompt can ground items by parent function (e.g.
        'on a fume hood vs. on a bookshelf')."""
        # Prefer Stage 8's geometry_progress.json (already keyed by name).
        run_dir = os.path.dirname(self.output_dir) if self.output_dir else None
        if run_dir:
            candidate = os.path.join(
                run_dir, "stage6_geometry", "geometry_progress.json"
            )
            if os.path.exists(candidate):
                try:
                    with open(candidate, "r", encoding="utf-8") as f:
                        gj = json.load(f)
                    for obj in gj.get("detailed_objects", []):
                        name = obj.get("name")
                        otype = obj.get("object_type")
                        if name and otype:
                            self.parent_type_map[name] = otype
                except Exception:
                    pass

        # Fall back to Stage 7's describe_output.json.
        if not self.parent_type_map and run_dir:
            candidate = os.path.join(
                run_dir, "stage5_describe", "describe_output.json"
            )
            if os.path.exists(candidate):
                try:
                    with open(candidate, "r", encoding="utf-8") as f:
                        dj = json.load(f)
                    for obj in dj.get("objects", []):
                        name = obj.get("name")
                        otype = obj.get("object_type")
                        if name and otype:
                            self.parent_type_map[name] = otype
                except Exception:
                    pass

        # Memory fallback (stage5_describe)
        if not self.parent_type_map and self.use_memory:
            entry = self.memory.get_latest(
                stage="stage5_describe", type="result"
            )
            if entry and entry.content:
                try:
                    payload = json.loads(entry.content)
                    for obj in payload.get("objects", []):
                        name = obj.get("name")
                        otype = obj.get("object_type")
                        if name and otype:
                            self.parent_type_map[name] = otype
                except json.JSONDecodeError:
                    pass

    def _load_room_style(self) -> None:
        """Pull the room style (from Stage 7) for prompt context."""
        run_dir = os.path.dirname(self.output_dir) if self.output_dir else None
        if run_dir:
            candidate = os.path.join(
                run_dir, "stage5_describe", "describe_output.json"
            )
            if os.path.exists(candidate):
                try:
                    with open(candidate, "r", encoding="utf-8") as f:
                        dj = json.load(f)
                    style = dj.get("room_style")
                    if isinstance(style, dict):
                        self.room_style = style
                        return
                except Exception:
                    pass

        if self.use_memory:
            entry = self.memory.get_latest(
                stage="stage5_describe", type="result"
            )
            if entry and entry.content:
                try:
                    payload = json.loads(entry.content)
                    style = payload.get("room_style")
                    if isinstance(style, dict):
                        self.room_style = style
                except json.JSONDecodeError:
                    pass

    def _load_image_path(self) -> None:
        if self.image_path and os.path.exists(self.image_path):
            return
        if not self.use_memory:
            return
        entry = self.memory.get_latest(stage="stage1", type="result")
        if entry and entry.metadata.get("image_path"):
            self.image_path = entry.metadata["image_path"]

    # ----------------------------------------------------------------- batching
    def _build_batches(self) -> List[List[SmallItemInput]]:
        """Group items into batches. We sort by `parent_name` first so each
        batch tends to contain related items (consistent material/color
        guesses across an entire bench's contents)."""
        sorted_items = sorted(self.items, key=lambda x: (x.parent_name, x.name))
        batches: List[List[SmallItemInput]] = []
        for i in range(0, len(sorted_items), self.batch_size):
            batches.append(sorted_items[i : i + self.batch_size])
        return batches

    def _items_payload(self, batch: List[SmallItemInput]) -> List[Dict[str, Any]]:
        payload = []
        for it in batch:
            payload.append({
                "name": it.name,
                "item_type": it.item_type,
                "parent_name": it.parent_name,
                "parent_type": self.parent_type_map.get(it.parent_name, ""),
                "shape": it.shape,
                "size": [round(float(v), 3) for v in it.size],
                "world_location": [
                    round(float(v), 3) for v in it.world_location
                ],
                "rotation_z": round(float(it.rotation_z), 4),
                "color_hint": it.color_hint,
                "description": it.description,
            })
        return payload

    # ---------------------------------------------------------------- LLM call
    def _build_user_content(
        self, batch: List[SmallItemInput], mime: str, b64: str
    ) -> List[Dict[str, Any]]:
        room_style_text = (
            json.dumps(self.room_style, ensure_ascii=False, indent=2)
            if self.room_style
            else "{}"
        )
        items_text = json.dumps(
            self._items_payload(batch), ensure_ascii=False, indent=2
        )
        scene_context = ""
        if self._is_industrial_scene():
            scene_context = (
                "scene_context:\n"
                + json.dumps(self.scene_type_info, ensure_ascii=False, indent=2)
                + "\n\n"
            )
        user_text = (
            "room_style:\n"
            f"{room_style_text}\n\n"
            f"{scene_context}"
            "small_items_batch:\n"
            f"{items_text}\n\n"
            "Return JSON only. One entry per item, keyed by 'small_objects'."
        )
        return [
            {"type": "image_url",
             "image_url": {"url": f"data:{mime};base64,{b64}"}},
            {"type": "text", "text": user_text},
        ]

    def _extract_json(self, text: str) -> Optional[Dict[str, Any]]:
        if not text:
            return None
        # Direct
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        # Fenced
        for pat in (r"```json\s*([\s\S]*?)\s*```", r"```\s*([\s\S]*?)\s*```"):
            m = re.search(pat, text)
            if m:
                try:
                    return json.loads(m.group(1))
                except json.JSONDecodeError:
                    continue
        # Greedy first object
        m = re.search(r"\{[\s\S]*\}", text)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
        return None

    def _describe_batch(
        self,
        batch_idx: int,
        batch: List[SmallItemInput],
        mime: str,
        b64: str,
    ) -> List[SmallItemDescribed]:
        """Run one LLM call for a batch and return the described records.
        On hard failure, returns fallback records derived from Stage 7
        hints so the pipeline still produces output for every input."""
        last_err: Optional[Exception] = None
        for attempt in range(1, MAX_BATCH_ATTEMPTS + 1):
            try:
                user_content = self._build_user_content(batch, mime, b64)
                messages = [
                    SystemMessage(content=self.system_prompt),
                    HumanMessage(content=user_content),
                ]
                response = self.llm.invoke(messages)
                payload = self._extract_json(response)
                if not payload or "small_objects" not in payload:
                    raise ValueError(
                        "LLM did not return a `small_objects` list"
                    )

                desc_map = {
                    str(rec.get("name", "")): rec
                    for rec in payload.get("small_objects", [])
                }
                out: List[SmallItemDescribed] = []
                for it in batch:
                    rec = desc_map.get(it.name)
                    if rec:
                        out.append(self._merge(it, rec))
                    else:
                        # Item missing from LLM response → fall back.
                        self._log(
                            f"Batch {batch_idx}: item '{it.name}' missing "
                            "from LLM response, using fallback",
                            "warning",
                        )
                        out.append(self._fallback(it))
                self._log(
                    f"Batch {batch_idx}: {len(out)} items described "
                    f"(attempt {attempt})",
                    "batch",
                )
                return out
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                self._log(
                    f"Batch {batch_idx} attempt {attempt}/"
                    f"{MAX_BATCH_ATTEMPTS} failed: {exc}",
                    "warning",
                )
                time.sleep(min(2.0 * attempt, 6.0))

        # All retries failed → fall back for the entire batch.
        self._log(
            f"Batch {batch_idx} permanently failed ({last_err}); "
            "writing fallback descriptions",
            "error",
        )
        return [self._fallback(it) for it in batch]

    @staticmethod
    def _merge(
        item: SmallItemInput, rec: Dict[str, Any]
    ) -> SmallItemDescribed:
        def _s(key: str, default: str = "") -> str:
            v = rec.get(key, default)
            return v if isinstance(v, str) and v.strip() else default

        return SmallItemDescribed(
            name=item.name,
            parent_name=item.parent_name,
            plane_id=item.plane_id,
            plane_type=item.plane_type,
            shape=item.shape,
            size=item.size,
            world_location=item.world_location,
            rotation_z=item.rotation_z,
            source_item_type=item.item_type,
            source_color_hint=item.color_hint,
            source_description=item.description,
            object_type=_s("object_type", item.item_type or "object"),
            appearance=_s("appearance", item.description or ""),
            material_description=_s("material_description", ""),
            color_description=_s("color_description", item.color_hint),
            description=_s("description", item.description),
            part_hierarchy_hint=_s("part_hierarchy_hint", ""),
            should_detail=True,
        )

    @staticmethod
    def _fallback(item: SmallItemInput) -> SmallItemDescribed:
        """When the LLM omits or fails on an item, produce a usable record
        from Stage 7 hints alone."""
        return SmallItemDescribed(
            name=item.name,
            parent_name=item.parent_name,
            plane_id=item.plane_id,
            plane_type=item.plane_type,
            shape=item.shape,
            size=item.size,
            world_location=item.world_location,
            rotation_z=item.rotation_z,
            source_item_type=item.item_type,
            source_color_hint=item.color_hint,
            source_description=item.description,
            object_type=item.item_type or "object",
            appearance=item.description or "",
            material_description="",
            color_description=item.color_hint,
            description=item.description or "",
            part_hierarchy_hint="",
            should_detail=True,
        )

    # ----------------------------------------------------------------- driver
    def _save_results(self) -> str:
        """Persist `small_describe_output.json` and update Memory."""
        os.makedirs(self.output_dir, exist_ok=True)
        out_path = os.path.join(self.output_dir, "small_describe_output.json")

        payload = {
            "room_style": self.room_style,
            "small_objects": [d.to_dict() for d in self.described],
            "summary": {
                "total_items": len(self.described),
                "should_detail_count": sum(
                    1 for d in self.described if d.should_detail
                ),
                "image_path": self.image_path,
                "scene_type_info": self.scene_type_info,
                "generated_at": datetime.now().isoformat(),
            },
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        if self.use_memory:
            self.memory.add(
                stage="stage8_small_describe",
                type="result",
                content=json.dumps(payload, ensure_ascii=False, indent=2),
                metadata={
                    "title": "Stage Small Describe - Per-item descriptions",
                    "summary": (
                        f"{len(self.described)} small items described"
                    ),
                    "output_file": out_path,
                    "image_path": self.image_path,
                    "total_items": len(self.described),
                    "scene_type_info": self.scene_type_info,
                },
                tags=["stage8_small_describe", "small_objects",
                      "object_description"],
            )

        return out_path

    def run(self) -> Tuple[bool, Dict[str, Any]]:
        print("\n" + "=" * 60)
        print("🔬 Stage Small Describe - small-object descriptions")
        print("=" * 60)

        if not self._load_prompt():
            return False, {}
        if not self._load_small_objects_json():
            return False, {}
        self._apply_scene_prompt_addendum()
        self._load_image_path()
        if not self.image_path or not os.path.exists(self.image_path):
            self._log(
                "Reference image not found; describe needs visual grounding.",
                "error",
            )
            return False, {}
        self._load_parent_type_map()
        self._load_room_style()

        if not self.items:
            self._log("No small items to describe; nothing to do.", "warning")
            return True, {"total_items": 0}

        b64, mime = self._encode_image(self.image_path)
        batches = self._build_batches()
        self._log(
            f"Describing {len(self.items)} items in {len(batches)} batches "
            f"(batch_size={self.batch_size}, parallel={self.parallel})",
            "step",
        )

        described: List[SmallItemDescribed] = []
        if self.parallel <= 1 or len(batches) == 1:
            for idx, batch in enumerate(batches, start=1):
                described.extend(
                    self._describe_batch(idx, batch, mime, b64)
                )
        else:
            with ThreadPoolExecutor(max_workers=self.parallel) as pool:
                future_map = {
                    pool.submit(
                        self._describe_batch, idx, batch, mime, b64
                    ): (idx, batch)
                    for idx, batch in enumerate(batches, start=1)
                }
                for fut in as_completed(future_map):
                    described.extend(fut.result())

        # Restore the original item order (input ordering matters for the
        # downstream geometry stage and for diff-friendly outputs).
        order = {it.name: i for i, it in enumerate(self.items)}
        described.sort(key=lambda d: order.get(d.name, 1 << 30))
        self.described = described

        out_path = self._save_results()
        self._log(f"Saved describe output: {out_path}", "success")
        return True, {
            "total_items": len(self.described),
            "output_file": out_path,
        }


# ==============================================================================
# CLI
# ==============================================================================
def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Stage Small Describe: describe Stage-7 small objects "
        "with an LLM for downstream detailed geometry."
    )
    p.add_argument("--image", "-i", required=False,
                   help="Reference top-down image path")
    p.add_argument("--small-objects-json",
                   help="Path to Stage 7 small_objects.json (defaults to "
                        "Memory / sibling run directory).")
    p.add_argument("--output-dir", "-o", required=False,
                   help="Output directory (defaults to "
                        "pipeline_output/stage8_small_describe/)")
    p.add_argument("--no-memory", action="store_true",
                   help="Disable Memory read/write")
    p.add_argument("--memory-file", default="agent_memory.jsonl")
    p.add_argument("--model", default=None,
                   help="LLM model override")
    p.add_argument("--base-url", default=None,
                   help="LLM endpoint override")
    p.add_argument("--api-key", default=None,
                   help="LLM API key override")
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                   help=f"Items per LLM call (default {DEFAULT_BATCH_SIZE})")
    p.add_argument("--parallel", type=int, default=DEFAULT_PARALLEL,
                   help=f"Parallel LLM workers (default {DEFAULT_PARALLEL})")
    p.add_argument("--quiet", "-q", action="store_true",
                   help="Suppress info logs")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_argparser().parse_args(argv)
    runner = StageSmallDescribeRunner(
        image_path=args.image,
        small_objects_json_path=args.small_objects_json,
        output_dir=args.output_dir,
        use_memory=not args.no_memory,
        verbose=not args.quiet,
        memory_file=args.memory_file,
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        batch_size=args.batch_size,
        parallel=args.parallel,
    )
    success, summary = runner.run()
    if args.quiet:
        return 0 if success else 1
    print("\nSummary:", json.dumps(summary, ensure_ascii=False))
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
