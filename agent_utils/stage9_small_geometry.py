"""
Stage Small Geometry - Detailed composite geometry for small objects (Stage 9)
================================================================================

Optional pipeline stage that follows Stage 8 (small describe). Per Stage 7
item, it asks an LLM to emit ≤5 primitive shapes that fit inside the item's
bounding box, persists them incrementally, then rewrites Stage 7's flat
`create_box` / `create_cylinder` calls into `create_detailed_object_small`
calls backed by an injected `DETAILED_GEOMETRY_SMALL` dict.

Mirrors Stage 8's pattern (`stage6_geometry.py`):
    - per-item LLM call with retries
    - ThreadPoolExecutor parallelism
    - `small_geometry_progress.json` incremental save (--resume friendly)
    - separate code-generation pass that rewrites the base script

Memory stage tag: `stage9_small_geometry`.
Output dir:       `pipeline_output/<run>/stage9_small_geometry/`
Output files:     `small_geometry_progress.json`, `small_geometry_output.py`
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.join(current_dir, "stage3"))

from stage3.core import LLMClient  # noqa: E402
from memory import Memory  # noqa: E402
from langchain_core.messages import HumanMessage, SystemMessage  # noqa: E402


# ==============================================================================
# Configuration
# ==============================================================================
PROMPT_PATH = os.path.join(
    current_dir, "..", "agent_prompt", "Stage9_small_geometry_task"
)

DEFAULT_PARALLEL = 8
MAX_LLM_ATTEMPTS = 3
RETRY_DELAY_SEC = 1.5

# Hard caps for the LLM's emitted primitives. These mirror the prompt and
# provide a defensive layer in case the model breaks the contract.
MAX_PARTS = 5
MIN_DIM = 0.005  # 5 mm
ALLOWED_SHAPES = ("box", "cylinder", "sphere", "cone")


# ==============================================================================
# Data structures
# ==============================================================================
@dataclass
class SmallPart:
    name: str
    shape: str
    relative_location: Tuple[float, float, float]
    dimensions: Tuple[float, float, float]
    rotation: Tuple[float, float, float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "shape": self.shape,
            "relative_location": list(self.relative_location),
            "dimensions": list(self.dimensions),
            "rotation": list(self.rotation),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SmallPart":
        loc = d.get("relative_location") or d.get("loc") or [0.0, 0.0, 0.0]
        dim = d.get("dimensions") or d.get("dim") or [0.0, 0.0, 0.0]
        rot = d.get("rotation") or d.get("rot") or [0.0, 0.0, 0.0]
        return cls(
            name=str(d.get("name", "part")),
            shape=str(d.get("shape", "box")),
            relative_location=tuple(float(x) for x in loc)[:3],
            dimensions=tuple(float(x) for x in dim)[:3],
            rotation=tuple(float(x) for x in rot)[:3],
        )


@dataclass
class SmallDetailedItem:
    """Per-item progress record: source describe fields + parts + state."""
    # Identity & placement (from Stage 7 / 8)
    name: str
    parent_name: str
    shape: str
    size: Tuple[float, float, float]
    world_location: Tuple[float, float, float]
    rotation_z: float

    # Describe context (kept for downstream stages and debugging)
    object_type: str = ""
    material_description: str = ""
    color_description: str = ""
    description: str = ""
    part_hierarchy_hint: str = ""

    # Generation state
    generated: bool = False
    failure_reason: str = ""
    parts: List[SmallPart] = field(default_factory=list)
    attempts: int = 0

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["size"] = list(self.size)
        d["world_location"] = list(self.world_location)
        d["parts"] = [p.to_dict() for p in self.parts]
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SmallDetailedItem":
        return cls(
            name=str(d.get("name", "")),
            parent_name=str(d.get("parent_name", "")),
            shape=str(d.get("shape", "box")),
            size=tuple(float(x) for x in d.get("size", [0, 0, 0]))[:3],
            world_location=tuple(
                float(x) for x in d.get("world_location", [0, 0, 0])
            )[:3],
            rotation_z=float(d.get("rotation_z", 0.0)),
            object_type=str(d.get("object_type", "")),
            material_description=str(d.get("material_description", "")),
            color_description=str(d.get("color_description", "")),
            description=str(d.get("description", "")),
            part_hierarchy_hint=str(d.get("part_hierarchy_hint", "")),
            generated=bool(d.get("generated", False)),
            failure_reason=str(d.get("failure_reason", "")),
            parts=[SmallPart.from_dict(p) for p in d.get("parts", [])],
            attempts=int(d.get("attempts", 0)),
        )


# ==============================================================================
# Stage Runner
# ==============================================================================
class StageSmallGeometryRunner:
    """Generate per-item composite geometry for Stage-8 small objects."""

    def __init__(
        self,
        image_path: Optional[str] = None,
        describe_json_path: Optional[str] = None,
        base_code_path: Optional[str] = None,
        output_dir: Optional[str] = None,
        use_memory: bool = True,
        verbose: bool = True,
        memory_file: str = "agent_memory.jsonl",
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        parallel: int = DEFAULT_PARALLEL,
        max_attempts: int = MAX_LLM_ATTEMPTS,
        retry_delay_sec: float = RETRY_DELAY_SEC,
    ):
        self.image_path = image_path  # accepted for parity / future use
        self.describe_json_path = describe_json_path
        self.base_code_path = base_code_path
        self.output_dir = output_dir or os.path.join(
            current_dir, "pipeline_output", "stage9_small_geometry"
        )
        self.use_memory = use_memory
        self.verbose = verbose
        self.parallel = max(1, int(parallel))
        self.max_attempts = max(1, int(max_attempts))
        self.retry_delay_sec = float(retry_delay_sec)

        self.memory = (
            Memory(workspace_dir=current_dir, memory_file=memory_file)
            if use_memory
            else None
        )
        self.llm = LLMClient(model=model, base_url=base_url, api_key=api_key)
        self.scene_type_info = self._load_scene_type_info()

        # Loaded state
        self.system_prompt: Optional[str] = None
        self.describe_payload: Dict[str, Any] = {}
        self.items: List[SmallDetailedItem] = []
        self.base_code: str = ""
        self._save_lock = threading.Lock()
        self.progress_path = os.path.join(
            self.output_dir, "small_geometry_progress.json"
        )

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
                print(f"Stage9: cannot read scene_type ({exc}); using generic small-geometry prompt")
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
INDUSTRIAL / FACTORY SMALL-GEOMETRY ADDENDUM
===============================================================================
Scene subtype: {subtype}
For factory-floor small objects, keep geometry functional, low-detail, and inside
bbox. Use at most 5 parts.

Useful templates:
- wrench / spanner: 1 slim handle box or cylinder + 1 small open head box/cylinder.
- screwdriver: 1 horizontal cylinder shaft + 1 handle cylinder/box.
- caliper: 1 thin beam box + 2 jaw boxes + optional small slider box.
- gauge / dial gauge: 1 small cylinder face + 1 stem cylinder + optional base.
- parts tray / fastener cup: shallow box or cylinder with thin rim.
- small parts bin / tote box: open rectangular box, front lip, optional label panel.
- carton: one box with thin top flap or tape strip.
- label roll / tape roll: short cylinder ring/disc.
- clipboard / tablet: thin rectangular slab, optional clip/bezel.
- barcode scanner: handle box/cylinder + angled head box.
- workpiece / machined part: simple metal block/cylinder with one distinguishing cut/slot if space allows.
- jig / fixture: base plate + clamp blocks/posts, no loose tools attached.

Avoid decorative residential geometry such as candles, vases, plants, pillows,
photo frames, ornaments, and bowls in industrial scenes.
"""

    def _apply_scene_prompt_addendum(self) -> None:
        if not self.system_prompt:
            return
        marker = "INDUSTRIAL / FACTORY SMALL-GEOMETRY ADDENDUM"
        if self._is_industrial_scene() and marker not in self.system_prompt:
            self.system_prompt += self._industrial_prompt_addendum()

    # -------------------------------------------------------------------- log
    def _log(self, msg: str, level: str = "info"):
        if not self.verbose:
            return
        prefix = {
            "info": "ℹ️",
            "success": "✅",
            "warning": "⚠️",
            "error": "❌",
            "step": "📋",
            "item": "🧩",
        }.get(level, "")
        print(f"{prefix} {msg}")

    # ------------------------------------------------------------ data loading
    def _load_prompt(self) -> bool:
        if not os.path.exists(PROMPT_PATH):
            self._log(f"Prompt template missing: {PROMPT_PATH}", "error")
            return False
        with open(PROMPT_PATH, "r", encoding="utf-8") as f:
            self.system_prompt = f.read()
        return True

    def _load_describe(self) -> bool:
        """Load Stage 8 small_describe_output.json from explicit path,
        Memory, or sibling run directory."""
        data: Optional[Dict[str, Any]] = None

        if self.describe_json_path and os.path.exists(self.describe_json_path):
            with open(self.describe_json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._log(
                f"small_describe_output.json: {self.describe_json_path}",
                "success",
            )

        if data is None and self.use_memory:
            entry = self.memory.get_latest(
                stage="stage8_small_describe", type="result"
            )
            if entry:
                meta_path = entry.metadata.get("output_file")
                if meta_path and os.path.exists(meta_path):
                    with open(meta_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    self._log(
                        f"small_describe_output.json: from Memory metadata "
                        f"({meta_path})",
                        "success",
                    )
                elif entry.content:
                    try:
                        data = json.loads(entry.content)
                        self._log(
                            "small_describe_output.json: Memory content",
                            "success",
                        )
                    except json.JSONDecodeError:
                        pass

        if data is None:
            run_dir = os.path.dirname(self.output_dir) if self.output_dir else None
            if run_dir:
                candidate = os.path.join(
                    run_dir,
                    "stage8_small_describe",
                    "small_describe_output.json",
                )
                if os.path.exists(candidate):
                    with open(candidate, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    self._log(
                        f"small_describe_output.json: sibling ({candidate})",
                        "success",
                    )

        if data is None:
            self._log(
                "Cannot find Stage 8 small_describe_output.json. Run "
                "stage8_small_describe first or pass --describe-json.",
                "error",
            )
            return False

        self.describe_payload = data
        self._refresh_scene_type_from_payload(data if isinstance(data, dict) else {})
        return True

    def _load_base_code(self) -> bool:
        """Load Stage 7 small_objects_output.py — the base script we will
        rewrite. Falls back to Memory or sibling run directory."""
        path = self.base_code_path

        if not path:
            run_dir = os.path.dirname(self.output_dir) if self.output_dir else None
            if run_dir:
                candidate = os.path.join(
                    run_dir,
                    "stage7_small_objects",
                    "small_objects_output.py",
                )
                if os.path.exists(candidate):
                    path = candidate

        if not path and self.use_memory:
            entry = self.memory.get_latest(
                stage="stage7_small_objects", type="result"
            )
            if entry:
                meta_path = entry.metadata.get("output_file")
                if meta_path and os.path.exists(meta_path):
                    path = meta_path

        if not path or not os.path.exists(path):
            self._log(
                "Cannot find Stage 7 small_objects_output.py. Run "
                "stage7_small_objects first or pass --base-code.",
                "error",
            )
            return False

        with open(path, "r", encoding="utf-8") as f:
            self.base_code = f.read()
        self.base_code_path = path
        self._log(f"Base code: {path}", "success")
        return True

    def _init_items_from_describe(self) -> None:
        """Build the item list, honoring the describe stage's
        `should_detail` flag (currently always True under strategy='all')."""
        records: List[Dict[str, Any]] = self.describe_payload.get(
            "small_objects", []
        )
        out: List[SmallDetailedItem] = []
        for rec in records:
            if not rec.get("should_detail", True):
                continue
            out.append(
                SmallDetailedItem(
                    name=str(rec.get("name", "")),
                    parent_name=str(rec.get("parent_name", "")),
                    shape=str(rec.get("shape", "box")),
                    size=tuple(
                        float(x) for x in rec.get("size", [0, 0, 0])
                    )[:3],
                    world_location=tuple(
                        float(x)
                        for x in rec.get("world_location", [0, 0, 0])
                    )[:3],
                    rotation_z=float(rec.get("rotation_z", 0.0)),
                    object_type=str(rec.get("object_type", "")),
                    material_description=str(
                        rec.get("material_description", "")
                    ),
                    color_description=str(rec.get("color_description", "")),
                    description=str(rec.get("description", "")),
                    part_hierarchy_hint=str(
                        rec.get("part_hierarchy_hint", "")
                    ),
                )
            )
        self.items = out

    def _load_progress(self) -> None:
        """If a partial run exists, merge `generated=True` items into the
        in-memory list so --resume can skip them."""
        if not os.path.exists(self.progress_path):
            return
        try:
            with open(self.progress_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            prior = {
                rec.get("name"): SmallDetailedItem.from_dict(rec)
                for rec in payload.get("items", [])
            }
        except Exception as exc:  # noqa: BLE001
            self._log(f"Failed to load progress: {exc}", "warning")
            return

        merged = 0
        for it in self.items:
            cached = prior.get(it.name)
            if cached and cached.generated and cached.parts:
                it.parts = cached.parts
                it.generated = True
                it.attempts = cached.attempts
                merged += 1
        if merged:
            self._log(f"Resumed {merged} items from prior progress", "info")

    def _save_progress(self) -> None:
        with self._save_lock:
            os.makedirs(self.output_dir, exist_ok=True)
            payload = {
                "summary": {
                    "total_items": len(self.items),
                    "generated_items": sum(
                        1 for it in self.items if it.generated
                    ),
                    "last_updated": datetime.now().isoformat(),
                },
                "items": [it.to_dict() for it in self.items],
            }
            with open(self.progress_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------ LLM
    def _build_item_prompt(self, it: SmallDetailedItem) -> str:
        payload = {
            "name": it.name,
            "object_type": it.object_type,
            "shape": it.shape,
            "size": [round(float(v), 4) for v in it.size],
            "rotation_z": round(float(it.rotation_z), 4),
            "material_description": it.material_description,
            "color_description": it.color_description,
            "description": it.description,
            "part_hierarchy_hint": it.part_hierarchy_hint,
            "parent_name": it.parent_name,
        }
        if self._is_industrial_scene():
            payload["scene_context"] = self.scene_type_info
        return (
            "item:\n"
            f"{json.dumps(payload, ensure_ascii=False, indent=2)}\n\n"
            "Return JSON only with a 'parts' array."
        )

    def _extract_json(self, text: str) -> Optional[Dict[str, Any]]:
        if not text:
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        for pat in (r"```json\s*([\s\S]*?)\s*```", r"```\s*([\s\S]*?)\s*```"):
            m = re.search(pat, text)
            if m:
                try:
                    return json.loads(m.group(1))
                except json.JSONDecodeError:
                    continue
        m = re.search(r"\{[\s\S]*\}", text)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
        return None

    @staticmethod
    def _validate_parts(
        parts: List[SmallPart],
        bbox: Tuple[float, float, float],
    ) -> Tuple[bool, str]:
        """Return (ok, reason). Reasons are recorded on the item for
        debugging when the LLM refuses to stay inside the bbox."""
        if not parts:
            return False, "no parts"
        if len(parts) > MAX_PARTS:
            return False, f"too many parts ({len(parts)} > {MAX_PARTS})"

        W, D, H = bbox
        half = (W / 2.0, D / 2.0, H / 2.0)
        for p in parts:
            if p.shape not in ALLOWED_SHAPES:
                return False, f"bad shape '{p.shape}'"
            if any(d <= 0 for d in p.dimensions):
                return False, "non-positive dimension"
            if any(d < MIN_DIM for d in p.dimensions):
                # Soft tolerance: clamp on the way out (handled below).
                pass

            # Rotated cylinders/cones swap which axis carries `depth`.
            dx, dy, dz = p.dimensions
            rx, ry, rz = p.rotation
            # Treat horizontal cylinder/cone rotations as a 90° swap of
            # (height, radius). Only consider primary-axis rotations.
            if p.shape in ("cylinder", "cone"):
                if abs(abs(ry) - math.pi / 2) < 0.2:  # ~90° around Y
                    dx, dz = dz, dx
                elif abs(abs(rx) - math.pi / 2) < 0.2:  # ~90° around X
                    dy, dz = dz, dy

            for axis_idx, (loc, dim) in enumerate(
                zip(p.relative_location, (dx, dy, dz))
            ):
                if abs(loc) + dim / 2.0 > half[axis_idx] + 1e-3:
                    return False, (
                        f"part '{p.name}' axis {axis_idx} out of bbox "
                        f"(|{loc:.3f}|+{dim/2:.3f} > {half[axis_idx]:.3f})"
                    )
        return True, ""

    @staticmethod
    def _clamp_parts(parts: List[SmallPart]) -> List[SmallPart]:
        clamped = []
        for p in parts:
            dims = tuple(max(float(d), MIN_DIM) for d in p.dimensions)
            clamped.append(
                SmallPart(
                    name=p.name,
                    shape=p.shape,
                    relative_location=tuple(
                        float(v) for v in p.relative_location
                    )[:3],
                    dimensions=dims,
                    rotation=tuple(float(v) for v in p.rotation)[:3],
                )
            )
        return clamped

    def _generate_one(self, it: SmallDetailedItem) -> bool:
        if it.generated and it.parts:
            return True

        last_reason = ""
        for attempt in range(1, self.max_attempts + 1):
            it.attempts = attempt
            try:
                messages = [
                    SystemMessage(content=self.system_prompt),
                    HumanMessage(content=self._build_item_prompt(it)),
                ]
                response = self.llm.invoke(messages)
                payload = self._extract_json(response)
                if not payload or "parts" not in payload:
                    last_reason = "no 'parts' in LLM output"
                else:
                    raw_parts = payload.get("parts", []) or []
                    parts = [SmallPart.from_dict(p) for p in raw_parts]
                    parts = self._clamp_parts(parts)
                    ok, reason = self._validate_parts(parts, it.size)
                    if ok:
                        it.parts = parts
                        it.generated = True
                        it.failure_reason = ""
                        self._save_progress()
                        return True
                    last_reason = reason
            except Exception as exc:  # noqa: BLE001
                last_reason = f"exception: {exc}"

            self._log(
                f"  '{it.name}' attempt {attempt}/{self.max_attempts} "
                f"failed: {last_reason}",
                "warning",
            )
            if attempt < self.max_attempts:
                time.sleep(self.retry_delay_sec * attempt)

        it.generated = False
        it.failure_reason = last_reason
        self._save_progress()
        return False

    # -------------------------------------------------------------- driver
    def _print_progress(self, done: int, total: int, elapsed: float) -> None:
        rate = done / elapsed if elapsed > 0 else 0
        eta = (total - done) / rate if rate > 0 else float("inf")
        self._log(
            f"Progress: {done}/{total} items "
            f"({rate:.1f}/s, ETA {eta:5.1f}s)",
            "step",
        )

    def _run_llm_pass(self) -> Tuple[int, int]:
        """Iterate self.items, generating geometry for every not-yet-generated
        entry. Returns (success_count, failure_count)."""
        todo = [it for it in self.items if not (it.generated and it.parts)]
        if not todo:
            self._log("Nothing to generate (all items already done)", "info")
            return (sum(1 for it in self.items if it.generated), 0)

        self._log(
            f"Generating geometry for {len(todo)} items "
            f"(parallel={self.parallel})",
            "step",
        )

        ok_count = 0
        fail_count = 0
        start = time.time()
        done = 0

        if self.parallel <= 1:
            for it in todo:
                if self._generate_one(it):
                    ok_count += 1
                else:
                    fail_count += 1
                done += 1
                if done % 5 == 0 or done == len(todo):
                    self._print_progress(done, len(todo), time.time() - start)
        else:
            with ThreadPoolExecutor(max_workers=self.parallel) as pool:
                futures = {pool.submit(self._generate_one, it): it for it in todo}
                for fut in as_completed(futures):
                    if fut.result():
                        ok_count += 1
                    else:
                        fail_count += 1
                    done += 1
                    if done % 5 == 0 or done == len(todo):
                        self._print_progress(
                            done, len(todo), time.time() - start
                        )

        # Account for items that came in already-generated (resume path).
        ok_count = sum(1 for it in self.items if it.generated)
        fail_count = sum(1 for it in self.items if not it.generated)
        return ok_count, fail_count

    # ----------------------------------------------------- code rewriting
    # (defined in stage9_small_geometry_codegen.py mixin section below)
    from_codegen = True  # marker for `_emit_code` extension

    def _emit_code(self) -> str:
        """Rewrite the Stage 7 base script to use detailed geometry where
        available. Implementation lives in `_codegen_emit` to keep this
        file readable; we delegate but do not split the module."""
        return _codegen_emit(self)

    def _save_outputs(self) -> str:
        os.makedirs(self.output_dir, exist_ok=True)
        # Final progress snapshot
        self._save_progress()

        code = self._emit_code()
        out_path = os.path.join(self.output_dir, "small_geometry_output.py")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(code)
        self._log(f"Small geometry script: {out_path}", "success")

        if self.use_memory:
            self.memory.add(
                stage="stage9_small_geometry",
                type="result",
                content=code,
                metadata={
                    "title": "Stage Small Geometry - composite geometry",
                    "summary": (
                        f"{sum(1 for it in self.items if it.generated)} "
                        f"of {len(self.items)} small items detailed"
                    ),
                    "output_file": out_path,
                    "progress_file": self.progress_path,
                    "total_items": len(self.items),
                    "generated_items": sum(
                        1 for it in self.items if it.generated
                    ),
                },
                tags=["stage9_small_geometry", "small_objects", "geometry"],
            )
        return out_path

    def run(
        self,
        resume: bool = True,
        generate_code: bool = True,
    ) -> Tuple[bool, Dict[str, Any]]:
        print("\n" + "=" * 60)
        print("🧩 Stage Small Geometry - composite primitives")
        print("=" * 60)

        if not self._load_prompt():
            return False, {}
        if not self._load_describe():
            return False, {}
        self._apply_scene_prompt_addendum()
        if not self._load_base_code():
            return False, {}

        self._init_items_from_describe()
        if resume:
            self._load_progress()

        if not self.items:
            self._log("No small items to detail; nothing to do.", "warning")
            return True, {"total_items": 0, "generated_items": 0}

        ok, fail = self._run_llm_pass()
        self._log(
            f"LLM pass done — generated={ok}, failed={fail}, "
            f"total={len(self.items)}",
            "success" if fail == 0 else "warning",
        )

        out_path = None
        if generate_code:
            out_path = self._save_outputs()

        return True, {
            "total_items": len(self.items),
            "generated_items": ok,
            "failed_items": fail,
            "output_file": out_path,
        }


# ==============================================================================
# Code generation (split out for readability; same module)
# ==============================================================================
_SMALL_SECTION_MARKER = (
    "# Small objects appended by stage7_small_objects.py"
)
_HELPER_BLOCK = """

# ==============================================================================
# Detailed geometry for small objects (auto-generated by stage9_small_geometry)
# ==============================================================================
import bmesh

DETAILED_GEOMETRY_SMALL = {GEOMETRY_DICT}


def create_detailed_object_small(
    name, location=None, rotation=None, material=None, collection=None
):
    \"\"\"Composite-primitive replacement for create_box/create_cylinder when a
    Stage-9 detailed entry exists in DETAILED_GEOMETRY_SMALL.

    Falls back to a single parent empty when an entry is missing; callers
    should still pass `location`/`rotation` derived from the original
    Stage-7 placement so the assembly lands in the right spot.\"\"\"
    data = DETAILED_GEOMETRY_SMALL.get(name)
    if not data:
        return None

    center = location if location is not None else data.get("center", (0, 0, 0))
    base_rot = (
        rotation if rotation is not None else data.get("rotation", (0, 0, 0))
    )
    parts = data.get("parts", [])

    parent = bpy.data.objects.new(name, None)
    parent.empty_display_type = "PLAIN_AXES"
    parent.empty_display_size = 0.05
    parent.location = center
    parent.rotation_euler = base_rot
    if collection:
        collection.objects.link(parent)
    else:
        bpy.context.scene.collection.objects.link(parent)

    for part in parts:
        ptype = part.get("shape") or part.get("type") or "box"
        pname = f"{name}_{part.get('name', 'part')}"
        ploc = part.get("relative_location") or part.get("loc") or (0, 0, 0)
        pdim = part.get("dimensions") or part.get("dim") or (0.01, 0.01, 0.01)
        prot = part.get("rotation") or part.get("rot") or (0, 0, 0)

        mesh = bpy.data.meshes.new(pname + "_mesh")
        bm = bmesh.new()
        if ptype == "box":
            bmesh.ops.create_cube(bm, size=1.0)
        elif ptype == "cylinder":
            bmesh.ops.create_cone(
                bm, cap_ends=True, cap_tris=False,
                segments=24, radius1=0.5, radius2=0.5, depth=1.0,
            )
        elif ptype == "sphere":
            bmesh.ops.create_uvsphere(
                bm, u_segments=24, v_segments=12, radius=0.5,
            )
        elif ptype == "cone":
            bmesh.ops.create_cone(
                bm, cap_ends=True, cap_tris=False,
                segments=24, radius1=0.5, radius2=0.0, depth=1.0,
            )
        else:
            bmesh.ops.create_cube(bm, size=1.0)
        bm.to_mesh(mesh)
        bm.free()

        obj = bpy.data.objects.new(pname, mesh)
        obj.location = ploc
        obj.dimensions = pdim
        obj.rotation_euler = [
            r * 3.14159265 / 180 if abs(r) > 6.3 else r for r in prot
        ]
        obj.parent = parent
        if material:
            obj.data.materials.append(material)
        if collection:
            collection.objects.link(obj)
        else:
            bpy.context.scene.collection.objects.link(obj)

    return parent
"""


def _format_geometry_dict(items: List[SmallDetailedItem]) -> str:
    """Pretty-print DETAILED_GEOMETRY_SMALL as a Python literal."""
    lines = ["{"]
    for it in items:
        if not (it.generated and it.parts):
            continue
        lines.append(f'    "{it.name}": {{')
        lines.append(f'        "center": {list(it.world_location)},')
        lines.append(
            f'        "rotation": [0.0, 0.0, {float(it.rotation_z):.6f}],'
        )
        lines.append('        "parts": [')
        for p in it.parts:
            lines.append(
                '            {'
                f'"shape": "{p.shape}", '
                f'"name": "{p.name}", '
                f'"relative_location": {list(p.relative_location)}, '
                f'"dimensions": {list(p.dimensions)}, '
                f'"rotation": {list(p.rotation)}'
                '},'
            )
        lines.append('        ],')
        lines.append('    },')
    lines.append('}')
    return "\n".join(lines)


def _build_call_rewriter(generated_names: set):
    """Return a function that rewrites a single
    `create_box("Name", loc, dim, rotation=..., material=..., collection=...)`
    or `create_cylinder(...)` call line into the matching
    `create_detailed_object_small(...)` call (when `Name` is in
    `generated_names`)."""

    call_re = re.compile(
        r"^(?P<indent>\s*)create_(?P<kind>box|cylinder)\(\s*"
        r"['\"](?P<name>[^'\"]+)['\"]\s*,\s*"
        r"(?P<loc>\([^)]*\)|[^,]+)\s*,\s*"
        r"(?P<dim>\([^)]*\)|[^,]+)\s*"
        r"(?P<rest>,.*)?\)\s*$"
    )

    def rewrite(line: str) -> Optional[str]:
        m = call_re.match(line.rstrip("\n"))
        if not m:
            return None
        name = m.group("name")
        if name not in generated_names:
            return None
        indent = m.group("indent")
        loc = m.group("loc").strip()
        rest = m.group("rest") or ""
        # Strip the leading ',' off `rest` so we can re-emit cleanly.
        rest = rest.lstrip(",").strip()
        # The rest may carry rotation=..., material=..., collection=...,
        # show_direction=...; we keep them verbatim but drop dimensions
        # (already captured by the dict).
        # Filter out keyword args that detailed helper does not accept.
        filtered = []
        for kv in _split_kwargs(rest):
            kv_stripped = kv.strip()
            if not kv_stripped:
                continue
            key = kv_stripped.split("=", 1)[0].strip()
            if key in ("rotation", "material", "collection"):
                filtered.append(kv_stripped)
            # silently drop show_direction / extras not supported here
        kwargs_text = (", " + ", ".join(filtered)) if filtered else ""
        return (
            f"{indent}create_detailed_object_small('{name}', "
            f"location={loc}{kwargs_text})"
        )

    return rewrite


def _split_kwargs(text: str) -> List[str]:
    """Split a kwargs string on top-level commas (ignoring those inside
    parens)."""
    out: List[str] = []
    depth = 0
    current = []
    for ch in text:
        if ch == "(":
            depth += 1
            current.append(ch)
        elif ch == ")":
            depth -= 1
            current.append(ch)
        elif ch == "," and depth == 0:
            out.append("".join(current))
            current = []
        else:
            current.append(ch)
    if current:
        out.append("".join(current))
    return out


def _codegen_emit(runner: "StageSmallGeometryRunner") -> str:
    """Build the final small_geometry_output.py contents."""
    base = runner.base_code
    items = runner.items
    generated_names = {it.name for it in items if it.generated and it.parts}

    # 1) Insert the helper block at the end of the small-object section.
    #    We anchor on the Stage 7 marker comment.
    if _SMALL_SECTION_MARKER not in base:
        # No 7 section found — append helper at the end of the file.
        anchor_end_idx = len(base)
        prefix = base
        suffix = ""
    else:
        anchor_end_idx = base.index(_SMALL_SECTION_MARKER)
        prefix = base[:anchor_end_idx]
        suffix = base[anchor_end_idx:]

    # 2) In the suffix (Stage 7 section), rewrite create_box / create_cylinder
    #    calls whose name appears in generated_names.
    rewrite_call = _build_call_rewriter(generated_names)
    suffix_lines = suffix.splitlines()
    rewritten_lines: List[str] = []
    rewrites = 0
    for line in suffix_lines:
        new_line = rewrite_call(line) if generated_names else None
        if new_line is not None:
            rewritten_lines.append(new_line)
            rewrites += 1
        else:
            rewritten_lines.append(line)
    rewritten_suffix = "\n".join(rewritten_lines)

    # 3) Compose final code: base prefix + helper block + rewritten suffix.
    helper = _HELPER_BLOCK.replace(
        "{GEOMETRY_DICT}", _format_geometry_dict(items)
    )

    runner._log(
        f"Rewrote {rewrites} small-object calls to "
        "create_detailed_object_small()",
        "info",
    )

    return prefix + helper + "\n" + rewritten_suffix


# ==============================================================================
# CLI
# ==============================================================================
def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Stage Small Geometry: detailed composite primitives "
        "for Stage-8 small objects."
    )
    p.add_argument("--image", "-i", required=False,
                   help="Reference image (currently unused; kept for parity)")
    p.add_argument("--describe-json",
                   help="Path to small_describe_output.json (defaults to "
                        "Memory / sibling run dir)")
    p.add_argument("--base-code",
                   help="Path to Stage 7 small_objects_output.py")
    p.add_argument("--output-dir", "-o", required=False,
                   help="Output directory")
    p.add_argument("--no-memory", action="store_true",
                   help="Disable Memory read/write")
    p.add_argument("--memory-file", default="agent_memory.jsonl")
    p.add_argument("--model", default=None)
    p.add_argument("--base-url", default=None)
    p.add_argument("--api-key", default=None)
    p.add_argument("--parallel", type=int, default=DEFAULT_PARALLEL,
                   help=f"Parallel LLM workers (default {DEFAULT_PARALLEL})")
    p.add_argument("--max-attempts", type=int, default=MAX_LLM_ATTEMPTS)
    p.add_argument("--retry-delay", type=float, default=RETRY_DELAY_SEC)
    p.add_argument("--no-code", action="store_true",
                   help="Run LLM only; skip code generation.")
    p.add_argument("--no-resume", action="store_true",
                   help="Ignore progress.json and regenerate from scratch.")
    p.add_argument("--quiet", "-q", action="store_true")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_argparser().parse_args(argv)
    runner = StageSmallGeometryRunner(
        image_path=args.image,
        describe_json_path=args.describe_json,
        base_code_path=args.base_code,
        output_dir=args.output_dir,
        use_memory=not args.no_memory,
        verbose=not args.quiet,
        memory_file=args.memory_file,
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        parallel=args.parallel,
        max_attempts=args.max_attempts,
        retry_delay_sec=args.retry_delay,
    )
    success, summary = runner.run(
        resume=not args.no_resume,
        generate_code=not args.no_code,
    )
    if not args.quiet:
        print("\nSummary:", json.dumps(summary, ensure_ascii=False))
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
