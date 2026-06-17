"""
Stage Material - Per-Part Material & Texture Generation
========================================================

Generates detailed PBR materials for every part of every object produced by
Stage 6 (stage6_geometry) and the optional small-object stage
(stage7_small_objects, runs after Stage 7), as well as enhanced floor and wall
materials. The output is a Blender Python script that is fully consistent with
the geometry code, but with rich per-part materials applied.

Pipeline position:
    Stage 6 (geometry) -> [Stage 7 small_objects] -> **Stage 10 Material**
    -> Stage Texture -> Stage Render (lighting)

Usage:
    cd /Users/yangyixuan/SceneGen_Agent/agent_utils

    # Run with Memory (prefers stage7_small_objects > stage6_geometry automatically)
    python stage10_material.py --image /path/to/image.png

    # Specify geometry code explicitly
    python stage10_material.py --image /path/to/image.png \\
        --geometry-code /path/to/geometry_output.py

    # Show Memory status
    python stage10_material.py --status
"""

import os
import sys
import re
import json
import base64
import argparse
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.join(current_dir, "stage3"))

from stage3.core import LLMClient, PromptManager, extract_python_from_response
from memory import Memory
from langchain_core.messages import HumanMessage, SystemMessage


# ==============================================================================
# Stage Material Runner
# ==============================================================================
class StageMaterialRunner:
    """Generate per-part PBR materials for geometry objects, floor, and walls."""

    def __init__(
        self,
        image_path: str = None,
        geometry_code_path: str = None,
        output_dir: str = None,
        use_memory: bool = True,
        verbose: bool = True,
        memory_file: str = "agent_memory.jsonl",
        model: str = None,
        base_url: str = None,
        api_key: str = None,
        batch_size: int = 6,
        parallel: int = 4,
        max_attempts: int = 3,
    ):
        self.image_path = image_path
        self.geometry_code_path = geometry_code_path
        self.output_dir = output_dir or os.path.join(
            current_dir, "pipeline_output", "stage10_material"
        )
        self.use_memory = use_memory
        self.verbose = verbose
        # Option-C batching: scene_palette pre-pass (single call) + parallel
        # batched per-part material calls. Each batch carries ~`batch_size`
        # objects (parts unrolled), `parallel` controls concurrent workers.
        self.batch_size = max(1, int(batch_size))
        self.parallel = max(1, int(parallel))
        self.max_attempts = max(1, int(max_attempts))

        self.memory = (
            Memory(workspace_dir=current_dir, memory_file=memory_file)
            if use_memory
            else None
        )
        self.llm = LLMClient(model=model, base_url=base_url, api_key=api_key)
        self.scene_type_info = self._load_scene_type_info()

        self.geometry_code: Optional[str] = None
        self.describe_data: Dict = {}
        self.scene_palette: Dict = {}
        self.part_materials: Dict[str, Dict[str, Dict]] = {}
        self.floor_material: Dict = {}
        self.wall_material: Dict = {}
        # Small-object integration (Stage 8 + 9 — opt-in via
        # --detail-small-objects in unified_pipeline). When the loaded
        # geometry code contains DETAILED_GEOMETRY_SMALL, Stage 9 also
        # generates per-part materials for these items.
        self.small_describe_map: Dict[str, Dict] = {}
        self.small_geo_objects: Dict[str, List[str]] = {}
        self.small_part_materials: Dict[str, Dict[str, Dict]] = {}

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    def _log(self, msg: str, level: str = "info"):
        if not self.verbose:
            return
        prefix = {
            "info": "ℹ️",
            "success": "✅",
            "warning": "⚠️",
            "error": "❌",
            "step": "📋",
            "material": "🎨",
            "save": "💾",
        }.get(level, "")
        print(f"{prefix} {msg}")

    # ------------------------------------------------------------------
    # Scene-type helpers
    # ------------------------------------------------------------------
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
        except Exception as exc:  # noqa: BLE001
            if self.verbose:
                print(
                    f"Stage10: cannot read scene_type ({exc}); "
                    "using generic material prompt"
                )
            return fallback

    def _is_industrial_scene(self) -> bool:
        return (
            (self.scene_type_info or {}).get("scene_type") == "industrial"
            and float((self.scene_type_info or {}).get("confidence", 0.0) or 0.0) >= 0.5
        )

    def _industrial_material_context(self) -> str:
        if not self._is_industrial_scene():
            return ""
        subtype = (self.scene_type_info or {}).get("industrial_subtype") or "general"
        return f"""
INDUSTRIAL / FACTORY MATERIAL CONTEXT
- Scene subtype: {subtype}. Treat the image as a manufacturing, logistics, or technical facility, not a residential room.
- Default large floor areas to sealed concrete or epoxy-coated concrete unless the image clearly shows another industrial surface.
- Machine enclosures and control cabinets usually use powder_coat or painted_metal in off-white, grey, blue-grey, or green-grey.
- Structural frames, rack uprights, robot bases, guards, and conveyors use painted_metal, brushed_aluminum, stainless_steel, or powder_coat.
- Conveyor belts, rollers, feet, bumpers, cable covers, and mats use black rubber or dark pvc where applicable.
- Safety rails, guard edges, warning posts, bollards, and hazard-marked parts use safety yellow or red accents when visible.
- Screens and indicator LEDs use glass or emission; emergency stop buttons are red plastic.
- Bins, totes, trays, caps, tags, and cable ducts are plastic or pvc.
- Avoid residential defaults such as hardwood, carpet, rugs, wallpaper, decorative fabric, sofas, and ornamental wood unless explicitly visible.
"""

    @staticmethod
    def _industrial_floor_material() -> Dict:
        return {
            "material_type": "concrete",
            "base_color": [0.48, 0.49, 0.47, 1.0],
            "roughness": 0.78,
            "metallic": 0.0,
            "specular": 0.28,
            "pattern": "solid",
            "pattern_scale": 3.0,
            "pattern_color2": [0.40, 0.41, 0.40, 1.0],
            "bump_strength": 0.06,
            "description": "sealed light grey industrial concrete floor with subtle scuffs and anti-slip texture",
        }

    @staticmethod
    def _industrial_wall_material() -> Dict:
        return {
            "material_type": "paint",
            "base_color": [0.78, 0.80, 0.78, 1.0],
            "roughness": 0.86,
            "metallic": 0.0,
            "specular": 0.25,
            "finish": "matte",
            "bump_strength": 0.03,
            "wall_visual_intensity": "subtle",
            "description": "plain light grey painted industrial wall or concrete partition",
        }

    def _sanitize_industrial_floor_wall(
        self, floor_mat: Dict, wall_mat: Dict
    ) -> Tuple[Dict, Dict]:
        if not self._is_industrial_scene():
            return floor_mat, wall_mat

        floor = dict(floor_mat or {})
        wall = dict(wall_mat or {})

        bad_floor_types = {"hardwood", "carpet", "laminate", "marble"}
        bad_floor_patterns = {"plank", "herringbone", "hexagonal"}
        floor_type = str(floor.get("material_type", "")).lower()
        floor_pattern = str(floor.get("pattern", "")).lower()
        if floor_type in bad_floor_types:
            floor = self._industrial_floor_material()
        else:
            base = self._industrial_floor_material()
            for key, value in base.items():
                floor.setdefault(key, value)
            if floor_pattern in bad_floor_patterns:
                floor["pattern"] = "solid"
            if floor.get("material_type") not in {"concrete", "stone", "tile"}:
                floor["material_type"] = "concrete"

        bad_wall_types = {"wallpaper", "wood_panel"}
        wall_type = str(wall.get("material_type", "")).lower()
        if wall_type in bad_wall_types:
            wall = self._industrial_wall_material()
        else:
            base = self._industrial_wall_material()
            for key, value in base.items():
                wall.setdefault(key, value)
            if wall.get("material_type") not in {"paint", "concrete", "plaster", "brick"}:
                wall["material_type"] = "paint"
            wall["wall_visual_intensity"] = "subtle"

        return floor, wall

    # ------------------------------------------------------------------
    # Image encoding
    # ------------------------------------------------------------------
    def _encode_image(self, path: str) -> Tuple[str, str]:
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        ext = os.path.splitext(path)[1].lower()
        mime = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
        }.get(ext, "image/png")
        return b64, mime

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------
    def _load_data(self) -> bool:
        self._log("Loading data...", "step")

        # 1) Geometry code
        # Priority order (highest first):
        #   explicit path > Memory(stage9_small_geometry) >
        #   Memory(stage7_small_objects) > Memory(stage6_geometry) > sibling
        #   stage9_small_geometry > sibling stage7_small_objects > sibling
        #   stage6_geometry > stage4 > stage3
        # `stage9_small_geometry` is the optional Stage 9 output that
        # rewrites Stage 7's flat create_box/cylinder calls into
        # create_detailed_object_small (with a DETAILED_GEOMETRY_SMALL
        # dict). Preferring it ensures downstream material / texture /
        # render see the most-decorated base. When 9 was skipped (the
        # default), stage7_small_objects is used instead.
        if self.geometry_code_path and os.path.exists(self.geometry_code_path):
            with open(self.geometry_code_path, "r", encoding="utf-8") as f:
                self.geometry_code = f.read()
            self._log(f"Geometry code: {self.geometry_code_path}", "success")
        elif self.use_memory:
            for pref_stage in (
                "stage9_small_geometry",
                "stage7_small_objects",
                "stage6_geometry",
            ):
                entry = self.memory.get_latest(stage=pref_stage, type="result")
                if not entry:
                    continue
                code_path = entry.metadata.get("output_file")
                if code_path and os.path.exists(code_path):
                    with open(code_path, "r", encoding="utf-8") as f:
                        self.geometry_code = f.read()
                    self._log(
                        f"Geometry code: from Memory ({pref_stage}) - {code_path}",
                        "success",
                    )
                    break

        if not self.geometry_code:
            run_dir = os.path.dirname(self.output_dir)
            for candidate in (
                os.path.join(run_dir, "stage9_small_geometry", "small_geometry_output.py"),
                os.path.join(run_dir, "stage7_small_objects", "small_objects_output.py"),
                os.path.join(run_dir, "stage6_geometry", "geometry_output.py"),
                os.path.join(run_dir, "stage4", "stage4_output.py"),
                os.path.join(run_dir, "stage4", "stage4_clean.py"),
                os.path.join(run_dir, "stage3", "stage3_output.py"),
            ):
                if os.path.exists(candidate):
                    with open(candidate, "r", encoding="utf-8") as f:
                        self.geometry_code = f.read()
                    self._log(f"Geometry code: from sibling - {candidate}", "success")
                    break

        if not self.geometry_code:
            self._log("Geometry code not found!", "error")
            return False

        # 1b) Optional: Stage 8 small_describe_output.json (gives the
        #     LLM per-item descriptive context — object_type, material,
        #     color, description, part_hierarchy_hint). Silently absent
        #     when --detail-small-objects is off.
        self.small_describe_map = self._build_small_describe_map()

        # 2) Reference image
        if not self.image_path and self.use_memory:
            stage1_entry = self.memory.get_latest(stage="stage1", type="result")
            if stage1_entry:
                self.image_path = stage1_entry.metadata.get("image_path")

        if not (self.image_path and os.path.exists(self.image_path)):
            self._log("Reference image not found!", "error")
            return False
        self._log(f"Reference image: {self.image_path}", "success")

        # 3) Describe data (for object descriptions)
        if self.use_memory:
            desc_entry = self.memory.get_latest(
                stage="stage5_describe", type="result"
            )
            if desc_entry:
                try:
                    self.describe_data = json.loads(desc_entry.content)
                    self._log("Describe data: from Memory", "success")
                except Exception:
                    pass

        if not self.describe_data:
            run_dir = os.path.dirname(self.output_dir)
            desc_path = os.path.join(
                run_dir, "stage5_describe", "describe_output.json"
            )
            if os.path.exists(desc_path):
                with open(desc_path, "r", encoding="utf-8") as f:
                    self.describe_data = json.load(f)
                self._log(f"Describe data: {desc_path}", "success")

        return True

    # ------------------------------------------------------------------
    # Parsing helpers
    # ------------------------------------------------------------------
    def _parse_geometry_objects(self) -> Dict[str, List[str]]:
        """Extract ``{object_name: [part_name, ...]}`` from
        ``DETAILED_GEOMETRY``.

        Implementation notes
        --------------------
        Earlier versions used a non-greedy regex ``"parts":\\s*\\[(.*?)\\]``
        which truncated at the first ``]`` found INSIDE the first part's
        ``"loc": [...]`` array. That meant Stage 10 saw only ONE part per
        object even though Stage 6 emitted 5–8 parts each, so the LLM only
        produced one material per object and the rest fell back to a
        Stage-3 mono-color material. We now extract the whole
        ``DETAILED_GEOMETRY = { ... }`` block by brace-balanced walk
        (string-literal aware) and parse it with ``ast.literal_eval``,
        which is robust for the JSON-like Python dict Stage 6 emits.
        """
        import ast

        objects: Dict[str, List[str]] = {}
        code = self.geometry_code or ""
        m = re.search(r"DETAILED_GEOMETRY\s*=\s*\{", code)
        if not m:
            return objects

        start = m.end() - 1  # position of the opening '{'
        depth = 0
        in_str = False
        str_ch = ""
        end = -1
        i = start
        while i < len(code):
            ch = code[i]
            if in_str:
                if ch == "\\":
                    i += 2
                    continue
                if ch == str_ch:
                    in_str = False
            else:
                if ch in ("\"", "'"):
                    in_str = True
                    str_ch = ch
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break
            i += 1

        if end < 0:
            return objects

        block = code[start:end]
        try:
            parsed = ast.literal_eval(block)
        except Exception as exc:  # noqa: BLE001
            self._log(
                f"DETAILED_GEOMETRY ast.literal_eval failed: {exc}",
                "warning",
            )
            return objects

        if not isinstance(parsed, dict):
            return objects

        for obj_name, info in parsed.items():
            if not isinstance(info, dict):
                continue
            parts = info.get("parts", []) or []
            names = [
                p.get("name")
                for p in parts
                if isinstance(p, dict) and p.get("name")
            ]
            if names:
                objects[obj_name] = names
        return objects

    def _build_describe_map(self) -> Dict[str, Dict]:
        """Map object name -> describe info from stage5_describe output."""
        desc_map: Dict[str, Dict] = {}
        for obj in self.describe_data.get("objects", []):
            desc_map[obj.get("name", "")] = obj
        return desc_map

    # ------------------------------------------------------------------
    # Stage 8 / 9 small-object support
    # ------------------------------------------------------------------
    def _build_small_describe_map(self) -> Dict[str, Dict]:
        """Load `small_describe_output.json` (Stage 8) and index by item
        name. Empty dict when the file is not present (the default — opt-in
        via `--detail-small-objects`)."""
        data: Optional[Dict] = None

        if self.use_memory:
            entry = self.memory.get_latest(
                stage="stage8_small_describe", type="result"
            )
            if entry:
                meta_path = entry.metadata.get("output_file")
                if meta_path and os.path.exists(meta_path):
                    try:
                        with open(meta_path, "r", encoding="utf-8") as f:
                            data = json.load(f)
                    except Exception:  # noqa: BLE001
                        pass

        if data is None:
            run_dir = os.path.dirname(self.output_dir) if self.output_dir else None
            if run_dir:
                cand = os.path.join(
                    run_dir, "stage8_small_describe", "small_describe_output.json"
                )
                if os.path.exists(cand):
                    try:
                        with open(cand, "r", encoding="utf-8") as f:
                            data = json.load(f)
                    except Exception:  # noqa: BLE001
                        pass

        if not data:
            return {}

        mp: Dict[str, Dict] = {}
        for rec in data.get("small_objects", []) or []:
            name = rec.get("name")
            if name:
                mp[name] = rec
        if mp:
            self._log(
                f"small_describe records: {len(mp)} items",
                "info",
            )
        return mp

    def _parse_small_geometry_objects(self) -> Dict[str, List[str]]:
        """Extract ``{small_obj_name: [part_name, ...]}`` from
        ``DETAILED_GEOMETRY_SMALL`` in the loaded geometry code.

        Returns ``{}`` when Stage 9 was not run — Stage 10 then silently
        skips small-object material generation."""
        import ast

        objects: Dict[str, List[str]] = {}
        code = self.geometry_code or ""
        m = re.search(r"DETAILED_GEOMETRY_SMALL\s*=\s*\{", code)
        if not m:
            return objects

        start = m.end() - 1
        depth = 0
        in_str = False
        str_ch = ""
        end = -1
        i = start
        while i < len(code):
            ch = code[i]
            if in_str:
                if ch == "\\":
                    i += 2
                    continue
                if ch == str_ch:
                    in_str = False
            else:
                if ch in ("\"", "'"):
                    in_str = True
                    str_ch = ch
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break
            i += 1

        if end < 0:
            return objects

        block = code[start:end]
        try:
            parsed = ast.literal_eval(block)
        except Exception as exc:  # noqa: BLE001
            self._log(
                f"DETAILED_GEOMETRY_SMALL parse failed: {exc}", "warning"
            )
            return objects

        if not isinstance(parsed, dict):
            return objects

        for obj_name, info in parsed.items():
            if not isinstance(info, dict):
                continue
            parts = info.get("parts", []) or []
            names = [
                p.get("name")
                for p in parts
                if isinstance(p, dict) and p.get("name")
            ]
            if names:
                objects[obj_name] = names
        return objects

    def _generate_small_part_materials(
        self,
        small_geo: Dict[str, List[str]],
    ) -> Dict[str, Dict[str, Dict]]:
        """Pass 2 for small objects: batched + parallel + palette-anchored.
        Mirrors :py:meth:`_generate_part_materials` but with a small-object
        flavored prompt (glassware/instruments/electronics/etc.)."""
        self._log(
            f"Small Pass 2: per-part materials (batch_size={self.batch_size}, "
            f"parallel={self.parallel})",
            "material",
        )

        names = list(small_geo.keys())
        if not names:
            return {}

        batches: List[List[str]] = []
        for i in range(0, len(names), self.batch_size):
            batches.append(names[i:i + self.batch_size])
        self._log(
            f"  {len(names)} small items -> {len(batches)} batches",
            "info",
        )

        results: Dict[str, Dict[str, Dict]] = {}
        results_lock = __import__("threading").Lock()

        def _run_batch(batch_idx: int, item_names: List[str]) -> None:
            sub = {n: small_geo[n] for n in item_names}
            batch_result = self._llm_batch_small_part_materials(
                sub, batch_idx, len(batches)
            )
            with results_lock:
                for k, v in batch_result.items():
                    results.setdefault(k, {}).update(v)

        from concurrent.futures import ThreadPoolExecutor, as_completed

        if self.parallel <= 1 or len(batches) == 1:
            for i, b in enumerate(batches):
                _run_batch(i, b)
        else:
            with ThreadPoolExecutor(max_workers=self.parallel) as pool:
                futures = {
                    pool.submit(_run_batch, i, b): i
                    for i, b in enumerate(batches)
                }
                for fut in as_completed(futures):
                    fut.result()

        total_parts = sum(len(v) for v in results.values())
        expected_parts = sum(len(v) for v in small_geo.values())
        self._log(
            f"Small per-part materials: {len(results)}/{len(small_geo)} items, "
            f"{total_parts}/{expected_parts} parts",
            "success" if total_parts == expected_parts else "warning",
        )
        return results

    def _llm_batch_small_part_materials(
        self,
        sub_geo: Dict[str, List[str]],
        batch_idx: int,
        total_batches: int,
    ) -> Dict[str, Dict[str, Dict]]:
        """One batch call for small objects — small-object-flavored prompt
        (glassware → transmission, instruments → anodized, electronics →
        plastic shell + screen, lab supplies → paper/cardboard, etc.)."""
        palette = self.scene_palette or {}
        allowed_types_str = " | ".join(self._MATERIAL_TYPES)

        # Unroll every part into a flat list using the same per-part
        # structure as the major-object batcher.
        part_records: List[str] = []
        expected_parts: List[Tuple[str, str]] = []
        for item_name, part_names in sub_geo.items():
            desc = self.small_describe_map.get(item_name, {}) or {}
            obj_header = (
                f"ITEM {item_name}: object_type={desc.get('object_type', '?')}; "
                f"appearance={desc.get('appearance', 'N/A')}; "
                f"material_hint={desc.get('material_description', 'N/A')}; "
                f"color_hint={desc.get('color_description', 'N/A')}; "
                f"description={desc.get('description', 'N/A')}; "
                f"parent={desc.get('parent_name', '?')}"
            )
            part_records.append(obj_header)
            for pn in part_names:
                part_records.append(
                    f"  PART {item_name}.{pn}: assign one material"
                )
                expected_parts.append((item_name, pn))

        palette_text = (
            json.dumps(palette, ensure_ascii=False, indent=2)
            if palette
            else "(no palette anchor — be self-consistent)"
        )
        industrial_context = self._industrial_material_context()

        sys_prompt = f"""You are a Blender PBR material expert specialising in SMALL on-surface items (laboratory glassware, instruments, computer peripherals, kitchen utensils, decor, paperwork, etc.). Assign realistic PBR material parameters to EVERY part listed below — one entry per part.

SCENE PALETTE (cross-batch anchor; reuse colors where roles match):
{palette_text}

Allowed material_type values (use these tokens EXACTLY):
{allowed_types_str}

SMALL-OBJECT GUIDANCE
1. Glassware (beaker / flask / vial / petri dish / cylinder / cuvette / test tube): material_type="glass", base_color near (0.90,0.95,0.98,1.0), roughness <= 0.06, metallic=0.0. The render engine handles transmission downstream.
2. Lab instruments (microscope / centrifuge / spectrometer / pH meter / hotplate / stirrer):
   - Main body / chassis: "anodized_aluminum" or "painted_metal" (dark grey or off-white), roughness 0.4-0.6.
   - Eyepieces / nozzles / optical tubes: "chrome" or "stainless_steel".
   - Buttons / dials: "plastic" or "rubber".
   - LCD / digital screen face: "emission" or dark "glass".
3. Pipettes / pipette racks: rack body = "abs plastic" → use "plastic" with roughness 0.45; pipette barrel = "plastic" tinted in palette accent_rgba (or color hint from describe).
4. Tube / sample racks: body = "plastic" or "painted_metal"; if tubes are visible, tubes = "plastic" (snap caps) or "glass".
5. Computer monitor: screen = "emission" with low intensity OR dark glass (≈0.05, 0.05, 0.05, 1); rear shell = "plastic" (dark grey); stand = "plastic" or "painted_metal".
6. Computer tower / printer: body = "plastic" (light/dark grey, palette-consistent); front panel = same; indicator LEDs = "emission" (if visible).
7. Keyboard / mouse: "plastic", roughness 0.4-0.55.
8. Paperwork / notebook / lab notebook / printout: material_type="paint" (matte) with base_color ~(0.92,0.92,0.88,1.0), roughness 0.85.
9. Cardboard / storage box: material_type="paint" with kraft tone (0.7,0.55,0.4,1), roughness 0.85.
10. Lab mats / silicone pads / rubber feet: "rubber".
11. Stainless steel small items (scoops, tweezers, scalpels): "stainless_steel" or "chrome".
12. Reagent bottles with colored cap: bottle body = "plastic" or "glass" depending on appearance hint; cap = "plastic" with cap color.
13. GENERIC PRINCIPLE: prefer the MOST SPECIFIC token. NEVER default everything to "plastic" — that is the original failure mode this prompt fixes.

{industrial_context}
HARD RULES (same as major-object batcher)
- ONE entry per part. Do not omit any (objectName, partName) pair.
- LINEAR RGB, 0.0-1.0.
- Palette is GUIDANCE for cross-batch consistency but small items can deviate when their nature is intrinsically different (e.g. red lab-tape on a stainless rack — keep the red).

{self._MATERIAL_REFERENCE}

OUTPUT FORMAT (STRICT JSON, no markdown fences):
{{
  "small_part_materials": {{
    "<ItemName>": {{
      "<part_name>": {{
        "base_color": [r, g, b, 1.0],
        "roughness": <float 0..1>,
        "metallic":  <float 0..1>,
        "specular":  <float 0..1>,
        "material_type": "<one of allowed tokens>"
      }}
    }}
  }}
}}
EVERY (item, part) listed below MUST appear in the output."""

        user_text = (
            f"Small-object batch {batch_idx + 1}/{total_batches}. "
            f"Total parts in this batch: {len(expected_parts)}.\n\n"
            + "\n".join(part_records)
        )

        b64, mime = self._encode_image(self.image_path)
        messages = [
            SystemMessage(content=sys_prompt),
            HumanMessage(
                content=[
                    {"type": "text",
                     "text": "Reference image (verify materials):"},
                    {"type": "image_url",
                     "image_url": {"url": f"data:{mime};base64,{b64}"}},
                    {"type": "text", "text": user_text},
                ]
            ),
        ]

        accumulated: Dict[str, Dict[str, Dict]] = {}

        for attempt in range(1, self.max_attempts + 1):
            try:
                response = self.llm.invoke(messages)
                parsed = self._extract_json(response)
            except Exception as exc:  # noqa: BLE001
                self._log(
                    f"  small-batch {batch_idx + 1} attempt {attempt} "
                    f"LLM error: {exc}", "warning"
                )
                parsed = None

            payload = None
            if isinstance(parsed, dict):
                # Accept both 'small_part_materials' and 'part_materials'
                # in case the model copies the major prompt style.
                payload = parsed.get("small_part_materials") or parsed.get(
                    "part_materials"
                )

            if isinstance(payload, dict):
                for item_name, parts_dict in payload.items():
                    if not isinstance(parts_dict, dict):
                        continue
                    bucket = accumulated.setdefault(item_name, {})
                    for part_name, mat in parts_dict.items():
                        if isinstance(mat, dict):
                            bucket[part_name] = mat

            missing: List[Tuple[str, str]] = []
            for item_name, pname in expected_parts:
                if pname not in accumulated.get(item_name, {}):
                    missing.append((item_name, pname))

            if not missing:
                self._log(
                    f"  small-batch {batch_idx + 1}/{total_batches}: "
                    f"all {len(expected_parts)} parts covered "
                    f"(attempt {attempt})",
                    "info",
                )
                return accumulated

            self._log(
                f"  small-batch {batch_idx + 1} attempt {attempt}: "
                f"missing {len(missing)}/{len(expected_parts)} parts",
                "warning",
            )
            focused = (
                "Previous response was incomplete. Return the SAME JSON "
                "shape (small_part_materials) but ONLY for these parts:\n"
            )
            for item_name, pname in missing:
                focused += f"  PART {item_name}.{pname}\n"
            messages = [
                SystemMessage(content=sys_prompt),
                HumanMessage(
                    content=[
                        {"type": "text",
                         "text": "Reference image:"},
                        {"type": "image_url",
                         "image_url": {"url": f"data:{mime};base64,{b64}"}},
                        {"type": "text", "text": focused},
                    ]
                ),
            ]

        if accumulated:
            covered = sum(len(v) for v in accumulated.values())
            self._log(
                f"  small-batch {batch_idx + 1}: partial result "
                f"({covered}/{len(expected_parts)} parts)",
                "warning",
            )
        return accumulated

    def _extract_scene_info(self) -> Dict:
        info = {"scene_w": 8.0, "scene_d": 6.0, "wall_h": 2.8}
        m = re.search(r"SCENE_W\s*=\s*([\d.]+)", self.geometry_code)
        if m:
            info["scene_w"] = float(m.group(1))
        m = re.search(r"SCENE_D\s*=\s*([\d.]+)", self.geometry_code)
        if m:
            info["scene_d"] = float(m.group(1))
        m = re.search(r"WALL_H\s*=\s*([\d.]+)", self.geometry_code)
        if m:
            info["wall_h"] = float(m.group(1))
        return info

    # ------------------------------------------------------------------
    # LLM: per-part material generation
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Material vocabulary
    # ------------------------------------------------------------------
    # Allowed material_type values. Wider than v1 to cover real lab /
    # kitchen / office fixtures whose materials don't reduce to
    # plastic/paint/metal without losing fidelity.
    _MATERIAL_TYPES: Tuple[str, ...] = (
        "fabric", "wood", "metal", "glass", "marble", "plastic",
        "leather", "ceramic", "stone", "wicker", "rattan", "paint",
        # added in v2 — needed for real PBR realism:
        "epoxy_resin",          # lab worktops, solid surface
        "stainless_steel",      # equipment frames, sinks
        "brushed_aluminum",     # appliance shells
        "chrome",               # taps, knobs, microscope tubes
        "anodized_aluminum",    # matte instrument bodies
        "powder_coat",          # cabinet doors, equipment chassis
        "painted_mdf",          # furniture panels
        "painted_metal",        # similar to powder_coat but glossier
        "porcelain",            # sinks, toilets
        "rubber",               # gaskets, stool feet, mats
        "mirror",               # silvered surfaces (no transmission)
        "acrylic",              # transparent panels (treated as glass)
        "pvc",                  # tubing / vinyl
        "concrete",             # walls, floors
        "tile",                 # ceramic tile
        "emission",             # LED indicators, lit screens
    )

    _MATERIAL_REFERENCE = """PBR REFERENCE VALUES (use the closest match):
- fabric/cloth: roughness=0.8-0.95, metallic=0.0, specular=0.3-0.5
- wood (polished): roughness=0.3-0.5, metallic=0.0, specular=0.5
- wood (raw/matte): roughness=0.6-0.8, metallic=0.0, specular=0.3
- metal (generic polished): roughness=0.1-0.3, metallic=0.9-1.0, specular=0.5
- stainless_steel (brushed): roughness=0.30-0.45, metallic=1.0, specular=0.5, base ~ (0.78,0.79,0.81,1)
- brushed_aluminum: roughness=0.35-0.50, metallic=1.0, specular=0.5, base ~ (0.82,0.83,0.85,1)
- chrome: roughness=0.02-0.10, metallic=1.0, specular=0.5, base ~ (0.85,0.87,0.90,1)
- anodized_aluminum: roughness=0.50-0.70, metallic=0.85, specular=0.45, base usually tinted dark/teal/red
- powder_coat: roughness=0.55-0.80, metallic=0.0, specular=0.30; tints to cabinet color
- painted_metal: roughness=0.30-0.55, metallic=0.10-0.30, specular=0.50; gloss varies
- painted_mdf: roughness=0.40-0.70, metallic=0.0, specular=0.35
- mirror: roughness=0.0, metallic=1.0, specular=0.5, base ~ (0.95,0.95,0.95,1)
- glass: roughness=0.0-0.08, metallic=0.0, specular=0.5, base ~ (0.90,0.95,0.98,1) — engine uses transmission
- acrylic: same handling as glass (transmission)
- marble/stone/granite: roughness=0.20-0.45, metallic=0.0, specular=0.5
- epoxy_resin (lab worktop): roughness=0.35-0.55, metallic=0.0, specular=0.45, base usually warm-grey / off-white
- leather: roughness=0.4-0.6, metallic=0.0, specular=0.4
- ceramic/porcelain: roughness=0.10-0.25, metallic=0.0, specular=0.5, base usually near-white
- tile: roughness=0.10-0.30, metallic=0.0, specular=0.5
- wicker/rattan: roughness=0.7-0.9, metallic=0.0, specular=0.2
- plastic (generic): roughness=0.3-0.55, metallic=0.0, specular=0.5
- rubber: roughness=0.80-0.95, metallic=0.0, specular=0.20
- pvc: roughness=0.35-0.60, metallic=0.0, specular=0.45
- concrete: roughness=0.70-0.90, metallic=0.0, specular=0.30
- paint (matte): roughness=0.8-0.95, metallic=0.0, specular=0.3
- paint (glossy): roughness=0.1-0.3, metallic=0.0, specular=0.5
- emission: metallic=0.0, roughness=0.5, specular=0.3 (downstream stage adds emission)"""

    # ------------------------------------------------------------------
    # Pass 1: scene-wide palette anchor
    # ------------------------------------------------------------------
    def _generate_scene_palette(
        self,
        geo_objects: Dict[str, List[str]],
        desc_map: Dict[str, Dict],
    ) -> Dict:
        """Single LLM call that yields a scene-wide palette anchor.

        The result is injected into every Pass-2 batch so that material
        choices stay stylistically consistent across batches (e.g. all
        cabinets share the same powder-coat color, every chrome handle
        matches every chrome tap, the worktop color is constant)."""
        self._log("Pass 1: scene palette pre-pass", "material")

        room_style = self.describe_data.get("room_style", {}) or {}
        obj_lines = []
        for name, parts in geo_objects.items():
            desc = desc_map.get(name, {})
            obj_lines.append(
                f"- {name}: type={desc.get('object_type', '?')}, "
                f"material_hint={desc.get('material_description', '?')}, "
                f"color_hint={desc.get('color_description', '?')}, "
                f"parts={len(parts)}"
            )

        allowed_types_str = " | ".join(self._MATERIAL_TYPES)
        industrial_context = self._industrial_material_context()
        sys_prompt = f"""You are a PBR scene-palette curator. Given a reference image, a room-style summary, and an inventory of objects, produce ONE shared PALETTE that every downstream per-part material decision will reference. This palette is the source-of-truth for cross-object consistency.

Allowed material_type values (use these tokens EXACTLY): {allowed_types_str}

Output JSON only:
{{
  "scene_palette": {{
    "style_summary": "<one sentence>",
    "primary_metal_type": <one of stainless_steel|chrome|brushed_aluminum|anodized_aluminum|painted_metal|null>,
    "primary_metal_rgba": [r,g,b,1.0],
    "primary_metal_roughness": <0.0-1.0>,

    "secondary_metal_type": <same enum or null>,
    "secondary_metal_rgba": [r,g,b,1.0],

    "primary_wood_type": <"oak"|"walnut"|"pine"|"maple"|"cherry"|null>,
    "primary_wood_rgba": [r,g,b,1.0],
    "primary_wood_roughness": <0.0-1.0>,

    "worktop_material_type": <"epoxy_resin"|"marble"|"granite"|"stainless_steel"|"wood"|"laminate"|"tile"|"plastic">,
    "worktop_rgba": [r,g,b,1.0],
    "worktop_roughness": <0.0-1.0>,

    "cabinet_finish_type": <"powder_coat"|"painted_mdf"|"wood_veneer"|"laminate"|"stainless_steel">,
    "cabinet_rgba": [r,g,b,1.0],
    "cabinet_roughness": <0.0-1.0>,

    "fabric_rgba": [r,g,b,1.0],
    "fabric_roughness": <0.0-1.0>,

    "accent_rgba": [r,g,b,1.0],
    "wall_rgba": [r,g,b,1.0],
    "floor_rgba": [r,g,b,1.0],

    "glass_handling": <"transmission_full"|"transmission_partial"|"opaque">
  }}
}}

Guidelines:
- Inspect the reference image carefully and extract dominant tones (not lazy grey).
- If a slot does not apply (e.g. no wood in a steel lab), set the *_type to null and pick a neutral rgba.
- All RGBA values are LINEAR (NOT sRGB) and in [0.0, 1.0].
- "primary_*" = the appearance shared by most instances of that material in the scene.
- The palette is GUIDANCE, downstream stages may deviate when an object is clearly a different material; but for any part whose role is a worktop/cabinet/frame/etc. the corresponding palette entry MUST be used.
{industrial_context}"""

        style_summary_in = ""
        if room_style:
            style_summary_in = (
                f"\nROOM STYLE: name={room_style.get('style_name', 'unknown')}, "
                f"palette={room_style.get('color_palette', [])}, "
                f"mood={room_style.get('mood', '')}"
            )
        user_text = (
            "Object inventory (with hints from Stage 7):\n"
            + "\n".join(obj_lines)
            + style_summary_in
        )

        b64, mime = self._encode_image(self.image_path)
        messages = [
            SystemMessage(content=sys_prompt),
            HumanMessage(
                content=[
                    {"type": "text",
                     "text": "Reference image (extract dominant colors carefully):"},
                    {"type": "image_url",
                     "image_url": {"url": f"data:{mime};base64,{b64}"}},
                    {"type": "text", "text": user_text},
                ]
            ),
        ]

        for attempt in range(1, self.max_attempts + 1):
            try:
                response = self.llm.invoke(messages)
                parsed = self._extract_json(response)
                if parsed and isinstance(parsed.get("scene_palette"), dict):
                    sp = parsed["scene_palette"]
                    self._log(
                        f"Scene palette: style='{sp.get('style_summary', '')}' "
                        f"worktop={sp.get('worktop_material_type')} "
                        f"primary_metal={sp.get('primary_metal_type')}",
                        "success",
                    )
                    return sp
                self._log(
                    f"Scene palette attempt {attempt}: missing 'scene_palette'",
                    "warning",
                )
            except Exception as exc:  # noqa: BLE001
                self._log(
                    f"Scene palette attempt {attempt} error: {exc}",
                    "warning",
                )
        return {}

    # ------------------------------------------------------------------
    # Pass 2: per-part materials, batched + parallel, anchored on palette
    # ------------------------------------------------------------------
    def _generate_part_materials(
        self,
        geo_objects: Dict[str, List[str]],
        desc_map: Dict[str, Dict],
    ) -> Dict[str, Dict[str, Dict]]:
        """Assign PBR materials to EVERY part using:
        (1) palette anchor from `_generate_scene_palette`,
        (2) per-part unrolled prompt (no nesting),
        (3) parallel batches with retry.

        Returns ``{object_name: {part_name: mat_dict, ...}, ...}``."""
        self._log(
            f"Pass 2: per-part materials (batch_size={self.batch_size}, "
            f"parallel={self.parallel})",
            "material",
        )

        names = list(geo_objects.keys())
        if not names:
            return {}

        # Group objects into batches preserving order.
        batches: List[List[str]] = []
        for i in range(0, len(names), self.batch_size):
            batches.append(names[i:i + self.batch_size])

        self._log(
            f"  {len(names)} objects -> {len(batches)} batches",
            "info",
        )

        results: Dict[str, Dict[str, Dict]] = {}
        results_lock = __import__("threading").Lock()

        def _run_batch(batch_idx: int, obj_names: List[str]) -> None:
            sub_geo = {n: geo_objects[n] for n in obj_names}
            batch_result = self._llm_batch_part_materials(
                sub_geo, desc_map, batch_idx, len(batches)
            )
            with results_lock:
                for k, v in batch_result.items():
                    results.setdefault(k, {}).update(v)

        from concurrent.futures import ThreadPoolExecutor, as_completed

        if self.parallel <= 1 or len(batches) == 1:
            for i, b in enumerate(batches):
                _run_batch(i, b)
        else:
            with ThreadPoolExecutor(max_workers=self.parallel) as pool:
                futures = {
                    pool.submit(_run_batch, i, b): i
                    for i, b in enumerate(batches)
                }
                for fut in as_completed(futures):
                    fut.result()

        total_parts = sum(len(v) for v in results.values())
        expected_parts = sum(len(v) for v in geo_objects.values())
        self._log(
            f"Per-part materials: {len(results)}/{len(geo_objects)} objects, "
            f"{total_parts}/{expected_parts} parts",
            "success" if total_parts == expected_parts else "warning",
        )
        return results

    def _llm_batch_part_materials(
        self,
        sub_geo: Dict[str, List[str]],
        desc_map: Dict[str, Dict],
        batch_idx: int,
        total_batches: int,
    ) -> Dict[str, Dict[str, Dict]]:
        """One batch call: ask LLM for every part in this batch, retry on
        missing keys."""
        palette = self.scene_palette or {}
        room_style = self.describe_data.get("room_style", {}) or {}
        allowed_types_str = " | ".join(self._MATERIAL_TYPES)

        # Unroll every part into a flat list of records; the LLM is asked
        # to return ONE entry per record. This kills the "homogenize one
        # material across all parts" failure mode of nested prompts.
        part_records: List[str] = []
        expected_parts: List[Tuple[str, str]] = []
        for obj_name, part_names in sub_geo.items():
            desc = desc_map.get(obj_name, {})
            obj_header = (
                f"OBJECT {obj_name}: type={desc.get('object_type', 'unknown')}; "
                f"description={desc.get('description', 'N/A')}; "
                f"material_hint={desc.get('material_description', 'N/A')}; "
                f"color_hint={desc.get('color_description', 'N/A')}"
            )
            part_records.append(obj_header)
            for pn in part_names:
                part_records.append(
                    f"  PART {obj_name}.{pn}: assign one material"
                )
                expected_parts.append((obj_name, pn))

        palette_text = (
            json.dumps(palette, ensure_ascii=False, indent=2)
            if palette
            else "(no palette anchor — be self-consistent)"
        )

        style_context = ""
        if room_style:
            style_context = (
                f"\nROOM STYLE: name={room_style.get('style_name', 'unknown')}, "
                f"palette={room_style.get('color_palette', [])}, "
                f"mood={room_style.get('mood', '')}\n"
            )
        industrial_context = self._industrial_material_context()

        sys_prompt = f"""You are a Blender PBR material expert. Assign realistic PBR material parameters to EVERY part listed below, one entry per part.

SCENE PALETTE (the source-of-truth — DO NOT contradict for matching part roles):
{palette_text}
{style_context}
Allowed material_type values (use these tokens EXACTLY):
{allowed_types_str}

HARD RULES
1. Output ONE entry per part. Do NOT collapse "drawer_1..4" into a single entry — give each its own record (they may share values, but the keys must be present).
2. Different STRUCTURAL roles within the same object MUST get different materials when appropriate. Examples:
   - bench: worktop_slab=epoxy_resin, base_cabinet=powder_coat, toe_kick=powder_coat (darker), drawer_handle_*=chrome, drawer_*=powder_coat.
   - microscope: base=anodized_aluminum (dark), arm=anodized_aluminum, eyepiece=chrome, light_emitter=emission, stage_clip=stainless_steel.
   - fume_hood: front_sash=glass (transmission), frame=stainless_steel, side_panel=powder_coat, worktop=epoxy_resin.
   - sofa: seat=fabric, legs=wood/metal, frame=wood.
3. GLASS / ACRYLIC DETECTION (mandatory): any part whose name contains "glass", "window_pane", "transparent", "display_front", "sash", "view_port", "screen_face" MUST be material_type="glass" (or "acrylic") with base_color ~ (0.90,0.95,0.98,1.0), roughness <= 0.08, metallic=0.0. The render engine handles transmission downstream.
4. METAL DETECTION: chrome handles, faucets, knobs, taps, microscope tubes -> material_type="chrome", metallic=1.0, roughness<=0.1. Equipment frames, fume hood frames, sinks -> "stainless_steel". Cabinets/equipment shells with a colored matte paint -> "powder_coat" (metallic=0.0).
5. Match palette colors: any part whose role is a worktop/benchtop/tabletop in a lab/kitchen -> palette.worktop_rgba + palette.worktop_material_type. Any cabinet door / drawer face -> palette.cabinet_rgba + palette.cabinet_finish_type. Any chrome detail -> palette.primary_metal_* when it's chrome, otherwise palette.secondary_metal_*.
6. Colors in LINEAR RGB, 0.0-1.0 (not sRGB 0-255).
7. NEVER default an unknown lab/kitchen surface to plain "plastic". Pick the most specific token from the allowed list.

{industrial_context}
{self._MATERIAL_REFERENCE}

OUTPUT FORMAT (STRICT JSON, no markdown fences):
{{
  "part_materials": {{
    "<ObjectName>": {{
      "<part_name>": {{
        "base_color": [r, g, b, 1.0],
        "roughness": <float 0..1>,
        "metallic":  <float 0..1>,
        "specular":  <float 0..1>,
        "material_type": "<one of allowed tokens>"
      }}
    }}
  }}
}}
EVERY (object, part) listed below MUST appear in the output."""

        user_text = (
            f"Batch {batch_idx + 1}/{total_batches}. "
            f"Assign one PBR material per PART below. "
            f"Total parts in this batch: {len(expected_parts)}.\n\n"
            + "\n".join(part_records)
        )

        b64, mime = self._encode_image(self.image_path)
        messages = [
            SystemMessage(content=sys_prompt),
            HumanMessage(
                content=[
                    {"type": "text",
                     "text": "Reference image (color/material verification):"},
                    {"type": "image_url",
                     "image_url": {"url": f"data:{mime};base64,{b64}"}},
                    {"type": "text", "text": user_text},
                ]
            ),
        ]

        accumulated: Dict[str, Dict[str, Dict]] = {}

        for attempt in range(1, self.max_attempts + 1):
            try:
                response = self.llm.invoke(messages)
                parsed = self._extract_json(response)
            except Exception as exc:  # noqa: BLE001
                self._log(
                    f"  batch {batch_idx + 1} attempt {attempt} LLM error: "
                    f"{exc}", "warning"
                )
                parsed = None

            if parsed and isinstance(parsed.get("part_materials"), dict):
                pm = parsed["part_materials"]
                for obj_name, parts_dict in pm.items():
                    if not isinstance(parts_dict, dict):
                        continue
                    bucket = accumulated.setdefault(obj_name, {})
                    for part_name, mat in parts_dict.items():
                        if isinstance(mat, dict):
                            bucket[part_name] = mat

            # Identify missing parts.
            missing: List[Tuple[str, str]] = []
            for obj_name, pname in expected_parts:
                if pname not in accumulated.get(obj_name, {}):
                    missing.append((obj_name, pname))

            if not missing:
                self._log(
                    f"  batch {batch_idx + 1}/{total_batches}: "
                    f"all {len(expected_parts)} parts covered "
                    f"(attempt {attempt})",
                    "info",
                )
                return accumulated

            self._log(
                f"  batch {batch_idx + 1} attempt {attempt}: "
                f"missing {len(missing)}/{len(expected_parts)} parts, "
                f"retrying with focused request",
                "warning",
            )
            # Build a focused follow-up: re-ask only for the missing parts.
            focused = "Previous response was incomplete. Return the SAME JSON shape but ONLY for these parts:\n"
            for obj_name, pname in missing:
                focused += f"  PART {obj_name}.{pname}\n"
            messages = [
                SystemMessage(content=sys_prompt),
                HumanMessage(
                    content=[
                        {"type": "text",
                         "text": "Reference image:"},
                        {"type": "image_url",
                         "image_url": {"url": f"data:{mime};base64,{b64}"}},
                        {"type": "text", "text": focused},
                    ]
                ),
            ]

        if accumulated:
            covered = sum(len(v) for v in accumulated.values())
            self._log(
                f"  batch {batch_idx + 1}: partial result "
                f"({covered}/{len(expected_parts)} parts)",
                "warning",
            )
        else:
            self._log(
                f"  batch {batch_idx + 1}: NO parts returned", "error"
            )
        return accumulated

    # ------------------------------------------------------------------
    # LLM: floor & wall materials
    # ------------------------------------------------------------------
    def _generate_floor_wall_materials(self) -> Tuple[Dict, Dict]:
        """Ask LLM for detailed floor and wall PBR materials."""
        self._log("Generating floor & wall materials...", "material")

        room_style = self.describe_data.get("room_style", {})
        style_hint = room_style.get("style_name", "modern")
        keywords = ", ".join(room_style.get("keywords", []))

        industrial_context = self._industrial_material_context()
        system_prompt = f"""You are a Blender PBR material expert specializing in architectural surfaces.
Analyze the provided top-down floor plan image and generate detailed PBR material definitions for the FLOOR and WALLS.

Room style: {style_hint}
Style keywords: {keywords}
{industrial_context}
Output ONLY valid JSON:
{{
  "floor_material": {{
    "material_type": "hardwood|tile|marble|concrete|carpet|stone|laminate",
    "base_color": [R, G, B, 1.0],
    "roughness": 0.0-1.0,
    "metallic": 0.0,
    "specular": 0.0-1.0,
    "pattern": "plank|herringbone|checker|hexagonal|solid|random_tile",
    "pattern_scale": 1.0,
    "pattern_color2": [R, G, B, 1.0],
    "bump_strength": 0.0-0.3,
    "description": "brief description of the floor"
  }},
  "wall_material": {{
    "material_type": "paint|wallpaper|brick|concrete|plaster|wood_panel",
    "base_color": [R, G, B, 1.0],
    "roughness": 0.0-1.0,
    "metallic": 0.0,
    "specular": 0.0-1.0,
    "finish": "matte|eggshell|satin|semi_gloss|gloss",
    "bump_strength": 0.0-0.1,
    "description": "brief description of the walls"
  }}
}}

Color values are in linear RGB 0.0-1.0.
Determine floor and wall materials from what you see in the image.
For industrial scenes, avoid residential materials: no hardwood, carpet, rugs, wallpaper, or decorative wall panels unless explicitly visible."""

        b64, mime = self._encode_image(self.image_path)
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(
                content=[
                    {"type": "text", "text": "Analyze the floor and wall materials:"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{b64}"},
                    },
                ]
            ),
        ]

        if self._is_industrial_scene():
            floor_mat = self._industrial_floor_material()
            wall_mat = self._industrial_wall_material()
        else:
            floor_mat = self._default_floor_material()
            wall_mat = self._default_wall_material()

        try:
            response = self.llm.invoke(messages)
            result = self._extract_json(response)
            if result:
                if "floor_material" in result:
                    floor_mat = result["floor_material"]
                    self._log(
                        f"Floor: {floor_mat.get('material_type', '?')} "
                        f"- {floor_mat.get('description', '')}",
                        "success",
                    )
                if "wall_material" in result:
                    wall_mat = result["wall_material"]
                    self._log(
                        f"Wall: {wall_mat.get('material_type', '?')} "
                        f"- {wall_mat.get('description', '')}",
                        "success",
                    )
        except Exception as e:
            self._log(f"Floor/wall material error: {e}", "error")

        return self._sanitize_industrial_floor_wall(floor_mat, wall_mat)

    # ------------------------------------------------------------------
    # Code generation: inject materials
    # ------------------------------------------------------------------
    def _inject_materials(self) -> str:
        """Build modified Blender code with per-part PBR materials."""
        code = self.geometry_code

        # 1) Replace or add create_pbr_material function
        pbr_func = self._build_pbr_function()

        old_func_pattern = re.compile(
            r"def create_material\s*\([^)]*\):\s*\n(?:.*\n)*?(?=\ndef |\n# ===)",
        )
        if old_func_pattern.search(code):
            code = old_func_pattern.sub(pbr_func + "\n\n", code, count=1)
        else:
            import_end = code.rfind("import ")
            if import_end > 0:
                line_end = code.find("\n", import_end)
                code = code[: line_end + 1] + "\n" + pbr_func + "\n" + code[line_end + 1 :]

        if "create_material = create_pbr_material" not in code:
            code = code.replace(
                pbr_func,
                pbr_func + "\ncreate_material = create_pbr_material\n",
                1,
            )

        # 2) Build PART_MATERIALS dict
        pm_code = self._build_part_materials_dict()

        # 3) Build floor/wall material code
        fw_code = self._build_floor_wall_code()

        # 4) Replace create_detailed_object with material-aware version
        new_cdo = self._build_create_detailed_object()

        # 4b) Small-object integration (Stage 8 / 9).
        #     `has_small` is True when:
        #       (a) the loaded geometry code defined DETAILED_GEOMETRY_SMALL
        #           AND create_detailed_object_small, AND
        #       (b) Stage 10 generated per-part materials for those items.
        #     When false we leave the original Stage-9 helper alone, so
        #     small items keep the caller-supplied flat material.
        has_small = bool(self.small_part_materials)
        small_pm_code = ""
        new_cdo_small = ""
        if has_small:
            small_pm_code = self._build_small_part_materials_dict()
            new_cdo_small = self._build_create_detailed_object_small()
            # Remove the existing (non-material-aware) helper defined in
            # the Stage 9 base code; we will re-emit a material-aware
            # version below.
            code = re.sub(
                r"\ndef create_detailed_object_small\([^)]*\):.*?(?=\n(?:def |# ===|if __name__))",
                "",
                code,
                count=1,
                flags=re.DOTALL,
            )

        # 5) Find injection point (before DETAILED_GEOMETRY or before run_layout_engine)
        markers = [
            "# === MAIN LAYOUT ENGINE ===",
            "def run_layout_engine():",
        ]

        # Remove old create_detailed_object if present
        code = re.sub(
            r"\ndef create_detailed_object\([^)]*\):.*?(?=\n(?:def |# ===|if __name__))",
            "",
            code,
            count=1,
            flags=re.DOTALL,
        )

        inject_point = None
        for marker in markers:
            pos = code.find(marker)
            if pos > 0:
                inject_point = pos
                break

        injection_parts = [pm_code, fw_code, new_cdo]
        if has_small:
            injection_parts.extend([small_pm_code, new_cdo_small])
        injection = "\n" + "\n\n".join(injection_parts) + "\n\n"

        if inject_point:
            code = code[:inject_point] + injection + code[inject_point:]
        else:
            geo_marker = "DETAILED_GEOMETRY = {"
            pos = code.find(geo_marker)
            if pos > 0:
                end_of_geo = code.find("\n\n", pos + len(geo_marker))
                if end_of_geo < 0:
                    end_of_geo = len(code)
                code = code[: end_of_geo + 1] + injection + code[end_of_geo + 1 :]

        # 6) Replace floor/wall material calls in run_layout_engine
        code = self._replace_floor_wall_in_layout(code)

        # 7) Add material setup call inside run_layout_engine
        code = self._add_material_setup_call(code)

        return code

    def _build_pbr_function(self) -> str:
        return '''_BUMP_PROFILES = {
    "fabric":            {"scale": 300.0, "detail": 4.0, "roughness": 0.6, "strength": 0.10},
    "cloth":             {"scale": 300.0, "detail": 4.0, "roughness": 0.6, "strength": 0.10},
    "wood":              {"scale":  80.0, "detail": 8.0, "roughness": 0.4, "strength": 0.15},
    "metal":             {"scale": 500.0, "detail": 2.0, "roughness": 0.3, "strength": 0.03},
    "stainless_steel":   {"scale": 600.0, "detail": 2.0, "roughness": 0.3, "strength": 0.02},
    "brushed_aluminum":  {"scale": 800.0, "detail": 2.0, "roughness": 0.3, "strength": 0.02},
    "chrome":            {"scale": 800.0, "detail": 1.0, "roughness": 0.0, "strength": 0.005},
    "anodized_aluminum": {"scale": 400.0, "detail": 2.0, "roughness": 0.5, "strength": 0.02},
    "painted_metal":     {"scale": 350.0, "detail": 2.0, "roughness": 0.4, "strength": 0.025},
    "powder_coat":       {"scale": 250.0, "detail": 3.0, "roughness": 0.6, "strength": 0.04},
    "painted_mdf":       {"scale": 280.0, "detail": 2.0, "roughness": 0.5, "strength": 0.03},
    "epoxy_resin":       {"scale": 200.0, "detail": 6.0, "roughness": 0.45,"strength": 0.05},
    "marble":            {"scale":  30.0, "detail": 6.0, "roughness": 0.5, "strength": 0.08},
    "stone":             {"scale":  40.0, "detail": 8.0, "roughness": 0.7, "strength": 0.25},
    "leather":           {"scale": 150.0, "detail": 4.0, "roughness": 0.5, "strength": 0.20},
    "ceramic":           {"scale": 200.0, "detail": 2.0, "roughness": 0.2, "strength": 0.04},
    "porcelain":         {"scale": 250.0, "detail": 2.0, "roughness": 0.2, "strength": 0.03},
    "tile":              {"scale": 200.0, "detail": 2.0, "roughness": 0.2, "strength": 0.04},
    "plastic":           {"scale": 400.0, "detail": 2.0, "roughness": 0.4, "strength": 0.02},
    "acrylic":           {"scale": 800.0, "detail": 1.0, "roughness": 0.0, "strength": 0.005},
    "pvc":               {"scale": 450.0, "detail": 2.0, "roughness": 0.4, "strength": 0.02},
    "rubber":            {"scale": 350.0, "detail": 3.0, "roughness": 0.8, "strength": 0.10},
    "wicker":            {"scale": 200.0, "detail": 6.0, "roughness": 0.7, "strength": 0.30},
    "rattan":            {"scale": 200.0, "detail": 6.0, "roughness": 0.7, "strength": 0.30},
    "paint":             {"scale": 350.0, "detail": 2.0, "roughness": 0.5, "strength": 0.03},
    "concrete":          {"scale":  60.0, "detail": 8.0, "roughness": 0.7, "strength": 0.20},
    "default":           {"scale": 250.0, "detail": 4.0, "roughness": 0.5, "strength": 0.04},
}
_NO_BUMP_TYPES = {"glass", "mirror", "emission", "led", "screen", "acrylic", "chrome"}


def _add_procedural_bump(nodes, links, bsdf, profile):
    """Wire a Noise -> Bump chain into ``bsdf.Normal`` for surface micro-detail.

    Generated coordinates keep the pattern locked to the mesh, so it does
    not drift when objects move. Strengths are intentionally subtle
    (<= 0.30) so the material reads as the LLM-authored color/roughness
    rather than turning into a noise texture.
    """
    tex_coord = nodes.new(type="ShaderNodeTexCoord")
    tex_coord.location = (-700, -180)
    noise = nodes.new(type="ShaderNodeTexNoise")
    noise.location = (-450, -180)
    noise.inputs["Scale"].default_value = float(profile["scale"])
    noise.inputs["Detail"].default_value = float(profile["detail"])
    if "Roughness" in noise.inputs:
        noise.inputs["Roughness"].default_value = float(profile["roughness"])
    bump = nodes.new(type="ShaderNodeBump")
    bump.location = (-200, -180)
    bump.inputs["Strength"].default_value = float(profile["strength"])
    bump.inputs["Distance"].default_value = 0.02
    links.new(tex_coord.outputs["Generated"], noise.inputs["Vector"])
    links.new(noise.outputs["Fac"], bump.inputs["Height"])
    links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])


def create_pbr_material(name, base_color, roughness=0.5, metallic=0.0, specular=0.5, alpha=None, transmission=0.0, ior=1.45, material_type=None):
    """Create a Principled BSDF material with PBR properties.

    ``transmission`` > 0 enables glass-like see-through behaviour (low
    roughness, high transmission, blended alpha). Safe to pass on Blender
    3.x (``Transmission Weight``) and 2.9x (``Transmission``).

    ``material_type`` (e.g. ``"wood"``, ``"fabric"``, ``"metal"``) selects a
    procedural Noise+Bump profile that gives the surface fine micro-detail
    (kills "plastic look"). Glass / emission / unknown types skip bump.
    """
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    node_output = nodes.new(type='ShaderNodeOutputMaterial')
    node_output.location = (400, 0)
    bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
    bsdf.location = (0, 0)
    links.new(bsdf.outputs['BSDF'], node_output.inputs['Surface'])
    bsdf.inputs['Base Color'].default_value = base_color
    bsdf.inputs['Roughness'].default_value = roughness
    bsdf.inputs['Metallic'].default_value = metallic
    spec_input = bsdf.inputs.get('Specular IOR Level') or bsdf.inputs.get('Specular')
    if spec_input:
        spec_input.default_value = specular
    if transmission and transmission > 0:
        tx_input = (bsdf.inputs.get('Transmission Weight')
                    or bsdf.inputs.get('Transmission'))
        if tx_input:
            tx_input.default_value = float(transmission)
        ior_input = bsdf.inputs.get('IOR')
        if ior_input:
            ior_input.default_value = float(ior)
        # Keep the surface visibly transparent in Eevee/Workbench too —
        # Cycles already respects transmission, but the viewport/report
        # renderer uses the material's blend method.
        if hasattr(mat, 'blend_method'):
            mat.blend_method = 'BLEND'
        if hasattr(mat, 'show_transparent_back'):
            mat.show_transparent_back = True
    if alpha is not None:
        bsdf.inputs['Alpha'].default_value = alpha
        if hasattr(mat, 'blend_method'):
            mat.blend_method = 'BLEND'
    mt = str(material_type or "").strip().lower()
    is_glassy = bool(transmission and transmission > 0)
    if mt and mt not in _NO_BUMP_TYPES and not is_glassy:
        profile = _BUMP_PROFILES.get(mt, _BUMP_PROFILES["default"])
        try:
            _add_procedural_bump(nodes, links, bsdf, profile)
        except Exception as _bump_err:
            print("[stage10_material] Bump injection skipped for " + str(name) + ": " + str(_bump_err))
    return mat'''

    def _build_small_part_materials_dict(self) -> str:
        """Pretty-print SMALL_PART_MATERIALS Python literal (same shape as
        PART_MATERIALS but a separate dict so downstream code can tell
        them apart)."""
        lines = [
            "# ==============================================================================",
            "# PER-PART MATERIAL DATA FOR SMALL OBJECTS (Stage 8 / 9 integration)",
            "# ==============================================================================",
            "",
            "SMALL_PART_MATERIALS = {",
        ]
        for item_name, parts_dict in self.small_part_materials.items():
            lines.append(f'    "{item_name}": {{')
            for part_name, mp in parts_dict.items():
                bc = mp.get("base_color", [0.5, 0.5, 0.5, 1.0])
                ro = mp.get("roughness", 0.5)
                me = mp.get("metallic", 0.0)
                sp = mp.get("specular", 0.5)
                mt = mp.get("material_type", "unknown")
                bc_str = (
                    f"({bc[0]:.3f}, {bc[1]:.3f}, {bc[2]:.3f}, "
                    f"{bc[3] if len(bc) > 3 else 1.0:.1f})"
                )
                lines.append(
                    f'        "{part_name}": {{"base_color": {bc_str}, '
                    f'"roughness": {ro}, "metallic": {me}, "specular": {sp}, '
                    f'"type": "{mt}"}},'
                )
            lines.append("    },")
        lines.append("}")
        return "\n".join(lines)

    def _build_create_detailed_object_small(self) -> str:
        """Material-aware override of the Stage 9 helper. Looks up
        SMALL_PART_MATERIALS first; falls back to caller's ``material``
        arg (preserves Stage 7 / 9 behaviour when an entry is
        missing). Handles glass / acrylic transmission and mirror
        identically to ``create_detailed_object``."""
        return '''def create_detailed_object_small(name, location=None, rotation=None, material=None, collection=None):
    """Create a small composite object with per-part PBR materials.

    Stage 10 (material) injects SMALL_PART_MATERIALS keyed by item name.
    When an entry exists, every part gets its own LLM-authored material;
    otherwise the caller-supplied ``material`` argument is applied to
    all parts (preserves Stage 7 / 9 default behaviour)."""
    data = DETAILED_GEOMETRY_SMALL.get(name)
    if not data:
        return None

    center = location if location is not None else data.get("center", (0, 0, 0))
    base_rot = rotation if rotation is not None else data.get("rotation", (0, 0, 0))
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

    has_part_mats = name in SMALL_PART_MATERIALS

    for part in parts:
        ptype = part.get("shape") or part.get("type") or "box"
        pname_short = part.get("name", "part")
        pname = f"{name}_{pname_short}"
        ploc = part.get("relative_location") or part.get("loc") or (0, 0, 0)
        pdim = part.get("dimensions") or part.get("dim") or (0.01, 0.01, 0.01)
        prot = part.get("rotation") or part.get("rot") or (0, 0, 0)

        mesh = bpy.data.meshes.new(pname + "_mesh")
        bm = bmesh.new()
        if ptype == "box":
            bmesh.ops.create_cube(bm, size=1.0)
        elif ptype == "cylinder":
            bmesh.ops.create_cone(bm, cap_ends=True, cap_tris=False, segments=24, radius1=0.5, radius2=0.5, depth=1.0)
        elif ptype == "sphere":
            bmesh.ops.create_uvsphere(bm, u_segments=24, v_segments=12, radius=0.5)
        elif ptype == "cone":
            bmesh.ops.create_cone(bm, cap_ends=True, cap_tris=False, segments=24, radius1=0.5, radius2=0.0, depth=1.0)
        else:
            bmesh.ops.create_cube(bm, size=1.0)
        bm.to_mesh(mesh)
        bm.free()

        obj = bpy.data.objects.new(pname, mesh)
        obj.location = ploc
        obj.dimensions = pdim
        obj.rotation_euler = [r * 3.14159265 / 180 if abs(r) > 6.3 else r for r in prot]
        obj.parent = parent

        _pn_lower = str(pname_short).lower()
        _name_is_glass = ("glass" in _pn_lower
                          or "window_pane" in _pn_lower
                          or "transparent" in _pn_lower)

        if has_part_mats and pname_short in SMALL_PART_MATERIALS[name]:
            mp = SMALL_PART_MATERIALS[name][pname_short]
            mp_type = str(mp.get("type", "")).lower()
            is_glass = (mp_type in ("glass", "acrylic")) or _name_is_glass
            is_mirror = (mp_type == "mirror")
            if is_glass:
                part_mat = create_pbr_material(
                    pname + "_mat",
                    mp["base_color"],
                    roughness=max(0.0, min(0.15, mp.get("roughness", 0.05))),
                    metallic=0.0,
                    specular=0.5,
                    alpha=0.25,
                    transmission=0.95,
                    ior=1.45,
                    material_type="glass",
                )
            elif is_mirror:
                part_mat = create_pbr_material(
                    pname + "_mat",
                    mp.get("base_color", (0.95, 0.95, 0.95, 1.0)),
                    roughness=max(0.0, min(0.05, mp.get("roughness", 0.0))),
                    metallic=1.0,
                    specular=0.5,
                    material_type="mirror",
                )
            else:
                part_mat = create_pbr_material(
                    pname + "_mat",
                    mp["base_color"],
                    roughness=mp.get("roughness", 0.5),
                    metallic=mp.get("metallic", 0.0),
                    specular=mp.get("specular", 0.5),
                    material_type=mp_type,
                )
            obj.data.materials.append(part_mat)
        elif _name_is_glass:
            part_mat = create_pbr_material(
                pname + "_mat",
                (0.90, 0.95, 0.98, 1.0),
                roughness=0.05,
                metallic=0.0,
                specular=0.5,
                alpha=0.25,
                transmission=0.95,
                ior=1.45,
                material_type="glass",
            )
            obj.data.materials.append(part_mat)
        elif material:
            obj.data.materials.append(material)

        if collection:
            collection.objects.link(obj)
        else:
            bpy.context.scene.collection.objects.link(obj)

    return parent'''

    def _build_part_materials_dict(self) -> str:
        lines = [
            "# ==============================================================================",
            "# PER-PART MATERIAL DATA (Auto-generated by Stage Material)",
            "# ==============================================================================",
            "",
            "PART_MATERIALS = {",
        ]
        for obj_name, parts_dict in self.part_materials.items():
            lines.append(f'    "{obj_name}": {{')
            for part_name, mat_props in parts_dict.items():
                bc = mat_props.get("base_color", [0.5, 0.5, 0.5, 1.0])
                ro = mat_props.get("roughness", 0.5)
                me = mat_props.get("metallic", 0.0)
                sp = mat_props.get("specular", 0.5)
                mt = mat_props.get("material_type", "unknown")
                bc_str = f"({bc[0]:.3f}, {bc[1]:.3f}, {bc[2]:.3f}, {bc[3] if len(bc) > 3 else 1.0:.1f})"
                lines.append(
                    f'        "{part_name}": {{"base_color": {bc_str}, '
                    f'"roughness": {ro}, "metallic": {me}, "specular": {sp}, '
                    f'"type": "{mt}"}},'
                )
            lines.append("    },")
        lines.append("}")
        return "\n".join(lines)

    def _build_floor_wall_code(self) -> str:
        fm = self.floor_material
        wm = self.wall_material

        fbc = fm.get("base_color", [0.5, 0.35, 0.2, 1.0])
        fbc_str = f"({fbc[0]:.3f}, {fbc[1]:.3f}, {fbc[2]:.3f}, {fbc[3] if len(fbc) > 3 else 1.0:.1f})"
        fpc = fm.get("pattern_color2", [0.45, 0.3, 0.18, 1.0])
        fpc_str = f"({fpc[0]:.3f}, {fpc[1]:.3f}, {fpc[2]:.3f}, {fpc[3] if len(fpc) > 3 else 1.0:.1f})"

        wbc = wm.get("base_color", [0.95, 0.93, 0.9, 1.0])
        wbc_str = f"({wbc[0]:.3f}, {wbc[1]:.3f}, {wbc[2]:.3f}, {wbc[3] if len(wbc) > 3 else 1.0:.1f})"

        lines = [
            "# ==============================================================================",
            "# FLOOR & WALL MATERIAL DATA",
            "# ==============================================================================",
            "",
            "FLOOR_MAT_DATA = {",
            f'    "material_type": "{fm.get("material_type", "hardwood")}",',
            f'    "base_color": {fbc_str},',
            f'    "roughness": {fm.get("roughness", 0.45)},',
            f'    "metallic": {fm.get("metallic", 0.0)},',
            f'    "specular": {fm.get("specular", 0.5)},',
            f'    "pattern": "{fm.get("pattern", "plank")}",',
            f'    "pattern_scale": {fm.get("pattern_scale", 4.0)},',
            f'    "pattern_color2": {fpc_str},',
            f'    "bump_strength": {fm.get("bump_strength", 0.1)},',
            "}",
            "",
            "WALL_MAT_DATA = {",
            f'    "material_type": "{wm.get("material_type", "paint")}",',
            f'    "base_color": {wbc_str},',
            f'    "roughness": {wm.get("roughness", 0.9)},',
            f'    "metallic": {wm.get("metallic", 0.0)},',
            f'    "specular": {wm.get("specular", 0.3)},',
            f'    "finish": "{wm.get("finish", "matte")}",',
            f'    "bump_strength": {wm.get("bump_strength", 0.05)},',
            "}",
            "",
            "",
            "def create_floor_material():",
            '    """Create a detailed floor material with optional pattern."""',
            '    d = FLOOR_MAT_DATA',
            '    mat = bpy.data.materials.new(name="Floor_PBR")',
            '    mat.use_nodes = True',
            '    nodes = mat.node_tree.nodes',
            '    links = mat.node_tree.links',
            '    nodes.clear()',
            '    node_output = nodes.new(type="ShaderNodeOutputMaterial")',
            '    node_output.location = (600, 0)',
            '    bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")',
            '    bsdf.location = (300, 0)',
            '    links.new(bsdf.outputs["BSDF"], node_output.inputs["Surface"])',
            '    bsdf.inputs["Roughness"].default_value = d["roughness"]',
            '    bsdf.inputs["Metallic"].default_value = d["metallic"]',
            '    spec_in = bsdf.inputs.get("Specular IOR Level") or bsdf.inputs.get("Specular")',
            '    if spec_in:',
            '        spec_in.default_value = d["specular"]',
            '    pat = d.get("pattern", "solid")',
            '    if pat in ("plank", "checker", "herringbone", "hexagonal"):'
            '',
            '        tex_coord = nodes.new(type="ShaderNodeTexCoord")',
            '        tex_coord.location = (-600, 0)',
            '        mapping = nodes.new(type="ShaderNodeMapping")',
            '        mapping.location = (-400, 0)',
            '        links.new(tex_coord.outputs["Object"], mapping.inputs["Vector"])',
            '        scale_val = d.get("pattern_scale", 4.0)',
            '        mapping.inputs["Scale"].default_value = (scale_val, scale_val, scale_val)',
            '        noise = nodes.new(type="ShaderNodeTexNoise")',
            '        noise.location = (-200, -200)',
            '        noise.inputs["Scale"].default_value = scale_val * 2',
            '        noise.inputs["Detail"].default_value = 8.0',
            '        links.new(mapping.outputs["Vector"], noise.inputs["Vector"])',
            '        color_ramp = nodes.new(type="ShaderNodeValToRGB")',
            '        color_ramp.location = (0, -200)',
            '        color_ramp.color_ramp.elements[0].color = d["base_color"]',
            '        color_ramp.color_ramp.elements[1].color = d.get("pattern_color2", d["base_color"])',
            '        links.new(noise.outputs["Fac"], color_ramp.inputs["Fac"])',
            '        links.new(color_ramp.outputs["Color"], bsdf.inputs["Base Color"])',
            '        bump = nodes.new(type="ShaderNodeBump")',
            '        bump.location = (0, -400)',
            '        bump.inputs["Strength"].default_value = d.get("bump_strength", 0.1)',
            '        links.new(noise.outputs["Fac"], bump.inputs["Height"])',
            '        links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])',
            "    else:",
            '        bsdf.inputs["Base Color"].default_value = d["base_color"]',
            "    return mat",
            "",
            "",
            "def create_wall_material():",
            '    """Create a detailed wall material."""',
            '    d = WALL_MAT_DATA',
            '    mat = bpy.data.materials.new(name="Wall_PBR")',
            '    mat.use_nodes = True',
            '    nodes = mat.node_tree.nodes',
            '    links = mat.node_tree.links',
            '    nodes.clear()',
            '    node_output = nodes.new(type="ShaderNodeOutputMaterial")',
            '    node_output.location = (600, 0)',
            '    bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")',
            '    bsdf.location = (300, 0)',
            '    links.new(bsdf.outputs["BSDF"], node_output.inputs["Surface"])',
            '    bsdf.inputs["Base Color"].default_value = d["base_color"]',
            '    bsdf.inputs["Roughness"].default_value = d["roughness"]',
            '    bsdf.inputs["Metallic"].default_value = d["metallic"]',
            '    spec_in = bsdf.inputs.get("Specular IOR Level") or bsdf.inputs.get("Specular")',
            '    if spec_in:',
            '        spec_in.default_value = d["specular"]',
            '    if d.get("bump_strength", 0) > 0:',
            '        tex_coord = nodes.new(type="ShaderNodeTexCoord")',
            '        tex_coord.location = (-400, 0)',
            '        noise = nodes.new(type="ShaderNodeTexNoise")',
            '        noise.location = (-200, -200)',
            '        noise.inputs["Scale"].default_value = 50.0',
            '        noise.inputs["Detail"].default_value = 2.0',
            '        links.new(tex_coord.outputs["Object"], noise.inputs["Vector"])',
            '        bump = nodes.new(type="ShaderNodeBump")',
            '        bump.location = (0, -200)',
            '        bump.inputs["Strength"].default_value = d["bump_strength"]',
            '        links.new(noise.outputs["Fac"], bump.inputs["Height"])',
            '        links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])',
            "    return mat",
        ]
        return "\n".join(lines)

    def _build_create_detailed_object(self) -> str:
        return '''def create_detailed_object(name, location=None, rotation=None, material=None, collection=None):
    """Create an object with detailed geometry AND per-part PBR materials.

    If PART_MATERIALS contains entries for this object, each part gets its own
    material; otherwise the fallback *material* argument is applied uniformly.
    """
    if name not in DETAILED_GEOMETRY:
        return None

    data = DETAILED_GEOMETRY[name]
    center = location if location is not None else data["center"]
    base_rot = rotation if rotation is not None else data["rotation"]
    parts = data["parts"]

    parent = bpy.data.objects.new(name, None)
    parent.empty_display_type = "PLAIN_AXES"
    parent.empty_display_size = 0.1
    parent.location = center
    parent.rotation_euler = base_rot

    if collection:
        collection.objects.link(parent)
    else:
        bpy.context.scene.collection.objects.link(parent)

    has_part_mats = name in PART_MATERIALS

    for part in parts:
        ptype = part["type"]
        pname = f"{name}_{part['name']}"
        ploc = part["loc"]
        pdim = part["dim"]
        prot = part["rot"]

        mesh = bpy.data.meshes.new(pname + "_mesh")
        bm = bmesh.new()

        if ptype == "box":
            bmesh.ops.create_cube(bm, size=1.0)
        elif ptype == "cylinder":
            bmesh.ops.create_cone(bm, cap_ends=True, cap_tris=False, segments=32, radius1=0.5, radius2=0.5, depth=1.0)
        elif ptype == "sphere":
            bmesh.ops.create_uvsphere(bm, u_segments=32, v_segments=16, radius=0.5)
        elif ptype == "cone":
            bmesh.ops.create_cone(bm, cap_ends=True, cap_tris=False, segments=32, radius1=0.5, radius2=0.0, depth=1.0)
        else:
            bmesh.ops.create_cube(bm, size=1.0)

        bm.to_mesh(mesh)
        bm.free()

        obj = bpy.data.objects.new(pname, mesh)
        obj.location = ploc
        obj.dimensions = pdim
        obj.rotation_euler = [r * 3.14159265 / 180 if abs(r) > 6.3 else r for r in prot]
        obj.parent = parent

        # Glass fallback: if the LLM forgot to mark a clearly-glass part
        # with material_type="glass", but the part name contains the
        # token "glass", we still render it as glass. This robustly
        # covers glass-front cabinet doors, display panels, etc.
        _part_name_lower = str(part.get("name", "")).lower()
        _name_is_glass = ("glass" in _part_name_lower
                          or "window_pane" in _part_name_lower
                          or "transparent" in _part_name_lower)

        if has_part_mats and part["name"] in PART_MATERIALS[name]:
            mp = PART_MATERIALS[name][part["name"]]
            mp_type = str(mp.get("type", "")).lower()
            # Acrylic and glass share the same transmission handling.
            is_glass = (mp_type in ("glass", "acrylic")) or _name_is_glass
            is_mirror = (mp_type == "mirror")
            if is_glass:
                # Glass / acrylic: low roughness, high transmission, low
                # alpha, slight tint from the LLM-chosen base_color.
                part_mat = create_pbr_material(
                    pname + "_mat",
                    mp["base_color"],
                    roughness=max(0.0, min(0.15, mp.get("roughness", 0.05))),
                    metallic=0.0,
                    specular=0.5,
                    alpha=0.25,
                    transmission=0.95,
                    ior=1.45,
                    material_type="glass",
                )
            elif is_mirror:
                # Mirror: perfect reflector, no transmission.
                part_mat = create_pbr_material(
                    pname + "_mat",
                    mp.get("base_color", (0.95, 0.95, 0.95, 1.0)),
                    roughness=max(0.0, min(0.05, mp.get("roughness", 0.0))),
                    metallic=1.0,
                    specular=0.5,
                    material_type="mirror",
                )
            else:
                part_mat = create_pbr_material(
                    pname + "_mat",
                    mp["base_color"],
                    roughness=mp.get("roughness", 0.5),
                    metallic=mp.get("metallic", 0.0),
                    specular=mp.get("specular", 0.5),
                    material_type=mp_type,
                )
            obj.data.materials.append(part_mat)
        elif _name_is_glass:
            # No per-part PBR entry at all, but the name screams glass.
            # Build a neutral clear-glass material so it doesn't render
            # as opaque wood inherited from the object's fallback mat.
            part_mat = create_pbr_material(
                pname + "_mat",
                (0.90, 0.95, 0.98, 1.0),
                roughness=0.05,
                metallic=0.0,
                specular=0.5,
                alpha=0.25,
                transmission=0.95,
                ior=1.45,
                material_type="glass",
            )
            obj.data.materials.append(part_mat)
        elif material:
            obj.data.materials.append(material)

        if collection:
            collection.objects.link(obj)
        else:
            bpy.context.scene.collection.objects.link(obj)

    return parent'''

    def _replace_floor_wall_in_layout(self, code: str) -> str:
        """Replace floor/wall material creation calls in run_layout_engine."""
        lines = code.split("\n")
        for i, line in enumerate(lines):
            stripped = line.strip()
            if re.match(
                r'mat_\w*[Ff]loor\w*\s*=\s*create_(material|pbr_material)\s*\(',
                stripped,
            ):
                indent = len(line) - len(line.lstrip())
                varname = stripped.split("=")[0].strip()
                lines[i] = " " * indent + f"{varname} = create_floor_material()"
            elif re.match(
                r'mat_\w*[Ww]all\w*\s*=\s*create_(material|pbr_material)\s*\(',
                stripped,
            ):
                indent = len(line) - len(line.lstrip())
                varname = stripped.split("=")[0].strip()
                lines[i] = " " * indent + f"{varname} = create_wall_material()"
        return "\n".join(lines)

    def _add_material_setup_call(self, code: str) -> str:
        """Replace existing simple material defs with PBR versions in run_layout_engine."""
        lines = code.split("\n")
        mat_pattern = re.compile(
            r'^(\s+)(mat_\w+)\s*=\s*create_material\s*\(\s*["\']([^"\']+)["\']\s*,\s*\(([^)]+)\)\s*\)'
        )

        for i, line in enumerate(lines):
            m = mat_pattern.match(line)
            if m:
                indent = m.group(1)
                var = m.group(2)
                name = m.group(3)
                color_str = m.group(4)

                if "floor" in var.lower() or "wall" in var.lower():
                    continue

                try:
                    vals = [float(x.strip()) for x in color_str.split(",")]
                    if len(vals) == 3:
                        vals.append(1.0)
                    color_tuple = f"({vals[0]:.3f}, {vals[1]:.3f}, {vals[2]:.3f}, {vals[3]:.1f})"
                except Exception:
                    color_tuple = f"({color_str})"

                lines[i] = (
                    f'{indent}{var} = create_pbr_material("{name}", {color_tuple})'
                )

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Syntax verification
    # ------------------------------------------------------------------
    def _verify_and_fix_syntax(self, code: str) -> str:
        from blender_code_syntax_fix import fix_empty_if_show_direction_before_return
        code = fix_empty_if_show_direction_before_return(code)

        MAX_PASSES = 5
        for pass_num in range(MAX_PASSES):
            try:
                compile(code, "<material_output>", "exec")
                self._log(
                    f"Syntax OK ({code.count(chr(10)) + 1} lines)", "success"
                )
                return code
            except SyntaxError as e:
                self._log(
                    f"Syntax error pass {pass_num+1} (line {e.lineno}): {e.msg}",
                    "warning",
                )
                lines = code.split("\n")
                if not e.lineno or e.lineno > len(lines):
                    break
                err_idx = e.lineno - 1
                err_line = lines[err_idx]
                fixed = False

                if "unexpected indent" in str(e.msg):
                    prev = lines[err_idx - 1] if err_idx > 0 else ""
                    if prev.rstrip().endswith(")") and (
                        "create_detailed_object(" in prev
                        or "create_box(" in prev
                        or "create_cylinder(" in prev
                    ):
                        lines[err_idx] = ""
                        fixed = True

                rot_start = err_line.find("rotation=(")
                mat_pos = err_line.find(
                    "material=", rot_start + 1 if rot_start >= 0 else 0
                )
                if not fixed and rot_start >= 0 and mat_pos > rot_start:
                    between = err_line[rot_start + len("rotation=(") : mat_pos]
                    raw = between.rstrip().rstrip(",").rstrip()
                    depth = 0
                    components = []
                    cur = []
                    for ch in raw:
                        if ch == "(":
                            depth += 1
                        elif ch == ")":
                            depth -= 1
                        if ch == "," and depth == 0:
                            components.append("".join(cur).strip())
                            cur = []
                        else:
                            cur.append(ch)
                    if cur:
                        last = "".join(cur).strip()
                        if last:
                            components.append(last)
                    while len(components) < 3:
                        components.append("0")
                    fix = f'rotation=({", ".join(components[:3])}), material='
                    lines[err_idx] = (
                        err_line[:rot_start]
                        + fix
                        + err_line[mat_pos + len("material=") :]
                    )
                    fixed = True

                if not fixed:
                    self._log(
                        f"Cannot auto-fix line {e.lineno}: "
                        f"{err_line.strip()[:80]}",
                        "warning",
                    )
                    break
                code = "\n".join(lines)

        return code

    # ------------------------------------------------------------------
    # JSON extraction
    # ------------------------------------------------------------------
    def _extract_json(self, text: str) -> Optional[Dict]:
        try:
            return json.loads(text)
        except Exception:
            pass
        for pat in (
            r"```json\s*([\s\S]*?)\s*```",
            r"```\s*([\s\S]*?)\s*```",
            r"\{[\s\S]*\}",
        ):
            m = re.search(pat, text)
            if m:
                try:
                    s = m.group(1) if "```" in pat else m.group(0)
                    return json.loads(s)
                except Exception:
                    continue
        return None

    # ------------------------------------------------------------------
    # Defaults
    # ------------------------------------------------------------------
    @staticmethod
    def _default_floor_material() -> Dict:
        return {
            "material_type": "hardwood",
            "base_color": [0.36, 0.22, 0.12, 1.0],
            "roughness": 0.45,
            "metallic": 0.0,
            "specular": 0.5,
            "pattern": "plank",
            "pattern_scale": 4.0,
            "pattern_color2": [0.32, 0.19, 0.10, 1.0],
            "bump_strength": 0.1,
        }

    @staticmethod
    def _default_wall_material() -> Dict:
        return {
            "material_type": "paint",
            "base_color": [0.9, 0.88, 0.85, 1.0],
            "roughness": 0.9,
            "metallic": 0.0,
            "specular": 0.3,
            "finish": "matte",
            "bump_strength": 0.02,
        }

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    def _save_results(self, code: str) -> str:
        os.makedirs(self.output_dir, exist_ok=True)

        try:
            from stage_clean_arrows import ArrowCleaner
            cleaner = ArrowCleaner(output_dir=self.output_dir, verbose=self.verbose)
            code = cleaner.clean_code(code)
        except Exception as e:
            self._log(f"Arrow cleanup skipped: {e}", "warning")

        output_path = os.path.join(self.output_dir, "material_output.py")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(code)
        self._log(f"Code saved: {output_path}", "save")

        config = {
            "scene_type_info": self.scene_type_info,
            "scene_palette": self.scene_palette,
            "part_materials": self.part_materials,
            "small_part_materials": self.small_part_materials,
            "floor_material": self.floor_material,
            "wall_material": self.wall_material,
            "summary": {
                "total_objects": len(self.part_materials),
                "total_parts": sum(
                    len(v) for v in self.part_materials.values()
                ),
                "total_small_items": len(self.small_part_materials),
                "total_small_parts": sum(
                    len(v) for v in self.small_part_materials.values()
                ),
                "generated": datetime.now().isoformat(),
                "scene_type": (self.scene_type_info or {}).get("scene_type"),
                "industrial_subtype": (self.scene_type_info or {}).get("industrial_subtype"),
            },
        }
        config_path = os.path.join(self.output_dir, "material_config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        self._log(f"Material config: {config_path}", "save")

        if self.use_memory:
            small_summary = ""
            if self.small_part_materials:
                small_summary = (
                    f" + {len(self.small_part_materials)} small items "
                    f"({sum(len(v) for v in self.small_part_materials.values())} parts)"
                )
            self.memory.add(
                stage="stage10_material",
                type="result",
                content=code,
                metadata={
                    "title": "Stage Material - Per-Part PBR Materials",
                    "summary": (
                        f"{len(self.part_materials)} objects, "
                        f"{sum(len(v) for v in self.part_materials.values())} parts"
                        + small_summary
                    ),
                    "output_file": output_path,
                    "config_file": config_path,
                    "image_path": self.image_path,
                    "small_objects_integrated": bool(self.small_part_materials),
                    "scene_type_info": self.scene_type_info,
                },
                tags=[
                    "stage10_material",
                    "blender_code",
                    "pbr_materials",
                    "per_part",
                ],
            )
            self._log("Saved to Memory", "success")

        return code

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------
    def run(self) -> Tuple[bool, Optional[str]]:
        print("\n" + "=" * 60)
        print("🎨 Stage Material - Per-Part PBR Material Generation")
        print("=" * 60)

        if not self._load_data():
            return False, None

        # Parse geometry objects
        print("\n--- Part A: Parse Geometry ---")
        geo_objects = self._parse_geometry_objects()
        desc_map = self._build_describe_map()
        self._log(
            f"Found {len(geo_objects)} objects with "
            f"{sum(len(v) for v in geo_objects.values())} total parts",
            "info",
        )

        # Pass 1: scene palette pre-pass (cross-object anchor for consistency)
        print("\n--- Part B0: Scene Palette ---")
        self.scene_palette = self._generate_scene_palette(geo_objects, desc_map)

        # Pass 2: per-part materials, batched + parallel, anchored on palette
        print("\n--- Part B: Per-Part Materials ---")
        self.part_materials = self._generate_part_materials(geo_objects, desc_map)

        # Pass 2 (small objects): only when DETAILED_GEOMETRY_SMALL is
        # present in the loaded geometry code (Stage 8/9 ran). When
        # 9 was skipped, this block is a no-op and small items keep the
        # caller-supplied flat material from Stage 7.
        self.small_geo_objects = self._parse_small_geometry_objects()
        if self.small_geo_objects:
            print("\n--- Part B': Small-Object Per-Part Materials ---")
            self._log(
                f"Found {len(self.small_geo_objects)} small items with "
                f"{sum(len(v) for v in self.small_geo_objects.values())} "
                f"total parts (Stage 9 detailed)",
                "info",
            )
            self.small_part_materials = self._generate_small_part_materials(
                self.small_geo_objects
            )

        # Generate floor & wall materials
        print("\n--- Part C: Floor & Wall Materials ---")
        self.floor_material, self.wall_material = (
            self._generate_floor_wall_materials()
        )

        # Inject into code
        print("\n--- Part D: Code Generation ---")
        code = self._inject_materials()
        code = self._verify_and_fix_syntax(code)

        # Save
        print("\n--- Part E: Save ---")
        code = self._save_results(code)

        print("\n" + "=" * 60)
        print("✅ Stage Material Complete!")
        print(f"   Objects with per-part materials: {len(self.part_materials)}")
        print(
            f"   Total material-assigned parts: "
            f"{sum(len(v) for v in self.part_materials.values())}"
        )
        if self.small_part_materials:
            print(
                f"   Small items with per-part materials: "
                f"{len(self.small_part_materials)} "
                f"({sum(len(v) for v in self.small_part_materials.values())} parts)"
            )
        print(f"   Output: {self.output_dir}")
        print("=" * 60)

        return True, code


# ==============================================================================
# CLI
# ==============================================================================
def show_memory_status():
    memory = Memory(workspace_dir=current_dir)
    print("=" * 60)
    print("📋 Memory Status")
    print("=" * 60)
    for stage in [
        "stage1", "stage2", "stage3", "stage4",
        "stage5_describe", "stage6_geometry", "stage7_small_objects",
        "stage10_material", "stage12_render",
    ]:
        entry = memory.get_latest(stage=stage, type="result")
        if entry:
            title = entry.metadata.get("title", "untitled")
            from datetime import datetime as dt
            ts = dt.fromtimestamp(entry.timestamp).strftime("%m-%d %H:%M")
            print(f"✅ {stage}: {title} ({ts})")
        else:
            print(f"❌ {stage}: No data")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Stage Material - Per-Part PBR Material Generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python stage10_material.py --image /path/to/image.png
  python stage10_material.py --image img.png --geometry-code geometry_output.py
  python stage10_material.py --status
""",
    )
    parser.add_argument("--image", "-i", help="Reference image path")
    parser.add_argument(
        "--geometry-code", "-g", help="Geometry code path (default: from Memory)"
    )
    parser.add_argument("--output-dir", "-o", default=None, help="Output directory")
    parser.add_argument(
        "--no-memory", action="store_true", help="Disable Memory system"
    )
    parser.add_argument(
        "--status", "-s", action="store_true", help="Show Memory status"
    )
    parser.add_argument(
        "--batch-size", type=int, default=6,
        help="Stage 10 Pass 2: objects per LLM call (default 6)",
    )
    parser.add_argument(
        "--parallel", type=int, default=4,
        help="Stage 10 Pass 2: parallel batch workers (default 4)",
    )
    parser.add_argument(
        "--max-attempts", type=int, default=3,
        help="Stage 10: max LLM retry attempts per batch (default 3)",
    )

    args = parser.parse_args()

    if args.status:
        show_memory_status()
        return 0

    runner = StageMaterialRunner(
        image_path=args.image,
        geometry_code_path=args.geometry_code,
        output_dir=args.output_dir,
        use_memory=not args.no_memory,
        batch_size=args.batch_size,
        parallel=args.parallel,
        max_attempts=args.max_attempts,
    )
    success, code = runner.run()
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
