"""
Stage Geometry - Detailed Geometry Generation Stage (Incremental)
=================================================================

Features:
1. INCREMENTAL: Generate one object at a time, save immediately to JSON
2. CENTER ALIGNMENT: Object center matches original bounding box center
3. FRONT FACING: Object front faces -Y direction
4. NO LOSS: Process all objects, with progress tracking

Usage:
    cd /Users/yangyixuan/SceneGen_Agent/agent_utils
    
    # Generate all objects (incremental, saves after each)
    python stage6_geometry.py --describe-json /path/to/describe_output.json
    
    # Generate specific objects by index (0-based)
    python stage6_geometry.py --indices 0 1 2
    
    # Generate specific objects by name
    python stage6_geometry.py --names "King_Bed" "Armchair_North"
    
    # Resume from last progress (skip already generated)
    python stage6_geometry.py --resume
    
    # Show progress
    python stage6_geometry.py --progress

Author: Auto-generated
"""

import os
import sys
import re
import json
import base64
import argparse
import threading
import time
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict, field
from datetime import datetime

# Path setup
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.join(current_dir, "stage3"))

from stage3.core import LLMClient, extract_python_from_response
from memory import Memory
from langchain_core.messages import HumanMessage, SystemMessage


# ==============================================================================
# Stage 4 wall-mounted policy
# ==============================================================================
# Stage 4 marks every wall-mounted addition (paintings, mirrors, clocks,
# curtain rods, etc.) in `wall_objects.json`. By default Stage 8 keeps those
# as a single Stage-4 bbox because they are essentially flat 2D decorations
# that gain nothing from detailed composite geometry.
#
# However, some wall-mounted items ARE structurally 3D (e.g. drying racks,
# wall shelves with brackets, pegboards with hooks, coat racks). For those we
# WANT Stage 8 to run normally and emit `create_detailed_object` parts.
#
# A wall object whose lower-cased name contains any of the substrings below
# is treated as "structured" and removed from the Stage 8 skip set.
_STRUCTURED_WALL_KEYWORDS: Tuple[str, ...] = (
    "rack",        # drying_rack, coat_rack, bike_rack, tool_rack, wine_rack ...
    "shelf",       # wall_shelf, floating_shelf, corner_shelf ...
    "pegboard",
    "hook",        # coat_hook strip / hook rail
    "organizer",   # wall_organizer
    "cabinet",     # wall_cabinet (has interior volume)
    "cubby",
)


def _is_structured_wall_name(name: str) -> bool:
    """Return True when the wall-mounted object has real 3D structure and
    should therefore still receive detailed geometry from Stage 8."""
    if not name:
        return False
    lowered = name.lower()
    return any(kw in lowered for kw in _STRUCTURED_WALL_KEYWORDS)


# ==============================================================================
# Data Structures
# ==============================================================================
@dataclass
class PrimitiveShape:
    """A single primitive shape (box, cylinder, sphere, cone)"""
    shape_type: str              # "box", "cylinder", "sphere", "cone"
    name: str                    # Part name (e.g., "seat", "backrest", "leg_1")
    relative_location: Tuple[float, float, float]  # Relative to object center
    dimensions: Tuple[float, float, float]         # Width, depth, height
    rotation: Tuple[float, float, float] = (0, 0, 0)  # Rotation in radians
    
    def to_dict(self) -> dict:
        return {
            "shape_type": self.shape_type,
            "name": self.name,
            "relative_location": list(self.relative_location),
            "dimensions": list(self.dimensions),
            "rotation": list(self.rotation)
        }


@dataclass
class DetailedObject:
    """An object composed of multiple primitives"""
    name: str                    # Object name
    object_type: str             # Object type (armchair, bed, lamp, etc.)
    center_location: Tuple[float, float, float]  # Center location (matches original bbox center)
    base_rotation: Tuple[float, float, float]    # Base rotation from original
    bounding_dimensions: Tuple[float, float, float]  # Original bounding box dimensions
    parts: List[PrimitiveShape] = field(default_factory=list)  # Component parts
    material_description: str = ""
    color_description: str = ""
    description: str = ""
    generated: bool = False      # Whether geometry has been generated
    source_stage: str = ""
    source_object_id: str = ""
    parent_code_match_name: str = ""
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "object_type": self.object_type,
            "center_location": list(self.center_location),
            "base_rotation": list(self.base_rotation),
            "bounding_dimensions": list(self.bounding_dimensions),
            "parts": [p.to_dict() for p in self.parts],
            "material_description": self.material_description,
            "color_description": self.color_description,
            "description": self.description,
            "generated": self.generated,
            "source_stage": self.source_stage,
            "source_object_id": self.source_object_id,
            "parent_code_match_name": self.parent_code_match_name
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> 'DetailedObject':
        parts = [PrimitiveShape(
            shape_type=p["shape_type"],
            name=p["name"],
            relative_location=tuple(p["relative_location"]),
            dimensions=tuple(p["dimensions"]),
            rotation=tuple(p.get("rotation", [0, 0, 0]))
        ) for p in d.get("parts", [])]
        
        return cls(
            name=d["name"],
            object_type=d.get("object_type", ""),
            center_location=tuple(d.get("center_location", [0, 0, 0])),
            base_rotation=tuple(d.get("base_rotation", [0, 0, 0])),
            bounding_dimensions=tuple(d.get("bounding_dimensions", [1, 1, 1])),
            parts=parts,
            material_description=d.get("material_description", ""),
            color_description=d.get("color_description", ""),
            description=d.get("description", ""),
            generated=d.get("generated", False),
            source_stage=d.get("source_stage", ""),
            source_object_id=d.get("source_object_id", ""),
            parent_code_match_name=d.get("parent_code_match_name", "")
        )


# ==============================================================================
# Stage Geometry Runner (Incremental)
# ==============================================================================
class StageGeometryRunner:
    """Stage Geometry Runner - Incremental detailed geometry generation"""
    
    def __init__(
        self,
        describe_json_path: str = None,
        output_dir: str = None,
        use_memory: bool = True,
        verbose: bool = True,
        memory_file: str = "agent_memory.jsonl",
        geometry_max_attempts: int = 3,
        geometry_retry_delay_sec: float = 2.0,
        model: str = None,
        base_url: str = None,
        api_key: str = None,
    ):
        self.describe_json_path = describe_json_path
        self.output_dir = output_dir or os.path.join(current_dir, "pipeline_output", "stage6_geometry")
        self.use_memory = use_memory
        self.verbose = verbose
        self.geometry_max_attempts = max(1, int(geometry_max_attempts))
        self.geometry_retry_delay_sec = max(0.0, float(geometry_retry_delay_sec))

        # Initialize
        self.memory = Memory(workspace_dir=current_dir, memory_file=memory_file) if use_memory else None
        self.llm = LLMClient(model=model, base_url=base_url, api_key=api_key)
        self.scene_type_info = self._load_scene_type_info()
        
        # Data
        self.describe_data: Dict = {}
        self.source_objects: List[Dict] = []  # Original objects from describe
        self.detailed_objects: List[DetailedObject] = []  # Generated objects
        self.room_style: Dict = {}
        self.orphan_placement_report: Dict[str, List[Dict[str, Any]]] = {
            "added": [],
            "skipped": [],
        }
        
        # Progress file
        self.progress_file = os.path.join(self.output_dir, "geometry_progress.json")
        self._save_lock = threading.Lock()
    
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
                print(f"Stage6: cannot read scene_type ({exc}); using generic geometry prompts")
            return fallback

    def _is_industrial_scene(self) -> bool:
        return (
            (self.scene_type_info or {}).get("scene_type") == "industrial"
            and float((self.scene_type_info or {}).get("confidence", 0.0) or 0.0) >= 0.5
        )

    def _log(self, msg: str, level: str = "info"):
        if self.verbose:
            prefix = {
                "info": "ℹ️", 
                "success": "✅", 
                "warning": "⚠️", 
                "error": "❌", 
                "step": "📋",
                "object": "🪑",
                "geometry": "📐",
                "save": "💾",
                "progress": "📊"
            }.get(level, "")
            print(f"{prefix} {msg}")
    
    def _load_describe_data(self) -> bool:
        """Load describe data"""
        self._log("Loading describe data...", "step")
        
        # 1. Load describe JSON
        if self.describe_json_path and os.path.exists(self.describe_json_path):
            with open(self.describe_json_path, "r", encoding="utf-8") as f:
                self.describe_data = json.load(f)
            self._log(f"Describe JSON: {self.describe_json_path}", "success")
        elif self.use_memory:
            entry = self.memory.get_latest(stage="stage5_describe", type="result")
            if entry:
                self.describe_data = json.loads(entry.content)
                self._log("Describe data: from Memory", "success")
        
        if not self.describe_data:
            self._log("Object description data not found!", "error")
            self._log("   Please make sure Stage 7 (stage5_describe) has been run, or use --describe-json to specify", "error")
            return False
        
        self.source_objects = self.describe_data.get("objects", [])
        self.room_style = self.describe_data.get("room_style", {})
        
        self._log(f"Found {len(self.source_objects)} source objects", "success")

        # Filter out Stage 4 wall-mounted additions. Flat 2D-style decorations
        # (paintings, mirrors, clocks, curtain rods, ...) must stay as a
        # single Stage-4 bbox -- detailed composite geometry would gain
        # nothing. Their original create_box calls in the Stage 4 code are
        # left untouched during code integration (only objects present in
        # DETAILED_GEOMETRY get their bbox call replaced by
        # create_detailed_object).
        #
        # Structurally 3D wall items (drying racks, wall shelves, pegboards,
        # coat racks, wall cabinets, ...) are kept in `source_objects` so
        # Stage 8 still produces detailed parts for them.
        wall_names = self._load_wall_object_names()
        if wall_names:
            flat_wall_names = {n for n in wall_names if not _is_structured_wall_name(n)}
            structured_wall_names = wall_names - flat_wall_names

            if structured_wall_names:
                self._log(
                    f"Keeping {len(structured_wall_names)} structured wall-mounted "
                    f"objects in Stage 8: {sorted(structured_wall_names)[:6]}"
                    + ("..." if len(structured_wall_names) > 6 else ""),
                    "info",
                )

            if flat_wall_names:
                before = len(self.source_objects)
                self.source_objects = [
                    obj for obj in self.source_objects
                    if obj.get("name") not in flat_wall_names
                ]
                skipped = before - len(self.source_objects)
                if skipped:
                    self._log(
                        f"Skipping {skipped} flat wall-mounted objects from "
                        f"Stage 4 (kept as bbox): "
                        f"{sorted(flat_wall_names)[:6]}"
                        + ("..." if len(flat_wall_names) > 6 else ""),
                        "info",
                    )

        self._augment_source_objects_with_orphans()
        return True

    @staticmethod
    def _normalize_match_text(text: str) -> str:
        text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", text or "")
        text = text.lower()
        text = re.sub(r"\bobj[_\s-]*\d+\b", " ", text)
        text = re.sub(r"[^a-z0-9]+", " ", text)
        stop = {
            "the", "a", "an", "and", "of", "on", "in", "near", "with",
            "section", "area", "zone", "object", "item", "small",
            "lab", "laboratory", "bench", "worktop", "top",
        }
        tokens = [t for t in text.split() if t and t not in stop]
        return " ".join(tokens)

    @staticmethod
    def _safe_identifier(text: str) -> str:
        safe = re.sub(r"[^A-Za-z0-9]+", "_", text or "object").strip("_")
        return safe or "object"

    def _find_source_object_by_name(self, candidate_name: str) -> Optional[Dict[str, Any]]:
        if not candidate_name:
            return None
        cand_norm = self._normalize_match_text(candidate_name)
        cand_tokens = set(cand_norm.split())
        if not cand_tokens:
            return None

        best_obj = None
        best_score = 0.0
        for obj in self.source_objects:
            if obj.get("source_stage") == "stage7_orphan":
                continue
            code_norm = self._normalize_match_text(obj.get("name", ""))
            code_tokens = set(code_norm.split())
            if not code_tokens:
                continue
            if cand_norm == code_norm:
                return obj
            overlap = len(cand_tokens & code_tokens)
            score = overlap / max(len(cand_tokens), len(code_tokens))
            if cand_tokens.issubset(code_tokens) or code_tokens.issubset(cand_tokens):
                score = max(score, 0.75)
            if score > best_score:
                best_score = score
                best_obj = obj

        return best_obj if best_score >= 0.45 else None

    def _estimate_orphan_dimensions(self, orphan: Dict[str, Any]) -> Tuple[float, float, float]:
        text = " ".join([
            orphan.get("name", ""),
            orphan.get("object_type", ""),
            orphan.get("description", ""),
        ]).lower()

        checks = [
            (("well plate", "microplate", "plate"), (0.16, 0.11, 0.02)),
            (("tube rack", "test tube rack"), (0.30, 0.12, 0.08)),
            (("pipette rack", "pipette"), (0.24, 0.08, 0.16)),
            (("notebook", "paper", "document"), (0.26, 0.18, 0.025)),
            (("keyboard",), (0.36, 0.12, 0.035)),
            (("monitor", "screen"), (0.40, 0.08, 0.30)),
            (("mouse",), (0.08, 0.05, 0.03)),
            (("centrifuge",), (0.36, 0.30, 0.25)),
            (("vortex", "mixer", "stirrer", "hot plate", "balance"), (0.30, 0.24, 0.14)),
            (("bottle", "flask", "beaker", "glassware", "container"), (0.12, 0.12, 0.20)),
            (("box", "storage"), (0.30, 0.22, 0.16)),
            (("rack",), (0.28, 0.12, 0.12)),
        ]
        for keywords, dims in checks:
            if any(keyword in text for keyword in keywords):
                return dims
        return (0.22, 0.16, 0.10)

    def _orphan_world_location(
        self,
        parent: Dict[str, Any],
        dims: Tuple[float, float, float],
        slot_index: int,
    ) -> Tuple[float, float, float]:
        parent_loc = parent.get("location", [0, 0, 0])
        parent_dim = parent.get("dimensions", [1, 1, 1])
        parent_rot = parent.get("rotation", [0, 0, 0])

        pw = max(float(parent_dim[0]), dims[0] + 0.05)
        pd = max(float(parent_dim[1]), dims[1] + 0.05)
        ph = float(parent_dim[2])
        theta = float(parent_rot[2]) if len(parent_rot) >= 3 else 0.0

        # Fill the support surface in a stable left-to-right grid.
        cols = max(1, min(5, int(pw / max(dims[0] + 0.08, 0.18))))
        row = slot_index // cols
        col = slot_index % cols
        usable_w = max(pw - dims[0] - 0.10, 0.05)
        usable_d = max(pd - dims[1] - 0.10, 0.05)
        local_x = -usable_w / 2 + (col + 0.5) * usable_w / cols
        local_y = -usable_d / 2 + (min(row, 3) + 0.5) * usable_d / 4

        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        world_x = float(parent_loc[0]) + cos_t * local_x - sin_t * local_y
        world_y = float(parent_loc[1]) + sin_t * local_x + cos_t * local_y
        world_z = float(parent_loc[2]) + ph / 2 + dims[2] / 2
        return (world_x, world_y, world_z)

    def _augment_source_objects_with_orphans(self):
        """Convert locatable Stage 7 orphan inventory entries into Stage 8 sources.

        Only surface objects with a matched parent bbox are safe to place here.
        Floor/wall orphans stay in the report unless a future stage provides
        explicit coordinates for them.
        """
        orphans = self.describe_data.get("orphan_objects", []) or []
        if not orphans:
            return

        existing_names = {obj.get("name") for obj in self.source_objects}
        parent_slots: Dict[str, int] = {}
        synthetic_objects: List[Dict[str, Any]] = []

        for orphan in orphans:
            if orphan.get("represented_in_code"):
                continue
            if orphan.get("placement_type") != "surface":
                self.orphan_placement_report["skipped"].append({
                    "name": orphan.get("name"),
                    "reason": "only surface orphans can be placed without explicit coordinates",
                })
                continue

            parent_match = orphan.get("parent_code_match_name")
            parent = self._find_source_object_by_name(parent_match)
            if not parent:
                parent = self._find_source_object_by_name(orphan.get("parent_name"))
            if not parent:
                self.orphan_placement_report["skipped"].append({
                    "name": orphan.get("name"),
                    "reason": "no matching parent bbox in Stage 7 objects",
                    "parent_name": orphan.get("parent_name"),
                })
                continue

            parent_name = parent.get("name", "")
            slot = parent_slots.get(parent_name, 0)
            parent_slots[parent_name] = slot + 1
            dims = self._estimate_orphan_dimensions(orphan)
            loc = self._orphan_world_location(parent, dims, slot)
            rotation = parent.get("rotation", [0, 0, 0])

            base_name = self._safe_identifier(orphan.get("name", "orphan"))
            source_id = self._safe_identifier(orphan.get("source_object_id", "unknown"))
            name = f"Orphan_{source_id}_{base_name}"
            suffix = 2
            while name in existing_names:
                name = f"Orphan_{source_id}_{base_name}_{suffix}"
                suffix += 1
            existing_names.add(name)

            synthetic = {
                "name": name,
                "shape": "box",
                "location": list(loc),
                "dimensions": list(dims),
                "rotation": rotation,
                "rotation_degrees": [0, 0, 0],
                "material_name": "",
                "collection": parent.get("collection", ""),
                "description": orphan.get("description", ""),
                "object_type": orphan.get("object_type") or orphan.get("name") or "surface object",
                "appearance": orphan.get("appearance", ""),
                "material_description": orphan.get("material_description", ""),
                "color_description": orphan.get("color_description", ""),
                "source_stage": "stage7_orphan",
                "source_object_id": orphan.get("source_object_id", ""),
                "original_name": orphan.get("name", ""),
                "parent_name": orphan.get("parent_name", ""),
                "parent_code_match_name": parent_name,
                "orphan_reason": orphan.get("orphan_reason", ""),
            }
            synthetic_objects.append(synthetic)
            self.orphan_placement_report["added"].append({
                "name": name,
                "original_name": orphan.get("name"),
                "parent_code_match_name": parent_name,
                "location": list(loc),
                "dimensions": list(dims),
            })

        if synthetic_objects:
            self.source_objects.extend(synthetic_objects)
            self._log(
                f"Added {len(synthetic_objects)} Stage 7 orphan surface objects "
                "to Stage 8 geometry sources",
                "success",
            )
        if self.orphan_placement_report["skipped"]:
            self._log(
                f"Skipped {len(self.orphan_placement_report['skipped'])} "
                "orphan objects without safe placement",
                "warning",
            )

    def _load_wall_object_names(self) -> set:
        """Load the set of object names that Stage 4 added as wall-mounted
        decorations. Returns an empty set when no such list is available —
        in that case Stage 8 runs its original behaviour and processes every
        describe object.

        Priority:
            1. Memory `stage4` metadata -> `wall_object_names`
            2. Sibling file `{run_dir}/stage4/wall_objects.json`
            3. (no fallback — do NOT guess from Stage 1 `placement_type`)
        """
        # 1) Memory
        if self.use_memory:
            try:
                entry = self.memory.get_latest(stage="stage4", type="result")
                if entry:
                    names = entry.metadata.get("wall_object_names")
                    if names:
                        self._log(
                            "Wall-object list: from Memory (stage4)",
                            "success",
                        )
                        return set(names)
                    # Fallback: the metadata may point to a sibling json.
                    meta_path = entry.metadata.get("wall_objects_json")
                    if meta_path and os.path.exists(meta_path):
                        try:
                            with open(meta_path, "r", encoding="utf-8") as f:
                                data = json.load(f)
                            names = data.get("wall_object_names", [])
                            if names:
                                self._log(
                                    f"Wall-object list: from Memory metadata "
                                    f"path ({meta_path})",
                                    "success",
                                )
                                return set(names)
                        except Exception:
                            pass
            except Exception:
                pass

        # 2) Sibling file under the same run dir
        run_dir = os.path.dirname(self.output_dir) if self.output_dir else None
        if run_dir:
            candidate = os.path.join(run_dir, "stage4", "wall_objects.json")
            if os.path.exists(candidate):
                try:
                    with open(candidate, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    names = data.get("wall_object_names", [])
                    if names:
                        self._log(
                            f"Wall-object list: from sibling ({candidate})",
                            "success",
                        )
                        return set(names)
                except Exception as e:
                    self._log(
                        f"Failed to read {candidate}: {e}", "warning"
                    )

        return set()
    
    def _load_progress(self) -> bool:
        """Load existing progress"""
        os.makedirs(self.output_dir, exist_ok=True)
        
        if os.path.exists(self.progress_file):
            try:
                with open(self.progress_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                self.detailed_objects = [
                    DetailedObject.from_dict(d) 
                    for d in data.get("detailed_objects", [])
                ]
                self._log(f"Loaded progress: {len(self.detailed_objects)} objects", "progress")
                return True
            except Exception as e:
                self._log(f"Failed to load progress: {e}", "warning")
        
        return False
    
    def _save_progress(self):
        """Save current progress to JSON (thread-safe)"""
        with self._save_lock:
            os.makedirs(self.output_dir, exist_ok=True)
            
            data = {
                "room_style": self.room_style,
                "detailed_objects": [obj.to_dict() for obj in self.detailed_objects],
                "orphan_placement_report": self.orphan_placement_report,
                "summary": {
                    "total_source_objects": len(self.source_objects),
                    "generated_count": sum(1 for obj in self.detailed_objects if obj.generated),
                    "total_parts": sum(len(obj.parts) for obj in self.detailed_objects if obj.generated),
                    "orphan_objects_added": len(self.orphan_placement_report.get("added", [])),
                    "orphan_objects_skipped": len(self.orphan_placement_report.get("skipped", [])),
                    "last_updated": datetime.now().isoformat()
                }
            }
            
            with open(self.progress_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
    
    def _init_detailed_objects(self):
        """Initialize detailed objects from source objects (preserving center point).
        
        Only keeps objects that exist in the current source (describe_output).
        Stale objects from previous runs are discarded.
        """
        source_names = {src.get("name", "Unknown") for src in self.source_objects}
        existing_map = {obj.name: obj for obj in self.detailed_objects}
        
        stale_count = sum(1 for n in existing_map if n not in source_names)
        if stale_count > 0:
            self._log(f"Removing {stale_count} stale objects from previous runs", "warning")
        
        new_objects: List[DetailedObject] = []
        reused = 0
        created = 0
        
        for src in self.source_objects:
            name = src.get("name", "Unknown")
            location = src.get("location", [0, 0, 0])
            dimensions = src.get("dimensions", [1, 1, 1])
            rotation = src.get("rotation", [0, 0, 0])
            
            if name in existing_map and existing_map[name].generated:
                existing = existing_map[name]
                # Keep generated parts, but refresh the bbox transform from the
                # latest Stage 7 parse. This is critical when Stage 3/4 used
                # positional rotation args that an older parser missed: code-only
                # regeneration must realign generated details and orphan props
                # without re-calling the geometry LLM.
                existing.center_location = tuple(location)
                existing.base_rotation = tuple(rotation)
                existing.bounding_dimensions = tuple(dimensions)
                existing.material_description = src.get("material_description", existing.material_description)
                existing.color_description = src.get("color_description", existing.color_description)
                existing.description = src.get("description", existing.description)
                existing.source_stage = src.get("source_stage", existing.source_stage)
                existing.source_object_id = src.get("source_object_id", existing.source_object_id)
                existing.parent_code_match_name = src.get("parent_code_match_name", existing.parent_code_match_name)
                new_objects.append(existing)
                reused += 1
                continue

            center_location = tuple(location)
            
            obj = DetailedObject(
                name=name,
                object_type=src.get("object_type", "object"),
                center_location=center_location,
                base_rotation=tuple(rotation),
                bounding_dimensions=tuple(dimensions),
                material_description=src.get("material_description", ""),
                color_description=src.get("color_description", ""),
                description=src.get("description", ""),
                generated=False,
                source_stage=src.get("source_stage", ""),
                source_object_id=src.get("source_object_id", ""),
                parent_code_match_name=src.get("parent_code_match_name", "")
            )
            new_objects.append(obj)
            created += 1
        
        self.detailed_objects = new_objects
        self._apply_deterministic_geometry_fixes()
        self._log(f"Initialized {len(self.detailed_objects)} objects ({reused} reused, {created} new)", "info")

    def _apply_deterministic_geometry_fixes(self):
        """Apply exact geometry for objects where LLM axis guesses are risky."""
        fixed = 0
        for obj in self.detailed_objects:
            if self._fix_wall_pegboard_geometry(obj):
                fixed += 1
        if fixed:
            self._log(f"Applied deterministic geometry fixes to {fixed} objects", "info")

    def _fix_wall_pegboard_geometry(self, obj: DetailedObject) -> bool:
        """Rebuild wall-mounted pegboard/drying rack parts in bbox axes.

        Stage 4 wall-mounted racks often have bbox dimensions like
        X=0.10, Y=0.80, Z=0.80 on the west/east wall. LLM geometry sometimes
        treats the thin X axis as the visible width and compresses the rack
        to a narrow strip. This deterministic version keeps the face on the
        two large bbox axes and uses the thin axis only for wall thickness /
        peg protrusion.
        """
        label = f"{obj.name} {obj.object_type}".lower()
        if "pegboard" not in label and "drying rack" not in label:
            return False

        w, d, h = obj.bounding_dimensions
        dims = [float(w), float(d), float(h)]
        thin_axis = min(range(3), key=lambda idx: dims[idx])
        if dims[thin_axis] > 0.25:
            return False

        # The current pipeline only needs the common wall-mounted cases:
        # thin X -> visible face is YZ; thin Y -> visible face is XZ.
        # Keep Z as height, because floor/wall scenes are Z-up.
        if thin_axis == 0:
            thickness = max(min(dims[0] * 0.22, 0.025), 0.012)
            face_w = max(dims[1] * 0.92, 0.10)
            face_h = max(dims[2] * 0.92, 0.10)
            peg_len = max(dims[0] * 0.75, 0.05)
            panel_x = -dims[0] * 0.30

            parts: List[PrimitiveShape] = [
                PrimitiveShape("box", "back_panel", (panel_x, 0, 0), (thickness, face_w, face_h)),
                PrimitiveShape("box", "top_frame_rail", (panel_x, 0, face_h / 2), (thickness * 1.5, face_w, 0.025)),
                PrimitiveShape("box", "bottom_frame_rail", (panel_x, 0, -face_h / 2), (thickness * 1.5, face_w, 0.025)),
                PrimitiveShape("box", "left_side_frame_rail", (panel_x, -face_w / 2, 0), (thickness * 1.5, 0.025, face_h)),
                PrimitiveShape("box", "right_side_frame_rail", (panel_x, face_w / 2, 0), (thickness * 1.5, 0.025, face_h)),
            ]
            y_positions = [-face_w * 0.30, 0.0, face_w * 0.30]
            z_positions = [face_h * 0.30, face_h * 0.10, -face_h * 0.10, -face_h * 0.30]
            for row_idx, z in enumerate(z_positions, start=1):
                for col_idx, y in enumerate(y_positions, start=1):
                    parts.append(PrimitiveShape(
                        "cylinder",
                        f"peg_r{row_idx}_c{col_idx}",
                        (panel_x + peg_len * 0.45, y, z),
                        (0.014, 0.014, peg_len),
                        (0, 1.5708, 0),
                    ))
        elif thin_axis == 1:
            thickness = max(min(dims[1] * 0.22, 0.025), 0.012)
            face_w = max(dims[0] * 0.92, 0.10)
            face_h = max(dims[2] * 0.92, 0.10)
            peg_len = max(dims[1] * 0.75, 0.05)
            panel_y = -dims[1] * 0.30

            parts = [
                PrimitiveShape("box", "back_panel", (0, panel_y, 0), (face_w, thickness, face_h)),
                PrimitiveShape("box", "top_frame_rail", (0, panel_y, face_h / 2), (face_w, thickness * 1.5, 0.025)),
                PrimitiveShape("box", "bottom_frame_rail", (0, panel_y, -face_h / 2), (face_w, thickness * 1.5, 0.025)),
                PrimitiveShape("box", "left_side_frame_rail", (-face_w / 2, panel_y, 0), (0.025, thickness * 1.5, face_h)),
                PrimitiveShape("box", "right_side_frame_rail", (face_w / 2, panel_y, 0), (0.025, thickness * 1.5, face_h)),
            ]
            x_positions = [-face_w * 0.30, 0.0, face_w * 0.30]
            z_positions = [face_h * 0.30, face_h * 0.10, -face_h * 0.10, -face_h * 0.30]
            for row_idx, z in enumerate(z_positions, start=1):
                for col_idx, x in enumerate(x_positions, start=1):
                    parts.append(PrimitiveShape(
                        "cylinder",
                        f"peg_r{row_idx}_c{col_idx}",
                        (x, panel_y + peg_len * 0.45, z),
                        (0.014, 0.014, peg_len),
                        (1.5708, 0, 0),
                    ))
        else:
            return False

        obj.parts = parts
        obj.generated = True
        return True
    
    def _lab_double_deck_geometry_hint(
        self, name: str, dims: Tuple[float, float, float]
    ) -> str:
        """Extra instructions when Stage 4 split a double-deck bench into parts.
        Without this, the LLM often turns a thin *_upper_shelf bbox into a
        freestanding rack with legs (violates Z extent) or shrinks *_worktop."""
        w, d, h = dims[0], dims[1], dims[2]
        if w < 1e-3 or d < 1e-3:
            return ""
        n = name.lower()
        if n.endswith("_upper_shelf") and h <= 0.12:
            return (
                "LAB DOUBLE-DECK — UPPER SHELF PLANK ONLY:\n"
                f"- This bbox IS the thin reagent shelf board (~{w:.2f} m × {d:.2f} m × {h:.2f} m).\n"
                "- Emit EXACTLY ONE box: dimensions ≈ bbox, relative_location [0, 0, 0].\n"
                "- NO vertical legs, NO posts, NO multi-tier shelving, NO cabinet interior.\n"
                "- Ignore generic shelf/cabinet guidelines below that ask for many parts.\n"
            )
        if n.endswith("_worktop") and h >= 0.5:
            return (
                "LAB DOUBLE-DECK — WORKTOP + CABINET VOLUME:\n"
                f"- This bbox is the merged cabinet + slab ({w:.2f} × {d:.2f} × {h:.2f} m).\n"
                "- Fill it with ONE main box (or a thin-shell cabinet) matching the full "
                "width/depth/height; do NOT replace the whole volume with only a 0.03–0.05 m slab.\n"
            )
        if "_post_" in n and max(w, d) <= 0.12 and h <= 0.65:
            return (
                "LAB DOUBLE-DECK — SHELF POST:\n"
                f"- Narrow vertical post (~{w:.2f} × {d:.2f} × {h:.2f} m). "
                "Use ONE box or ONE upright cylinder spanning the bbox; at most one tiny cap.\n"
                "- Stay inside the Z half-extent; no brackets sticking far outside.\n"
            )
        return ""

    def _looks_industrial_object(self, obj: DetailedObject) -> bool:
        s = " ".join([
            obj.name or "",
            obj.object_type or "",
            obj.description or "",
            obj.material_description or "",
        ]).lower()
        keywords = (
            "cnc", "machining center", "machining cell", "machine tool",
            "lathe", "milling machine", "grinder station", "press brake",
            "hydraulic press", "stamping press", "injection molding",
            "molding machine", "conveyor", "assembly line", "production line",
            "robot arm", "robot cell", "cobot", "safety fence", "safety cage",
            "guard fence", "guard rail", "pallet rack", "storage rack",
            "parts rack", "bin rack", "parts bin", "tool cabinet",
            "control cabinet", "electrical cabinet", "switchgear",
            "industrial workbench", "assembly bench", "inspection table",
            "packing table", "server rack", "compressor", "utility skid",
            "agv", "forklift",
        )
        return any(k in s for k in keywords)

    def _industrial_geometry_hint(self, obj: DetailedObject) -> str:
        if not (self._is_industrial_scene() or self._looks_industrial_object(obj)):
            return ""
        label = " ".join([
            obj.name or "",
            obj.object_type or "",
            obj.description or "",
        ]).lower()
        hints = [
            "INDUSTRIAL EQUIPMENT GEOMETRY RULES:",
            "- Preserve factory semantics. Do not turn machinery into generic residential cabinets, desks, wardrobes, or decor.",
            "- Use structural parts that read from top-down and 3/4 views: enclosures, frames, rails, belts, posts, panels, doors, control panels, feet, and safety guards.",
            "- Keep loose tools, workpieces, bins, clipboards, tablets, cables, and paperwork OUT of the main object geometry unless they are physically attached. Those are small objects handled later.",
            "- Use part names that imply material: glass_window, rubber_belt, steel_frame, safety_yellow_rail, red_estop_button, control_panel, acrylic_guard.",
        ]
        if any(k in label for k in ("cnc", "machining center", "machine center", "lathe", "milling", "mill", "grinder", "press", "molding machine")):
            hints.extend([
                "ENCLOSED / OPEN MACHINE:",
                "- Model a main machine_base or enclosure_body filling most of the bbox.",
                "- Add front/side access door panels as thin boxes on the front face (y ~= -depth/2).",
                "- Add a distinct control_panel box on the operator side; small attached buttons/lights are allowed.",
                "- If a viewing window is visible, name it glass_window and keep it thin.",
                "- Do not create tabletop clutter such as tools or parts as part of the machine body.",
            ])
        if "conveyor" in label or "assembly line" in label or "production line" in label:
            hints.extend([
                "CONVEYOR / ASSEMBLY LINE:",
                "- Long axis must be along X in canonical pose; base_rotation will orient it in world.",
                "- Emit a dark rubber_belt top slab near the bbox top, metal side_rails, support legs, and a frame below.",
                "- Optional rollers should be a small representative set, not dozens of tiny repeated parts.",
                "- Do not place workpieces or bins on the belt as part geometry.",
            ])
        if "robot" in label or "cobot" in label:
            hints.extend([
                "ROBOT / ROBOT CELL:",
                "- Use a pedestal/base, lower_arm, upper_arm, wrist, and end_effector if the bbox represents a robot arm.",
                "- If the bbox represents the whole cell, include safety_fence posts/rails around the perimeter and a simple robot pedestal inside.",
                "- Keep the arm raised within the bbox and avoid flat furniture-like shapes.",
            ])
        if "safety fence" in label or "guard fence" in label or "guard rail" in label or "safety cage" in label:
            hints.extend([
                "SAFETY FENCE / GUARD:",
                "- Model as thin vertical posts plus horizontal rails or mesh panels, not a solid wall.",
                "- Use repeated but limited posts (4-8 max) and thin rail boxes spanning between them.",
                "- Leave the interior visually open.",
            ])
        if "pallet rack" in label or "storage rack" in label or "parts rack" in label or "shelving rack" in label or "server rack" in label:
            hints.extend([
                "RACK / SERVER RACK:",
                "- Use open vertical uprights and horizontal shelves/beams; do not make a single solid cabinet.",
                "- Rack shelves should be usable horizontal planes for later small-object/bin placement.",
                "- Server racks may have a front mesh/glass door panel but should remain tall equipment racks.",
            ])
        if "control cabinet" in label or "electrical cabinet" in label or "control panel" in label or "switchgear" in label:
            hints.extend([
                "CONTROL / ELECTRICAL CABINET:",
                "- Tall metal cabinet body with front door seam, handle, indicator lights, and attached red emergency-stop button if visible.",
                "- Keep controls attached to the front face; do not generate loose consoles around it.",
            ])
        if "workbench" in label or "work table" in label or "assembly bench" in label or "inspection table" in label or "packing table" in label:
            hints.extend([
                "INDUSTRIAL WORKBENCH / INSPECTION TABLE:",
                "- Generate only the bench/table body: top slab, metal frame, legs, lower shelf or drawers if visible.",
                "- Do not generate tools, parts trays, jigs, monitors, or workpieces on top as part geometry.",
                "- The top slab must reach the bbox top so Stage7 can detect the support surface.",
            ])
        if "pallet" in label and "rack" not in label:
            hints.extend([
                "PALLET / MATERIAL PLATFORM:",
                "- Low slatted pallet: several parallel deck boards and a few cross runners, not one solid block.",
            ])
        return "\n".join(hints) + "\n"

    def _generate_geometry_for_object(self, obj: DetailedObject) -> bool:
        """Generate detailed geometry for a single object"""
        self._log(f"Generating: {obj.name} ({obj.object_type})", "geometry")
        
        dims = obj.bounding_dimensions
        
        # --- BACKUP of original prompt (before orientation fix) ---
        # The old prompt did NOT inform the LLM about base_rotation, causing
        # elongated objects (barbells, dumbbells, foam rollers, etc.) to appear
        # skewed when rendered — the LLM would guess an orientation from the
        # top-down image while the code later applies base_rotation on top.
        # Fix: tell the LLM to always model in canonical (axis-aligned) pose;
        # the pipeline applies base_rotation afterwards.
        # --- END BACKUP ---

        rot_deg = [round(r * 180 / 3.14159265, 1) for r in obj.base_rotation]
        has_rotation = any(abs(r) > 0.01 for r in obj.base_rotation)
        rotation_note = ""
        if has_rotation:
            rotation_note = f"""
IMPORTANT - OBJECT ORIENTATION:
This object has a base rotation of {rot_deg} degrees (X, Y, Z) that will be
applied AFTER your geometry is placed. You must model the object in its
CANONICAL (axis-aligned) pose as if it has NO rotation:
- For elongated objects (barbells, benches, rollers, rods, beams):
  Model the LONG axis along X (width direction). Do NOT tilt or rotate parts
  to match how it looks in a top-down image.
- For flat objects (mats, rugs, plates): Model them flat on XY plane.
- The pipeline will rotate your geometry by {rot_deg} degrees automatically.
- Therefore: set "rotation" to [0, 0, 0] for ALL parts unless the part
  itself is structurally angled (e.g., a backrest tilted relative to a seat).
"""

        industrial_hint = self._industrial_geometry_hint(obj)
        industrial_system_addendum = ""
        if industrial_hint:
            industrial_system_addendum = """
INDUSTRIAL / FACTORY EQUIPMENT ADDENDUM:
- When the object is a CNC, machine, conveyor, robot, safety fence, rack, control cabinet, pallet, workbench, inspection station, utility skid, or server rack, follow the industrial hint in the runtime input.
- Industrial equipment should be built from structural primitives: enclosures, frames, rails, panels, belts, posts, shelves, doors, vents, control panels, and attached buttons/lights.
- Avoid residential furniture semantics and avoid loose surface clutter as body geometry.
- Open racks, guard fences, and machine frames should remain visibly open; do not collapse them into solid blocks unless the real object is an enclosed machine.
"""

        system_prompt = f"""You are a 3D modeling expert. Generate primitive shapes to create a realistic representation of the given object.

CRITICAL RULES:
1. Use ONLY: "box", "cylinder", "sphere", "cone"
2. All coordinates are RELATIVE to the object's GEOMETRIC CENTER (0, 0, 0)
3. Z-axis is UP, Y-axis is DEPTH (front=-Y, back=+Y), X-axis is WIDTH
4. The FRONT of the object faces -Y direction
5. Total bounding box: width={dims[0]:.2f}, depth={dims[1]:.2f}, height={dims[2]:.2f}

COORDINATE SYSTEM (VERY IMPORTANT):
- The parent object is placed at the GEOMETRIC CENTER of the bounding box
- X ranges from -{dims[0]/2:.2f} to +{dims[0]/2:.2f} (left to right)
- Y ranges from -{dims[1]/2:.2f} to +{dims[1]/2:.2f} (front to back, FRONT is -Y)
- Z ranges from -{dims[2]/2:.2f} to +{dims[2]/2:.2f} (bottom to top)
- Z=0 is the VERTICAL CENTER, not the bottom!
- Bottom of object is at Z=-{dims[2]/2:.2f}, Top is at Z=+{dims[2]/2:.2f}
{rotation_note}
CYLINDER / CONE ORIENTATION (VERY IMPORTANT):
- Blender creates cylinders/cones with height along Z-axis by default.
- "dimensions" is [width_X, depth_Y, height_Z].
- For an UPRIGHT cylinder (e.g., table leg): height goes in dim[2], radius in dim[0] and dim[1].
  Example: dim=[0.05, 0.05, 0.8] — thin cylinder standing up, 0.8m tall.
- For a HORIZONTAL cylinder (e.g., barbell shaft along X): you MUST still put height in dim[2]
  and then set rotation to [0, 1.5708, 0] (90° around Y in RADIANS) to tip it horizontal.
  Example: dim=[0.03, 0.03, 1.2], rotation=[0, 1.5708, 0] — cylinder 1.2m long along X.
- For a HORIZONTAL cylinder along Y: dim=[0.03, 0.03, 1.0], rotation=[1.5708, 0, 0].

ROTATION RULES:
- "rotation" values MUST be in RADIANS (e.g., 1.5708 = 90°, 3.1416 = 180°).
  NEVER use degrees like 90 or 180. Always use radians.
- Keep "rotation" as [0, 0, 0] for most parts.
- Only use non-zero rotation when a part is structurally angled relative to
  the object body (e.g., a tilted backrest, horizontal bar, angled armrest).
- NEVER rotate parts to match an observed angle from a top-down image.
  The object's world-space orientation is handled separately by base_rotation.

Example for a chair (height=0.9):
- Seat at Z=-0.05 (slightly below center)
- Legs at Z=-0.35 (near bottom, which is -0.45)
- Backrest at Z=+0.20 (above center)

SHELVED / DISPLAY / CABINET INTERIOR (VERY IMPORTANT):
If the object_type OR description refers to any of these — hutch, china
hutch / cabinet, display cabinet, display case, curio, bookcase,
bookshelf, etagere, open shelf, floating shelf, wall shelf, armoire,
pantry, glass-front cabinet, media console with open compartments — then
the interior MUST contain explicit HORIZONTAL SHELF parts. Items will be
placed on those shelves by a later stage, and it needs a usable flat top
on each tier.

Concretely:
- Do NOT model the body as a single big solid box. If you emit a
  "cabinet_body" / "upper_body" / "display_body" block, leave it thin
  (a <= 0.04 m outer shell: thin back panel + thin side panels + thin
  top + thin bottom) OR omit the body volume entirely and rely on the
  individual frame parts to imply it. The centre must be HOLLOW.
- Emit one part per interior shelf level, named
  "shelf_1", "shelf_2", "shelf_3" (or "upper_shelf_1", etc. when the
  piece has distinct upper/lower compartments).
- Each shelf part: type="box", thickness (dim[2]) between 0.02 and 0.04,
  width and depth at least 80 % of the interior width / depth, located
  INSIDE the body (not at its very top or bottom).
- Typical shelf counts:
  * 1.3 m tall upper cabinet   -> 3 internal shelves
  * 0.8 m tall lower cabinet   -> 1-2 internal shelves
  * floor-to-ceiling bookcase  -> 4-5 internal shelves
  * floating / wall shelf      -> 0 (the slab itself is the surface)
- Space the shelves roughly evenly across the interior height.

GLASS / TRANSPARENT PARTS NAMING (CRITICAL):
If the description mentions "glass", "glass-fronted", "glass door",
"transparent", "crystal", "display glass", "windowed", or similar, any
door / panel / front MUST have the token "glass" in its part name
(examples: "upper_glass_door_left", "glass_panel", "front_glass",
"display_glass_door"). The material stage uses this token to assign a
transparent glass shader; without it the door renders as opaque wood.
Keep the part a thin box (thickness <= 0.03 m) located on the front
face of the cabinet (y ≈ -depth/2).

SURFACE-BEARING FURNITURE — BODY ONLY (CRITICAL):
If the object is a bench, table, desk, island, worktop, countertop, counter,
nightstand, vanity, console, dining table, kitchen island, cabinet top, or any
furniture whose primary role is to HOLD things on a horizontal top surface:
- Generate parts for the STRUCTURAL BODY ONLY: worktop / countertop slab, base
  cabinet body, drawers, doors, handles, legs, support posts, toe kick, back
  panel, side panels, integrated sinks / basins / built-in faucets, integrated
  wall-attached fixtures.
- NEVER generate parts that represent loose items placed ON TOP of the surface,
  even if the description ("holds a centrifuge", "with a computer setup",
  "surface holds bottles and racks", "topped with a microscope", etc.) mentions
  them. Those items are independent objects handled by other stages — modeling
  them here creates duplicate and overlapping geometry and visible floating.
- Forbidden surface-clutter part names (non-exhaustive): monitor, screen,
  keyboard, mouse, laptop, computer, microscope, centrifuge, vortex, mixer,
  stirrer, balance, hot_plate, bottle, flask, beaker, vial, reagent, test_tube,
  rack, tube_rack, pipette, plate, dish, cup, mug, glassware, notebook, paper,
  document, book, folder, binder, decor, ornament, vase, plant, lamp, candle,
  basket, box, storage_box, equipment_box, papers, food, dishware.
- If the description repeatedly describes tabletop clutter, you must STILL
  ignore it — produce ONLY the furniture body. The clutter is captured
  elsewhere by Stage7_small_objects and orphan placement.

WORKTOP TOP MUST FILL THE BBOX (CRITICAL):
For any surface-bearing furniture above, the worktop / countertop / desk-top
slab (the topmost large horizontal part) MUST have its TOP face at exactly
z = +{dims[2]/2:.3f} m (the bbox top). Do not leave empty headroom above the
slab — that empty headroom is what gets filled with phantom clutter by mistake
and causes downstream objects to float.
Concretely:
- Model the worktop slab as a single thin box with
  loc_z + dim_z/2 == +{dims[2]/2:.3f}.
- Slab thickness (dim_z) ≈ 0.04 – 0.08 m.
- The base cabinet body fills the volume below the slab, from
  z = -{dims[2]/2:.3f} (bbox bottom) up to the underside of the slab.

{industrial_system_addendum}
Output ONLY valid JSON:
{{
  "parts": [
    {{
      "shape_type": "box|cylinder|sphere|cone",
      "name": "descriptive_part_name",
      "relative_location": [x, y, z],
      "dimensions": [width_X, depth_Y, height_Z],
      "rotation": [rx, ry, rz]
    }}
  ]
}}

IMPORTANT for "dimensions":
- For box: [width_X, depth_Y, height_Z] — straightforward.
- For cylinder/cone: [diameter_X, diameter_Y, height_Z]. The height is ALWAYS dim[2].
  To orient horizontally, use "rotation" in radians, NOT by swapping dimensions.
- For sphere: [dX, dY, dZ] — typically all equal.

Part count guidelines:
- Simple objects (book, pillow, rug): 1-2 parts
- Medium objects (nightstand, bench): 3-6 parts  
- Complex objects (armchair, bed, wardrobe): 5-12 parts
- Shelved / display cabinets, hutches, bookcases, pantries: 10-18 parts
  (outer thin frame + 3-5 shelves + multiple doors / glass panels)
"""

        rotation_reminder = ""
        if has_rotation:
            rotation_reminder = f"""- This object has base_rotation={rot_deg}° applied by the pipeline.
  Model it in CANONICAL axis-aligned pose. Do NOT add rotation to parts to
  match the top-down image orientation."""

        lab_hint = self._lab_double_deck_geometry_hint(obj.name, dims)
        lab_block = f"{lab_hint}\n" if lab_hint else ""
        industrial_block = f"{industrial_hint}\n" if industrial_hint else ""

        user_content = f"""Generate geometry for:

**Name**: {obj.name}
**Type**: {obj.object_type}
**Bounding Box**: width={dims[0]:.2f}m, depth={dims[1]:.2f}m, height={dims[2]:.2f}m
**Description**: {obj.description}
**Material**: {obj.material_description}
**Color**: {obj.color_description}

{lab_block}{industrial_block}REMEMBER:
- (0,0,0) is the GEOMETRIC CENTER of the bounding box
- Z ranges from -{dims[2]/2:.2f} (bottom) to +{dims[2]/2:.2f} (top)
- Front faces -Y direction
{rotation_reminder}
"""
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_content)
        ]

        last_err: Optional[str] = None
        max_attempts = self.geometry_max_attempts

        for attempt in range(1, max_attempts + 1):
            if attempt > 1:
                self._log(
                    f"  Retry {attempt}/{max_attempts} for {obj.name} "
                    f"(after {self.geometry_retry_delay_sec}s)...",
                    "warning",
                )
                time.sleep(self.geometry_retry_delay_sec)

            try:
                response = self.llm.invoke(messages)
                result = self._extract_json(response)

                if result and "parts" in result:
                    obj.parts = []
                    for p in result["parts"]:
                        rel_loc = p.get("relative_location", [0, 0, 0])
                        dims_part = p.get("dimensions", [0.1, 0.1, 0.1])

                        obj.parts.append(PrimitiveShape(
                            shape_type=p.get("shape_type", "box"),
                            name=p.get("name", "part"),
                            relative_location=tuple(rel_loc),
                            dimensions=tuple(dims_part),
                            rotation=tuple(p.get("rotation", [0, 0, 0]))
                        ))

                    obj.generated = True
                    self._log(f"  -> {len(obj.parts)} parts generated", "success")
                    if attempt > 1:
                        self._log(f"  -> Succeeded on attempt {attempt}", "success")
                    return True

                last_err = "Failed to parse response or missing parts"
                self._log(f"  -> {last_err} (attempt {attempt}/{max_attempts})", "warning")

            except Exception as e:
                last_err = str(e)
                self._log(f"  -> Error: {e} (attempt {attempt}/{max_attempts})", "error")

        self._log(
            f"  -> Giving up on {obj.name} after {max_attempts} attempt(s). Last: {last_err}",
            "error",
        )
        return False
    
    def _extract_json(self, text: str) -> dict:
        """Extract JSON from response"""
        try:
            return json.loads(text)
        except:
            pass
        
        patterns = [
            r'```json\s*([\s\S]*?)\s*```',
            r'```\s*([\s\S]*?)\s*```',
            r'\{[\s\S]*\}'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    json_str = match.group(1) if '```' in pattern else match.group(0)
                    return json.loads(json_str)
                except:
                    continue
        return None
    
    def generate_single(self, index: int) -> bool:
        """Generate geometry for a single object by index"""
        if index < 0 or index >= len(self.detailed_objects):
            self._log(f"Invalid index: {index}", "error")
            return False
        
        obj = self.detailed_objects[index]
        success = self._generate_geometry_for_object(obj)
        
        if success:
            self._save_progress()
            self._log(f"Progress saved ({index + 1}/{len(self.detailed_objects)})", "save")
        
        return success
    
    def generate_by_name(self, name: str) -> bool:
        """Generate geometry for a single object by name"""
        for i, obj in enumerate(self.detailed_objects):
            if obj.name == name:
                return self.generate_single(i)
        
        self._log(f"Object not found: {name}", "error")
        return False
    
    def generate_all(self, skip_generated: bool = True) -> int:
        """Generate geometry for all objects (incremental)"""
        generated_count = 0
        
        for i, obj in enumerate(self.detailed_objects):
            if skip_generated and obj.generated:
                self._log(f"[{i+1}/{len(self.detailed_objects)}] Skipping {obj.name} (already generated)", "info")
                continue
            
            self._log(f"[{i+1}/{len(self.detailed_objects)}] Processing {obj.name}...", "step")
            
            if self._generate_geometry_for_object(obj):
                generated_count += 1
                self._save_progress()
                self._log(f"  Progress saved", "save")
        
        return generated_count
    
    def generate_all_parallel(self, skip_generated: bool = True, max_workers: int = 4) -> int:
        """Generate geometry for all objects in parallel using thread pool."""
        pending = []
        for i, obj in enumerate(self.detailed_objects):
            if skip_generated and obj.generated:
                self._log(f"[{i+1}/{len(self.detailed_objects)}] Skipping {obj.name} (already generated)", "info")
                continue
            pending.append((i, obj))
        
        if not pending:
            self._log("All objects already generated, nothing to do", "info")
            return 0
        
        total = len(self.detailed_objects)
        self._log(f"Parallel generation: {len(pending)} objects with {max_workers} workers", "step")
        
        generated_count = 0
        counter_lock = threading.Lock()
        
        def _worker(idx_obj):
            i, obj = idx_obj
            self._log(f"[{i+1}/{total}] Processing {obj.name}...", "step")
            success = self._generate_geometry_for_object(obj)
            if success:
                self._save_progress()
            return i, obj.name, success
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_worker, item): item for item in pending}
            
            for future in as_completed(futures):
                try:
                    i, name, success = future.result()
                    if success:
                        with counter_lock:
                            generated_count += 1
                        done = sum(1 for o in self.detailed_objects if o.generated)
                        self._log(f"  Done: {name} (progress: {done}/{total})", "save")
                    else:
                        self._log(f"  Failed: {name}", "warning")
                except Exception as e:
                    item = futures[future]
                    self._log(f"  Worker error for {item[1].name}: {e}", "error")
        
        return generated_count
    
    def generate_indices(self, indices: List[int]) -> int:
        """Generate geometry for specific indices"""
        generated_count = 0
        
        for idx in indices:
            if self.generate_single(idx):
                generated_count += 1
        
        return generated_count
    
    def generate_names(self, names: List[str]) -> int:
        """Generate geometry for specific names"""
        generated_count = 0
        
        for name in names:
            if self.generate_by_name(name):
                generated_count += 1
        
        return generated_count
    
    def show_progress(self):
        """Show current progress"""
        print("\n" + "=" * 60)
        print("📊 Geometry Generation Progress")
        print("=" * 60)
        
        total = len(self.detailed_objects)
        generated = sum(1 for obj in self.detailed_objects if obj.generated)
        pending = total - generated
        
        print(f"Total objects: {total}")
        print(f"Generated:     {generated} ✅")
        print(f"Pending:       {pending} ⏳")
        print(f"Progress:      {generated/total*100:.1f}%")
        
        print("\n--- Object Status ---")
        for i, obj in enumerate(self.detailed_objects):
            status = "✅" if obj.generated else "⏳"
            parts = f"({len(obj.parts)} parts)" if obj.generated else ""
            print(f"  [{i:2d}] {status} {obj.name:30s} {obj.object_type:20s} {parts}")
        
        print("=" * 60)
    
    def _extract_location_expr(self, call_str: str) -> Optional[str]:
        """
        Extract the location expression (2nd positional arg) from a create_box/create_cylinder call.
        
        Handles:
        - Tuple literals: create_box("Name", (1, 2, 3), ...)
        - Variable refs:  create_cylinder("Name", table_loc, ...)
        - Complex exprs:  create_box("Name", (loc[0], loc[1], 0.5), ...)
        """
        import re
        
        name_end = call_str.find('"', call_str.find('"') + 1)
        if name_end < 0:
            return None
        
        comma_after_name = call_str.find(',', name_end)
        if comma_after_name < 0:
            return None
        
        rest = call_str[comma_after_name + 1:].lstrip()
        
        if rest.startswith('('):
            depth = 0
            start_idx = comma_after_name + 1 + (len(call_str[comma_after_name + 1:]) - len(rest))
            for j in range(start_idx, len(call_str)):
                if call_str[j] == '(':
                    depth += 1
                elif call_str[j] == ')':
                    depth -= 1
                    if depth == 0:
                        return call_str[start_idx:j + 1].strip()
        else:
            m = re.match(r'([A-Za-z_]\w*(?:\[[^\]]+\])*)', rest)
            if m:
                return m.group(1).strip()
        
        return None

    def _split_top_level_call_args(self, call_str: str) -> List[str]:
        """Split a function call's argument list on top-level commas only."""
        open_idx = call_str.find('(')
        if open_idx < 0:
            return []

        args_text = call_str[open_idx + 1:]
        if args_text.endswith(')'):
            args_text = args_text[:-1]

        args: List[str] = []
        start = 0
        depth = 0
        in_string = None
        escape = False

        for idx, ch in enumerate(args_text):
            if escape:
                escape = False
                continue
            if in_string is not None:
                if ch == '\\':
                    escape = True
                elif ch == in_string:
                    in_string = None
                continue
            if ch in ("'", '"'):
                in_string = ch
                continue
            if ch in "([{":
                depth += 1
                continue
            if ch in ")]}":
                depth -= 1
                continue
            if ch == "," and depth == 0:
                arg = args_text[start:idx].strip()
                if arg:
                    args.append(arg)
                start = idx + 1

        tail = args_text[start:].strip()
        if tail:
            args.append(tail)
        return args

    @staticmethod
    def _is_keyword_arg(arg: str, keyword: str) -> bool:
        return bool(re.match(rf"^\s*{re.escape(keyword)}\s*=", arg or ""))
    
    def _build_replacement_line(self, full_call: str, name: str) -> Optional[str]:
        """Build a create_detailed_object replacement line from an original create_box/create_cylinder call.
        
        Args:
            full_call: The complete function call string (may span multiple lines).
            name: The DETAILED_GEOMETRY key name to use.
        """
        import re

        first_line = full_call.split('\n')[0]
        func_start = first_line.find('create_box(')
        if func_start < 0:
            func_start = first_line.find('create_cylinder(')
        if func_start < 0:
            return None

        call_str = full_call[full_call.find('create_box('):] if 'create_box(' in full_call else full_call[full_call.find('create_cylinder('):]
        call_args = self._split_top_level_call_args(call_str)
        loc_expr = call_args[1] if len(call_args) > 1 else self._extract_location_expr(call_str)

        rot_match = re.search(r'rotation\s*=\s*(\([^)]+\))', full_call)
        rot_expr = rot_match.group(1) if rot_match else None
        if not rot_expr and len(call_args) > 3 and "=" not in call_args[3]:
            rot_expr = call_args[3].strip()

        mat_match = re.search(r'material\s*=\s*(\w+)', full_call)
        material = mat_match.group(1) if mat_match else "None"
        if not mat_match and len(call_args) > 4 and "=" not in call_args[4]:
            material = call_args[4].strip()

        col_match = re.search(r'collection\s*=\s*(\w+)', full_call)
        collection = col_match.group(1) if col_match else "None"
        if not col_match and len(call_args) > 5 and "=" not in call_args[5]:
            collection = call_args[5].strip()

        indent = len(first_line) - len(first_line.lstrip())
        indent_str = ' ' * indent

        parts = [f'"{name}"']
        if loc_expr:
            parts.append(f'location={loc_expr}')
        if rot_expr:
            parts.append(f'rotation={rot_expr}')
        parts.append(f'material={material}')
        parts.append(f'collection={collection}')

        return f'{indent_str}create_detailed_object({", ".join(parts)})'

    def _append_orphan_object_calls(
        self,
        code: str,
        orphan_objects: List[DetailedObject],
    ) -> str:
        """Append create calls for generated objects that did not exist in base code."""
        if not orphan_objects:
            return code

        lines = [
            "",
            "    # Stage 7 semantic inventory orphans promoted into Stage 8 geometry",
        ]
        for obj in sorted(orphan_objects, key=lambda o: o.name):
            loc = tuple(obj.center_location)
            rot = tuple(obj.base_rotation)
            parts = [f'"{obj.name}"', f"location={loc}"]
            if any(abs(r) > 1e-6 for r in rot):
                parts.append(f"rotation={rot}")
            parts.extend(["material=None", "collection=None"])
            lines.append(f"    create_detailed_object({', '.join(parts)})")

        block = "\n".join(lines) + "\n"
        marker = '\nif __name__ == "__main__":'
        if marker in code:
            return code.replace(marker, "\n" + block + marker, 1)

        # Fallback for unusual scripts without a main guard. These calls will
        # run after the base layout function has executed if the script calls it
        # at top level; otherwise they are harmless no-ops until called manually.
        top_level = [
            "",
            "# Stage 7 semantic inventory orphans promoted into Stage 8 geometry",
        ]
        for obj in sorted(orphan_objects, key=lambda o: o.name):
            loc = tuple(obj.center_location)
            rot = tuple(obj.base_rotation)
            parts = [f'"{obj.name}"', f"location={loc}"]
            if any(abs(r) > 1e-6 for r in rot):
                parts.append(f"rotation={rot}")
            parts.extend(["material=None", "collection=None"])
            top_level.append(f"create_detailed_object({', '.join(parts)})")
        return code.rstrip() + "\n" + "\n".join(top_level) + "\n"
    
    def generate_blender_code(self, base_code_path: str = None) -> str:
        """
        Generate Blender Python code that integrates detailed geometry into the original scene.
        
        Args:
            base_code_path: Path to the original scene code (e.g., stage4_output.py)
        """
        self._log("Generating integrated Blender code...", "step")
        
        base_code = None
        
        # 1) Explicit path
        if base_code_path and os.path.exists(base_code_path):
            with open(base_code_path, "r", encoding="utf-8") as f:
                base_code = f.read()
            self._log(f"Base code from file: {base_code_path}", "info")
        
        # 2) Fallback: read from Memory (stage4 -> stage3)
        if not base_code and self.memory:
            for stage_name in ("stage4", "stage3"):
                entry = self.memory.get_latest(stage=stage_name, type="result")
                if entry and isinstance(entry.content, str) and "bpy" in entry.content:
                    base_code = entry.content
                    self._log(f"Base code from Memory ({stage_name})", "info")
                    break
        
        # 3) Fallback: try output_dir sibling (same run directory)
        if not base_code:
            run_dir = os.path.dirname(self.output_dir)
            for candidate in (
                os.path.join(run_dir, "stage4", "stage4_output.py"),
                os.path.join(run_dir, "stage4", "stage4_clean.py"),
                os.path.join(run_dir, "stage3", "stage3_output.py"),
            ):
                if os.path.exists(candidate):
                    with open(candidate, "r", encoding="utf-8") as f:
                        base_code = f.read()
                    self._log(f"Base code from sibling: {candidate}", "info")
                    break
        
        if not base_code:
            self._log("Base code not found from any source (file / Memory / sibling dirs)", "error")
            return ""
        
        # Get generated objects map
        generated_map = {obj.name: obj for obj in self.detailed_objects if obj.generated}
        self._log(f"Replacing {len(generated_map)} objects with detailed geometry", "info")
        
        # Build the detailed geometry data
        geometry_data_lines = [
            '',
            '# ==============================================================================',
            '# DETAILED GEOMETRY DATA (Auto-generated)',
            '# ==============================================================================',
            '',
            'import bmesh',
            '',
            'DETAILED_GEOMETRY = {'
        ]
        
        for name, obj in generated_map.items():
            geometry_data_lines.append(f'    "{name}": {{')
            geometry_data_lines.append(f'        "center": {list(obj.center_location)},')
            geometry_data_lines.append(f'        "rotation": {list(obj.base_rotation)},')
            geometry_data_lines.append(f'        "parts": [')
            for part in obj.parts:
                geometry_data_lines.append(f'            {{"type": "{part.shape_type}", "name": "{part.name}", "loc": {list(part.relative_location)}, "dim": {list(part.dimensions)}, "rot": {list(part.rotation)}}},')
            geometry_data_lines.append('        ]')
            geometry_data_lines.append('    },')
        
        geometry_data_lines.append('}')
        
        # Build the detailed object creation function
        geometry_func_lines = [
            '',
            'def create_detailed_object(name, location=None, rotation=None, material=None, collection=None):',
            '    """Create an object with detailed geometry instead of simple bbox.',
            '    ',
            '    Args:',
            '        name: Object name (must exist in DETAILED_GEOMETRY)',
            '        location: Override position (x, y, z). If None, uses DETAILED_GEOMETRY center.',
            '        rotation: Override rotation (rx, ry, rz). If None, uses DETAILED_GEOMETRY rotation.',
            '        material: Material to apply to all parts.',
            '        collection: Collection to link the object to.',
            '    """',
            '    if name not in DETAILED_GEOMETRY:',
            '        return None',
            '    ',
            '    data = DETAILED_GEOMETRY[name]',
            '    center = location if location is not None else data["center"]',
            '    base_rot = rotation if rotation is not None else data["rotation"]',
            '    parts = data["parts"]',
            '    ',
            '    # Create parent empty at the object center',
            '    parent = bpy.data.objects.new(name, None)',
            '    parent.empty_display_type = "PLAIN_AXES"',
            '    parent.empty_display_size = 0.1',
            '    parent.location = center',
            '    parent.rotation_euler = base_rot',
            '    ',
            '    if collection:',
            '        collection.objects.link(parent)',
            '    else:',
            '        bpy.context.scene.collection.objects.link(parent)',
            '    ',
            '    # Create each part',
            '    for part in parts:',
            '        ptype = part["type"]',
            '        pname = f"{name}_{part[\'name\']}"',
            '        ploc = part["loc"]',
            '        pdim = part["dim"]',
            '        prot = part["rot"]',
            '        ',
            '        mesh = bpy.data.meshes.new(pname + "_mesh")',
            '        bm = bmesh.new()',
            '        ',
            '        if ptype == "box":',
            '            bmesh.ops.create_cube(bm, size=1.0)',
            '        elif ptype == "cylinder":',
            '            bmesh.ops.create_cone(bm, cap_ends=True, cap_tris=False, segments=32, radius1=0.5, radius2=0.5, depth=1.0)',
            '        elif ptype == "sphere":',
            '            bmesh.ops.create_uvsphere(bm, u_segments=32, v_segments=16, radius=0.5)',
            '        elif ptype == "cone":',
            '            bmesh.ops.create_cone(bm, cap_ends=True, cap_tris=False, segments=32, radius1=0.5, radius2=0.0, depth=1.0)',
            '        else:',
            '            bmesh.ops.create_cube(bm, size=1.0)',
            '        ',
            '        bm.to_mesh(mesh)',
            '        bm.free()',
            '        ',
            '        obj = bpy.data.objects.new(pname, mesh)',
            '        obj.location = ploc',
            '        obj.dimensions = pdim',
            '        obj.rotation_euler = [r * 3.14159265 / 180 if abs(r) > 6.3 else r for r in prot]',
            '        obj.parent = parent',
            '        ',
            '        if material:',
            '            obj.data.materials.append(material)',
            '        ',
            '        if collection:',
            '            collection.objects.link(obj)',
            '        else:',
            '            bpy.context.scene.collection.objects.link(obj)',
            '    ',
            '    return parent',
            '',
        ]
        
        # Now modify the base code to use detailed geometry where available
        # CRITICAL: Preserve the original position and rotation expressions!
        # Handle both:
        #   1. Static-name calls: create_box("Name", ...)
        #   2. Loop-generated calls: for i, ... : create_box(f"Name_{i}", ...)
        
        modified_code = base_code
        import re
        
        replaced_names = set()
        
        def _extract_full_call(lines, start_idx):
            """Extract a full (possibly multi-line) function call starting at start_idx.
            Returns (full_call_str, end_idx) where end_idx is the last line index consumed."""
            paren_depth = 0
            call_lines = []
            for j in range(start_idx, len(lines)):
                call_lines.append(lines[j])
                paren_depth += lines[j].count('(') - lines[j].count(')')
                if paren_depth <= 0:
                    return '\n'.join(call_lines), j
            return '\n'.join(call_lines), len(lines) - 1
        
        def _fuzzy_match_name(code_name: str, geo_name: str) -> bool:
            """Check if a geometry name matches a code name, allowing fuzzy suffix matching."""
            if code_name == geo_name:
                return True
            cn = code_name.lower().replace(' ', '_')
            gn = geo_name.lower().replace(' ', '_')
            if cn == gn:
                return True
            cn_parts = cn.split('_')
            gn_parts = gn.split('_')
            if len(cn_parts) >= 2 and len(gn_parts) >= 2:
                if cn_parts[0] == gn_parts[0] and cn_parts[1] == gn_parts[1]:
                    cn_suffix = '_'.join(cn_parts[2:])
                    gn_suffix = '_'.join(gn_parts[2:])
                    if cn_suffix.startswith(gn_suffix) or gn_suffix.startswith(cn_suffix):
                        return True
            return False

        def _extract_static_call_name(full_call: str) -> Optional[str]:
            """Extract the first string argument from a create_box/create_cylinder call."""
            m = re.search(r'create_(?:box|cylinder)\(\s*["\']([^"\']+)["\']', full_call, re.S)
            return m.group(1) if m else None

        def _scan_balanced_assign_rhs(lines, start_idx, max_lookahead=200):
            """Return ``(rhs_text, end_idx)`` for an assignment whose RHS may
            span many lines (list / dict / tuple / set literals).

            Tracks paren/bracket/brace depth across lines and skips chars
            inside string literals so closing tokens inside strings (e.g.
            ``"obj_]_weird"``) don't mis-terminate the scan.

            This replaces an older `if ']' in lines[ll]: break` heuristic
            that capped at 15 lines and only handled list literals — broken
            for the GPT-5.x dict-literal pattern::

                chair_positions = {       # 8 entries — 8 lines, happens to fit
                    "obj_003_…": (-0.65, 0.65),
                    …
                }
                for name, (x, y) in chair_positions.items():
                    create_box(name, …)

            and outright wrong for ≥15-entry dicts (silent name truncation).
            """
            if start_idx >= len(lines):
                return '', start_idx
            head = lines[start_idx]
            eq_pos = head.find('=')
            if eq_pos < 0:
                return '', start_idx

            rhs_pieces = [head[eq_pos + 1:]]
            depth = 0
            in_string = None  # None | "'" | '"'
            escape = False

            def consume(text):
                nonlocal depth, in_string, escape
                for ch in text:
                    if escape:
                        escape = False
                        continue
                    if in_string is not None:
                        if ch == '\\':
                            escape = True
                        elif ch == in_string:
                            in_string = None
                        continue
                    if ch in ("'", '"'):
                        in_string = ch
                        continue
                    if ch in '([{':
                        depth += 1
                    elif ch in ')]}':
                        depth -= 1

            consume(rhs_pieces[0])
            end_idx = start_idx

            if depth > 0:
                hard_cap = min(start_idx + max_lookahead, len(lines))
                for ll in range(start_idx + 1, hard_cap):
                    rhs_pieces.append(lines[ll])
                    consume(lines[ll])
                    end_idx = ll
                    if depth <= 0:
                        break

            return '\n'.join(rhs_pieces), end_idx
        
        # --- Pass 1: Replace static-name calls ---
        for name in generated_map.keys():
            lines = modified_code.split('\n')
            found = False
            for i, line in enumerate(lines):
                if 'create_box(' in line or 'create_cylinder(' in line:
                    full_call, end_idx = _extract_full_call(lines, i)
                    code_name = _extract_static_call_name(full_call)
                    if not code_name or not _fuzzy_match_name(code_name, name):
                        continue
                    new_line = self._build_replacement_line(full_call, name)
                    if new_line:
                        lines[i:end_idx + 1] = [new_line]
                        replaced_names.add(name)
                        self._log(f"  Replaced (static): {name} <- {code_name}", "info")
                        found = True
                        modified_code = '\n'.join(lines)
                    break
        
        # --- Pass 2: Replace for-loop-generated calls ---
        # Find names in generated_map that have a common prefix pattern like "Dining_Chair_01", "Dining_Chair_02"
        # Group them by prefix (e.g., "Dining_Chair_" -> ["Dining_Chair_01", ...])
        unreplaced = {n: obj for n, obj in generated_map.items() if n not in replaced_names}
        
        if unreplaced:
            # Group by prefix: find common base name pattern
            # e.g., "Dining_Chair_01" -> "Dining_Chair_", "Plant_01" -> "Plant_", "Plant_SW" -> "Plant_"
            prefix_groups: Dict[str, List[str]] = {}
            for name in unreplaced:
                # Try multiple splitting strategies
                # 1. Strip trailing digits: Plant_01 -> Plant_
                m = re.match(r'^(.+_)\d+$', name)
                if m:
                    prefix = m.group(1)
                    prefix_groups.setdefault(prefix, []).append(name)
                    continue
                # 2. Strip trailing word after last underscore: Plant_SW -> Plant_
                m = re.match(r'^(.+_)[A-Za-z]+$', name)
                if m:
                    prefix = m.group(1)
                    prefix_groups.setdefault(prefix, []).append(name)
                    continue
                # 3. No pattern - standalone name
                prefix_groups.setdefault(name + '_', []).append(name)
            
            # Merge groups with the same prefix
            # e.g., Plant_01 and Plant_SW both map to "Plant_"
            merged: Dict[str, List[str]] = {}
            for prefix, names_list in prefix_groups.items():
                merged.setdefault(prefix, []).extend(names_list)
            prefix_groups = merged
            
            lines = modified_code.split('\n')
            
            for prefix, names_in_group in prefix_groups.items():
                # Find the for-loop that generates these objects
                # Look for lines containing f-strings with this prefix
                # e.g., f"Dining_Chair_{i+1:02}" or f"Leather_Armchair_{i+1}"
                loop_start = None
                loop_end = None
                loop_indent = 0
                
                for i, line in enumerate(lines):
                    stripped = line.lstrip()
                    # Check if this is a for-loop header
                    if stripped.startswith('for ') and stripped.endswith(':'):
                        # Scan the loop body for the prefix in f-strings
                        body_indent = len(lines[i+1]) - len(lines[i+1].lstrip()) if i + 1 < len(lines) and lines[i+1].strip() else 0
                        body_lines_range = []
                        for j in range(i + 1, len(lines)):
                            if not lines[j].strip():
                                body_lines_range.append(j)
                                continue
                            cur_indent = len(lines[j]) - len(lines[j].lstrip())
                            if cur_indent >= body_indent:
                                body_lines_range.append(j)
                            else:
                                break
                        
                        body_text = '\n'.join(lines[j] for j in body_lines_range)
                        escaped_prefix = re.escape(prefix.rstrip('_'))
                        
                        # Check if this loop generates our objects. Multiple patterns:
                        # 1. f-string: f"Prefix_{i+1}" or f"Prefix_{i+1:02}"
                        # 2. Variable name in loop data: ("Plant_01", ...) with create_xxx(name, ...)
                        is_match = False
                        
                        # Pattern 1: f-string containing prefix (may have obj_xxx_ prefix before it)
                        if re.search(rf'f["\'][^"\']*' + escaped_prefix, body_text):
                            is_match = True
                        
                        # Pattern 2: loop iterates over data containing our object names
                        # Check if any of our group names appear in the loop header's iterable data
                        if not is_match:
                            # Look for the iterable in the for-loop header
                            for_match = re.search(r'for\s+.+\s+in\s+(\w+)', stripped)
                            if for_match:
                                iter_var = for_match.group(1)
                                # Find the list definition for this variable above the loop
                                for k in range(i - 1, max(i - 20, -1), -1):
                                    if k < 0:
                                        break
                                    if f'{iter_var}' in lines[k] and '=' in lines[k]:
                                        # Collect the FULL RHS of the assignment
                                        # (list / dict / tuple) by tracking
                                        # paren/bracket/brace balance — no
                                        # 15-line truncation, no '`]`' false
                                        # terminator.
                                        list_text, _assign_end = _scan_balanced_assign_rhs(lines, k)
                                        if any(f'"{n}"' in list_text or f"'{n}'" in list_text for n in names_in_group):
                                            is_match = True
                                        break
                        
                        if is_match:
                            # For Pattern 2, also include the list definition lines
                            list_def_start = i
                            if for_match:
                                iter_var = for_match.group(1)
                                for k in range(i - 1, max(i - 20, -1), -1):
                                    if k < 0:
                                        break
                                    if f'{iter_var}' in lines[k] and '=' in lines[k]:
                                        # Include from list definition to end of loop
                                        list_def_start = k
                                        # Also skip any blank/comment lines before the list
                                        while list_def_start > 0 and (not lines[list_def_start - 1].strip() or lines[list_def_start - 1].strip().startswith('#')):
                                            list_def_start -= 1
                                        break
                            
                            loop_start = list_def_start
                            loop_end = body_lines_range[-1] if body_lines_range else i
                            loop_indent = len(lines[i]) - len(lines[i].lstrip())
                            break
                
                if loop_start is not None:
                    body_text = '\n'.join(lines[loop_start:loop_end + 1])
                    indent_str = ' ' * loop_indent
                    assigned_locals = set(re.findall(
                        r'^\s*([A-Za-z_]\w*)\s*=', body_text, re.MULTILINE
                    ))

                    def _stable_material_name(material_name: str) -> str:
                        if material_name not in assigned_locals:
                            return material_name

                        assign_match = re.search(
                            rf'^\s*{re.escape(material_name)}\s*=\s*(.+)$',
                            body_text,
                            re.MULTILINE,
                        )
                        if assign_match:
                            rhs = assign_match.group(1)
                            for candidate in re.findall(r'\bmat_[A-Za-z0-9_]+\b', rhs):
                                if candidate not in assigned_locals:
                                    return candidate
                        return "None"
                    
                    # Collect ALL prefix groups whose objects live in this same loop
                    all_names_in_loop = list(names_in_group)
                    for other_prefix, other_names in prefix_groups.items():
                        if other_prefix == prefix:
                            continue
                        for oname in other_names:
                            if oname in replaced_names:
                                continue
                            escaped_oprefix = re.escape(other_prefix.rstrip('_'))
                            if re.search(rf'f["\'][^"\']*' + escaped_oprefix, body_text):
                                all_names_in_loop.append(oname)
                    
                    replacement_lines = [f'{indent_str}# Detailed geometry (replaced from loop)']
                    
                    for obj_name in sorted(set(all_names_in_loop)):
                        if obj_name not in generated_map:
                            continue
                        obj = generated_map[obj_name]
                        loc = list(obj.center_location)
                        rot = list(obj.base_rotation)
                        
                        # Find the original material for this specific object type
                        obj_mat = "None"
                        obj_col = "None"
                        code_name_escaped = re.escape(obj_name.rsplit('_', 1)[0] if '_' in obj_name else obj_name)
                        mat_in_body = re.search(
                            rf'create_(?:box|cylinder)\(.*{code_name_escaped}.*material\s*=\s*(\w+)',
                            body_text, re.DOTALL)
                        if mat_in_body:
                            obj_mat = mat_in_body.group(1)
                        else:
                            mat_fallback = re.search(r'material\s*=\s*(\w+)', body_text)
                            obj_mat = mat_fallback.group(1) if mat_fallback else "None"
                        obj_mat = _stable_material_name(obj_mat)
                        col_in_body = re.search(r'collection\s*=\s*(\w+)', body_text)
                        obj_col = col_in_body.group(1) if col_in_body else "None"
                        
                        parts = [f'"{obj_name}"']
                        parts.append(f'location={tuple(loc)}')
                        if any(r != 0 for r in rot):
                            parts.append(f'rotation={tuple(rot)}')
                        parts.append(f'material={obj_mat}')
                        parts.append(f'collection={obj_col}')
                        
                        replacement_lines.append(f'{indent_str}create_detailed_object({", ".join(parts)})')
                        replaced_names.add(obj_name)
                    
                    lines[loop_start:loop_end + 1] = replacement_lines
                    self._log(f"  Replaced (loop): {len(all_names_in_loop)} objects from loop at lines {loop_start}-{loop_end}", "info")
                    
                    # Rebuild modified_code since line indices shifted
                    modified_code = '\n'.join(lines)
                    lines = modified_code.split('\n')
        
        # Log unreplaced objects
        still_unreplaced = {n for n in generated_map if n not in replaced_names}
        orphan_to_append = [
            generated_map[name]
            for name in still_unreplaced
            if generated_map[name].source_stage == "stage7_orphan"
        ]
        if orphan_to_append:
            modified_code = self._append_orphan_object_calls(
                modified_code,
                orphan_to_append,
            )
            for obj in orphan_to_append:
                replaced_names.add(obj.name)
            self._log(
                f"  Appended {len(orphan_to_append)} Stage 7 orphan objects",
                "info",
            )
        remaining_unreplaced = {n for n in generated_map if n not in replaced_names}
        if remaining_unreplaced:
            self._log(f"  WARNING: Could not replace: {remaining_unreplaced}", "warning")
        
        # Insert the geometry data and function after the imports
        # Find where to insert (after the helper functions, before run_layout_engine)
        insert_marker = "# === MAIN LAYOUT ENGINE ==="
        if insert_marker in modified_code:
            insert_pos = modified_code.find(insert_marker)
            modified_code = (
                modified_code[:insert_pos] + 
                '\n'.join(geometry_data_lines) + '\n' +
                '\n'.join(geometry_func_lines) + '\n\n' +
                modified_code[insert_pos:]
            )
        else:
            # Fallback: insert before "def run_layout_engine"
            insert_marker2 = "def run_layout_engine():"
            insert_pos = modified_code.find(insert_marker2)
            if insert_pos > 0:
                modified_code = (
                    modified_code[:insert_pos] + 
                    '\n'.join(geometry_data_lines) + '\n' +
                    '\n'.join(geometry_func_lines) + '\n\n' +
                    modified_code[insert_pos:]
                )
        
        # Add header comment
        header = f'''"""
Stage Geometry Output - Integrated Scene with Detailed Geometry
===============================================================
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Base scene: {base_code_path}
Objects with detailed geometry: {len(generated_map)}
Total detailed parts: {sum(len(obj.parts) for obj in generated_map.values())}

This file integrates detailed geometry into the original scene.
Objects with detailed geometry are replaced, others remain as simple bbox.
"""

'''
        
        # Remove original docstring/comments at the top if any
        if modified_code.startswith('import'):
            modified_code = header + modified_code
        else:
            # Find first import
            first_import = modified_code.find('import ')
            if first_import > 0:
                modified_code = header + modified_code[first_import:]
            else:
                modified_code = header + modified_code
        
        return modified_code
    
    def _verify_and_fix_syntax(self, code: str) -> str:
        """Multi-pass syntax validation and auto-fix for generated Blender code."""
        import re

        from blender_code_syntax_fix import fix_empty_if_show_direction_before_return

        code = fix_empty_if_show_direction_before_return(code)

        hits = re.findall(r'material\s*=\s*create_material\s*(?=[,\)])', code)
        if hits:
            code = re.sub(r'material\s*=\s*create_material\s*(?=[,\)])', 'material=None', code)
            self._log(f"  Fixed {len(hits)}x material=create_material (function ref → None)", "warning")

        MAX_PASSES = 10
        for pass_num in range(MAX_PASSES):
            try:
                compile(code, '<geometry_output>', 'exec')
                if pass_num > 0:
                    self._log(f"  Syntax OK after {pass_num} fix pass(es)", "success")
                return code
            except SyntaxError as e:
                err_line = e.lineno or 0
                err_msg = str(e.msg) if e.msg else ""
                lines = code.split('\n')
                self._log(f"  Syntax pass {pass_num+1}: {err_msg} at line {err_line}", "warning")
                
                fixed = False
                
                # Fix 1: Broken call where create_xxx(..., material=None, collection=None)
                # is followed by dangling continuation lines forming a second set of args.
                # Merge the prev complete call + all dangling lines into one correct call.
                if 'unexpected indent' in err_msg:
                    if 0 < err_line <= len(lines) and err_line >= 2:
                        prev_line = lines[err_line - 2]
                        prev_stripped = prev_line.rstrip()
                        if (prev_stripped.endswith(')') and 
                            ('create_detailed_object(' in prev_line or 'create_box(' in prev_line or 'create_cylinder(' in prev_line)):
                            prev_indent = len(prev_line) - len(prev_line.lstrip())
                            dangling_end = err_line - 1
                            for j in range(err_line - 1, len(lines)):
                                s = lines[j].strip()
                                if not s:
                                    break
                                cur_indent = len(lines[j]) - len(lines[j].lstrip())
                                if cur_indent <= prev_indent:
                                    break
                                dangling_end = j
                                if s.endswith(')'):
                                    break
                            
                            dangling_text = ' '.join(lines[j].strip() for j in range(err_line - 1, dangling_end + 1))
                            dim_var = re.match(r'^(\w+(?:_\w+)*)\s*,\s*', dangling_text)
                            if dim_var:
                                dangling_text = dangling_text[dim_var.end():]
                            
                            base_call = prev_stripped
                            if re.search(r',\s*material\s*=\s*None\s*,\s*collection\s*=\s*None\s*\)\s*$', base_call):
                                base_call = re.sub(r',\s*material\s*=\s*None\s*,\s*collection\s*=\s*None\s*\)\s*$', '', base_call)
                            else:
                                base_call = base_call.rstrip(')')
                            
                            merged = base_call + ', ' + dangling_text
                            lines[err_line - 2] = merged
                            for j in range(err_line - 1, dangling_end + 1):
                                lines[j] = ''
                            code = '\n'.join(lines)
                            self._log(f"  Fixed broken call at lines {err_line}-{dangling_end+1}", "info")
                            fixed = True
                
                # Fix 2: Unclosed rotation tuple — rotation=(value, material=...
                # Handles nested parens like math.radians(20)
                if not fixed and ('invalid syntax' in err_msg or 'was never closed' in err_msg or 'unmatched' in err_msg):
                    for idx in range(max(0, err_line - 3), min(len(lines), err_line + 2)):
                        rot_start = lines[idx].find('rotation=(')
                        mat_pos = lines[idx].find('material=', rot_start + 1 if rot_start >= 0 else 0)
                        if rot_start >= 0 and mat_pos > rot_start:
                            between = lines[idx][rot_start + len('rotation=('):mat_pos]
                            raw_between = between.rstrip().rstrip(',').rstrip()
                            d = 0
                            rot_closed = False
                            for ch in raw_between:
                                if ch == '(':
                                    d += 1
                                elif ch == ')':
                                    d -= 1
                                    if d < 0:
                                        rot_closed = True
                                        break
                            if rot_closed:
                                continue
                            depth = 0
                            components = []
                            current = []
                            for ch in raw_between:
                                if ch == '(':
                                    depth += 1
                                elif ch == ')':
                                    depth -= 1
                                if ch == ',' and depth == 0:
                                    components.append(''.join(current).strip())
                                    current = []
                                else:
                                    current.append(ch)
                            if current:
                                last = ''.join(current).strip()
                                if last:
                                    components.append(last)
                            while len(components) < 3:
                                components.append('0')
                            fixed_rot = f'rotation=({", ".join(components[:3])}), material='
                            new_line = lines[idx][:rot_start] + fixed_rot + lines[idx][mat_pos + len('material='):]
                            lines[idx] = new_line
                            code = '\n'.join(lines)
                            self._log(f"  Fixed unclosed rotation at line {idx+1}", "info")
                            fixed = True
                            break
                
                if not fixed:
                    self._log(f"  Could not auto-fix syntax error: {err_msg} at line {err_line}", "error")
                    break
        
        return code
    
    def save_blender_code(self, base_code_path: str = None):
        """Save Blender code to file"""
        code = self.generate_blender_code(base_code_path)
        
        if not code:
            self._log("Failed to generate code", "error")
            return None
        
        code = self._verify_and_fix_syntax(code)

        try:
            from stage_clean_arrows import ArrowCleaner
            cleaner = ArrowCleaner(output_dir=self.output_dir, verbose=self.verbose)
            code = cleaner.clean_code(code)
            code = self._verify_and_fix_syntax(code)
        except Exception as e:
            self._log(f"Direction arrow cleanup skipped: {e}", "warning")

        code_path = os.path.join(self.output_dir, "geometry_output.py")

        with open(code_path, "w", encoding="utf-8") as f:
            f.write(code)
        
        self._log(f"Blender code saved: {code_path}", "success")
        return code_path
    
    def run(self, 
            indices: List[int] = None, 
            names: List[str] = None, 
            resume: bool = False,
            generate_code: bool = True,
            base_code_path: str = None,
            parallel: int = 0) -> Tuple[bool, int]:
        """
        Run Stage Geometry
        
        Args:
            indices: Specific indices to generate
            names: Specific names to generate
            resume: Skip already generated objects
            generate_code: Generate Blender code after processing
            base_code_path: Path to original scene code (for integration)
            parallel: Number of parallel workers (0 = sequential)
        
        Returns:
            (success, generated_count)
        """
        print("\n" + "=" * 60)
        print("📐 Stage Geometry - Incremental Detailed Geometry Generation")
        print("=" * 60)
        
        # 1. Load describe data
        if not self._load_describe_data():
            return False, 0
        
        # 2. Load existing progress
        self._load_progress()
        
        # 3. Initialize objects from source
        self._init_detailed_objects()
        self._save_progress()
        
        # 4. Generate geometry
        print("\n--- Geometry Generation ---")
        
        if indices:
            generated_count = self.generate_indices(indices)
        elif names:
            generated_count = self.generate_names(names)
        elif parallel > 0:
            generated_count = self.generate_all_parallel(skip_generated=resume, max_workers=parallel)
        else:
            generated_count = self.generate_all(skip_generated=resume)
        
        # 5. Show progress
        self.show_progress()
        
        # 6. Generate Blender code (integrated with original scene)
        if generate_code:
            print("\n--- Code Generation (Integrated) ---")
            self.save_blender_code(base_code_path)
        
        # 7. Save to Memory
        if self.use_memory:
            generated_objects = [obj for obj in self.detailed_objects if obj.generated]
            code_output_path = os.path.join(self.output_dir, "geometry_output.py")
            self.memory.add(
                stage="stage6_geometry",
                type="result",
                content=json.dumps({
                    "detailed_objects": [obj.to_dict() for obj in generated_objects],
                    "orphan_placement_report": self.orphan_placement_report,
                }, indent=2),
                metadata={
                    "title": "Stage Geometry - Detailed Geometry",
                    "summary": f"{len(generated_objects)} objects, {sum(len(obj.parts) for obj in generated_objects)} parts",
                    "total_objects": len(generated_objects),
                    "orphan_objects_added": len(self.orphan_placement_report.get("added", [])),
                    "orphan_objects_skipped": len(self.orphan_placement_report.get("skipped", [])),
                    "output_file": code_output_path
                },
                tags=["stage6_geometry", "detailed_geometry"]
            )
        
        print("\n" + "=" * 60)
        print("✅ Stage Geometry Complete!")
        print(f"   Generated this run: {generated_count}")
        print(f"   Total generated: {sum(1 for obj in self.detailed_objects if obj.generated)}/{len(self.detailed_objects)}")
        print(f"   Output: {self.output_dir}")
        print("=" * 60)
        
        return True, generated_count


# ==============================================================================
# Command Line Entry
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Stage Geometry - Incremental Detailed Geometry Generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate all objects (skip already generated)
  python stage6_geometry.py --resume
  
  # Generate all objects (regenerate all)
  python stage6_geometry.py
  
  # Generate specific objects by index
  python stage6_geometry.py --indices 0 1 2
  
  # Generate specific objects by name
  python stage6_geometry.py --names "King_Bed" "Armchair_North"
  
  # Show current progress
  python stage6_geometry.py --progress
  
  # Specify base scene code (for integration)
  python stage6_geometry.py --resume --base-code /path/to/stage4_output.py
  
  # Only regenerate Blender code (no LLM calls)
  python stage6_geometry.py --code-only
"""
    )
    
    parser.add_argument("--describe-json", "-d", help="Path to describe_output.json")
    parser.add_argument("--base-code", "-b", help="Path to base scene code (e.g., stage4_output.py)")
    parser.add_argument("--output-dir", "-o", default=None, help="Output directory")
    parser.add_argument("--indices", "-i", type=int, nargs="+", help="Generate specific indices")
    parser.add_argument("--names", "-n", nargs="+", help="Generate specific object names")
    parser.add_argument("--resume", "-r", action="store_true", help="Skip already generated objects")
    parser.add_argument("--progress", "-p", action="store_true", help="Show progress only")
    parser.add_argument("--code-only", action="store_true", help="Only regenerate Blender code (no LLM)")
    parser.add_argument("--no-code", action="store_true", help="Don't generate Blender code")
    parser.add_argument("--no-memory", action="store_true", help="Don't use Memory system")
    parser.add_argument(
        "--geometry-max-attempts",
        type=int,
        default=3,
        help="Max LLM attempts per object on failure (default: 3)",
    )
    parser.add_argument(
        "--geometry-retry-delay",
        type=float,
        default=2.0,
        help="Seconds to wait between retries (default: 2.0)",
    )
    args = parser.parse_args()

    runner = StageGeometryRunner(
        describe_json_path=args.describe_json,
        output_dir=args.output_dir,
        use_memory=not args.no_memory,
        geometry_max_attempts=args.geometry_max_attempts,
        geometry_retry_delay_sec=args.geometry_retry_delay,
    )
    
    # Show progress only
    if args.progress:
        runner._load_describe_data()
        runner._load_progress()
        runner._init_detailed_objects()
        runner.show_progress()
        return 0
    
    # Code only mode - just regenerate Blender code from existing progress
    if args.code_only:
        runner._load_describe_data()
        runner._load_progress()
        runner._init_detailed_objects()
        runner._save_progress()
        runner.show_progress()
        print("\n--- Regenerating Blender Code ---")
        runner.save_blender_code(args.base_code)
        return 0
    
    success, count = runner.run(
        indices=args.indices,
        names=args.names,
        resume=args.resume,
        generate_code=not args.no_code,
        base_code_path=args.base_code
    )
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
