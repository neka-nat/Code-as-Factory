"""
Stage Texture - Real Texture Map Generation via NanoBanana (Gemini Image)
==========================================================================

Generates real texture map PNGs for floor, walls, and wall art using the
Gemini generateContent image API (default: gemini-3.1-flash-image).
Then rewrites the Blender code produced by Stage 10 (stage10_material) to load
those textures via ShaderNodeTexImage, so the final render uses photographic
textures instead of purely procedural shaders.

Pipeline position:
    Stage 10 (stage10_material, parameters)
        -> Stage 11 (stage11_texture, real textures)  <-- THIS FILE
            -> Stage 12 (stage12_render, lighting + render settings)

Inputs (from Memory):
    - stage10_material : material_output.py (code) + floor/wall material JSON
    - stage5_describe : objects list + room_style
    - stage1         : image_path

Outputs:
    - <run_dir>/stage11_texture/images/floor.png
    - <run_dir>/stage11_texture/images/wall.png
    - <run_dir>/stage11_texture/images/art_<obj_id>.png (one per detected wall art)
    - <run_dir>/stage11_texture/texture_output.py (Blender code with texture maps)
    - <run_dir>/stage11_texture/texture_manifest.json

Memory entry:
    stage="stage11_texture", type="result"
        content  = final code
        metadata = {output_file, manifest, image_path, ...}
"""

import os
import re
import sys
import math
import json
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.join(project_root, "image_prompt_gen"))

from memory import Memory  # noqa: E402

# Reuse the nanobanana client from image_prompt_gen
try:
    from topdown_room_image_generator import _call_gemini_image  # noqa: E402
    GEMINI_IMAGE_AVAILABLE = True
except Exception as e:
    print(f"[stage11_texture] warning: _call_gemini_image import failed: {e}")
    GEMINI_IMAGE_AVAILABLE = False


# =============================================================================
# Defaults (reuse render_gen.py conventions)
# =============================================================================
DEFAULT_IMAGE_MODEL = os.environ.get("SCENEGEN_TEXTURE_MODEL") or "gemini-3.1-flash-image"
DEFAULT_BASE_URL = (
    os.environ.get("SCENEGEN_TEXTURE_BASE_URL")
    or os.environ.get("GEMINI_IMAGE_BASE_URL")
    or os.environ.get("SCENEGEN_BASE_URL")
    or os.environ.get("GEMINI_BASE_URL")
    or "https://generativelanguage.googleapis.com"
)
DEFAULT_API_KEY = (
    os.environ.get("SCENEGEN_TEXTURE_API_KEY")
    or os.environ.get("SCENEGEN_API_KEY")
    or os.environ.get("GEMINI_API_KEY")
    or os.environ.get("OPENAI_API_KEY")
)

# Style names that tend to trigger Gemini's IMAGE_RECITATION filter because
# they are strongly associated with copyrighted designs (wallpaper brands,
# trademarked patterns, etc.). We replace them with generic descriptors when
# building primary prompts, and strip them entirely in fallback prompts.
COPYRIGHTED_STYLES = {
    "art deco": "ornate classic",
    "art nouveau": "ornate classic",
    "victorian": "classic traditional",
    "william morris": "floral classic",
    "rifle paper": "floral classic",
    "bohemian": "eclectic",
    "boho": "eclectic",
    "japandi": "warm minimalist",
    "memphis": "colorful geometric",
    "shibori": "indigo textured",
    "toile": "classic pastoral",
    "chinoiserie": "ornate floral",
    "tiffany": "colorful leaded",
}


def _sanitize_style_name(name: str) -> str:
    """Swap copyright-sensitive style names for generic descriptors."""
    if not name:
        return "modern"
    low = name.lower().strip()
    for bad, good in COPYRIGHTED_STYLES.items():
        if bad in low:
            return good
    return name


def _sanitize_description(text: str) -> str:
    """Strip copyright-sensitive style/brand names from free-text descriptions.

    Uses case-insensitive replacement so phrases like "Art Deco geometric pattern"
    or "William Morris floral" won't leak into the generation prompt.
    """
    if not text:
        return text
    out = text
    for bad, good in COPYRIGHTED_STYLES.items():
        out = re.sub(re.escape(bad), good, out, flags=re.IGNORECASE)
    return out


def _is_recitation_error(err: Exception) -> bool:
    """Detect Gemini's IMAGE_RECITATION / copyright-style refusal from the
    error message raised by ``_call_gemini_image``."""
    msg = str(err)
    markers = (
        "IMAGE_RECITATION",
        "could not generate the image",
        "Unable to show the generated image",
    )
    return any(m in msg for m in markers)


def _is_transient_network_error(err: Exception) -> bool:
    """Detect timeouts / connection resets / 5xx — errors where the correct
    action is to back off and retry rather than change the prompt."""
    msg = str(err).lower()
    markers = (
        "read timed out",
        "readtimeout",
        "connection aborted",
        "connection reset",
        "connectionerror",
        "connection refused",
        "remote end closed connection",
        "bad gateway",
        "service unavailable",
        "gateway timeout",
        "http 500",
        "http 502",
        "http 503",
        "http 504",
        "temporarily unavailable",
    )
    return any(m in msg for m in markers)


# Object types that count as wall art (matched case-insensitively as substrings).
# Kept specific enough to avoid false-positives like "bar cart" (contains "art"
# but not "artwork"/"art_piece").
ART_KEYWORDS = (
    "painting", "picture", "poster", "artwork", "mural",
    "wall_art", "wallart", "wall art", "wall hanging", "wall-hanging",
    "framed_print", "framed print", "canvas",
    "framed_art", "framed art", "framed_photo", "framed photo",
    "picture_frame", "picture frame", "poster_frame", "poster frame",
    "wall_decor", "wall decor", "wall_painting", "wall painting",
    "photo_frame", "photo frame", "photograph",
    "tapestry", "art_piece", "art piece",
    "wall_clock", "wall clock",
    # Pin / memo / cork / chalk / white / peg boards mounted on the wall
    "corkboard", "cork_board", "cork board",
    "pinboard", "pin_board", "pin board",
    "bulletin_board", "bulletin board",
    "notice_board", "notice board", "noticeboard",
    "memo_board", "memo board", "memoboard",
    "message_board", "message board",
    "chalkboard", "chalk_board", "chalk board",
    "whiteboard", "white_board", "white board",
    "pegboard", "peg_board", "peg board",
    # Decorative fabric / soft wall decor
    "wreath", "garland", "macrame", "dream_catcher", "dreamcatcher",
    "wall_tapestry", "wall tapestry", "wall_textile", "wall textile",
    "banner", "pennant", "flag_banner", "flag banner",
    # Plaques / signs / letters
    "plaque", "wall_sign", "wall sign", "wall_letter", "wall letter",
    "monogram",
    # Misc wall-mounted display
    "display_case", "display case", "shadow_box", "shadow box",
    "wall_panel", "wall panel",
)

# Fallback heuristic: object names like "Art_West", "Art_South_R", "Art_North_2"
# strongly imply wall art even when object_type doesn't match any keyword.
# Strict pattern: must start with "art" followed by a SEPARATOR (_, -, space),
# so we don't match "artichoke", "artifact", "artisan", "artistic", etc.
ART_NAME_PREFIX_RE = re.compile(r"^art[_\s\-]", re.IGNORECASE)

# Object types / names that indicate a floor rug / carpet. Matched as
# lowercase substrings against both object_type and name. Kept tight enough
# not to catch unrelated items ("rugby ball", "carpet-cleaner", etc. are not
# expected in home scenes but the substring check is still reasonable).
RUG_KEYWORDS = (
    "area rug", "area_rug", "arearug",
    "floor rug", "floor_rug",
    "throw rug", "throw_rug",
    "runner rug", "runner_rug", "rug_runner", "hallway rug",
    "kilim", "persian rug", "persian_rug", "oriental rug",
    "oriental_rug", "dhurrie", "dhurry",
    "accent rug", "accent_rug",
    "carpet", "carpet_tile", "shag rug", "shag_rug",
    "floor mat", "floor_mat", "doormat", "door mat", "door_mat",
    "rug",  # last (broadest) — keep after the more specific entries
)

# When relying ONLY on name-prefix (no keyword hit), reject objects whose
# object_type clearly identifies a piece of furniture. Prevents false
# positives like "Art_Deco_Table" (object_type="side table") or
# "Art_Nouveau_Chair" (object_type="accent chair").
NON_ART_OBJECT_TYPE_WORDS = (
    "table", "chair", "sofa", "couch", "bed", "desk", "cabinet", "shelf",
    "bookshelf", "lamp", "light", "cart", "stool", "bench", "ottoman",
    "dresser", "wardrobe", "nightstand", "rug", "carpet", "vase",
    "sculpture", "plant", "pot", "basket", "bin", "tray", "bowl",
    "pillow", "cushion", "blanket", "book", "magazine", "tv", "speaker",
    "fan", "clock", "radio", "fridge", "oven", "stove", "sink", "toilet",
    "bathtub", "counter", "island", "sideboard", "console", "pouf",
    "headboard", "footboard", "mattress", "mirror",
)


# =============================================================================
# Runner
# =============================================================================
class StageTextureRunner:
    """Generate real texture maps for floor / walls / wall art and inject
    ShaderNodeTexImage into the Blender code from Stage 10."""

    def __init__(
        self,
        image_path: Optional[str] = None,
        material_code_path: Optional[str] = None,
        output_dir: Optional[str] = None,
        use_memory: bool = True,
        verbose: bool = True,
        memory_file: str = "agent_memory.jsonl",
        # nanobanana config
        texture_model: str = DEFAULT_IMAGE_MODEL,
        texture_base_url: str = DEFAULT_BASE_URL,
        texture_api_key: str = DEFAULT_API_KEY,
        aspect_ratio: str = "1:1",
        image_size: str = "2K",
        parallel: int = 3,
        max_retries: int = 5,
        delay_sec: float = 1.5,
        # art selection
        max_wall_arts: int = 12,
        # wall visual intensity override: None -> parse from WALL_MAT_DATA /
        # material_config / default; "subtle" | "bold" | "mural_like" -> force.
        wall_intensity: Optional[str] = None,
    ):
        self.image_path = image_path
        self.material_code_path = material_code_path
        self.output_dir = output_dir or os.path.join(
            current_dir, "pipeline_output", "stage11_texture"
        )
        self.use_memory = use_memory
        self.verbose = verbose

        self.memory = (
            Memory(workspace_dir=current_dir, memory_file=memory_file)
            if use_memory
            else None
        )

        self.texture_model = texture_model
        self.texture_base_url = texture_base_url
        self.texture_api_key = texture_api_key
        self.aspect_ratio = aspect_ratio
        self.image_size = image_size
        self.parallel = max(1, parallel)
        self.max_retries = max_retries
        self.delay_sec = delay_sec
        self.max_wall_arts = max_wall_arts

        # Normalize intensity override. Empty / unknown -> None (use fallback chain).
        self.wall_intensity_override: Optional[str] = None
        if wall_intensity:
            v = str(wall_intensity).strip().lower()
            if v in ("subtle", "bold", "mural_like"):
                self.wall_intensity_override = v

        # Loaded data
        self.material_code: Optional[str] = None
        self.material_config: Dict = {}
        self.floor_material: Dict = {}
        self.wall_material: Dict = {}
        self.describe_data: Dict = {}
        self.scene_info: Dict = {"scene_w": 8.0, "scene_d": 6.0, "wall_h": 2.8}

        # Generated during run()
        self.images_dir = os.path.join(self.output_dir, "images")
        self.wall_arts: List[Dict] = []   # [{id, object_type, desc, size, path, ...}]
        self.rugs: List[Dict] = []        # [{id, object_type, desc, size (w,d), path, ...}]
        self.manifest: Dict = {}

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
            "image": "🖼️",
            "save": "💾",
        }.get(level, "")
        print(f"{prefix} {msg}")

    # ------------------------------------------------------------------
    # Load upstream data
    # ------------------------------------------------------------------
    def _load_data(self) -> bool:
        self._log("Loading data...", "step")

        # 1) Material code
        if self.material_code_path and os.path.exists(self.material_code_path):
            with open(self.material_code_path, "r", encoding="utf-8") as f:
                self.material_code = f.read()
            self._log(f"Material code: {self.material_code_path}", "success")
        elif self.use_memory:
            entry = self.memory.get_latest(stage="stage10_material", type="result")
            if entry:
                code_path = entry.metadata.get("output_file")
                if code_path and os.path.exists(code_path):
                    with open(code_path, "r", encoding="utf-8") as f:
                        self.material_code = f.read()
                    self._log(
                        f"Material code: from Memory - {code_path}", "success"
                    )
                elif entry.content:
                    self.material_code = entry.content
                    self._log("Material code: from Memory (content)", "success")

                cfg_path = entry.metadata.get("config_file")
                if cfg_path and os.path.exists(cfg_path):
                    try:
                        with open(cfg_path, "r", encoding="utf-8") as f:
                            self.material_config = json.load(f)
                    except Exception:
                        pass

        if not self.material_code:
            run_dir = os.path.dirname(self.output_dir)
            for candidate in (
                os.path.join(run_dir, "stage10_material", "material_output.py"),
                os.path.join(run_dir, "stage6_geometry", "geometry_output.py"),
            ):
                if os.path.exists(candidate):
                    with open(candidate, "r", encoding="utf-8") as f:
                        self.material_code = f.read()
                    self._log(f"Material code: sibling - {candidate}", "success")
                    break

        if not self.material_code:
            self._log("Material code not found (need Stage 10 first)", "error")
            return False

        # 2) Floor / wall material JSON
        #    Prefer material_config.json; otherwise parse from code constants.
        self.floor_material = self.material_config.get("floor_material", {})
        self.wall_material = self.material_config.get("wall_material", {})
        if not self.floor_material or not self.wall_material:
            fm, wm = self._parse_floor_wall_from_code(self.material_code)
            self.floor_material = self.floor_material or fm
            self.wall_material = self.wall_material or wm

        # 3) Describe data
        if self.use_memory:
            d_entry = self.memory.get_latest(stage="stage5_describe", type="result")
            if d_entry:
                try:
                    self.describe_data = json.loads(d_entry.content)
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

        # 4) Scene info (dimensions) extracted from code
        self.scene_info = self._extract_scene_info(self.material_code)
        self._log(
            f"Scene: {self.scene_info['scene_w']}m x {self.scene_info['scene_d']}m "
            f"x {self.scene_info['wall_h']}m",
            "info",
        )

        # 5) Image path for logging / potential reference
        if not self.image_path and self.use_memory:
            s1 = self.memory.get_latest(stage="stage1", type="result")
            if s1:
                self.image_path = s1.metadata.get("image_path")

        return True

    # ------------------------------------------------------------------
    # Helpers: parse floor/wall JSON from generated code constants
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_floor_wall_from_code(code: str) -> Tuple[Dict, Dict]:
        floor, wall = {}, {}
        for name, target in (("FLOOR_MAT_DATA", floor), ("WALL_MAT_DATA", wall)):
            m = re.search(rf"{name}\s*=\s*\{{(.*?)\n\}}", code, re.DOTALL)
            if not m:
                continue
            block = m.group(1)
            for key, rx in (
                ("material_type", r'"material_type"\s*:\s*"([^"]+)"'),
                ("pattern", r'"pattern"\s*:\s*"([^"]+)"'),
                ("finish", r'"finish"\s*:\s*"([^"]+)"'),
                ("description", r'"description"\s*:\s*"([^"]+)"'),
                ("wall_visual_intensity", r'"wall_visual_intensity"\s*:\s*"([^"]+)"'),
            ):
                mm = re.search(rx, block)
                if mm:
                    target[key] = mm.group(1)
            for key, rx in (
                ("roughness", r'"roughness"\s*:\s*([\d.]+)'),
                ("pattern_scale", r'"pattern_scale"\s*:\s*([\d.]+)'),
                ("bump_strength", r'"bump_strength"\s*:\s*([\d.]+)'),
            ):
                mm = re.search(rx, block)
                if mm:
                    target[key] = float(mm.group(1))
            mm = re.search(r'"base_color"\s*:\s*\(([^)]+)\)', block)
            if mm:
                vals = [float(x.strip()) for x in mm.group(1).split(",") if x.strip()]
                target["base_color"] = vals
        return floor, wall

    @staticmethod
    def _extract_scene_info(code: str) -> Dict:
        info = {"scene_w": 8.0, "scene_d": 6.0, "wall_h": 2.8}
        for key, rx in (
            ("scene_w", r"SCENE_W\s*=\s*([\d.]+)"),
            ("scene_d", r"SCENE_D\s*=\s*([\d.]+)"),
            ("wall_h", r"WALL_H\s*=\s*([\d.]+)"),
        ):
            m = re.search(rx, code)
            if m:
                info[key] = float(m.group(1))
        return info

    @staticmethod
    def _parse_wall_obj_from_code(
        code: Optional[str], obj_name: str,
    ) -> Optional[Tuple[
        Tuple[float, float, float],
        Optional[Tuple[float, float, float]],
        Tuple[float, float, float],
    ]]:
        """Resolve ``(location, dimensions, rotation_euler)`` for ``obj_name``
        in the Stage 10 base code.

        For wall-mounted objects, Stage 4's ``_enforce_wall_orientation``
        post-process locks ``rotation`` to ``(0, 0, 0)`` and re-orders
        ``dimensions`` so that the SHORTEST axis aligns with the wall's
        normal. That makes ``dimensions`` literally equal to the world AABB,
        and the smallest axis there unambiguously identifies the wall
        (north/south = Y-thin, east/west = X-thin). This is the signal we
        use to spawn the artwork plane.
        (Aside: ``obj.dimensions = dim`` AFTER a non-zero rotation does NOT
        give world AABB == dim — Blender just rescales each axis without
        undoing the rotation. That's exactly the bug Stage 4 enforces away
        by locking rotation to identity for wall objects.)

        Tuples like ``(math.pi/2, 0, -math.pi/2)`` and
        ``(SCENE_W/2 + WALL_T/2, 0, 0)`` are evaluated in a sandbox bound to
        ``math`` plus any top-level numeric constants found in the code.
        ``dimensions`` may be ``None`` for ``create_detailed_object`` calls
        which do not pass a size argument.
        """
        if not code or not obj_name:
            return None
        safe = re.escape(obj_name)

        ns: Dict[str, Any] = {"math": math}
        for cm in re.finditer(
            r"^\s*([A-Z_][A-Z0-9_]*)\s*=\s*([\-\d\.]+)\s*$",
            code,
            re.MULTILINE,
        ):
            try:
                ns[cm.group(1)] = float(cm.group(2))
            except ValueError:
                continue

        def _eval_tuple(expr: Optional[str]) -> Optional[Tuple[float, float, float]]:
            if not expr:
                return None
            try:
                val = eval(expr, {"__builtins__": {}}, ns)
            except Exception:
                return None
            if not isinstance(val, (tuple, list)) or len(val) < 3:
                return None
            try:
                return (float(val[0]), float(val[1]), float(val[2]))
            except (TypeError, ValueError):
                return None

        m = re.search(
            rf'create_(?:box|cylinder)\s*\(\s*["\']{safe}["\']\s*,\s*'
            rf'(\([^()]*\))\s*,\s*(\([^()]*\))'
            rf'(?:\s*,\s*rotation\s*=\s*(\([^()]*\)))?',
            code,
        )
        if m:
            loc = _eval_tuple(m.group(1))
            dim = _eval_tuple(m.group(2))
            rot = _eval_tuple(m.group(3)) or (0.0, 0.0, 0.0)
            if loc and dim:
                return (loc, dim, rot)

        m = re.search(
            rf'create_detailed_object\s*\(\s*["\']{safe}["\']\s*,\s*'
            rf'location\s*=\s*(\([^()]*\))'
            rf'(?:\s*,\s*rotation\s*=\s*(\([^()]*\)))?',
            code,
        )
        if m:
            loc = _eval_tuple(m.group(1))
            rot = _eval_tuple(m.group(2)) or (0.0, 0.0, 0.0)
            if loc:
                return (loc, None, rot)

        return None

    # ------------------------------------------------------------------
    # Wall art detection from Stage 5 describe data
    # ------------------------------------------------------------------
    def _detect_wall_arts(self) -> List[Dict]:
        objects = self.describe_data.get("objects", [])
        arts: List[Dict] = []
        for obj in objects:
            otype = (obj.get("object_type") or "").lower()
            raw_name = obj.get("name") or ""
            name = raw_name.lower()
            text = otype + " " + name
            if "mirror" in text:
                continue
            kw_hit = any(kw in text for kw in ART_KEYWORDS)
            # Fallback: obvious art-naming conventions like "Art_West", "Art-02"
            name_hit = bool(ART_NAME_PREFIX_RE.match(raw_name))
            # Guard: if ONLY name matched (no keyword hit) and object_type
            # clearly identifies furniture, treat as a false positive
            # (e.g. "Art_Deco_Table" / "side table").
            if name_hit and not kw_hit and any(
                w in otype for w in NON_ART_OBJECT_TYPE_WORDS
            ):
                continue
            if not (kw_hit or name_hit):
                continue

            # ------- Resolve location & dimensions -----------------------
            # Source of truth for an object that already exists in the Stage
            # 9 base code is its ``create_box`` call. Stage 4's
            # ``_enforce_wall_orientation`` post-process locks ``rotation``
            # to (0,0,0) and reorders ``dimensions`` so the smallest axis
            # aligns with the wall normal — therefore ``dimensions`` IS the
            # world AABB and its smallest axis identifies the wall.
            # Stage 5 describe data is used purely as a fallback.
            base_tx = self._parse_wall_obj_from_code(self.material_code, raw_name)

            loc_raw = obj.get("location") or obj.get("position") or (0.0, 0.0, 1.5)
            if isinstance(loc_raw, dict):
                lx = float(loc_raw.get("x", 0.0))
                ly = float(loc_raw.get("y", 0.0))
                lz = float(loc_raw.get("z", 1.5))
            elif isinstance(loc_raw, (list, tuple)) and len(loc_raw) >= 3:
                lx = float(loc_raw[0])
                ly = float(loc_raw[1])
                lz = float(loc_raw[2])
            else:
                lx, ly, lz = 0.0, 0.0, 1.5

            dims = obj.get("dimensions") or {}
            if isinstance(dims, (list, tuple)) and len(dims) >= 2:
                wx = float(dims[0])
                wy = float(dims[1])
                wz = float(dims[2]) if len(dims) > 2 else 0.04
            elif isinstance(dims, dict):
                wx = float(dims.get("x", dims.get("w", 1.0)))
                wy = float(dims.get("y", dims.get("d", 0.04)))
                wz = float(dims.get("z", dims.get("h", 0.8)))
            else:
                wx, wy, wz = 1.0, 0.04, 0.8

            if base_tx:
                base_loc, base_dim, _base_rot = base_tx
                if base_loc:
                    lx, ly, lz = base_loc
                if base_dim:
                    wx, wy, wz = (
                        float(base_dim[0]),
                        float(base_dim[1]),
                        float(base_dim[2]),
                    )

            _axis_dims = (abs(wx), abs(wy), abs(wz))
            _thickness_axis = min(range(3), key=lambda i: _axis_dims[i])
            _thickness_guess = _axis_dims[_thickness_axis]
            _dims_sorted = sorted(_axis_dims)
            _side_small, _side_big = _dims_sorted[1], _dims_sorted[2]
            _is_thin_plate = (
                _thickness_guess <= 0.10
                and _side_small >= 0.15
                and _side_big >= 0.15
            )

            # ------- Wall direction --------------------------------------
            # Stage 4 guarantees rotation=(0,0,0) for wall objects and dim
            # ordered so smallest axis aligns with the wall normal. So the
            # smallest axis here IS the wall-normal axis. The previous code
            # rotated the dims by Stage 5's ``rotation_z`` once more, which
            # for any 90°-multiple rotation flipped the thickness axis and
            # mis-classified east/west posters as north/south (or vice
            # versa). Use the AABB axis directly here.
            if _is_thin_plate and _thickness_axis in (0, 1):
                is_east_west = (_thickness_axis == 0)
            elif abs(wx) <= 0.10 and abs(wx) < abs(wy):
                # Cube-ish but slightly thinner along x → likely east/west.
                is_east_west = True
            elif abs(wy) <= 0.10 and abs(wy) < abs(wx):
                is_east_west = False
            else:
                # No clear thickness axis → fall back to the location
                # heuristic, but only consider it valid when the object is
                # actually close to a wall (≥ ~⅔ of half-room).
                is_east_west = abs(lx) > abs(ly)

            # ------- Frame size ------------------------------------------
            if _is_thin_plate and _thickness_axis == 0:
                frame_w = abs(wy)
                frame_h = abs(wz)
                thickness = max(0.02, min(0.08, _thickness_guess))
            elif _is_thin_plate and _thickness_axis == 1:
                frame_w = abs(wx)
                frame_h = abs(wz)
                thickness = max(0.02, min(0.08, _thickness_guess))
            else:
                horizontal_big = max(abs(wx), abs(wy))
                horizontal_small = min(abs(wx), abs(wy))
                frame_w = horizontal_big
                frame_h = (
                    abs(wz) if abs(wz) > 0.05
                    else max(horizontal_big, horizontal_small)
                )
                thickness = max(
                    0.02,
                    min(0.08, horizontal_small if horizontal_small < 0.2 else 0.03),
                )

            if frame_w < 0.05:
                frame_w = max(_side_big, 0.30)
            if frame_h < 0.05:
                frame_h = max(_side_big, 0.30)

            # ------- Plane rotation --------------------------------------
            # The wall-art mesh in spawn_wall_arts() is a unit quad lying in
            # the XZ-plane with its face normal pointing along -Y (vertices
            # ordered BL→BR→TR→TL, right-hand rule). We rotate around Z so
            # the visible face points INTO the room:
            #   North wall (y>0): visible -> -Y -> rot_z =   0°
            #   South wall (y<0): visible -> +Y -> rot_z = 180°
            #   East  wall (x>0): visible -> -X -> rot_z = -90°  (Rz(-90)·-Ŷ = -X̂)
            #   West  wall (x<0): visible -> +X -> rot_z = +90°  (Rz(+90)·-Ŷ = +X̂)
            # The previous implementation had east/west swapped, which
            # mirrored the artwork left-right (only invisible because the
            # plane is double-sided in Cycles).
            if is_east_west:
                rot_z = -90.0 if lx > 0 else 90.0
            else:
                rot_z = 180.0 if ly < 0 else 0.0
            wall_rotation = (0.0, 0.0, rot_z)

            arts.append({
                "id": obj.get("name", f"art_{len(arts)}"),
                "object_type": obj.get("object_type", "painting"),
                "description": obj.get("description", ""),
                "color": obj.get("color_description", ""),
                "material": obj.get("material_description", ""),
                "location": (lx, ly, lz),
                # Override any Stage 5 rotation: our wall-derived rotation
                # is the only one that keeps the plane flat against the wall.
                "rotation": wall_rotation,
                "frame_w": frame_w,
                "frame_h": frame_h,
                "thickness": thickness,
            })

            if len(arts) >= self.max_wall_arts:
                break

        self._log(
            f"Wall art detected: {len(arts)}/{self.max_wall_arts} max",
            "info",
        )
        return arts

    # ------------------------------------------------------------------
    # Rug detection
    # ------------------------------------------------------------------
    def _detect_rugs(self) -> List[Dict]:
        """Find floor rug / carpet objects that deserve a real texture map.

        Mirrors ``_detect_wall_arts`` but with different filters:
          * must match a rug keyword in object_type or name
          * must look like a flat floor piece: z-height clearly shorter than
            both horizontal sides (so we don't accidentally texture a tall
            "carpet roll" prop or a "rug cleaner" machine)
          * reports a 2-D size (width, depth) for aspect-aware prompting
        """
        objects = self.describe_data.get("objects", []) or []
        rugs: List[Dict] = []
        for obj in objects:
            otype = (obj.get("object_type") or "").lower()
            raw_name = obj.get("name") or ""
            name = raw_name.lower()
            text = otype + " " + name
            if not any(kw in text for kw in RUG_KEYWORDS):
                continue

            dims = obj.get("dimensions") or {}
            if isinstance(dims, (list, tuple)) and len(dims) >= 2:
                wx = float(dims[0])
                wy = float(dims[1])
                wz = float(dims[2]) if len(dims) > 2 else 0.02
            elif isinstance(dims, dict):
                wx = float(dims.get("x", dims.get("w", 1.0)))
                wy = float(dims.get("y", dims.get("d", 1.0)))
                wz = float(dims.get("z", dims.get("h", 0.02)))
            else:
                wx, wy, wz = 1.0, 1.0, 0.02

            wx, wy, wz = abs(wx), abs(wy), abs(wz)
            # Floor-rug geometry sanity: must be clearly a flat piece.
            # z-height must be the smallest axis AND well under both
            # horizontal sides (rugs are <=5 cm thick in our pipeline).
            if wz >= 0.15 or wz >= min(wx, wy) * 0.5:
                continue
            # Require a non-trivial footprint so we don't texture a tiny
            # sample swatch or a coaster mistagged as a "mat".
            if wx < 0.30 or wy < 0.30:
                continue

            loc_raw = obj.get("location") or obj.get("position") or (0.0, 0.0, 0.01)
            if isinstance(loc_raw, dict):
                lx = float(loc_raw.get("x", 0.0))
                ly = float(loc_raw.get("y", 0.0))
                lz = float(loc_raw.get("z", 0.01))
            elif isinstance(loc_raw, (list, tuple)) and len(loc_raw) >= 3:
                lx = float(loc_raw[0])
                ly = float(loc_raw[1])
                lz = float(loc_raw[2])
            else:
                lx, ly, lz = 0.0, 0.0, 0.01

            rugs.append({
                "id": obj.get("name", f"rug_{len(rugs)}"),
                "object_type": obj.get("object_type", "area rug"),
                "description": obj.get("description", ""),
                "color": obj.get("color_description", ""),
                "material": obj.get("material_description", ""),
                "location": (lx, ly, lz),
                "width": wx,   # along world-x before parent rotation
                "depth": wy,   # along world-y before parent rotation
                "thickness": wz,
            })

        self._log(f"Rugs detected: {len(rugs)}", "info")
        return rugs

    # ------------------------------------------------------------------
    # Prompt builders
    # ------------------------------------------------------------------
    def _prompt_floor(self) -> str:
        fm = self.floor_material
        style = (self.describe_data.get("room_style") or {})
        mt = fm.get("material_type", "hardwood")
        pat = _sanitize_description(fm.get("pattern", "plank"))
        raw_desc = fm.get("description", "") or f"{mt} floor in {pat} pattern"
        desc = _sanitize_description(raw_desc)
        bc = fm.get("base_color", [0.5, 0.35, 0.2])
        style_name = _sanitize_style_name(style.get("style_name", "modern"))
        mood = _sanitize_description(style.get("mood", ""))
        return (
            f"Seamless tileable PBR albedo texture of {mt} floor surface, "
            f"{pat} pattern, {desc}. "
            f"Dominant color roughly RGB({bc[0]:.2f}, {bc[1]:.2f}, {bc[2]:.2f}) in linear space. "
            f"Captured strictly top-down orthographic, 2K resolution, sharp fine details. "
            f"No lighting, no shadows, no perspective distortion, neutral white balance, "
            f"evenly lit. The texture must tile seamlessly when repeated. "
            f"Style context: {style_name} {mood}. No watermarks, no text."
        )

    def _prompt_floor_fallback(self) -> str:
        """Recitation-safe floor prompt: only material + color, no style names."""
        fm = self.floor_material
        mt = fm.get("material_type", "hardwood")
        bc = fm.get("base_color", [0.5, 0.35, 0.2])
        return (
            f"Seamless tileable PBR albedo texture of plain generic {mt} floor. "
            f"Dominant color roughly RGB({bc[0]:.2f}, {bc[1]:.2f}, {bc[2]:.2f}) in linear space. "
            f"Top-down orthographic, 2K resolution, uniform natural grain, "
            f"no lighting, no shadows, no perspective distortion, neutral white balance. "
            f"The texture must tile seamlessly when repeated. "
            f"Generic unbranded surface. No patterns referencing any existing design, "
            f"no text, no watermark, no logo."
        )

    def _wall_intensity(self) -> str:
        """Return normalized wall visual intensity: 'subtle' | 'bold' | 'mural_like'.

        Resolution order (highest priority first):
          1. Constructor override ``wall_intensity`` (typically wired from CLI)
          2. ``wall_visual_intensity`` parsed from WALL_MAT_DATA in scene code
          3. ``wall_visual_intensity`` on ``material_config``
          4. Default ``'subtle'`` — i.e. a plain wall unless the user asks
             explicitly for something more decorative. This reflects feedback
             that the previous ``'bold'`` default produced over-ornate walls.
        """
        if self.wall_intensity_override:
            return self.wall_intensity_override
        raw = (
            self.wall_material.get("wall_visual_intensity")
            or self.material_config.get("wall_visual_intensity")
            or "subtle"
        )
        val = str(raw).strip().lower()
        if val not in ("subtle", "bold", "mural_like"):
            val = "subtle"
        return val

    def _prompt_wall(self) -> str:
        wm = self.wall_material
        style = (self.describe_data.get("room_style") or {})
        mt = wm.get("material_type", "paint")
        finish = wm.get("finish", "matte")
        bc = wm.get("base_color", [0.95, 0.93, 0.9])
        intensity = self._wall_intensity()

        # Subtle mode: plain wall. Drop description / style / mood entirely
        # (they tend to drag the model back toward motifs like "floral
        # wallpaper" or "damask"). Actively negate patterns.
        if intensity == "subtle":
            return (
                f"Seamless tileable PBR albedo texture of a plain flat interior "
                f"wall, {mt} with {finish} finish. Uniform solid color with only "
                f"very faint natural surface grain. "
                f"Dominant color roughly RGB({bc[0]:.2f}, {bc[1]:.2f}, {bc[2]:.2f}) in linear space. "
                f"Front-facing flat, orthographic, 2K resolution. "
                f"ABSOLUTELY NO decorative patterns, NO wallpaper motifs, NO "
                f"damask, NO floral, NO geometric ornament, NO panels, NO "
                f"mouldings, NO borders, NO medallions, NO relief, NO metallic "
                f"accents. Plain, calm, minimal. "
                f"No lighting, no shadows, no perspective distortion, neutral "
                f"white balance. The texture must tile seamlessly when "
                f"repeated. No text, no watermark, no logo, no signatures."
            )

        # Bold / mural_like still pull in description + style for richness.
        raw_desc = wm.get("description", "") or f"{finish} {mt} wall surface"
        desc = _sanitize_description(raw_desc)
        style_name = _sanitize_style_name(style.get("style_name", "modern"))
        mood = _sanitize_description(style.get("mood", ""))

        base = (
            f"Seamless tileable PBR albedo texture of interior wall surface, "
            f"{mt} material with {finish} finish. {desc}. "
            f"Dominant color roughly RGB({bc[0]:.2f}, {bc[1]:.2f}, {bc[2]:.2f}) in linear space. "
            f"Captured front-facing flat, orthographic, 2K resolution. "
            f"No lighting, no shadows, no perspective distortion, neutral white balance. "
            f"The texture must tile seamlessly when repeated. "
        )

        if intensity == "mural_like":
            flourish = (
                f"Highly decorative feature wall: large-scale ornamental motif, "
                f"symmetrical composition with a central medallion and radiating "
                f"geometry, panel layout with ornamental borders, embossed "
                f"plaster-like relief with subtle metallic sheen in the raised areas, "
                f"hand-painted appearance. "
                f"Style context: {style_name} {mood}. "
                f"Original non-representational design, generic ornamental, "
                f"not copying any specific artwork, wallpaper brand, or designer. "
            )
        else:  # bold
            flourish = (
                f"Decorative feature wall: large-scale repeating ornamental motif, "
                f"embossed plaster-like relief, subtle metallic sheen in raised "
                f"areas, refined hand-crafted appearance. "
                f"Style context: {style_name} {mood}. "
                f"Original non-representational design, generic ornamental, "
                f"not copying any specific artwork, wallpaper brand, or designer. "
            )

        return base + flourish + "No watermarks, no text, no signatures."

    def _prompt_wall_medium(self) -> str:
        """Mid-tier fallback: keeps ornamental scale but drops all style names
        and free-text description. Meant for the first retry after an
        ``IMAGE_RECITATION`` refusal of the main prompt."""
        wm = self.wall_material
        mt = wm.get("material_type", "paint")
        finish = wm.get("finish", "matte")
        bc = wm.get("base_color", [0.95, 0.93, 0.9])
        intensity = self._wall_intensity()

        if intensity == "subtle":
            ornament = (
                "uniform solid color, only a very faint natural surface grain, "
                "no decorative patterns, no wallpaper motifs, no damask, no "
                "floral, no geometric ornament, no mouldings, no relief, "
            )
        else:
            ornament = (
                "large-scale abstract ornamental pattern with symmetric geometric "
                "shapes, soft embossed relief, gentle metallic highlights in the "
                "raised areas, generic unbranded decorative look, "
            )

        return (
            f"Seamless tileable PBR albedo texture of an interior wall, "
            f"{mt} with {finish} finish, {ornament}"
            f"dominant color roughly RGB({bc[0]:.2f}, {bc[1]:.2f}, {bc[2]:.2f}) in linear space. "
            f"Front-facing flat, orthographic, 2K resolution. "
            f"Original design, non-representational, not copying any existing "
            f"artwork, wallpaper brand, or designer. "
            f"No lighting, no shadows, no perspective distortion, neutral white balance. "
            f"The texture must tile seamlessly when repeated. "
            f"No text, no watermark, no logo, no signatures."
        )

    def _prompt_wall_fallback(self) -> str:
        """Recitation-safe wall prompt: plain material + color, no patterns, no style."""
        wm = self.wall_material
        mt = wm.get("material_type", "paint")
        finish = wm.get("finish", "matte")
        bc = wm.get("base_color", [0.95, 0.93, 0.9])
        return (
            f"Seamless tileable PBR albedo texture of a plain generic interior wall, "
            f"{mt} with {finish} finish, uniform solid surface with only very subtle "
            f"natural grain. "
            f"Dominant color roughly RGB({bc[0]:.2f}, {bc[1]:.2f}, {bc[2]:.2f}) in linear space. "
            f"Front-facing flat, orthographic, 2K resolution. "
            f"No decorative patterns, no figurative motifs, no geometric patterns, "
            f"no period or designer style references. "
            f"No lighting, no shadows, no perspective distortion, neutral white balance. "
            f"The texture must tile seamlessly when repeated. "
            f"Generic unbranded surface. No text, no watermark, no logo."
        )

    def _prompt_wall_art(self, art: Dict) -> str:
        style = (self.describe_data.get("room_style") or {})
        style_name = _sanitize_style_name(style.get("style_name", "modern"))
        palette = ", ".join(style.get("color_palette", []))
        obj_type = art.get("object_type", "painting")
        desc = _sanitize_description(art.get("description", ""))
        color = art.get("color", "")

        aspect = "portrait"
        try:
            if float(art.get("frame_w", 1)) > float(art.get("frame_h", 1)):
                aspect = "landscape"
        except Exception:
            pass

        # IMPORTANT: the rendered image is used as an image texture mapped onto
        # a thin rectangular plane in Blender. The plane itself provides the
        # physical frame; the IMAGE must be pure canvas content, full-bleed,
        # with NO visible frame, NO mat, NO white margin, NO surrounding wall.
        return (
            f"Full-bleed original {obj_type} artwork for a {style_name} interior. "
            f"{desc}. Color scheme aligned with: {color or palette}. "
            f"Orientation: {aspect}. "
            f"CRITICAL COMPOSITION: the artwork fills the ENTIRE rendered image "
            f"edge-to-edge. The rendered image IS the canvas content itself. "
            f"Absolutely NO visible picture frame, NO mat, NO border, NO white "
            f"margin, NO gray padding, NO background wall, NO drop shadow. "
            f"Photorealistic, crisp details, flat front-facing view, "
            f"camera perfectly perpendicular to the canvas plane, "
            f"no perspective distortion, no glare, no paper texture. "
            f"Original design, not reproducing any existing artwork, brand, "
            f"character, or designer. No text, no watermark, no signature."
        )

    def _prompt_wall_art_fallback(self, art: Dict) -> str:
        """Recitation-safe wall-art prompt: abstract, color-only, no style names.

        Also full-bleed: the entire image is canvas content, no frame.
        """
        obj_type = art.get("object_type", "painting")
        color = art.get("color", "") or ", ".join(
            (self.describe_data.get("room_style") or {}).get("color_palette", [])
        )

        aspect = "portrait"
        try:
            if float(art.get("frame_w", 1)) > float(art.get("frame_h", 1)):
                aspect = "landscape"
        except Exception:
            pass

        return (
            f"Full-bleed original abstract {obj_type} artwork, non-figurative "
            f"composition with simple shapes and gentle color gradients. "
            f"Color palette: {color or 'neutral warm tones'}. "
            f"Orientation: {aspect}. "
            f"CRITICAL COMPOSITION: the artwork fills the ENTIRE rendered image "
            f"edge-to-edge. The rendered image IS the canvas content itself. "
            f"Absolutely NO visible picture frame, NO mat, NO border, NO white "
            f"margin, NO gray padding, NO background wall, NO drop shadow. "
            f"Flat front-facing view, camera perpendicular to the canvas plane, "
            f"no perspective distortion, no glare, no paper texture. "
            f"Do not reproduce any existing artwork, brand, character, or designer. "
            f"No text, no watermark, no signature."
        )

    def _prompt_rug(self, rug: Dict) -> str:
        """Top-down rug / carpet texture prompt.

        The image is mapped onto a flat rectangular rug mesh and viewed
        from above in the final render, so it must be a pure rug surface:
        no floor context, no furniture shadow, no perspective.
        """
        style = (self.describe_data.get("room_style") or {})
        style_name = _sanitize_style_name(style.get("style_name", "modern"))
        palette = ", ".join(style.get("color_palette", []))
        obj_type = rug.get("object_type", "area rug")
        desc = _sanitize_description(rug.get("description", ""))
        color = rug.get("color", "") or palette
        material = _sanitize_description(rug.get("material", "")) or "wool"

        w = float(rug.get("width", 1.0))
        d = float(rug.get("depth", 1.0))
        aspect_word = (
            "roughly square" if abs(w - d) / max(w, d, 1e-3) < 0.1
            else ("landscape (wider than tall)" if w > d else "portrait (taller than wide)")
        )

        return (
            f"Full-bleed original {obj_type} texture for a {style_name} interior, "
            f"captured as a flat ORTHOGRAPHIC TOP-DOWN view of the rug surface. "
            f"{desc}. Colors aligned with: {color}. Material: {material}. "
            f"Proportions: {aspect_word}. "
            f"CRITICAL COMPOSITION: the rug fills the ENTIRE rendered image "
            f"edge-to-edge — the rendered image IS the rug pile itself. "
            f"Absolutely NO visible floor around the rug, NO furniture, NO "
            f"feet / legs, NO people, NO wall, NO margin, NO fringe spilling "
            f"off the edges, NO drop shadow, NO perspective foreshortening. "
            f"Camera perfectly perpendicular to the rug plane, as if scanned "
            f"flat. Soft, realistic pile and weave detail, natural wool / "
            f"fabric texture, gentle diffuse lighting, no glossy highlight. "
            f"Original design, not reproducing any existing rug, brand, "
            f"designer, character, or logo. No text, no watermark, no signature."
        )

    def _prompt_rug_fallback(self, rug: Dict) -> str:
        """Recitation-safe fallback rug prompt — abstract, color-led."""
        obj_type = rug.get("object_type", "rug")
        color = rug.get("color", "") or ", ".join(
            (self.describe_data.get("room_style") or {}).get("color_palette", [])
        )
        w = float(rug.get("width", 1.0))
        d = float(rug.get("depth", 1.0))
        aspect_word = (
            "roughly square" if abs(w - d) / max(w, d, 1e-3) < 0.1
            else ("landscape" if w > d else "portrait")
        )
        return (
            f"Full-bleed generic {obj_type} texture, flat orthographic "
            f"top-down view, gentle abstract repeat pattern in soft colors. "
            f"Color palette: {color or 'neutral warm tones'}. "
            f"Proportions: {aspect_word}. "
            f"CRITICAL COMPOSITION: the rug fills the ENTIRE rendered image "
            f"edge-to-edge. No visible floor, no furniture, no people, no "
            f"margin, no fringe off the edges, no drop shadow, no perspective. "
            f"Camera perpendicular to the rug plane. Realistic fabric pile. "
            f"Do not reproduce any existing rug, brand, designer, or logo. "
            f"No text, no watermark, no signature."
        )

    # ------------------------------------------------------------------
    # Image generation (parallel)
    # ------------------------------------------------------------------
    def _generate_one(
        self, name: str, prompt: str, filepath: str,
        aspect_ratio: Optional[str] = None,
        fallback_prompt: Optional[str] = None,
        fallback_prompts: Optional[List[str]] = None,
        *,
        max_retries: Optional[int] = None,
        request_timeout: Optional[int] = None,
        serial_pass: bool = False,
    ) -> Tuple[str, bool, str]:
        """Generate a single texture image with retries. Returns (name, ok, path).

        Fallback chain: on an ``IMAGE_RECITATION`` refusal we advance to the
        next entry in ``fallback_prompts`` (or ``fallback_prompt`` for
        backward compatibility) instead of wasting a retry on the same text.
        For transient errors we keep the current active prompt and just
        back off. Total attempts are still capped at ``max_retries``.

        Parameters
        ----------
        max_retries : optional override of ``self.max_retries`` — used by the
            recovery pass to give failed items a second life with more tries.
        request_timeout : per-HTTP-request timeout in seconds. When ``None``
            we default to 300s (the legacy behaviour). The recovery pass
            supplies a larger value so a slow-but-responding server can finish.
        serial_pass : True when this call is part of the recovery pass. Only
            affects logging so the user can tell the passes apart.
        """
        if os.path.exists(filepath):
            self._log(f"SKIP (exists): {os.path.basename(filepath)}", "info")
            return name, True, filepath

        # Build fallback queue: prefer list, else wrap single, else empty.
        if fallback_prompts:
            fb_queue: List[str] = [p for p in fallback_prompts if p]
        elif fallback_prompt:
            fb_queue = [fallback_prompt]
        else:
            fb_queue = []

        import time
        import random
        active_prompt = prompt
        tier = 0  # 0 = main, 1 = first fallback, ...
        total_tiers = 1 + len(fb_queue)

        retries = int(max_retries if max_retries is not None else self.max_retries)
        retries = max(1, retries)
        timeout_s = int(request_timeout if request_timeout is not None else 300)
        tag_prefix = "recovery " if serial_pass else ""

        for attempt in range(1, retries + 1):
            try:
                img_bytes = _call_gemini_image(
                    base_url=self.texture_base_url,
                    api_key=self.texture_api_key,
                    model=self.texture_model,
                    prompt_text=active_prompt,
                    aspect_ratio=aspect_ratio or self.aspect_ratio,
                    image_size=self.image_size,
                    timeout=timeout_s,
                )
                Path(filepath).write_bytes(img_bytes)
                tier_tag = f" (fallback tier {tier}/{total_tiers - 1})" if tier > 0 else ""
                serial_tag = " [recovery]" if serial_pass else ""
                self._log(
                    f"Saved {os.path.basename(filepath)} "
                    f"({len(img_bytes) // 1024} KB){tier_tag}{serial_tag}",
                    "image",
                )
                return name, True, filepath
            except Exception as e:
                is_recitation = _is_recitation_error(e)
                is_transient = _is_transient_network_error(e)
                if attempt < retries:
                    if is_recitation and fb_queue:
                        active_prompt = fb_queue.pop(0)
                        tier += 1
                        self._log(
                            f"[{name}] recitation-refused on {tag_prefix}attempt "
                            f"{attempt}/{retries}; switching to "
                            f"fallback tier {tier}/{total_tiers - 1}",
                            "warning",
                        )
                        wait = self.delay_sec
                    elif is_recitation and not fb_queue:
                        self._log(
                            f"[{name}] recitation-refused on {tag_prefix}attempt "
                            f"{attempt}/{retries}; fallback chain "
                            f"exhausted, retrying current prompt anyway",
                            "warning",
                        )
                        wait = self.delay_sec
                    elif is_transient:
                        # Gentle-but-growing wait with jitter so the queue
                        # doesn't thunder-herd the server the moment it
                        # recovers. Cap the base at 30s.
                        base = min(30.0, self.delay_sec * (2 ** (attempt - 1)) * 4.0)
                        wait = base * random.uniform(0.75, 1.35)
                        self._log(
                            f"[{name}] {tag_prefix}attempt {attempt}/{retries} "
                            f"network-timeout/5xx: {type(e).__name__}. "
                            f"retrying in {wait:.1f}s",
                            "warning",
                        )
                    else:
                        wait = self.delay_sec * (2 ** (attempt - 1))
                        wait *= random.uniform(0.85, 1.25)
                        self._log(
                            f"[{name}] {tag_prefix}attempt {attempt}/{retries} "
                            f"failed: {e}. retrying in {wait:.1f}s",
                            "warning",
                        )
                    time.sleep(wait)
                else:
                    self._log(f"[{name}] FAILED after {retries} attempts: {e}", "error")
                    return name, False, ""

        return name, False, ""

    def _generate_all_textures(self) -> Dict[str, str]:
        """Kick off all texture generations in parallel. Returns {task_name: filepath}."""
        os.makedirs(self.images_dir, exist_ok=True)

        # Task layout: (name, primary_prompt, filepath, aspect_ratio, fallback_chain)
        tasks: List[Tuple[str, str, str, Optional[str], List[str]]] = []

        # Floor and wall (always generated)
        tasks.append((
            "floor",
            self._prompt_floor(),
            os.path.join(self.images_dir, "floor.png"),
            "1:1",
            [self._prompt_floor_fallback()],
        ))
        tasks.append((
            "wall",
            self._prompt_wall(),
            os.path.join(self.images_dir, "wall.png"),
            "1:1",
            # Two-tier fallback: medium (keeps ornament, drops style names) →
            # plain (uniform generic wall). Lets us recover from recitation
            # without dropping straight to a boring wall.
            [self._prompt_wall_medium(), self._prompt_wall_fallback()],
        ))

        # Wall arts
        for art in self.wall_arts:
            safe_id = re.sub(r"[^A-Za-z0-9_]", "_", str(art["id"]))
            fp = os.path.join(self.images_dir, f"art_{safe_id}.png")
            art["path"] = fp  # record planned path
            fw = float(art.get("frame_w", 1.0) or 1.0)
            fh = float(art.get("frame_h", 1.0) or 1.0)
            ar = "16:9" if fw >= 1.4 * fh else ("9:16" if fh >= 1.4 * fw else "1:1")
            tasks.append((
                f"art:{safe_id}",
                self._prompt_wall_art(art),
                fp,
                ar,
                [self._prompt_wall_art_fallback(art)],
            ))

        # Rugs (flat top-down textures)
        for rug in self.rugs:
            safe_id = re.sub(r"[^A-Za-z0-9_]", "_", str(rug["id"]))
            fp = os.path.join(self.images_dir, f"rug_{safe_id}.png")
            rug["path"] = fp  # record planned path
            rw = float(rug.get("width", 1.0) or 1.0)
            rd = float(rug.get("depth", 1.0) or 1.0)
            ar = "16:9" if rw >= 1.4 * rd else ("9:16" if rd >= 1.4 * rw else "1:1")
            tasks.append((
                f"rug:{safe_id}",
                self._prompt_rug(rug),
                fp,
                ar,
                [self._prompt_rug_fallback(rug)],
            ))

        results: Dict[str, str] = {}
        if not GEMINI_IMAGE_AVAILABLE:
            self._log("_call_gemini_image unavailable, skipping generation", "error")
            return results

        self._log(
            f"Generating {len(tasks)} textures ({self.parallel} parallel, "
            f"up to {self.max_retries} retries each)...",
            "step",
        )

        # Pass 1: parallel — the common happy path.
        failed_tasks: List[Tuple[str, str, str, Optional[str], List[str]]] = []
        with ThreadPoolExecutor(max_workers=self.parallel) as ex:
            futures = {
                ex.submit(
                    self._generate_one, name, prompt, fp, ar, None, fb_chain
                ): (name, prompt, fp, ar, fb_chain)
                for (name, prompt, fp, ar, fb_chain) in tasks
            }
            for fut in as_completed(futures):
                task_meta = futures[fut]
                name, ok, path = fut.result()
                if ok:
                    results[name] = path
                else:
                    failed_tasks.append(task_meta)

        # Pass 2: serial recovery — for each still-failed item, re-run once
        # with single-threaded pacing, more retries, and a longer per-request
        # timeout. This is cheap and rescues most flaky-network failures
        # without polluting the parallel log.
        if failed_tasks:
            self._log(
                f"Recovery pass: {len(failed_tasks)} item(s) failed in parallel "
                f"pass, retrying serially (1-at-a-time, longer timeout)...",
                "warning",
            )
            recovery_retries = max(self.max_retries, 5)
            recovery_timeout = 600  # give the slow server twice as long per call
            import time as _time
            for (name, prompt, fp, ar, fb_chain) in failed_tasks:
                if os.path.exists(fp):
                    results[name] = fp
                    continue
                rec_name, ok, path = self._generate_one(
                    name, prompt, fp, ar,
                    fallback_prompts=fb_chain,
                    max_retries=recovery_retries,
                    request_timeout=recovery_timeout,
                    serial_pass=True,
                )
                if ok:
                    results[rec_name] = path
                else:
                    self._log(
                        f"[{rec_name}] UNRECOVERABLE after parallel + recovery passes",
                        "error",
                    )
                # Small spacer between serial calls to let the server breathe.
                _time.sleep(1.0)

        return results

    # ------------------------------------------------------------------
    # Code rewriting
    # ------------------------------------------------------------------
    def _rewrite_material_code(
        self, results: Dict[str, str]
    ) -> str:
        """Rewrite Stage 10 code: swap create_floor_material / create_wall_material
        with texture-image versions, remove detected wall-art legacy objects,
        and append wall-art plane rendering."""
        code = self.material_code

        scene_w = self.scene_info["scene_w"]
        scene_d = self.scene_info["scene_d"]

        # Tiling: one texture tile per ~2 meters
        floor_tile_x = max(1.0, scene_w / 2.0)
        floor_tile_y = max(1.0, scene_d / 2.0)
        wall_tile_x = max(1.0, scene_w / 2.0)
        wall_tile_y = max(1.0, self.scene_info["wall_h"] / 2.0)

        texture_dir = os.path.abspath(self.images_dir)
        # Always serialize absolute paths: Blender may be launched from any cwd.
        floor_path = os.path.abspath(results["floor"]) if results.get("floor") else ""
        wall_path = os.path.abspath(results["wall"]) if results.get("wall") else ""

        # 1) Insert a top-level TEXTURE_DIR / FLOOR_TEX / WALL_TEX constant
        header_block = (
            "\n# ==============================================================================\n"
            "# STAGE TEXTURE - Real texture maps generated via nanobanana\n"
            "# ==============================================================================\n"
            "import os as _tex_os\n"
            f"TEXTURE_DIR = r\"{texture_dir}\"\n"
            f"FLOOR_TEXTURE = r\"{floor_path}\"\n"
            f"WALL_TEXTURE = r\"{wall_path}\"\n"
            f"FLOOR_TILE = ({floor_tile_x:.3f}, {floor_tile_y:.3f})\n"
            f"WALL_TILE = ({wall_tile_x:.3f}, {wall_tile_y:.3f})\n"
        )
        code = self._insert_after_imports(code, header_block)

        # 2) Replace create_floor_material / create_wall_material bodies
        new_floor_fn = self._build_floor_material_fn()
        new_wall_fn = self._build_wall_material_fn()
        code = self._replace_function(code, "create_floor_material", new_floor_fn)
        code = self._replace_function(code, "create_wall_material", new_wall_fn)

        # 3) Wall-art: remove legacy simple objects and append plane renderer
        if self.wall_arts:
            art_ids = [a["id"] for a in self.wall_arts if a.get("path")]
            code = self._remove_legacy_art_calls(code, art_ids)
            code = self._append_wall_art_block(code, results)

        # 4) Rugs: append texture-swap block that overrides the rug's
        #    Stage 10 solid-color material with a real image texture.
        if self.rugs:
            code = self._append_rug_texture_block(code, results)

        return code

    @staticmethod
    def _insert_after_imports(code: str, block: str) -> str:
        """Insert block after the leading top-level import section.

        Do not search for the last import in the whole file: downstream stages
        can append helper imports near the bottom, after the
        ``if __name__ == "__main__"`` execution block. Inserting texture
        constants there makes ``create_floor_material`` reference
        ``FLOOR_TEXTURE`` before it has been defined.
        """
        lines = code.split("\n")
        last_import = -1
        i = 0

        # Skip a leading module docstring, if present. This keeps the texture
        # constants after the file's initial imports rather than before the
        # docstring.
        while i < len(lines) and not lines[i].strip():
            i += 1
        if i < len(lines) and lines[i].lstrip().startswith(('"""', "'''")):
            quote = lines[i].lstrip()[:3]
            if lines[i].lstrip().count(quote) >= 2:
                i += 1
            else:
                i += 1
                while i < len(lines):
                    if quote in lines[i]:
                        i += 1
                        break
                    i += 1

        for j in range(i, len(lines)):
            line = lines[j]
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if line.startswith(" ") or line.startswith("\t"):
                continue
            if stripped.startswith("import ") or stripped.startswith("from "):
                last_import = j
                continue
            if last_import >= 0:
                break
            # A non-import before any leading imports means this file has no
            # conventional import block. Put constants at the front.
            break

        if last_import < 0:
            # No leading import section; still make the constants available
            # before any top-level execution by placing them at the front.
            return block + "\n" + code
        return "\n".join(lines[: last_import + 1]) + "\n" + block + "\n" + "\n".join(
            lines[last_import + 1 :]
        )

    @staticmethod
    def _replace_function(code: str, func_name: str, new_body: str) -> str:
        """Replace `def func_name(...): ...` block with new_body.
        Fall back to appending if the function is not present."""
        pattern = re.compile(
            rf"(^|\n)def {re.escape(func_name)}\s*\([^)]*\):\s*\n"
            rf"(?:(?:[ \t]+.*|\s*)\n)+?(?=\ndef |\n# ===|\nif __name__|\Z)",
            re.MULTILINE,
        )
        if pattern.search(code):
            return pattern.sub("\n" + new_body.rstrip() + "\n\n", code, count=1)
        return code + "\n\n" + new_body + "\n"

    # ------------------------------------------------------------------
    # New material function bodies (image-based)
    # ------------------------------------------------------------------
    def _build_floor_material_fn(self) -> str:
        return '''def create_floor_material():
    """Floor material with real image texture (Stage Texture).

    Falls back to a solid PBR shader if the texture file is missing.
    """
    d = FLOOR_MAT_DATA
    mat = bpy.data.materials.new(name="Floor_PBR_Tex")
    mat.use_nodes = True
    nt = mat.node_tree
    nodes = nt.nodes
    links = nt.links
    nodes.clear()
    out = nodes.new(type="ShaderNodeOutputMaterial")
    out.location = (800, 0)
    bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")
    bsdf.location = (500, 0)
    links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    bsdf.inputs["Roughness"].default_value = d.get("roughness", 0.5)
    bsdf.inputs["Metallic"].default_value = d.get("metallic", 0.0)
    spec_in = bsdf.inputs.get("Specular IOR Level") or bsdf.inputs.get("Specular")
    if spec_in:
        spec_in.default_value = d.get("specular", 0.5)

    tex_path = FLOOR_TEXTURE
    if tex_path and _tex_os.path.exists(tex_path):
        tex_coord = nodes.new(type="ShaderNodeTexCoord")
        tex_coord.location = (-400, 0)
        mapping = nodes.new(type="ShaderNodeMapping")
        mapping.location = (-150, 0)
        sx, sy = FLOOR_TILE
        mapping.inputs["Scale"].default_value = (sx, sy, 1.0)
        tex_img = nodes.new(type="ShaderNodeTexImage")
        tex_img.location = (100, 0)
        try:
            tex_img.image = bpy.data.images.load(tex_path, check_existing=True)
        except Exception:
            tex_img.image = None
        tex_img.projection = "BOX"
        tex_img.projection_blend = 0.2
        links.new(tex_coord.outputs["Generated"], mapping.inputs["Vector"])
        links.new(mapping.outputs["Vector"], tex_img.inputs["Vector"])
        links.new(tex_img.outputs["Color"], bsdf.inputs["Base Color"])
        bs = d.get("bump_strength", 0.0)
        if bs and bs > 0:
            bump = nodes.new(type="ShaderNodeBump")
            bump.location = (250, -300)
            bump.inputs["Strength"].default_value = bs
            links.new(tex_img.outputs["Color"], bump.inputs["Height"])
            links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    else:
        bsdf.inputs["Base Color"].default_value = d.get(
            "base_color", (0.5, 0.35, 0.2, 1.0)
        )
    return mat'''

    def _build_wall_material_fn(self) -> str:
        return '''def create_wall_material():
    """Wall material with real image texture (Stage Texture).

    Falls back to a solid PBR shader if the texture file is missing.
    """
    d = WALL_MAT_DATA
    mat = bpy.data.materials.new(name="Wall_PBR_Tex")
    mat.use_nodes = True
    nt = mat.node_tree
    nodes = nt.nodes
    links = nt.links
    nodes.clear()
    out = nodes.new(type="ShaderNodeOutputMaterial")
    out.location = (800, 0)
    bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")
    bsdf.location = (500, 0)
    links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    bsdf.inputs["Roughness"].default_value = d.get("roughness", 0.9)
    bsdf.inputs["Metallic"].default_value = d.get("metallic", 0.0)
    spec_in = bsdf.inputs.get("Specular IOR Level") or bsdf.inputs.get("Specular")
    if spec_in:
        spec_in.default_value = d.get("specular", 0.3)

    tex_path = WALL_TEXTURE
    if tex_path and _tex_os.path.exists(tex_path):
        tex_coord = nodes.new(type="ShaderNodeTexCoord")
        tex_coord.location = (-400, 0)
        mapping = nodes.new(type="ShaderNodeMapping")
        mapping.location = (-150, 0)
        sx, sy = WALL_TILE
        mapping.inputs["Scale"].default_value = (sx, sy, 1.0)
        tex_img = nodes.new(type="ShaderNodeTexImage")
        tex_img.location = (100, 0)
        try:
            tex_img.image = bpy.data.images.load(tex_path, check_existing=True)
        except Exception:
            tex_img.image = None
        tex_img.projection = "BOX"
        tex_img.projection_blend = 0.2
        links.new(tex_coord.outputs["Generated"], mapping.inputs["Vector"])
        links.new(mapping.outputs["Vector"], tex_img.inputs["Vector"])
        links.new(tex_img.outputs["Color"], bsdf.inputs["Base Color"])
        bs = d.get("bump_strength", 0.0)
        if bs and bs > 0:
            bump = nodes.new(type="ShaderNodeBump")
            bump.location = (250, -300)
            bump.inputs["Strength"].default_value = bs
            links.new(tex_img.outputs["Color"], bump.inputs["Height"])
            links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    else:
        bsdf.inputs["Base Color"].default_value = d.get(
            "base_color", (0.95, 0.93, 0.9, 1.0)
        )
    return mat'''

    # ------------------------------------------------------------------
    # Wall art rewriting
    # ------------------------------------------------------------------
    def _remove_legacy_art_calls(self, code: str, art_ids: List[str]) -> str:
        """Remove create_detailed_object / create_box / create_cylinder calls
        that construct the placeholder wall-art objects we are replacing."""
        if not art_ids:
            return code

        lines = code.split("\n")
        out: List[str] = []
        skip_block = False
        block_indent = 0
        matched_ids = set()

        for line in lines:
            if skip_block:
                # Skip continuation lines of a bracketed call
                stripped = line.strip()
                if stripped.endswith(")") or stripped.endswith("),"):
                    skip_block = False
                continue

            matched = False
            for aid in art_ids:
                aid_q = re.escape(aid)
                if re.search(
                    rf'create_(?:detailed_object|box|cylinder|wall_art_plane)\s*\(\s*["\']{aid_q}["\']',
                    line,
                ):
                    matched = True
                    matched_ids.add(aid)
                    break

            if matched:
                # If call spans multiple lines, skip until we see the closing paren
                if line.count("(") > line.count(")"):
                    skip_block = True
                continue

            out.append(line)

        if matched_ids:
            self._log(
                f"Removed {len(matched_ids)} legacy art placeholder(s): "
                f"{', '.join(sorted(matched_ids))}",
                "info",
            )
        return "\n".join(out)

    def _append_wall_art_block(self, code: str, results: Dict[str, str]) -> str:
        """Append WALL_ART dict + helper fn + a call inside run_layout_engine()."""
        art_entries: List[str] = []
        for art in self.wall_arts:
            aid = art["id"]
            safe_id = re.sub(r"[^A-Za-z0-9_]", "_", str(aid))
            key = f"art:{safe_id}"
            path = results.get(key, "")
            if not path:
                continue
            path = os.path.abspath(path)

            loc = art.get("location") or (0.0, 0.0, 1.5)
            rot = art.get("rotation") or (0.0, 0.0, 0.0)
            loc_tup = self._tuple3_py_literal(loc, default=(0.0, 0.0, 1.5))
            rot_tup = self._tuple3_py_literal(rot, default=(0.0, 0.0, 0.0))
            fw = float(art.get("frame_w", 1.0) or 1.0)
            fh = float(art.get("frame_h", 1.0) or 1.0)
            th = float(art.get("thickness", 0.03) or 0.03)
            art_entries.append(
                f'    "{aid}": {{"path": r"{path}", '
                f'"size": ({fw:.3f}, {fh:.3f}), "thickness": {th:.3f}, '
                f'"location": {loc_tup}, "rotation": {rot_tup}}},'
            )

        if not art_entries:
            return code

        art_dict_block = (
            "\n# ==============================================================================\n"
            "# WALL ART (Stage Texture)\n"
            "# ==============================================================================\n"
            "WALL_ART = {\n" + "\n".join(art_entries) + "\n}\n"
        )

        helper_fn = self._build_wall_art_fn()

        injection = art_dict_block + "\n" + helper_fn + "\n\n"

        # Insert before run_layout_engine if present; else append.
        marker = "def run_layout_engine("
        pos = code.find(marker)
        if pos > 0:
            code = code[:pos] + injection + code[pos:]
        else:
            code = code + "\n" + injection

        # Add a call to spawn wall arts at the end of run_layout_engine body.
        code = self._inject_wall_art_call(code)
        return code

    @staticmethod
    def _tuple3_py_literal(val, default=(0.0, 0.0, 0.0)) -> str:
        try:
            if isinstance(val, (list, tuple)) and len(val) >= 3:
                return f"({float(val[0]):.4f}, {float(val[1]):.4f}, {float(val[2]):.4f})"
            if isinstance(val, dict):
                x = val.get("x", val.get(0, default[0]))
                y = val.get("y", val.get(1, default[1]))
                z = val.get("z", val.get(2, default[2]))
                return f"({float(x):.4f}, {float(y):.4f}, {float(z):.4f})"
        except Exception:
            pass
        return f"({default[0]:.4f}, {default[1]:.4f}, {default[2]:.4f})"

    def _build_wall_art_fn(self) -> str:
        # NOTE: Prior implementation used bmesh.ops.create_cube without adding
        # a UV layer, so ShaderNodeTexCoord -> UV output was all zeros and
        # every face sampled a single pixel (objects looked uniform-colored).
        # We now build a single-face plane with EXPLICIT UVs (0..1 each axis)
        # so the image texture actually shows. Backface culling is off in
        # Cycles by default so both sides of the plane display the image.
        return '''def spawn_wall_arts(collection=None):
    """Spawn wall art planes (textured quads) with explicit UV mapping.

    Each entry in WALL_ART is rendered as a single-face rectangular plane
    whose 4 UV corners are pinned to (0,0)-(1,1) so the corresponding
    image fills the whole quad. Missing images fall back to a neutral
    solid color so the scene still builds.
    """
    import bmesh as _bmesh
    for name, art in WALL_ART.items():
        w, h = art["size"]
        loc = art.get("location", (0.0, 0.0, 1.5))
        rot = art.get("rotation", (0.0, 0.0, 0.0))

        # Unit plane in the XZ plane at y=0. Vertex winding is
        # BL -> BR -> TR -> TL so UV coords line up left-to-right,
        # bottom-to-top with the source image.
        mesh = bpy.data.meshes.new(name + "_mesh")
        bm = _bmesh.new()
        v_bl = bm.verts.new((-0.5, 0.0, -0.5))
        v_br = bm.verts.new(( 0.5, 0.0, -0.5))
        v_tr = bm.verts.new(( 0.5, 0.0,  0.5))
        v_tl = bm.verts.new((-0.5, 0.0,  0.5))
        face = bm.faces.new((v_bl, v_br, v_tr, v_tl))
        uv_layer = bm.loops.layers.uv.new("UVMap")
        face.loops[0][uv_layer].uv = (0.0, 0.0)
        face.loops[1][uv_layer].uv = (1.0, 0.0)
        face.loops[2][uv_layer].uv = (1.0, 1.0)
        face.loops[3][uv_layer].uv = (0.0, 1.0)
        bm.to_mesh(mesh)
        bm.free()

        obj = bpy.data.objects.new(name, mesh)
        obj.location = loc
        # Use scale directly: obj.dimensions would crash on a flat plane
        # because the Y bbox is 0 (division-by-zero inside Blender).
        obj.scale = (w, 1.0, h)
        obj.rotation_euler = tuple(
            r * 3.14159265 / 180.0 if abs(r) > 6.3 else r for r in rot
        )

        mat = bpy.data.materials.new(name="Art_" + name)
        mat.use_nodes = True
        nt = mat.node_tree
        for n in list(nt.nodes):
            nt.nodes.remove(n)
        out_node = nt.nodes.new(type="ShaderNodeOutputMaterial")
        bsdf = nt.nodes.new(type="ShaderNodeBsdfPrincipled")
        bsdf.inputs["Roughness"].default_value = 0.55
        nt.links.new(bsdf.outputs["BSDF"], out_node.inputs["Surface"])
        img_path = art["path"]
        if img_path and _tex_os.path.exists(img_path):
            tex_img = nt.nodes.new(type="ShaderNodeTexImage")
            try:
                tex_img.image = bpy.data.images.load(img_path, check_existing=True)
            except Exception:
                tex_img.image = None
            tex_coord = nt.nodes.new(type="ShaderNodeTexCoord")
            nt.links.new(tex_coord.outputs["UV"], tex_img.inputs["Vector"])
            nt.links.new(tex_img.outputs["Color"], bsdf.inputs["Base Color"])
        else:
            bsdf.inputs["Base Color"].default_value = (0.85, 0.82, 0.78, 1.0)

        obj.data.materials.append(mat)
        if collection:
            collection.objects.link(obj)
        else:
            bpy.context.scene.collection.objects.link(obj)'''

    def _inject_wall_art_call(self, code: str) -> str:
        """Insert `spawn_wall_arts()` near the end of run_layout_engine()."""
        # Note: check for the CALL site only, not the `def spawn_wall_arts(`
        # definition, otherwise this guard is a no-op.
        if re.search(r"^\s+spawn_wall_arts\s*\(", code, re.MULTILINE):
            return code  # already injected

        lines = code.split("\n")
        fn_start = -1
        fn_indent = 0
        for i, line in enumerate(lines):
            if re.match(r"\s*def run_layout_engine\s*\(", line):
                fn_start = i
                fn_indent = len(line) - len(line.lstrip())
                break
        if fn_start < 0:
            # No engine; append a top-level call after WALL_ART block.
            return code + "\n\nspawn_wall_arts()\n"

        # Find end of fn body (next def / top-level line at same-or-less indent).
        end_idx = len(lines)
        for j in range(fn_start + 1, len(lines)):
            ln = lines[j]
            if ln.strip() == "":
                continue
            indent = len(ln) - len(ln.lstrip())
            if indent <= fn_indent and j > fn_start + 1:
                end_idx = j
                break

        insert_at = end_idx
        indent_str = " " * (fn_indent + 4)
        lines.insert(insert_at, f"{indent_str}spawn_wall_arts()")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Rug texture injection (post-hoc material swap on existing meshes)
    # ------------------------------------------------------------------
    def _append_rug_texture_block(
        self, code: str, results: Dict[str, str]
    ) -> str:
        """Append RUG_TEXTURES + apply_rug_textures() + a call in the engine.

        Unlike wall art (where we SPAWN new planes), the rug meshes already
        exist from Stage 10 (``create_detailed_object("Area_Rug", ...)``).
        We simply look up each rug's child parts by name after run, and
        replace their material with an image-textured one — non-invasive
        and immune to Stage 10's exact variable naming."""
        rug_entries: List[str] = []
        for rug in self.rugs:
            rid = rug["id"]
            safe_id = re.sub(r"[^A-Za-z0-9_]", "_", str(rid))
            key = f"rug:{safe_id}"
            path = results.get(key, "")
            if not path:
                continue
            path = os.path.abspath(path)
            w = float(rug.get("width", 1.0) or 1.0)
            d = float(rug.get("depth", 1.0) or 1.0)
            # The rug PNG is a single, complete top-down picture of the rug
            # (prompt explicitly forces "rug fills entire image edge-to-edge").
            # It must be stretched 1:1 onto the rug mesh, NOT tiled.
            tx, ty = 1.0, 1.0
            rug_entries.append(
                f'    "{rid}": {{"path": r"{path}", '
                f'"size": ({w:.3f}, {d:.3f}), '
                f'"tile": ({tx:.3f}, {ty:.3f})}},'
            )
        if not rug_entries:
            return code

        rug_dict_block = (
            "\n# ==============================================================================\n"
            "# RUG TEXTURES (Stage Texture)\n"
            "# ==============================================================================\n"
            "RUG_TEXTURES = {\n" + "\n".join(rug_entries) + "\n}\n"
        )
        helper_fn = self._build_rug_apply_fn()
        injection = rug_dict_block + "\n" + helper_fn + "\n\n"

        marker = "def run_layout_engine("
        pos = code.find(marker)
        if pos > 0:
            code = code[:pos] + injection + code[pos:]
        else:
            code = code + "\n" + injection

        code = self._inject_rug_apply_call(code)
        return code

    @staticmethod
    def _build_rug_apply_fn() -> str:
        """Build a helper that rewires matching rug objects' materials to
        use a real image texture, with safe fallback if the image is
        missing. We match rug parts by prefix (``<rug_name>_``) because
        ``create_detailed_object`` names each sub-mesh ``<parent>_<part>``.
        """
        return '''def _build_rug_image_material(mat_name, tex_path, tile):
    """Create a PrincipledBSDF material with an image texture for a rug."""
    mat = bpy.data.materials.new(name=mat_name)
    mat.use_nodes = True
    nt = mat.node_tree
    for _n in list(nt.nodes):
        nt.nodes.remove(_n)
    out = nt.nodes.new(type="ShaderNodeOutputMaterial")
    out.location = (800, 0)
    bsdf = nt.nodes.new(type="ShaderNodeBsdfPrincipled")
    bsdf.location = (500, 0)
    bsdf.inputs["Roughness"].default_value = 0.9
    bsdf.inputs["Metallic"].default_value = 0.0
    spec_in = bsdf.inputs.get("Specular IOR Level") or bsdf.inputs.get("Specular")
    if spec_in:
        spec_in.default_value = 0.2
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    if tex_path and _tex_os.path.exists(tex_path):
        tex_coord = nt.nodes.new(type="ShaderNodeTexCoord")
        tex_coord.location = (-400, 0)
        mapping = nt.nodes.new(type="ShaderNodeMapping")
        mapping.location = (-150, 0)
        # The rug image is one complete top-down picture of the rug, so we
        # stretch it 1:1 across the mesh AABB (no tiling, no repeats).
        # `tile` is preserved for backwards compat but is forced to (1,1) by
        # the writer; we still apply it so any non-(1,1) override works.
        sx, sy = tile
        mapping.inputs["Scale"].default_value = (sx, sy, 1.0)
        tex_img = nt.nodes.new(type="ShaderNodeTexImage")
        tex_img.location = (100, 0)
        try:
            tex_img.image = bpy.data.images.load(tex_path, check_existing=True)
        except Exception:
            tex_img.image = None
        # FLAT projection on the AABB-normalized "Generated" coordinates
        # maps the entire image once onto the rug's top-down footprint.
        # Thin side faces (rug is ~1-3 cm) get streaks but are invisible
        # from any normal viewing angle.
        tex_img.projection = "FLAT"
        tex_img.extension = "EXTEND"
        nt.links.new(tex_coord.outputs["Generated"], mapping.inputs["Vector"])
        nt.links.new(mapping.outputs["Vector"], tex_img.inputs["Vector"])
        nt.links.new(tex_img.outputs["Color"], bsdf.inputs["Base Color"])
    else:
        bsdf.inputs["Base Color"].default_value = (0.75, 0.72, 0.65, 1.0)
    return mat


def apply_rug_textures():
    """Override the Stage 10 solid-color rug material with a real texture.

    Scans `bpy.data.objects` for any object whose name matches one of the
    keys in ``RUG_TEXTURES`` (either exactly or as a parent prefix used by
    ``create_detailed_object`` for its sub-meshes) and swaps every material
    slot on those objects to the textured rug material.
    """
    for rug_name, info in RUG_TEXTURES.items():
        tex_path = info.get("path", "")
        tile = info.get("tile", (1.0, 1.0))
        mat = _build_rug_image_material(
            "Rug_Tex_" + rug_name, tex_path, tile
        )
        prefix = rug_name + "_"
        matched = 0
        for obj in list(bpy.data.objects):
            if obj.type != "MESH":
                continue
            # Match either the parent empty (no mesh) or any sub-mesh that
            # create_detailed_object named "<rug>_<part>".
            if obj.name == rug_name or obj.name.startswith(prefix):
                if obj.data is None or not hasattr(obj.data, "materials"):
                    continue
                slots = obj.data.materials
                if len(slots) == 0:
                    slots.append(mat)
                else:
                    for i in range(len(slots)):
                        slots[i] = mat
                matched += 1
        if matched == 0:
            print("[rug_texture] WARNING: no meshes matched '" + rug_name + "'")'''

    def _inject_rug_apply_call(self, code: str) -> str:
        """Insert `apply_rug_textures()` near the end of run_layout_engine()."""
        if re.search(r"^\s+apply_rug_textures\s*\(", code, re.MULTILINE):
            return code  # already injected

        lines = code.split("\n")
        fn_start = -1
        fn_indent = 0
        for i, line in enumerate(lines):
            if re.match(r"\s*def run_layout_engine\s*\(", line):
                fn_start = i
                fn_indent = len(line) - len(line.lstrip())
                break
        if fn_start < 0:
            return code + "\n\napply_rug_textures()\n"

        end_idx = len(lines)
        for j in range(fn_start + 1, len(lines)):
            ln = lines[j]
            if ln.strip() == "":
                continue
            indent = len(ln) - len(ln.lstrip())
            if indent <= fn_indent and j > fn_start + 1:
                end_idx = j
                break

        indent_str = " " * (fn_indent + 4)
        lines.insert(end_idx, f"{indent_str}apply_rug_textures()")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Residual syntax fix: empty `if ...:` followed (possibly after blank
    # lines) by a dedented statement. Inserts a `pass` body.
    # ------------------------------------------------------------------
    @staticmethod
    def _fix_empty_if_blocks(code: str) -> str:
        lines = code.split("\n")
        out: List[str] = []
        i = 0
        n = len(lines)
        while i < n:
            line = lines[i]
            stripped_right = line.rstrip()
            # Match a lone `if ...:` line with indentation
            m = re.match(r"^([ \t]+)(if\s.*:)\s*$", stripped_right)
            if m:
                if_indent = len(m.group(1).expandtabs())
                # Peek ahead: skip blank lines to find the next non-blank line
                j = i + 1
                while j < n and lines[j].strip() == "":
                    j += 1
                needs_pass = True
                if j < n:
                    next_line = lines[j]
                    next_indent = len(
                        next_line[: len(next_line) - len(next_line.lstrip())]
                        .expandtabs()
                    )
                    if next_indent > if_indent:
                        needs_pass = False

                if needs_pass:
                    out.append(line)
                    out.append(m.group(1) + "    pass")
                    i += 1
                    continue

            out.append(line)
            i += 1
        return "\n".join(out)

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    def _save_outputs(self, code: str, results: Dict[str, str]) -> str:
        os.makedirs(self.output_dir, exist_ok=True)

        # Defensive: fix residual empty `if show_direction:` blocks left behind
        # by arrow cleaning. Upstream's fixer requires `if` and `return` on
        # consecutive lines; we also handle cases with a blank line between them.
        try:
            from blender_code_syntax_fix import (
                fix_empty_if_show_direction_before_return,
            )
            code = fix_empty_if_show_direction_before_return(code)
        except Exception as e:
            self._log(f"Syntax fixer unavailable: {e}", "warning")
        code = self._fix_empty_if_blocks(code)

        try:
            compile(code, "<texture_output>", "exec")
            self._log(
                f"Syntax OK ({code.count(chr(10)) + 1} lines)", "success"
            )
        except SyntaxError as e:
            self._log(
                f"Syntax warning at line {e.lineno}: {e.msg} "
                f"(code still written for inspection)",
                "warning",
            )

        out_path = os.path.join(self.output_dir, "texture_output.py")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(code)
        self._log(f"Code saved: {out_path}", "save")

        self.manifest = {
            "floor": results.get("floor", ""),
            "wall": results.get("wall", ""),
            "wall_art": [
                {
                    "id": art["id"],
                    "path": results.get(
                        f"art:{re.sub(r'[^A-Za-z0-9_]', '_', str(art['id']))}", ""
                    ),
                    "size": [art.get("frame_w"), art.get("frame_h")],
                    "location": art.get("location"),
                    "rotation": art.get("rotation"),
                }
                for art in self.wall_arts
            ],
            "rugs": [
                {
                    "id": rug["id"],
                    "path": results.get(
                        f"rug:{re.sub(r'[^A-Za-z0-9_]', '_', str(rug['id']))}", ""
                    ),
                    "size": [rug.get("width"), rug.get("depth")],
                    "location": rug.get("location"),
                }
                for rug in self.rugs
            ],
            "texture_dir": os.path.abspath(self.images_dir),
            "model": self.texture_model,
            "generated_at": datetime.now().isoformat(),
        }
        mf_path = os.path.join(self.output_dir, "texture_manifest.json")
        with open(mf_path, "w", encoding="utf-8") as f:
            json.dump(self.manifest, f, ensure_ascii=False, indent=2)
        self._log(f"Manifest saved: {mf_path}", "save")

        if self.use_memory:
            self.memory.add(
                stage="stage11_texture",
                type="result",
                content=code,
                metadata={
                    "title": "Stage Texture - NanoBanana real textures",
                    "summary": (
                        f"floor+wall"
                        f"{'+' + str(len(self.wall_arts)) + 'art' if self.wall_arts else ''}"
                        f"{'+' + str(len(self.rugs)) + 'rug' if self.rugs else ''}, "
                        f"dir={os.path.basename(self.images_dir)}"
                    ),
                    "output_file": out_path,
                    "manifest_file": mf_path,
                    "texture_dir": os.path.abspath(self.images_dir),
                    "image_path": self.image_path,
                },
                tags=["stage11_texture", "blender_code", "nanobanana", "textures"],
            )
            self._log("Saved to Memory", "success")

        return out_path

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------
    def run(self) -> Tuple[bool, Optional[str]]:
        print("\n" + "=" * 60)
        print("🖼️  Stage Texture - Real Texture Generation (nanobanana)")
        print("=" * 60)

        if not self._load_data():
            return False, None

        self.wall_arts = self._detect_wall_arts()
        self.rugs = self._detect_rugs()

        results = self._generate_all_textures()
        if "floor" not in results or "wall" not in results:
            self._log(
                "Critical textures missing (floor/wall). Continuing with fallbacks.",
                "warning",
            )

        code = self._rewrite_material_code(results)
        out_path = self._save_outputs(code, results)
        return True, out_path


# =============================================================================
# CLI
# =============================================================================
def _parse_args():
    p = argparse.ArgumentParser(
        description="Stage Texture - generate real texture maps via nanobanana "
                    "and inject them into the Stage 10 Blender script."
    )
    p.add_argument("--image", help="Reference image path (optional, pulled from Memory if absent)")
    p.add_argument("--material-code", help="Path to material_output.py (defaults to Memory)")
    p.add_argument("--output-dir", help="Output dir (default: pipeline_output/stage11_texture)")
    p.add_argument("--memory-file", default="agent_memory.jsonl")
    p.add_argument("--no-memory", action="store_true")
    p.add_argument("--model", default=DEFAULT_IMAGE_MODEL)
    p.add_argument("--base-url", default=DEFAULT_BASE_URL)
    p.add_argument("--api-key", default=DEFAULT_API_KEY)
    p.add_argument("--image-size", default="2K")
    p.add_argument("--parallel", type=int, default=3)
    p.add_argument("--max-retries", type=int, default=5,
                   help="Max retries per texture in the parallel pass "
                        "(recovery pass always uses >=5).")
    p.add_argument("--max-wall-arts", type=int, default=6)
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def main():
    args = _parse_args()
    runner = StageTextureRunner(
        image_path=args.image,
        material_code_path=args.material_code,
        output_dir=args.output_dir,
        use_memory=not args.no_memory,
        verbose=not args.quiet,
        memory_file=args.memory_file,
        texture_model=args.model,
        texture_base_url=args.base_url,
        texture_api_key=args.api_key,
        image_size=args.image_size,
        parallel=args.parallel,
        max_retries=args.max_retries,
        max_wall_arts=args.max_wall_arts,
    )
    ok, out = runner.run()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
