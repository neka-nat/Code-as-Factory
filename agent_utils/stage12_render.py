"""Stage Render - Lighting & Render Settings Generation (Stage 12)

This stage ONLY handles lighting and Cycles render configuration.
Materials are handled entirely by Stage 10 (stage10_material.py).
"""
import os
import sys
import json
import base64
import argparse

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.join(current_dir, "stage3"))

from stage3.core import LLMClient, PromptManager, extract_python_from_response
from memory import Memory
from langchain_core.messages import HumanMessage, SystemMessage


class StageRenderRunner:
    """Stage 12 - Generate lighting and Cycles render settings.

    Reads scene code from upstream stages (stage10_material > stage6_geometry >
    stage4 > stage3), analyses the reference image for lighting, and injects
    a ``setup_lighting_and_render()`` function into the scene code.

    This stage does NOT touch materials — that responsibility belongs to
    Stage 10 (StageMaterialRunner).
    """

    def __init__(
        self,
        image_path: str = None,
        scene_code_path: str = None,
        output_dir: str = None,
        use_memory: bool = True,
        verbose: bool = True,
        memory_file: str = "agent_memory.jsonl",
        blender_path: str = "/Applications/Blender.app/Contents/MacOS/Blender",
        max_iterations: int = 0,
        target_score: float = 0.75,
        model: str = None,
        base_url: str = None,
        api_key: str = None,
    ):
        self.image_path = image_path
        self.scene_code_path = scene_code_path
        self.output_dir = output_dir or os.path.join(current_dir, "pipeline_output", "stage12_render")
        self.use_memory = use_memory
        self.verbose = verbose
        self.blender_path = blender_path
        self.max_iterations = max_iterations
        self.target_score = target_score

        self.memory = Memory(workspace_dir=current_dir, memory_file=memory_file) if use_memory else None
        self.prompts = PromptManager()
        self.llm = LLMClient(model=model, base_url=base_url, api_key=api_key)

        self.stage1_json = None
        self.stage2_json = None
        self.scene_code = None
        self.lighting_config = None

    def _log(self, msg: str, level: str = "info"):
        if self.verbose:
            prefix = {
                "info": "ℹ️",
                "success": "✅",
                "warning": "⚠️",
                "error": "❌",
                "step": "📋",
                "light": "💡",
            }.get(level, "")
            print(f"{prefix} {msg}")

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _is_python_code(text: str) -> bool:
        stripped = text.lstrip()
        if stripped.startswith('{') or stripped.startswith('['):
            return False
        python_markers = ['import ', 'def ', 'class ', 'bpy.', 'from ']
        return any(m in text[:2000] for m in python_markers)

    def _encode_image(self, path: str) -> tuple:
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        ext = os.path.splitext(path)[1].lower()
        mime = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}.get(ext, "image/png")
        return b64, mime

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_data(self) -> bool:
        self._log("Loading data...", "step")

        # 1. Scene code (priority: explicit path > stage11_texture > stage10_material
        #    > stage6_geometry > stage4 > stage3)
        if self.scene_code_path and os.path.exists(self.scene_code_path):
            with open(self.scene_code_path, "r", encoding="utf-8") as f:
                self.scene_code = f.read()
            self._log(f"Scene code: {self.scene_code_path} ({self.scene_code.count(chr(10)) + 1} lines)", "success")
        elif self.use_memory:
            for stage_name in ("stage11_texture", "stage10_material", "stage6_geometry"):
                entry = self.memory.get_latest(stage=stage_name, type="result")
                if entry:
                    code_path = entry.metadata.get("output_file")
                    if code_path and os.path.exists(code_path):
                        with open(code_path, "r", encoding="utf-8") as f:
                            self.scene_code = f.read()
                        self._log(f"Scene code: from Memory ({stage_name}) - {code_path}", "success")
                        break
                    elif isinstance(entry.content, str) and self._is_python_code(entry.content):
                        self.scene_code = entry.content
                        self._log(f"Scene code: from Memory ({stage_name} content)", "success")
                        break

            if self.scene_code and not self._is_python_code(self.scene_code):
                self._log("Stage content is JSON, not Python code; falling back", "warning")
                self.scene_code = None

            if not self.scene_code:
                for stage_name in ("stage4", "stage3"):
                    entry = self.memory.get_latest(stage=stage_name, type="result")
                    if entry:
                        self.scene_code = entry.content
                        self._log(f"Scene code: from Memory ({stage_name})", "success")
                        break

        if not self.scene_code:
            run_dir = os.path.dirname(self.output_dir)
            for candidate in (
                os.path.join(run_dir, "stage11_texture", "texture_output.py"),
                os.path.join(run_dir, "stage10_material", "material_output.py"),
                os.path.join(run_dir, "stage6_geometry", "geometry_output.py"),
                os.path.join(run_dir, "stage4", "stage4_output.py"),
                os.path.join(run_dir, "stage4", "stage4_clean.py"),
                os.path.join(run_dir, "stage3", "stage3_output.py"),
            ):
                if os.path.exists(candidate):
                    with open(candidate, "r", encoding="utf-8") as f:
                        self.scene_code = f.read()
                    self._log(f"Scene code: from directory - {candidate}", "success")
                    break

        if not self.scene_code:
            self._log("No scene code found! Use --scene-code to specify", "error")
            return False

        # 2. Reference image
        if not self.image_path and self.use_memory:
            stage1_entry = self.memory.get_latest(stage="stage1", type="result")
            if stage1_entry:
                self.image_path = stage1_entry.metadata.get("image_path")

        if self.image_path and os.path.exists(self.image_path):
            self._log(f"Reference image: {self.image_path}", "success")
        else:
            self._log("No reference image found! Lighting analysis requires an image", "error")
            return False

        # 3. Optional: Stage1/Stage2 JSON for object semantics
        if self.use_memory:
            stage1_entry = self.memory.get_latest(stage="stage1", type="result")
            if stage1_entry:
                self.stage1_json = stage1_entry.content
                self._log("Stage1 JSON: OK", "success")

            stage2_entry = self.memory.get_latest(stage="stage2", type="result")
            if stage2_entry:
                self.stage2_json = stage2_entry.content
                self._log("Stage2 JSON: OK", "success")

        return True

    # ------------------------------------------------------------------
    # Scene info extraction
    # ------------------------------------------------------------------

    def _extract_scene_info(self) -> dict:
        import re
        info = {
            "scene_w": 8.0, "scene_d": 6.0, "wall_h": 2.8,
            "lamp_objects": [], "window_objects": [],
        }
        if not self.scene_code:
            return info

        m = re.search(r'SCENE_W\s*=\s*([\d.]+)', self.scene_code)
        if m:
            info["scene_w"] = float(m.group(1))
        m = re.search(r'SCENE_D\s*=\s*([\d.]+)', self.scene_code)
        if m:
            info["scene_d"] = float(m.group(1))
        m = re.search(r'WALL_H\s*=\s*([\d.]+)', self.scene_code)
        if m:
            info["wall_h"] = float(m.group(1))

        for m in re.finditer(r'create_(?:cylinder|box|detailed_object)\(\s*"([^"]*[Ll]amp[^"]*)"\s*,\s*(?:location\s*=\s*)?\(([^)]+)\)', self.scene_code):
            name = m.group(1)
            try:
                coords = [c.strip() for c in m.group(2).split(',')]
                if len(coords) >= 3:
                    info["lamp_objects"].append({"name": name, "location_expr": m.group(2).strip()})
            except:
                pass

        info["window_objects"] = self._extract_windows(
            scene_w=info["scene_w"],
            scene_d=info["scene_d"],
            wall_h=info["wall_h"],
        )

        return info

    def _extract_windows(self, scene_w: float, scene_d: float, wall_h: float) -> list:
        """Detect window meshes in the scene code.

        Looks for ``create_box`` / ``create_detailed_object`` calls whose name
        token contains ``window`` (any case) or ``glass_door``. Each detected
        window returns a dict with ``name``, parsed ``location`` (cx, cy, cz)
        and ``size`` (w, d, h), plus a ``wall`` tag indicating which exterior
        wall it sits on (``"north"|"south"|"east"|"west"|"unknown"``).
        Used by Stage 12 to spawn Cycles light portals + a directional Sun
        aligned with the window aperture.

        Variable-aware: Stage 3 routinely writes window calls in terms of
        local helper variables, e.g.
        ``create_box("Window_East_1", (SCENE_W/2 + WALL_T/2, win_e1_y, 1.5), ...)``.
        We do a fixed-point pass over every top-level ``<name> = <expr>``
        assignment in the scene code, evaluating each in a sandboxed env
        seeded with ``SCENE_W/D/H``. A few iterations are enough for chains
        like ``WALL_T = 0.1`` → ``win_e1_y = SCENE_D/4 - WALL_T``.
        """
        import re
        windows: list = []
        if not self.scene_code:
            return windows

        env: dict = {
            "SCENE_W": scene_w, "SCENE_D": scene_d, "WALL_H": wall_h,
            "scene_w": scene_w, "scene_d": scene_d, "wall_h": wall_h,
            "__builtins__": {},
            # Allow common math functions in window expressions without
            # opening the door to arbitrary builtins.
            "min": min, "max": max, "abs": abs,
        }

        # Pull every top-level assignment. We require the LHS to be a bare
        # identifier on its own line (allow leading indent; reject augmented
        # assigns like ``+=``). RHS captured up to the end of the line.
        assign_re = re.compile(
            r'^[ \t]*([A-Za-z_][A-Za-z_0-9]*)\s*=\s*([^=#\n][^\n]*?)\s*$',
            re.MULTILINE,
        )
        raw_assignments: list = []
        for m in assign_re.finditer(self.scene_code):
            varname = m.group(1)
            rhs = m.group(2).rstrip()
            # Strip trailing inline comment if any
            if "#" in rhs:
                rhs = rhs.split("#", 1)[0].rstrip()
            # Reject obvious non-numeric RHSes early
            if rhs.startswith(("'", '"', "[", "{", "(")):
                continue
            if "lambda" in rhs or "def " in rhs:
                continue
            raw_assignments.append((varname, rhs))

        # Fixed-point: keep eval'ing until no new variable resolves.
        for _ in range(5):
            progressed = False
            for var, rhs in raw_assignments:
                if var in env:
                    continue
                try:
                    val = eval(rhs, env, {})  # noqa: S307 — sandboxed env
                except Exception:
                    continue
                if isinstance(val, (int, float)):
                    env[var] = float(val)
                    progressed = True
            if not progressed:
                break

        def _safe_eval_num_tuple(expr: str) -> list:
            parts = []
            for raw in expr.split(','):
                raw = raw.strip()
                if not raw:
                    parts.append(None)
                    continue
                try:
                    parts.append(float(raw))
                    continue
                except ValueError:
                    pass
                try:
                    val = eval(raw, env, {})  # noqa: S307
                    parts.append(float(val))
                except Exception:
                    parts.append(None)
            return parts

        pattern = re.compile(
            r'create_(?:box|detailed_object)\(\s*'
            r'"([^"]*(?:[Ww]indow|[Gg]lass[_ ][Dd]oor)[^"]*)"\s*,\s*'
            r'(?:location\s*=\s*)?\(([^)]+)\)'
            r'(?:\s*,\s*(?:size|dimensions)\s*=\s*\(([^)]+)\))?'
        )
        for m in pattern.finditer(self.scene_code):
            name = m.group(1)
            loc_expr = m.group(2).strip()
            size_expr = (m.group(3) or "").strip()
            # Stage 3 also writes positional 3-tuple form:
            #   create_box("Window_X", (loc), (size), material=...)
            # The (size) tuple is the 2nd positional arg; if our regex did
            # not catch it via the "size=" keyword form, reach for the next
            # parenthesized tuple right after the location tuple.
            if not size_expr:
                tail = self.scene_code[m.end():m.end() + 200]
                m2 = re.match(r'\s*,\s*\(([^)]+)\)', tail)
                if m2:
                    size_expr = m2.group(1).strip()
            loc = _safe_eval_num_tuple(loc_expr)
            size = _safe_eval_num_tuple(size_expr) if size_expr else []
            if len(loc) < 3 or any(v is None for v in loc[:3]):
                continue
            cx, cy, cz = loc[0], loc[1], loc[2]
            if len(size) < 3 or any(v is None for v in size[:3]):
                w_ = 1.5
                h_ = 1.5
                d_ = 0.1
            else:
                w_, d_, h_ = size[0], size[1], size[2]

            # Wall classification: tolerance 0.5m to be lenient with WALL_T
            # offsets — many Stage 3 scripts position windows slightly
            # outside the wall plane (cx = SCENE_W/2 + WALL_T/2).
            tol = 0.5
            wall = "unknown"
            best_dist = tol + 1.0
            for w_name, expected in (
                ("east",  scene_w / 2.0),
                ("west", -scene_w / 2.0),
                ("north", scene_d / 2.0),
                ("south",-scene_d / 2.0),
            ):
                d = abs(cx - expected) if w_name in ("east", "west") else abs(cy - expected)
                if d < tol and d < best_dist:
                    best_dist = d
                    wall = w_name

            windows.append({
                "name": name,
                "location": (cx, cy, cz),
                "size": (w_, d_, h_),
                "wall": wall,
            })

        seen = set()
        unique: list = []
        for w in windows:
            key = (w["name"], round(w["location"][0], 3),
                   round(w["location"][1], 3), round(w["location"][2], 3))
            if key in seen:
                continue
            seen.add(key)
            unique.append(w)
        return unique

    # ------------------------------------------------------------------
    # Lighting analysis (LLM)
    # ------------------------------------------------------------------

    def _analyze_lighting(self) -> dict:
        self._log("Analyzing image lighting...", "light")

        scene = self._extract_scene_info()
        w, d, h = scene["scene_w"], scene["scene_d"], scene["wall_h"]

        lamp_info = ""
        if scene["lamp_objects"]:
            lamp_lines = [f"  - {l['name']} at ({l['location_expr']})" for l in scene["lamp_objects"]]
            lamp_info = "\nExisting lamp objects in the scene:\n" + "\n".join(lamp_lines)

        system_prompt = f"""You are a lighting analysis expert. Analyze the provided top-down floor plan image and extract lighting information.

CRITICAL - SCENE COORDINATE SYSTEM:
- The scene uses Blender world coordinates with the ORIGIN (0, 0, 0) at the CENTER of the room floor.
- X axis: left/right. Range: [{-w/2:.2f}, {w/2:.2f}] (scene width = {w}m)
- Y axis: front/back. Range: [{-d/2:.2f}, {d/2:.2f}] (scene depth = {d}m)
- Z axis: up. Range: [0, {h}] (wall height = {h}m)
- North wall is at Y={d/2:.2f}, South wall at Y={-d/2:.2f}
- West wall is at X={-w/2:.2f}, East wall at X={w/2:.2f}
- Ceiling is at Z={h}
{lamp_info}

All light positions MUST use this coordinate system. Do NOT use image pixel coordinates.
For example, a ceiling light centered in the room should be at (0, 0, {h}).
A table lamp should be at the SAME X,Y as the table/lamp object, with Z slightly above the object top.

Output ONLY valid JSON (no markdown, no explanation), following this exact structure:
{{
  "lighting_analysis": {{
    "overall_mood": "warm_cozy | bright_modern | soft_natural | dramatic",
    "primary_light_source": {{
      "type": "natural_window | ceiling_pendant | recessed | mixed",
      "direction": "north | south | east | west | overhead",
      "color_temperature": 2700 | 4000 | 5500 | 6500,
      "intensity": "low | medium | high"
    }},
    "light_sources": [
      {{
        "id": "light_001",
        "type": "area | point | spot | sun",
        "blender_type": "AREA | POINT | SPOT | SUN",
        "position": {{"x": 0, "y": 0, "z": {h}}},
        "color_rgb": [255, 244, 229],
        "energy": 500,
        "size": 2.0,
        "notes": "description"
      }}
    ],
    "ambient_light": {{
      "color_rgb": [255, 250, 240],
      "strength": 0.3
    }},
    "shadow_softness": "hard | medium | soft"
  }}
}}

ENERGY GUIDELINES for Cycles renderer:
- SUN light: energy 3-8 (NOT hundreds)
- POINT light (table lamp): energy 20-80
- AREA light (ceiling fill): energy 100-300, size 3-6
- SPOT light: energy 50-200

Analyze shadows, highlights, and visible light fixtures to determine:
- Where light is coming from (windows, lamps)
- Color temperature (warm/cool)
- Light intensity distribution
"""

        b64, mime = self._encode_image(self.image_path)
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=[
                {"type": "text", "text": "Analyze this image for lighting information, output JSON:"},
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}
            ])
        ]

        try:
            response = self.llm.invoke(messages)
            lighting_config = self._extract_json(response)
            if lighting_config:
                self._log(f"Lighting analysis complete: {len(lighting_config.get('lighting_analysis', {}).get('light_sources', []))} light sources", "success")
                return lighting_config
            else:
                self._log("Lighting analysis failed, using defaults", "warning")
                return self._default_lighting()
        except Exception as e:
            self._log(f"Lighting analysis error: {e}", "error")
            return self._default_lighting()

    # ------------------------------------------------------------------
    # Code generation — lighting only
    # ------------------------------------------------------------------

    @staticmethod
    def _strip_lighting_self_calls(block: str, func_name: str) -> str:
        """Remove erroneous ``func_name()`` lines inside the lighting function body.

        The LLM often appends a self-call at the end of ``setup_lighting_and_render``;
        that would recurse infinitely and makes the global call-detector think the
        layout engine already invokes lighting (so no call is injected).
        """
        import re
        pat = re.compile(rf"^(\s*){re.escape(func_name)}\s*\(\s*\)\s*(#.*)?$")
        return "\n".join(line for line in block.split("\n") if not pat.match(line))

    @staticmethod
    def _run_layout_has_lighting_call(code: str, func_name: str) -> bool:
        """True if ``run_layout_engine`` body contains ``func_name()`` (not inside other defs)."""
        import re
        lines = code.split("\n")
        in_run = False
        run_indent = 0
        call_re = re.compile(rf"^\s*{re.escape(func_name)}\s*\(\s*\)")
        for line in lines:
            if "def run_layout_engine" in line:
                in_run = True
                run_indent = len(line) - len(line.lstrip())
                continue
            if not in_run:
                continue
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                cur = len(line) - len(line.lstrip())
                if cur <= run_indent:
                    break
            if call_re.search(line):
                return True
        return False

    def _build_fallback_lighting_function(self) -> str:
        """Build deterministic lighting code when the LLM adapter fails.

        Indoor-realism tuning (post 2026-05 brightness/shadow rework):
          * Cycles + adaptive sampling, samples=192, indirect clamp
          * AgX color management with global -0.3 EV exposure
          * World shader: Sky Texture (NISHITA) at low strength (0.18) acts
            as a soft daylight backdrop for reflections, NOT a primary fill.
            Skipped entirely when the room has no windows AND the LLM did
            not detect a natural light source (Sky would otherwise paint
            interior rooms blue).
          * NO independent Sun light. Sky NISHITA's built-in sun disk is the
            single directional component, sun_direction matched to the LLM
            analysis. This avoids the "double parallel shadow" artefact
            from stacking Sun + Sky + portals + ceiling Area.
          * Light portals (Area lights with cycles.is_portal=True) at every
            detected window aperture, oriented to face the room interior.
            Portals do not emit; they channel Sky/HDRI light into the room
            with an order-of-magnitude noise reduction.
          * LLM-analyzed Area lights are forced to ``shape='DISK'`` with a
            minimum size of half the room width (very soft shadows) and
            energy capped at 250 W to keep the LLM's lamps as the focal
            light without over-driving the scene.
        """
        scene = self._extract_scene_info()
        h = scene["wall_h"]
        sw = scene["scene_w"]
        sd = scene["scene_d"]
        analysis = (self.lighting_config or self._default_lighting()).get("lighting_analysis", {})

        def as_float(value, default):
            try:
                return float(value)
            except (TypeError, ValueError):
                return default

        def as_rgb(value, default):
            if not isinstance(value, (list, tuple)) or len(value) < 3:
                return default
            return [max(0, min(255, int(as_float(v, d)))) for v, d in zip(value[:3], default)]

        raw_sources = analysis.get("light_sources")
        if not isinstance(raw_sources, list):
            raw_sources = []
        if not raw_sources:
            raw_sources = self._default_lighting()["lighting_analysis"]["light_sources"]

        # Soft Area light minimum for ceiling-class lights: half the room's
        # shorter side, capped between 2.5 and 6 m (large emitter = soft
        # shadows). Wall-class Area lights ("window panes") get a smaller
        # default because we set them to a window-aspect rectangle later.
        area_min_size = max(2.5, min(6.0, min(sw, sd) * 0.5))

        # Threshold for deciding whether an Area light is "stuck to a wall".
        # WALL_T (wall thickness) is rarely > 0.4 m, so 0.6 m comfortably
        # captures lights whose center the LLM put just inside or just
        # outside the wall plane.
        WALL_TOL = 0.6
        # Threshold for "ceiling-class": light is in the upper 20 % of the
        # room height. This catches LLM "ceiling fill" Areas at z ≈ WALL_H
        # while leaving window-height lights (z ≈ 1.4) for the wall path.
        CEIL_Z_FRAC = 0.80

        def _classify_area(px: float, py: float, pz: float) -> str:
            # Ceiling first: a light at WALL_H — even if it also happens
            # to be within WALL_TOL of a wall — should fill from above.
            if pz >= h * CEIL_Z_FRAC:
                return "ceiling"
            # Pick the closest exterior wall (if any is within tolerance).
            cands = (
                ("east",  abs(px - sw / 2.0)),
                ("west",  abs(px - (-sw / 2.0))),
                ("north", abs(py - sd / 2.0)),
                ("south", abs(py - (-sd / 2.0))),
            )
            wall, dist = min(cands, key=lambda t: t[1])
            if dist < WALL_TOL:
                return wall
            return "ceiling"

        light_sources = []
        for idx, src in enumerate(raw_sources, start=1):
            if not isinstance(src, dict):
                continue
            pos = src.get("position") if isinstance(src.get("position"), dict) else {}
            blender_type = str(src.get("blender_type") or src.get("type") or "AREA").upper()
            if blender_type not in {"AREA", "POINT", "SPOT", "SUN"}:
                blender_type = "AREA"
            energy = as_float(src.get("energy"), 300.0)
            size = as_float(src.get("size"), 3.0)
            px = as_float(pos.get("x"), 0.0)
            py = as_float(pos.get("y"), 0.0)
            pz = as_float(pos.get("z"), h)

            area_class = "n/a"
            if blender_type == "AREA":
                area_class = _classify_area(px, py, pz)
                if area_class == "ceiling":
                    size = max(size, area_min_size)
                    energy = min(energy, 250.0)
                else:
                    # Wall-class window pane: roughly window-shaped
                    # rectangle covering a typical 0.6–2.4 m vertical span
                    # so the upper walls get direct light.
                    size = max(min(size, 3.0), 1.5)
                    energy = min(max(energy, 60.0), 280.0)
            light_sources.append({
                "id": f"stage11_light_{idx:02d}",
                "blender_type": blender_type,
                "area_class": area_class,
                "position": {"x": px, "y": py, "z": pz},
                "color_rgb": as_rgb(src.get("color_rgb"), [255, 244, 230]),
                "energy": energy,
                "size": size,
            })

        ambient = analysis.get("ambient_light") if isinstance(analysis.get("ambient_light"), dict) else {}
        ambient_color = as_rgb(ambient.get("color_rgb"), [255, 250, 245])
        ambient_strength = as_float(ambient.get("strength"), 0.3)

        # Build window portals data (no portal node if scene has no windows).
        windows = scene.get("window_objects", [])
        WINDOW_NORMAL = {
            "north": (0.0, -1.0, 0.0),
            "south": (0.0,  1.0, 0.0),
            "east":  (-1.0, 0.0, 0.0),
            "west":  ( 1.0, 0.0, 0.0),
        }
        portals = []
        for w in windows:
            cx, cy, cz = w["location"]
            ww, wd, wh = w["size"]
            wall = w.get("wall", "unknown")
            if wall == "unknown":
                if abs(abs(cx) - sw / 2.0) < 0.30:
                    wall = "east" if cx > 0 else "west"
                elif abs(abs(cy) - sd / 2.0) < 0.30:
                    wall = "north" if cy > 0 else "south"
                else:
                    wall = "north"
            normal = WINDOW_NORMAL.get(wall, (0.0, -1.0, 0.0))
            offset = 0.05
            px = cx + normal[0] * offset
            py = cy + normal[1] * offset
            pz = cz
            if wall in ("north", "south"):
                size_x = max(0.6, ww)
                size_y = max(0.6, wh)
            else:
                size_x = max(0.6, wd)
                size_y = max(0.6, wh)
            portals.append({
                "id": f"stage11_portal_{w['name']}",
                "location": (px, py, pz),
                "wall": wall,
                "size_x": size_x,
                "size_y": size_y,
            })

        primary = analysis.get("primary_light_source") if isinstance(analysis.get("primary_light_source"), dict) else {}
        primary_type = str(primary.get("type", "")).lower()
        primary_dir = str(primary.get("direction", "")).lower()
        primary_intensity = str(primary.get("intensity", "")).lower()
        overall_mood = str(analysis.get("overall_mood", "")).lower()
        SUN_DIR = {
            "north":    (0.0, -0.6, -0.8),
            "south":    (0.0,  0.6, -0.8),
            "east":     (-0.6, 0.0, -0.8),
            "west":     ( 0.6, 0.0, -0.8),
            "overhead": (0.1, -0.1, -1.0),
        }
        is_natural = (
            "natural_window" in primary_type
            or "window" in primary_type
            or "sun" in primary_type
            or "natural" in primary_type
            # 'mixed' or '' is what LLM emits when the room has both
            # window daylight + indoor lamps. Treat as natural for the
            # purposes of Sky / Sun direction selection — the Sky will
            # provide soft daylight, indoor lamps still come through
            # individually below.
            or primary_type in ("mixed", "")
        )
        # Also infer daylight from the per-source notes / overall mood,
        # since LLM is inconsistent about primary_type vs notes.
        sources_text = " ".join(
            str(s.get("notes", "")).lower()
            for s in raw_sources if isinstance(s, dict)
        )
        if not is_natural and any(
            kw in sources_text for kw in
            ("window", "daylight", "natural light", "sunlight", "skylight")
        ):
            is_natural = True
        if not is_natural and any(
            kw in overall_mood for kw in ("natural", "bright", "soft_natural", "daylight")
        ):
            is_natural = True

        sun_vec = SUN_DIR.get(primary_dir)
        if is_natural and sun_vec is None:
            if portals:
                p_wall = portals[0]["wall"]
                inverse = {"north": "south", "south": "north",
                           "east": "west", "west": "east"}.get(p_wall, "south")
                sun_vec = SUN_DIR.get(inverse, SUN_DIR["south"])
            else:
                sun_vec = SUN_DIR["south"]
        if sun_vec is None:
            sun_vec = SUN_DIR["overhead"]

        # Sky on whenever we have any reasonable daylight evidence:
        # detected windows OR LLM hints. Off only for confidently
        # windowless interior rooms (closets, internal bathrooms) where
        # LLM also offered no daylight cue — in that case Sky would
        # paint a blue ceiling with no physical justification.
        use_sky = bool(windows) or is_natural or primary_intensity == "high"

        # Sky sun disk intensity. NISHITA sun disk doubles as our directional
        # component: we lean on it instead of a separate Sun light to
        # eliminate double shadows. High-intensity daylight → 0.30, otherwise
        # 0.12 (still gives reflections their direction).
        if is_natural and primary_intensity == "high":
            sky_sun_intensity = 0.30
        else:
            sky_sun_intensity = 0.12
        # Sky 0.25 is a daylight-room backdrop: enough indirect bounce to
        # paint the upper walls/ceiling, dim enough that LLM's interior
        # lamps still register as warm focal lights.
        sky_strength = 0.25

        return f'''def setup_lighting_and_render():
    import bpy
    import math

    def rgb3(c):
        return (c[0] / 255.0, c[1] / 255.0, c[2] / 255.0)

    def rgba4(c):
        return (c[0] / 255.0, c[1] / 255.0, c[2] / 255.0, 1.0)

    scene = bpy.context.scene

    def ensure_topdown_camera():
        from mathutils import Vector

        points = []
        for obj in bpy.context.scene.objects:
            if obj.type in {'MESH', 'CURVE', 'SURFACE', 'FONT'}:
                try:
                    points.extend(obj.matrix_world @ Vector(corner) for corner in obj.bound_box)
                except Exception:
                    pass
        if points:
            min_x = min(p.x for p in points)
            max_x = max(p.x for p in points)
            min_y = min(p.y for p in points)
            max_y = max(p.y for p in points)
            max_z = max(p.z for p in points)
            center_x = (min_x + max_x) / 2.0
            center_y = (min_y + max_y) / 2.0
            span = max(max_x - min_x, max_y - min_y, 1.0)
        else:
            center_x = center_y = 0.0
            max_z = 0.0
            span = 8.0

        cam = scene.camera
        if cam is None or cam.name not in bpy.data.objects:
            bpy.ops.object.camera_add()
            cam = bpy.context.object
            cam.name = 'stage12_topdown_camera'
            scene.camera = cam

        cam.location = (center_x, center_y, max_z + max(8.0, span * 1.6))
        cam.rotation_euler = (0.0, 0.0, 0.0)
        cam.data.type = 'ORTHO'
        cam.data.ortho_scale = span * 1.15
        cam.data.clip_end = max(100.0, max_z + span * 4.0)
        return cam

    ensure_topdown_camera()

    try:
        scene.render.engine = 'CYCLES'
        scene.cycles.samples = 192
        scene.cycles.use_denoising = False
        if hasattr(scene.cycles, 'use_adaptive_sampling'):
            scene.cycles.use_adaptive_sampling = True
            scene.cycles.adaptive_threshold = 0.01
        if hasattr(scene.cycles, 'sample_clamp_indirect'):
            scene.cycles.sample_clamp_indirect = 10.0
    except Exception:
        scene.render.engine = 'BLENDER_EEVEE_NEXT'

    for obj in list(bpy.data.objects):
        if obj.type == 'LIGHT' and (
            obj.name.startswith('stage11_light_')
            or obj.name.startswith('stage11_portal_')
            or obj.name.startswith('stage11_sun_')
        ):
            bpy.data.objects.remove(obj, do_unlink=True)

    light_sources = {repr(light_sources)}
    for src in light_sources:
        pos = src["position"]
        bpy.ops.object.light_add(
            type=src["blender_type"],
            location=(pos["x"], pos["y"], pos["z"]),
        )
        light_obj = bpy.context.object
        light_obj.name = src["id"]
        light_obj.data.color = rgb3(src["color_rgb"])
        light_obj.data.energy = src["energy"]
        if hasattr(light_obj.data, "size"):
            light_obj.data.size = src["size"]
        if src["blender_type"] == 'AREA':
            ac = src.get("area_class", "ceiling")
            ld = light_obj.data
            if ac == 'ceiling':
                # Ceiling fill: round large emitter pointing straight down.
                # Default rotation already emits along -Z, so leave it.
                try:
                    ld.shape = 'DISK'
                except Exception:
                    pass
                if hasattr(ld, "size_y"):
                    try:
                        ld.size_y = float(src["size"])
                    except Exception:
                        pass
            else:
                # Wall-class "window pane" Area light: rectangular, oriented
                # to face the room INTERIOR (so its emit direction is -Z in
                # local space, rotated to point inward). Vertical span ≈
                # window height (0.6 m), horizontal span = src.size, so the
                # light covers the wall vertically from below the lamp band
                # up to the ceiling — fixing the dark upper-wall artefact.
                try:
                    ld.shape = 'RECTANGLE'
                except Exception:
                    pass
                if hasattr(ld, "size_y"):
                    try:
                        # 0.6 m is a typical window pane vertical thickness;
                        # the inverse-square fall-off then naturally lights
                        # the entire 2.4–2.8 m wall height.
                        ld.size_y = 0.6
                    except Exception:
                        pass
                # Rotate so emit (-Z local) points into the room.
                if ac == 'east':
                    light_obj.rotation_euler = (0.0, math.radians(90.0), 0.0)
                elif ac == 'west':
                    light_obj.rotation_euler = (0.0, math.radians(-90.0), 0.0)
                elif ac == 'north':
                    light_obj.rotation_euler = (math.radians(-90.0), 0.0, 0.0)
                elif ac == 'south':
                    light_obj.rotation_euler = (math.radians(90.0), 0.0, 0.0)

    portals = {repr(portals)}
    for p in portals:
        loc = p["location"]
        bpy.ops.object.light_add(type='AREA', location=loc)
        portal_obj = bpy.context.object
        portal_obj.name = p["id"]
        ld = portal_obj.data
        ld.shape = 'RECTANGLE'
        ld.size   = float(p["size_x"])
        if hasattr(ld, "size_y"):
            ld.size_y = float(p["size_y"])
        ld.energy = 0.0
        if hasattr(ld, "cycles") and hasattr(ld.cycles, "is_portal"):
            ld.cycles.is_portal = True
        wall = p["wall"]
        if wall == "north":
            portal_obj.rotation_euler = (math.radians(90.0),  0.0, 0.0)
        elif wall == "south":
            portal_obj.rotation_euler = (math.radians(-90.0), 0.0, 0.0)
        elif wall == "east":
            portal_obj.rotation_euler = (0.0, math.radians(-90.0), 0.0)
        elif wall == "west":
            portal_obj.rotation_euler = (0.0, math.radians(90.0),  0.0)

    # Note: as of the indoor-realism rework we no longer spawn an
    # independent SUN light. The Sky Texture (NISHITA) below carries the
    # directional component via its built-in sun disk, which gives ONE
    # soft, atmospherically-scattered shadow direction instead of the
    # double parallel shadows we got from Sun + Sky stacked together.
    sun_direction = {repr(sun_vec)}

    world = scene.world or bpy.data.worlds.new("World")
    scene.world = world
    world.use_nodes = True
    nt = world.node_tree
    for _n in list(nt.nodes):
        nt.nodes.remove(_n)
    world_output = nt.nodes.new(type="ShaderNodeOutputWorld")
    world_output.location = (600, 0)
    bg_node = nt.nodes.new(type="ShaderNodeBackground")
    bg_node.location = (300, 0)
    nt.links.new(bg_node.outputs[0], world_output.inputs[0])
    sky_ok = False
    use_sky = {repr(use_sky)}
    if use_sky:
        try:
            sky = nt.nodes.new(type="ShaderNodeTexSky")
            sky.location = (0, 0)
            if hasattr(sky, "sky_type"):
                try:
                    sky.sky_type = 'NISHITA'
                except Exception:
                    try:
                        sky.sky_type = 'HOSEK_WILKIE'
                    except Exception:
                        pass
            if hasattr(sky, "sun_direction"):
                try:
                    dx, dy, dz = sun_direction
                    from mathutils import Vector as _V
                    _sd = _V((-dx, -dy, -dz)).normalized()
                    sky.sun_direction = (_sd.x, _sd.y, _sd.z)
                except Exception:
                    pass
            if hasattr(sky, "sun_size"):
                sky.sun_size = 0.05
            if hasattr(sky, "sun_intensity"):
                sky.sun_intensity = {sky_sun_intensity}
            if hasattr(sky, "air_density"):
                sky.air_density = 1.0
            if hasattr(sky, "dust_density"):
                sky.dust_density = 0.5
            if hasattr(sky, "ozone_density"):
                sky.ozone_density = 1.0
            nt.links.new(sky.outputs["Color"], bg_node.inputs[0])
            bg_node.inputs[1].default_value = {sky_strength}
            sky_ok = True
        except Exception as e:
            print("[stage11] Sky Texture unavailable, using flat ambient: " + str(e))
    if not sky_ok:
        # Either an enclosed interior (no windows + no natural primary)
        # or NISHITA threw — fall back to a soft flat Background so we
        # don't render pitch-black ceilings.
        bg_node.inputs[0].default_value = rgba4({repr(ambient_color)})
        bg_node.inputs[1].default_value = 0.15

    try:
        scene.view_settings.view_transform = 'AgX'
        scene.view_settings.look = 'AgX - Medium High Contrast'
    except Exception:
        scene.view_settings.view_transform = 'Filmic'
        scene.view_settings.look = 'Medium High Contrast'
    if hasattr(scene.view_settings, 'exposure'):
        scene.view_settings.exposure = -0.3
    scene.render.resolution_x = 1024
    scene.render.resolution_y = 1024'''

    def _inject_lighting_function(self, lighting_func: str) -> str:
        """Insert a lighting function and call it from run_layout_engine()."""
        import re

        func_name = "setup_lighting_and_render"
        m = re.search(r"def (\w+)\s*\(", lighting_func)
        if m:
            func_name = m.group(1)
        lighting_func = self._strip_lighting_self_calls(lighting_func, func_name)

        modified_code = self.scene_code
        insert_markers = ["# === MAIN LAYOUT ENGINE ===", "def run_layout_engine():"]
        inserted = False
        for marker in insert_markers:
            if marker in modified_code:
                pos = modified_code.find(marker)
                modified_code = modified_code[:pos] + "\n" + lighting_func + "\n\n" + modified_code[pos:]
                inserted = True
                break
        if not inserted:
            modified_code = lighting_func + "\n\n" + modified_code

        if not self._run_layout_has_lighting_call(modified_code, func_name):
            lines = modified_code.split("\n")
            in_run_layout = False
            run_layout_indent = 0
            last_body_line = -1

            for i in range(len(lines)):
                stripped = lines[i].strip()
                if "def run_layout_engine" in lines[i]:
                    in_run_layout = True
                    run_layout_indent = len(lines[i]) - len(lines[i].lstrip())
                    continue
                if in_run_layout:
                    if stripped and not stripped.startswith("#"):
                        cur_indent = len(lines[i]) - len(lines[i].lstrip())
                        if cur_indent <= run_layout_indent and lines[i].strip():
                            break
                    if stripped:
                        last_body_line = i

            if last_body_line > 0:
                lines.insert(last_body_line + 1, "")
                lines.insert(last_body_line + 2, "    # === Lighting & Render ===")
                lines.insert(last_body_line + 3, f"    {func_name}()")

            modified_code = "\n".join(lines)

        return self._verify_and_fix_syntax(modified_code)

    def _fallback_lighting_code(self, reason: str) -> str:
        # The deterministic path is the default since the realism upgrade.
        # Reasons that originate from real LLM failures still get a
        # "warning" log; benign reasons ("deterministic-by-default ...")
        # log as info.
        level = "info" if reason.startswith("deterministic-by-default") else "warning"
        self._log(f"Lighting builder: {reason}", level)
        try:
            return self._inject_lighting_function(self._build_fallback_lighting_function())
        except Exception as exc:
            self._log(f"Lighting builder failed: {exc}", "error")
            return None

    def _generate_lighting_code(self) -> str:
        """Generate a setup_lighting_and_render() function and inject it into the scene code.

        IMPORTANT: As of the realism upgrade, code generation is deterministic
        by default. The LLM still performs image-based lighting analysis
        (via ``_analyze_lighting``) which produces the lighting JSON config
        (light source positions, color temperatures, ambient strength, ...);
        but the Python code that creates lights, the Sky Texture (NISHITA)
        world shader, window light portals, and the AgX color management is
        emitted by ``_build_fallback_lighting_function`` directly. This was
        done because the LLM-generated code path historically produced subtle
        enum / shader-socket / portal-flag bugs that required ever more
        brittle post-fixes, while the deterministic path is exhaustively
        unit-tested by us. Set ``self._force_deterministic_lighting = False``
        on the runner to opt back into the LLM code-gen path (legacy).
        """
        import re

        if getattr(self, "_force_deterministic_lighting", True):
            self._log("Using deterministic lighting builder (Sky+Portal+Sun+AgX)", "step")
            return self._fallback_lighting_code(
                "deterministic-by-default for realism upgrade")

        self._log("Generating lighting + render settings code (LLM path)...", "step")

        scene = self._extract_scene_info()
        w, d, h = scene["scene_w"], scene["scene_d"], scene["wall_h"]

        system_prompt = """You are a Blender Python expert. Generate a SINGLE Python code block containing a function
`setup_lighting_and_render()` that:
1. Creates all light sources (Area, Point, Sun, Spot) based on the lighting analysis
2. Sets up Cycles render engine with reasonable settings (samples=128, denoising=True)
3. Sets up the world/environment (ambient light, background color)

CRITICAL: Do NOT generate any material code. Materials are already handled by an earlier stage.
Only output lighting and render configuration.

ENERGY GUIDELINES for Cycles renderer:
- SUN light: energy 3-8 (NOT hundreds)
- POINT light (table lamp): energy 20-80
- AREA light (ceiling fill): energy 100-300, size 3-6
- SPOT light: energy 50-200
- For Blender 4.x: use `scene.cycles.preview_samples` (NOT `scene.render.preview_samples`)

COLOR ASSIGNMENT RULES (STRICT - wrong tuple length will raise ValueError):
- `light_data.color`  MUST be a 3-tuple RGB in 0.0-1.0, e.g. `(1.0, 0.96, 0.9)`.
  This is TRUE for POINT, SUN, SPOT and AREA lights. NEVER assign a 4-tuple here.
- `world.color`  is also a 3-tuple RGB.
- Shader node socket `default_value` (e.g. Background input 0, Principled BSDF
  Base Color) MUST be a 4-tuple RGBA, e.g. `(1.0, 1.0, 1.0, 1.0)`.
- DO NOT write a single `get_color()` / `rgb_to_float()` helper that returns a
  4-tuple and reuse it for BOTH light colors AND shader sockets. If you need a
  helper, write TWO:
    def rgb3(c): return (c[0]/255, c[1]/255, c[2]/255)          # lights / world
    def rgba4(c): return (c[0]/255, c[1]/255, c[2]/255, 1.0)    # shader sockets
  Then use `rgb3(...)` for every `*.color = ...` on a light/world, and
  `rgba4(...)` only for `inputs[i].default_value = ...`.

WORLD / ENVIRONMENT NODE RULES (STRICT - prevents AttributeError at runtime):
- DO NOT look up world nodes by name (e.g. `nodes.get("World Output")` or
  `nodes.get("Background")`). Names can be missing/renamed/localised, which
  returns `None` and crashes on `.inputs[0]`.
- ALWAYS look them up by node `type`, fall back to creating one, and ALWAYS
  ensure the link exists. Use this EXACT safe pattern:
    world = scene.world or bpy.data.worlds.new("World")
    scene.world = world
    world.use_nodes = True
    nt = world.node_tree
    world_output = next((n for n in nt.nodes if n.type == 'OUTPUT_WORLD'), None)
    if world_output is None:
        world_output = nt.nodes.new(type="ShaderNodeOutputWorld")
    bg_node = next((n for n in nt.nodes if n.type == 'BACKGROUND'), None)
    if bg_node is None:
        bg_node = nt.nodes.new(type="ShaderNodeBackground")
    if not any(l.from_node == bg_node and l.to_node == world_output
               for l in nt.links):
        nt.links.new(bg_node.outputs[0], world_output.inputs[0])
- Only AFTER the pattern above may you assign
  `bg_node.inputs[0].default_value = rgba4(ambient_color)` and
  `bg_node.inputs[1].default_value = strength`.

Output ONLY a single Python code block with the function definition."""

        user_text = f"""Generate lighting and render setup for this Blender scene.

## SCENE COORDINATE SYSTEM
- Origin (0,0,0) is at room center on the floor
- X range: [{-w/2:.2f}, {w/2:.2f}] (width={w}m)
- Y range: [{-d/2:.2f}, {d/2:.2f}] (depth={d}m)
- Z range: [0, {h}] (wall height={h}m)
- Ceiling at Z={h}, Floor at Z=0

## LIGHTING ANALYSIS
```json
{json.dumps(self.lighting_config, ensure_ascii=False, indent=2)}
```

IMPORTANT: Only generate lighting + render setup. NO material code.
"""

        b64, mime = self._encode_image(self.image_path)
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=[
                {"type": "text", "text": "Reference image:"},
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                {"type": "text", "text": user_text}
            ])
        ]

        try:
            response = self.llm.invoke(messages)

            raw_path = os.path.join(self.output_dir, "render_raw.txt")
            os.makedirs(self.output_dir, exist_ok=True)
            with open(raw_path, "w", encoding="utf-8") as f:
                f.write(response if response else "(empty)")

            # Extract code blocks
            code_blocks = re.findall(r'```python\s*([\s\S]*?)\s*```', response)
            if not code_blocks:
                code_blocks = re.findall(r'```\s*([\s\S]*?)\s*```', response)

            lighting_func = ""
            for block in code_blocks:
                if "def setup_lighting" in block or "def setup_render" in block or "bpy.context.scene.render" in block:
                    lighting_func = block.strip()
                    break
            if not lighting_func and code_blocks:
                lighting_func = code_blocks[0].strip()

            if not lighting_func:
                self._log("No lighting code extracted from LLM response", "error")
                return self._fallback_lighting_code("no code block extracted")

            # Wrap bare code in a function if needed
            if "def setup_lighting" not in lighting_func and "def setup_render" not in lighting_func:
                lighting_func = (
                    "def setup_lighting_and_render():\n"
                    + "\n".join("    " + l for l in lighting_func.split("\n"))
                )

            # Clean up: remove top-level calls and __main__ blocks from the function
            lr_lines = lighting_func.strip().split("\n")
            lr_clean = []
            for ll in lr_lines:
                s = ll.strip()
                if s and not ll.startswith(" ") and not ll.startswith("\t") and not ll.startswith("def ") and not ll.startswith("#"):
                    if "()" in s and "setup" in s.lower():
                        continue
                    if s.startswith("if __name__"):
                        continue
                lr_clean.append(ll)
            lighting_func = "\n".join(lr_clean)

            return self._inject_lighting_function(lighting_func)

        except Exception as e:
            self._log(f"Lighting code generation failed; falling back: {e}", "warning")
            return self._fallback_lighting_code(str(e))

    # ------------------------------------------------------------------
    # Syntax verification & Blender enum fixes
    # ------------------------------------------------------------------

    def _verify_and_fix_syntax(self, code: str) -> str:
        import re

        from blender_code_syntax_fix import fix_empty_if_show_direction_before_return

        code = fix_empty_if_show_direction_before_return(code)

        # IMPORTANT: run enum normalization BEFORE the first compile check.
        # Blender enum values like `view_settings.look = 'Medium High Contrast'`
        # are syntactically valid Python, so if we only fall through to the
        # enum fixer on SyntaxError, the fix gets skipped and the bad enum
        # explodes at runtime inside Blender. Doing it up front also costs
        # nothing when the enums are already correct (str.replace no-ops).
        code = self._fix_blender_enums(code)

        MAX_FIX_PASSES = 5
        for pass_num in range(MAX_FIX_PASSES):
            try:
                compile(code, '<string>', 'exec')
                self._log(f"Syntax OK ({code.count(chr(10)) + 1} lines)", "success")
                return code
            except SyntaxError as e:
                self._log(f"Syntax error (pass {pass_num+1}, line {e.lineno}): {e.msg}", "warning")
                lines = code.split('\n')
                if not e.lineno or e.lineno > len(lines):
                    break

                err_idx = e.lineno - 1
                err_line = lines[err_idx]
                prev_line = lines[err_idx - 1] if err_idx > 0 else ""
                fixed = False

                if (e.msg in ('unexpected indent', 'invalid syntax')
                        and err_line.strip() and not err_line.strip().startswith('#')
                        and prev_line.rstrip().endswith(')')
                        and ('material=' in err_line or 'rotation=' in err_line or 'collection=' in err_line)):
                    prev_stripped = prev_line.rstrip()
                    if prev_stripped.endswith(')'):
                        continuation = err_line.strip()
                        continuation = re.sub(r'^[\w_]+\s*,\s*', '', continuation)
                        new_prev = prev_stripped[:-1] + ', ' + continuation
                        lines[err_idx - 1] = new_prev
                        lines[err_idx] = ''
                        fixed = True
                        self._log(f"  Auto-fix: merged broken function call at line {e.lineno}", "info")

                for check_line_idx in ([err_idx - 1, err_idx] if err_idx > 0 else [err_idx]):
                    if fixed:
                        break
                    check_line = lines[check_line_idx]
                    rot_start = check_line.find('rotation=(')
                    mat_pos = check_line.find('material=', rot_start + 1 if rot_start >= 0 else 0)
                    if rot_start >= 0 and mat_pos > rot_start:
                        between = check_line[rot_start + len('rotation=('):mat_pos]
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
                        lines[check_line_idx] = check_line[:rot_start] + fixed_rot + check_line[mat_pos + len('material='):]
                        fixed = True
                        self._log(f"  Auto-fix: completed rotation tuple at line {check_line_idx + 1}", "info")

                if not fixed:
                    self._log(f"  Cannot auto-fix line {e.lineno}: {err_line.strip()[:80]}", "warning")
                    break

                code = '\n'.join(lines)

        try:
            compile(code, '<string>', 'exec')
            self._log(f"Syntax OK ({code.count(chr(10)) + 1} lines)", "success")
        except SyntaxError as e:
            self._log(f"Syntax error persists (line {e.lineno}): {e.msg} — saved but may need manual fix", "error")

        code = self._fix_blender_enums(code)

        return code

    @staticmethod
    def _fix_blender_enums(code: str) -> str:
        """Reconcile ``view_settings.look`` with ``view_settings.view_transform``.

        Blender's ``look`` enum is gated by the active ``view_transform``:

        - ``view_transform == 'AgX'``   → looks are prefixed,
          e.g. ``'AgX - Medium High Contrast'``, ``'AgX - Base Contrast'``.
        - ``view_transform == 'Filmic'`` (or ``'Standard'`` / other) → looks
          are bare short names: ``'None'``, ``'Very High Contrast'``,
          ``'High Contrast'``, ``'Medium High Contrast'``, ``'Medium Contrast'``,
          ``'Medium Low Contrast'``, ``'Low Contrast'``, ``'Very Low Contrast'``.

        LLMs and downstream code regularly emit a mismatched pair, which
        Blender rejects at runtime with a TypeError ("enum X not found in
        (...)"). These are syntactically valid Python strings, so the error
        only surfaces inside Blender — we patch them here before the code
        is ever executed.

        Strategy: find the (last) assigned ``view_transform`` and rewrite
        every ``.look = '...'`` to match it. If ``view_transform`` is not
        set we default to Filmic semantics (Stage 3/4's default).
        """
        import re

        vt_matches = list(re.finditer(
            r"\.view_transform\s*=\s*['\"]([^'\"]+)['\"]",
            code,
        ))
        view_transform = vt_matches[-1].group(1) if vt_matches else "Filmic"

        SHORT_LOOKS = {
            "None",
            "Very High Contrast",
            "High Contrast",
            "Medium High Contrast",
            "Medium Contrast",
            "Medium Low Contrast",
            "Low Contrast",
            "Very Low Contrast",
        }
        AGX_LOOKS = {
            "AgX - Punchy",
            "AgX - Greyscale",
            "AgX - Very High Contrast",
            "AgX - High Contrast",
            "AgX - Medium High Contrast",
            "AgX - Base Contrast",
            "AgX - Medium Low Contrast",
            "AgX - Low Contrast",
            "AgX - Very Low Contrast",
        }
        SHORT_TO_AGX = {
            "Very High Contrast":   "AgX - Very High Contrast",
            "High Contrast":        "AgX - High Contrast",
            "Medium High Contrast": "AgX - Medium High Contrast",
            "Medium Contrast":      "AgX - Base Contrast",
            "Medium Low Contrast":  "AgX - Medium Low Contrast",
            "Low Contrast":         "AgX - Low Contrast",
            "Very Low Contrast":    "AgX - Very Low Contrast",
            "None":                 "AgX - Base Contrast",
        }
        AGX_TO_SHORT = {
            "AgX - Very High Contrast":   "Very High Contrast",
            "AgX - High Contrast":        "High Contrast",
            "AgX - Medium High Contrast": "Medium High Contrast",
            "AgX - Base Contrast":        "Medium Contrast",
            "AgX - Medium Low Contrast":  "Medium Low Contrast",
            "AgX - Low Contrast":         "Low Contrast",
            "AgX - Very Low Contrast":    "Very Low Contrast",
            "AgX - Punchy":               "Medium High Contrast",
            "AgX - Greyscale":            "None",
        }

        use_agx = (view_transform == "AgX")

        def _normalize_look(value: str) -> str:
            if use_agx:
                if value in AGX_LOOKS:
                    return value
                return SHORT_TO_AGX.get(value, value)
            if value in SHORT_LOOKS:
                return value
            return AGX_TO_SHORT.get(value, value)

        def _fix_look_line(match: re.Match) -> str:
            lhs   = match.group("lhs")
            quote = match.group("q")
            value = match.group("value")
            return f"{lhs}{quote}{_normalize_look(value)}{quote}"

        code = re.sub(
            r"(?P<lhs>\.look\s*=\s*)(?P<q>['\"])(?P<value>[^'\"]+)(?P=q)",
            _fix_look_line,
            code,
        )

        return code

    # ------------------------------------------------------------------
    # JSON extraction & defaults
    # ------------------------------------------------------------------

    def _extract_json(self, text: str) -> dict:
        import re

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

    def _default_lighting(self) -> dict:
        return {
            "lighting_analysis": {
                "overall_mood": "warm_cozy",
                "primary_light_source": {
                    "type": "mixed",
                    "direction": "overhead",
                    "color_temperature": 4000,
                    "intensity": "medium"
                },
                "light_sources": [
                    {
                        "id": "light_main",
                        "type": "area",
                        "blender_type": "AREA",
                        "position": {"x": 0, "y": 0, "z": 2.8},
                        "color_rgb": [255, 244, 230],
                        "energy": 300,
                        "size": 3.0,
                        "notes": "Main ceiling light"
                    }
                ],
                "ambient_light": {
                    "color_rgb": [255, 250, 245],
                    "strength": 0.3
                },
                "shadow_softness": "soft"
            }
        }

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------

    def _save_results(self, code: str) -> str:
        """Save results. Returns final code (after optional arrow cleanup)."""
        os.makedirs(self.output_dir, exist_ok=True)

        try:
            from stage_clean_arrows import ArrowCleaner
            cleaner = ArrowCleaner(output_dir=self.output_dir, verbose=self.verbose)
            code = cleaner.clean_code(code)
        except Exception as e:
            self._log(f"Direction arrow cleanup skipped: {e}", "warning")

        output_path = os.path.join(self.output_dir, "render_output.py")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(code)
        self._log(f"Code saved: {output_path}", "success")

        lighting_path = os.path.join(self.output_dir, "render_lighting.json")
        with open(lighting_path, "w", encoding="utf-8") as f:
            json.dump(self.lighting_config, f, ensure_ascii=False, indent=2)
        self._log(f"Lighting config: {lighting_path}", "light")

        if self.use_memory:
            self.memory.add(
                stage="stage12_render",
                type="result",
                content=code,
                metadata={
                    "title": "Stage Render - Lighting & Render Settings",
                    "summary": f"{code.count(chr(10)) + 1} lines with lighting and render settings",
                    "output_file": output_path,
                    "lighting_file": lighting_path,
                    "image_path": self.image_path
                },
                tags=["stage12_render", "blender_code", "render", "lighting"]
            )
            self._log("Saved to Memory", "success")

        return code

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------

    def run(self) -> tuple:
        """Run Stage 12 — lighting & render settings.

        Returns:
            (success, code)
        """
        print("\n" + "=" * 60)
        print("💡 Stage 12 - Lighting & Render Settings")
        print("=" * 60)

        # 1. Load data
        if not self._load_data():
            return False, None

        # 2. Analyze lighting
        print("\n--- Lighting Analysis ---")
        self.lighting_config = self._analyze_lighting()

        # 3. Generate lighting code
        print("\n--- Code Generation ---")
        code = self._generate_lighting_code()
        if not code:
            return False, None

        # 4. Save results
        code = self._save_results(code)

        print("\n" + "=" * 60)
        print("✅ Stage 12 complete!")
        print(f"   Output: {self.output_dir}")
        print("=" * 60)

        return True, code


def show_memory_status():
    memory = Memory(workspace_dir=os.path.dirname(os.path.abspath(__file__)))

    print("=" * 60)
    print("Memory Status")
    print("=" * 60)

    for stage in ["stage1", "stage2", "stage3", "stage4", "stage10_material", "stage12_render"]:
        entry = memory.get_latest(stage=stage, type="result")
        if entry:
            title = entry.metadata.get("title", "untitled")
            from datetime import datetime
            time_str = datetime.fromtimestamp(entry.timestamp).strftime("%m-%d %H:%M")
            print(f"  {stage}: {title} ({time_str})")
        else:
            print(f"  {stage}: no data")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Stage 12 - Lighting & Render Settings")
    parser.add_argument("--image", "-i", required=True, help="Reference image path (required)")
    parser.add_argument("--scene-code", "-c", required=True, help="Scene code path (required)")
    parser.add_argument("--output-dir", "-o", default=None, help="Output directory")
    parser.add_argument("--no-memory", action="store_true", help="Disable Memory system")
    parser.add_argument("--status", "-s", action="store_true", help="Show Memory status")

    args = parser.parse_args()

    if args.status:
        show_memory_status()
        return 0

    runner = StageRenderRunner(
        image_path=args.image,
        scene_code_path=args.scene_code,
        output_dir=args.output_dir,
        use_memory=not args.no_memory
    )

    success, code = runner.run()
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
