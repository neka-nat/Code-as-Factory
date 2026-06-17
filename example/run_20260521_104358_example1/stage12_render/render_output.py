"""
Stage Geometry Output - Integrated Scene with Detailed Geometry
===============================================================
Generated: 2026-05-21 11:16:16
Base scene: None
Objects with detailed geometry: 19
Total detailed parts: 192

This file integrates detailed geometry into the original scene.
Objects with detailed geometry are replaced, others remain as simple bbox.
"""

import bpy
import math
import mathutils

# ==============================================================================
# STAGE TEXTURE - Real texture maps generated via nanobanana
# ==============================================================================
import os as _tex_os
TEXTURE_DIR = r"/Users/yangyixuan/Code-as-Room_github/agent_utils/pipeline_output/run_20260521_104358_example1/stage11_texture/images"
FLOOR_TEXTURE = r"/Users/yangyixuan/Code-as-Room_github/agent_utils/pipeline_output/run_20260521_104358_example1/stage11_texture/images/floor.png"
WALL_TEXTURE = r"/Users/yangyixuan/Code-as-Room_github/agent_utils/pipeline_output/run_20260521_104358_example1/stage11_texture/images/wall.png"
FLOOR_TILE = (3.750, 2.500)
WALL_TILE = (3.750, 1.400)


# === HELPER FUNCTIONS ===
def clear_scene():
    """Clear all objects and collections."""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    for mesh in bpy.data.meshes:
        bpy.data.meshes.remove(mesh)
    for mat in bpy.data.materials:
        bpy.data.materials.remove(mat)
    for coll in bpy.data.collections:
        if coll.name != "Scene Collection":
            bpy.data.collections.remove(coll)

_BUMP_PROFILES = {
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
    return mat
create_material = create_pbr_material



def create_box(name, location, dimensions, rotation=(0,0,0), material=None, collection=None, show_direction=False):
    """Create a box primitive with optional red arrow direction indicator."""
    bpy.ops.mesh.primitive_cube_add(size=1, location=location, rotation=rotation)
    obj = bpy.context.active_object
    obj.name = name
    obj.dimensions = dimensions
    
    if material:
        obj.data.materials.append(material)
    
    if collection:
        old_colls = list(obj.users_collection)
        collection.objects.link(obj)
        for c in old_colls:
            c.objects.unlink(obj)
    
    # Add red arrow on top of object pointing to -Y (front)
    if show_direction and dimensions[0] > 0.3 and dimensions[1] > 0.3:
        pass
    return obj
def create_cylinder(name, location, dimensions, rotation=(0,0,0), material=None, collection=None):
    """Create a cylinder primitive."""
    bpy.ops.mesh.primitive_cylinder_add(radius=1, depth=1, location=location, rotation=rotation)
    obj = bpy.context.active_object
    obj.name = name
    obj.dimensions = dimensions
    if material:
        obj.data.materials.append(material)
    if collection:
        old_colls = list(obj.users_collection)
        collection.objects.link(obj)
        for c in old_colls:
            c.objects.unlink(obj)
    return obj

def create_collection(name):
    """Create and return a collection."""
    coll = bpy.data.collections.new(name)
    bpy.context.scene.collection.children.link(coll)
    return coll


# ==============================================================================
# DETAILED GEOMETRY DATA (Auto-generated)
# ==============================================================================

import bmesh

DETAILED_GEOMETRY = {
    "Armchair_North": {
        "center": [-2.75, 1.0, 0.45],
        "rotation": [0.0, 0.0, 0.7853981633974483],
        "parts": [
            {"type": "box", "name": "plush_seat_cushion", "loc": [0.0, -0.11, -0.2], "dim": [0.54, 0.54, 0.18], "rot": [0, 0, 0]},
            {"type": "box", "name": "left_square_arm", "loc": [-0.33, -0.03, -0.105], "dim": [0.14, 0.68, 0.45], "rot": [0, 0, 0]},
            {"type": "box", "name": "right_square_arm", "loc": [0.33, -0.03, -0.105], "dim": [0.14, 0.68, 0.45], "rot": [0, 0, 0]},
            {"type": "box", "name": "tall_padded_back_panel", "loc": [0.0, 0.32, 0.09], "dim": [0.8, 0.16, 0.72], "rot": [0, 0, 0]},
            {"type": "box", "name": "inner_back_cushion", "loc": [0.0, 0.215, 0.04], "dim": [0.54, 0.1, 0.54], "rot": [0, 0, 0]},
            {"type": "box", "name": "front_lower_apron", "loc": [0.0, -0.34, -0.305], "dim": [0.54, 0.08, 0.15], "rot": [0, 0, 0]},
            {"type": "box", "name": "left_front_leg", "loc": [-0.27, -0.28, -0.405], "dim": [0.08, 0.08, 0.09], "rot": [0, 0, 0]},
            {"type": "box", "name": "right_front_leg", "loc": [0.27, -0.28, -0.405], "dim": [0.08, 0.08, 0.09], "rot": [0, 0, 0]},
            {"type": "box", "name": "left_rear_leg", "loc": [-0.27, 0.24, -0.405], "dim": [0.08, 0.08, 0.09], "rot": [0, 0, 0]},
            {"type": "box", "name": "right_rear_leg", "loc": [0.27, 0.24, -0.405], "dim": [0.08, 0.08, 0.09], "rot": [0, 0, 0]},
        ]
    },
    "Armchair_South": {
        "center": [-2.75, -1.0, 0.45],
        "rotation": [0.0, 0.0, 2.356194490192345],
        "parts": [
            {"type": "box", "name": "plush_seat_base", "loc": [0.0, -0.05, -0.28], "dim": [0.72, 0.62, 0.18], "rot": [0, 0, 0]},
            {"type": "box", "name": "soft_seat_cushion", "loc": [0.0, -0.1, -0.13], "dim": [0.56, 0.52, 0.14], "rot": [0, 0, 0]},
            {"type": "box", "name": "left_padded_arm", "loc": [-0.34, -0.02, -0.16], "dim": [0.12, 0.74, 0.42], "rot": [0, 0, 0]},
            {"type": "box", "name": "right_padded_arm", "loc": [0.34, -0.02, -0.16], "dim": [0.12, 0.74, 0.42], "rot": [0, 0, 0]},
            {"type": "box", "name": "tall_square_backrest", "loc": [0.0, 0.33, 0.04], "dim": [0.8, 0.14, 0.82], "rot": [0, 0, 0]},
            {"type": "box", "name": "inner_back_cushion", "loc": [0.0, 0.235, 0.1], "dim": [0.58, 0.07, 0.56], "rot": [0, 0, 0]},
            {"type": "box", "name": "front_plush_apron", "loc": [0.0, -0.36, -0.25], "dim": [0.72, 0.08, 0.24], "rot": [0, 0, 0]},
            {"type": "box", "name": "front_left_short_leg", "loc": [-0.27, -0.28, -0.41], "dim": [0.08, 0.08, 0.08], "rot": [0, 0, 0]},
            {"type": "box", "name": "front_right_short_leg", "loc": [0.27, -0.28, -0.41], "dim": [0.08, 0.08, 0.08], "rot": [0, 0, 0]},
            {"type": "box", "name": "rear_left_short_leg", "loc": [-0.27, 0.26, -0.41], "dim": [0.08, 0.08, 0.08], "rot": [0, 0, 0]},
            {"type": "box", "name": "rear_right_short_leg", "loc": [0.27, 0.26, -0.41], "dim": [0.08, 0.08, 0.08], "rot": [0, 0, 0]},
        ]
    },
    "Side_Table": {
        "center": [-2.75, 0.0, 0.25],
        "rotation": [0.0, 0.0, 0.0],
        "parts": [
            {"type": "box", "name": "square_tabletop_slab", "loc": [0, 0, 0.225], "dim": [0.5, 0.5, 0.05], "rot": [0, 0, 0]},
            {"type": "box", "name": "front_left_leg", "loc": [-0.2, -0.2, -0.025], "dim": [0.05, 0.05, 0.45], "rot": [0, 0, 0]},
            {"type": "box", "name": "front_right_leg", "loc": [0.2, -0.2, -0.025], "dim": [0.05, 0.05, 0.45], "rot": [0, 0, 0]},
            {"type": "box", "name": "back_left_leg", "loc": [-0.2, 0.2, -0.025], "dim": [0.05, 0.05, 0.45], "rot": [0, 0, 0]},
            {"type": "box", "name": "back_right_leg", "loc": [0.2, 0.2, -0.025], "dim": [0.05, 0.05, 0.45], "rot": [0, 0, 0]},
            {"type": "box", "name": "front_apron", "loc": [0, -0.225, 0.165], "dim": [0.4, 0.03, 0.07], "rot": [0, 0, 0]},
            {"type": "box", "name": "back_apron", "loc": [0, 0.225, 0.165], "dim": [0.4, 0.03, 0.07], "rot": [0, 0, 0]},
            {"type": "box", "name": "left_side_apron", "loc": [-0.225, 0, 0.165], "dim": [0.03, 0.4, 0.07], "rot": [0, 0, 0]},
            {"type": "box", "name": "right_side_apron", "loc": [0.225, 0, 0.165], "dim": [0.03, 0.4, 0.07], "rot": [0, 0, 0]},
        ]
    },
    "Floor_Lamp": {
        "center": [-3.45, 2.2, 0.8],
        "rotation": [0.0, 0.0, 0.0],
        "parts": [
            {"type": "cylinder", "name": "round_brass_base", "loc": [0, 0, -0.78], "dim": [0.32, 0.32, 0.04], "rot": [0, 0, 0]},
            {"type": "cylinder", "name": "slender_brass_pole", "loc": [0, 0, -0.14], "dim": [0.035, 0.035, 1.24], "rot": [0, 0, 0]},
            {"type": "cylinder", "name": "lower_brass_shade_ring", "loc": [0, 0, 0.49], "dim": [0.36, 0.36, 0.025], "rot": [0, 0, 0]},
            {"type": "cylinder", "name": "warm_white_glass_cylindrical_shade", "loc": [0, 0, 0.64], "dim": [0.34, 0.34, 0.3], "rot": [0, 0, 0]},
            {"type": "cylinder", "name": "inner_glowing_glass_diffuser", "loc": [0, 0, 0.64], "dim": [0.24, 0.24, 0.26], "rot": [0, 0, 0]},
            {"type": "cylinder", "name": "upper_brass_shade_ring", "loc": [0, 0, 0.79], "dim": [0.36, 0.36, 0.02], "rot": [0, 0, 0]},
            {"type": "sphere", "name": "small_brass_top_finial", "loc": [0, 0, 0.78], "dim": [0.04, 0.04, 0.04], "rot": [0, 0, 0]},
        ]
    },
    "Bed": {
        "center": [0.0, 1.45, 0.3],
        "rotation": [0.0, 0.0, 0.0],
        "parts": [
            {"type": "box", "name": "light_beige_upholstered_headboard", "loc": [0, 0.99, 0], "dim": [1.8, 0.12, 0.6], "rot": [0, 0, 0]},
            {"type": "box", "name": "bed_base_platform", "loc": [0, -0.08, -0.24], "dim": [1.72, 1.9, 0.12], "rot": [0, 0, 0]},
            {"type": "box", "name": "left_side_rail", "loc": [-0.87, -0.08, -0.14], "dim": [0.06, 1.92, 0.2], "rot": [0, 0, 0]},
            {"type": "box", "name": "right_side_rail", "loc": [0.87, -0.08, -0.14], "dim": [0.06, 1.92, 0.2], "rot": [0, 0, 0]},
            {"type": "box", "name": "front_foot_rail", "loc": [0, -1.01, -0.14], "dim": [1.78, 0.08, 0.2], "rot": [0, 0, 0]},
            {"type": "box", "name": "white_mattress", "loc": [0, -0.08, -0.07], "dim": [1.66, 1.82, 0.22], "rot": [0, 0, 0]},
            {"type": "box", "name": "white_top_linen_duvet", "loc": [0, -0.15, 0.065], "dim": [1.72, 1.62, 0.07], "rot": [0, 0, 0]},
            {"type": "box", "name": "taupe_folded_throw_blanket_at_foot", "loc": [0, -0.76, 0.115], "dim": [1.72, 0.36, 0.05], "rot": [0, 0, 0]},
            {"type": "box", "name": "left_taupe_pillow", "loc": [-0.43, 0.55, 0.14], "dim": [0.56, 0.32, 0.12], "rot": [0, 0, 0]},
            {"type": "box", "name": "right_taupe_pillow", "loc": [0.43, 0.55, 0.14], "dim": [0.56, 0.32, 0.12], "rot": [0, 0, 0]},
            {"type": "box", "name": "center_white_pillow", "loc": [0, 0.72, 0.12], "dim": [0.62, 0.26, 0.1], "rot": [0, 0, 0]},
            {"type": "cylinder", "name": "front_left_short_bed_leg", "loc": [-0.72, -0.88, -0.27], "dim": [0.08, 0.08, 0.06], "rot": [0, 0, 0]},
            {"type": "cylinder", "name": "front_right_short_bed_leg", "loc": [0.72, -0.88, -0.27], "dim": [0.08, 0.08, 0.06], "rot": [0, 0, 0]},
            {"type": "cylinder", "name": "back_left_short_bed_leg", "loc": [-0.72, 0.72, -0.27], "dim": [0.08, 0.08, 0.06], "rot": [0, 0, 0]},
            {"type": "cylinder", "name": "back_right_short_bed_leg", "loc": [0.72, 0.72, -0.27], "dim": [0.08, 0.08, 0.06], "rot": [0, 0, 0]},
        ]
    },
    "Nightstand_North": {
        "center": [-1.2, 2.3, 0.25],
        "rotation": [0.0, 0.0, 0.0],
        "parts": [
            {"type": "box", "name": "top_surface_slab", "loc": [0, 0, 0.225], "dim": [0.5, 0.4, 0.05], "rot": [0, 0, 0]},
            {"type": "box", "name": "main_wood_cabinet_body", "loc": [0, 0.01, -0.025], "dim": [0.46, 0.36, 0.45], "rot": [0, 0, 0]},
            {"type": "box", "name": "upper_drawer_front", "loc": [0, -0.19, 0.085], "dim": [0.42, 0.02, 0.13], "rot": [0, 0, 0]},
            {"type": "box", "name": "lower_drawer_front", "loc": [0, -0.19, -0.085], "dim": [0.42, 0.02, 0.13], "rot": [0, 0, 0]},
            {"type": "cylinder", "name": "upper_drawer_handle", "loc": [0, -0.191, 0.085], "dim": [0.018, 0.018, 0.22], "rot": [0, 1.5708, 0]},
            {"type": "cylinder", "name": "lower_drawer_handle", "loc": [0, -0.191, -0.085], "dim": [0.018, 0.018, 0.22], "rot": [0, 1.5708, 0]},
            {"type": "box", "name": "recessed_toe_kick", "loc": [0, -0.08, -0.225], "dim": [0.36, 0.16, 0.05], "rot": [0, 0, 0]},
        ]
    },
    "Nightstand_South": {
        "center": [1.2, 2.3, 0.25],
        "rotation": [0.0, 0.0, 0.0],
        "parts": [
            {"type": "box", "name": "top_worktop_slab", "loc": [0, 0, 0.225], "dim": [0.5, 0.4, 0.05], "rot": [0, 0, 0]},
            {"type": "box", "name": "left_side_panel", "loc": [-0.235, 0, -0.025], "dim": [0.03, 0.38, 0.45], "rot": [0, 0, 0]},
            {"type": "box", "name": "right_side_panel", "loc": [0.235, 0, -0.025], "dim": [0.03, 0.38, 0.45], "rot": [0, 0, 0]},
            {"type": "box", "name": "back_panel", "loc": [0, 0.185, -0.025], "dim": [0.47, 0.03, 0.45], "rot": [0, 0, 0]},
            {"type": "box", "name": "bottom_panel", "loc": [0, 0, -0.235], "dim": [0.47, 0.38, 0.03], "rot": [0, 0, 0]},
            {"type": "box", "name": "upper_drawer_front", "loc": [0, -0.1875, 0.08], "dim": [0.43, 0.025, 0.18], "rot": [0, 0, 0]},
            {"type": "box", "name": "lower_drawer_front", "loc": [0, -0.1875, -0.12], "dim": [0.43, 0.025, 0.2], "rot": [0, 0, 0]},
            {"type": "box", "name": "center_drawer_gap", "loc": [0, -0.199, -0.01], "dim": [0.43, 0.006, 0.015], "rot": [0, 0, 0]},
            {"type": "cylinder", "name": "upper_drawer_handle", "loc": [0, -0.194, 0.08], "dim": [0.012, 0.012, 0.18], "rot": [0, 1.5708, 0]},
            {"type": "cylinder", "name": "lower_drawer_handle", "loc": [0, -0.194, -0.12], "dim": [0.012, 0.012, 0.18], "rot": [0, 1.5708, 0]},
            {"type": "box", "name": "front_toe_kick", "loc": [0, -0.16, -0.225], "dim": [0.38, 0.04, 0.05], "rot": [0, 0, 0]},
        ]
    },
    "Bench": {
        "center": [0.0, 0.1999999999999999, 0.225],
        "rotation": [0.0, 0.0, 0.0],
        "parts": [
            {"type": "box", "name": "upholstered_seat_cushion", "loc": [0, 0, 0.185], "dim": [1.2, 0.4, 0.08], "rot": [0, 0, 0]},
            {"type": "box", "name": "upholstered_base_panel", "loc": [0, 0, -0.01], "dim": [1.08, 0.32, 0.31], "rot": [0, 0, 0]},
            {"type": "box", "name": "front_apron", "loc": [0, -0.185, 0.035], "dim": [1.12, 0.03, 0.16], "rot": [0, 0, 0]},
            {"type": "box", "name": "back_apron", "loc": [0, 0.185, 0.035], "dim": [1.12, 0.03, 0.16], "rot": [0, 0, 0]},
            {"type": "box", "name": "left_apron", "loc": [-0.565, 0, 0.035], "dim": [0.03, 0.34, 0.16], "rot": [0, 0, 0]},
            {"type": "box", "name": "right_apron", "loc": [0.565, 0, 0.035], "dim": [0.03, 0.34, 0.16], "rot": [0, 0, 0]},
            {"type": "cylinder", "name": "front_left_short_leg", "loc": [-0.48, -0.14, -0.145], "dim": [0.07, 0.07, 0.16], "rot": [0, 0, 0]},
            {"type": "cylinder", "name": "front_right_short_leg", "loc": [0.48, -0.14, -0.145], "dim": [0.07, 0.07, 0.16], "rot": [0, 0, 0]},
            {"type": "cylinder", "name": "back_left_short_leg", "loc": [-0.48, 0.14, -0.145], "dim": [0.07, 0.07, 0.16], "rot": [0, 0, 0]},
            {"type": "cylinder", "name": "back_right_short_leg", "loc": [0.48, 0.14, -0.145], "dim": [0.07, 0.07, 0.16], "rot": [0, 0, 0]},
        ]
    },
    "Media_Console": {
        "center": [0.0, -2.275, 0.25],
        "rotation": [0.0, 0.0, 3.141592653589793],
        "parts": [
            {"type": "box", "name": "top_worktop_slab", "loc": [0, 0, 0.225], "dim": [1.8, 0.45, 0.05], "rot": [0, 0, 0]},
            {"type": "box", "name": "main_wood_console_body", "loc": [0, 0, -0.025], "dim": [1.76, 0.41, 0.45], "rot": [0, 0, 0]},
            {"type": "box", "name": "front_left_door_panel", "loc": [-0.66, -0.215, -0.035], "dim": [0.4, 0.02, 0.34], "rot": [0, 0, 0]},
            {"type": "box", "name": "front_mid_left_drawer_panel", "loc": [-0.22, -0.215, -0.035], "dim": [0.4, 0.02, 0.34], "rot": [0, 0, 0]},
            {"type": "box", "name": "front_mid_right_drawer_panel", "loc": [0.22, -0.215, -0.035], "dim": [0.4, 0.02, 0.34], "rot": [0, 0, 0]},
            {"type": "box", "name": "front_right_door_panel", "loc": [0.66, -0.215, -0.035], "dim": [0.4, 0.02, 0.34], "rot": [0, 0, 0]},
            {"type": "box", "name": "thin_shadow_gap_between_front_panels_1", "loc": [-0.44, -0.226, -0.035], "dim": [0.015, 0.008, 0.35], "rot": [0, 0, 0]},
            {"type": "box", "name": "thin_shadow_gap_between_front_panels_2", "loc": [0, -0.226, -0.035], "dim": [0.015, 0.008, 0.35], "rot": [0, 0, 0]},
            {"type": "box", "name": "thin_shadow_gap_between_front_panels_3", "loc": [0.44, -0.226, -0.035], "dim": [0.015, 0.008, 0.35], "rot": [0, 0, 0]},
            {"type": "cylinder", "name": "left_panel_handle", "loc": [-0.66, -0.216, 0.055], "dim": [0.018, 0.018, 0.18], "rot": [0, 1.5708, 0]},
            {"type": "cylinder", "name": "mid_left_panel_handle", "loc": [-0.22, -0.216, 0.055], "dim": [0.018, 0.018, 0.18], "rot": [0, 1.5708, 0]},
            {"type": "cylinder", "name": "mid_right_panel_handle", "loc": [0.22, -0.216, 0.055], "dim": [0.018, 0.018, 0.18], "rot": [0, 1.5708, 0]},
            {"type": "cylinder", "name": "right_panel_handle", "loc": [0.66, -0.216, 0.055], "dim": [0.018, 0.018, 0.18], "rot": [0, 1.5708, 0]},
            {"type": "box", "name": "recessed_toe_kick", "loc": [0, -0.12, -0.225], "dim": [1.55, 0.2, 0.05], "rot": [0, 0, 0]},
        ]
    },
    "Plant": {
        "center": [-1.45, -2.2, 0.4],
        "rotation": [0.0, 0.0, 0.0],
        "parts": [
            {"type": "cylinder", "name": "dark_grey_circular_planter_body", "loc": [0, 0, -0.27], "dim": [0.34, 0.34, 0.26], "rot": [0, 0, 0]},
            {"type": "cylinder", "name": "dark_grey_planter_top_rim", "loc": [0, 0, -0.12], "dim": [0.38, 0.38, 0.04], "rot": [0, 0, 0]},
            {"type": "cylinder", "name": "dark_soil_surface", "loc": [0, 0, -0.09], "dim": [0.3, 0.3, 0.025], "rot": [0, 0, 0]},
            {"type": "cylinder", "name": "central_plant_stem", "loc": [0, 0, 0.035], "dim": [0.035, 0.035, 0.25], "rot": [0, 0, 0]},
            {"type": "cylinder", "name": "left_plant_stem", "loc": [-0.07, 0.02, 0.015], "dim": [0.022, 0.022, 0.2], "rot": [0, 0, 0]},
            {"type": "cylinder", "name": "right_plant_stem", "loc": [0.07, -0.015, 0.015], "dim": [0.022, 0.022, 0.2], "rot": [0, 0, 0]},
            {"type": "sphere", "name": "central_bushy_green_leaves", "loc": [0, 0, 0.18], "dim": [0.3, 0.3, 0.3], "rot": [0, 0, 0]},
            {"type": "sphere", "name": "top_bushy_green_leaves", "loc": [0.01, 0.01, 0.28], "dim": [0.24, 0.24, 0.24], "rot": [0, 0, 0]},
            {"type": "sphere", "name": "left_bushy_green_leaves", "loc": [-0.11, 0.015, 0.13], "dim": [0.25, 0.25, 0.25], "rot": [0, 0, 0]},
            {"type": "sphere", "name": "right_bushy_green_leaves", "loc": [0.11, -0.01, 0.14], "dim": [0.25, 0.25, 0.25], "rot": [0, 0, 0]},
            {"type": "sphere", "name": "front_bushy_green_leaves", "loc": [0, -0.11, 0.14], "dim": [0.24, 0.24, 0.24], "rot": [0, 0, 0]},
            {"type": "sphere", "name": "back_bushy_green_leaves", "loc": [0.005, 0.11, 0.13], "dim": [0.24, 0.24, 0.24], "rot": [0, 0, 0]},
        ]
    },
    "Wardrobe_East": {
        "center": [3.45, 0.9, 1.2],
        "rotation": [0.0, 0.0, -1.5707963267948966],
        "parts": [
            {"type": "box", "name": "left_side_panel", "loc": [-0.98, 0, 0], "dim": [0.04, 0.6, 2.4], "rot": [0, 0, 0]},
            {"type": "box", "name": "right_side_panel", "loc": [0.98, 0, 0], "dim": [0.04, 0.6, 2.4], "rot": [0, 0, 0]},
            {"type": "box", "name": "back_panel", "loc": [0, 0.28, 0], "dim": [2.0, 0.04, 2.4], "rot": [0, 0, 0]},
            {"type": "box", "name": "top_panel", "loc": [0, 0, 1.18], "dim": [2.0, 0.6, 0.04], "rot": [0, 0, 0]},
            {"type": "box", "name": "bottom_panel", "loc": [0, 0, -1.18], "dim": [2.0, 0.6, 0.04], "rot": [0, 0, 0]},
            {"type": "box", "name": "left_vertical_divider", "loc": [-0.34, 0, 0], "dim": [0.04, 0.56, 2.32], "rot": [0, 0, 0]},
            {"type": "box", "name": "right_vertical_divider", "loc": [0.46, 0, 0], "dim": [0.04, 0.56, 2.32], "rot": [0, 0, 0]},
            {"type": "box", "name": "left_shelf_1", "loc": [-0.66, -0.005, -0.55], "dim": [0.56, 0.54, 0.03], "rot": [0, 0, 0]},
            {"type": "box", "name": "left_shelf_2", "loc": [-0.66, -0.005, 0.05], "dim": [0.56, 0.54, 0.03], "rot": [0, 0, 0]},
            {"type": "box", "name": "left_shelf_3", "loc": [-0.66, -0.005, 0.65], "dim": [0.56, 0.54, 0.03], "rot": [0, 0, 0]},
            {"type": "box", "name": "right_shelf_1", "loc": [0.72, -0.005, -0.55], "dim": [0.44, 0.54, 0.03], "rot": [0, 0, 0]},
            {"type": "box", "name": "right_shelf_2", "loc": [0.72, -0.005, 0.05], "dim": [0.44, 0.54, 0.03], "rot": [0, 0, 0]},
            {"type": "box", "name": "right_shelf_3", "loc": [0.72, -0.005, 0.65], "dim": [0.44, 0.54, 0.03], "rot": [0, 0, 0]},
            {"type": "box", "name": "hanging_bay_upper_shelf", "loc": [0.06, -0.005, 0.95], "dim": [0.72, 0.54, 0.03], "rot": [0, 0, 0]},
            {"type": "cylinder", "name": "hanging_rail", "loc": [0.06, -0.08, 0.72], "dim": [0.035, 0.035, 0.68], "rot": [0, 1.5708, 0]},
            {"type": "box", "name": "hanging_clothing_left", "loc": [-0.16, -0.1, 0.32], "dim": [0.18, 0.06, 0.72], "rot": [0, 0, 0]},
            {"type": "box", "name": "hanging_clothing_center", "loc": [0.06, -0.1, 0.27], "dim": [0.2, 0.06, 0.82], "rot": [0, 0, 0]},
            {"type": "box", "name": "hanging_clothing_right", "loc": [0.28, -0.1, 0.34], "dim": [0.17, 0.06, 0.68], "rot": [0, 0, 0]},
        ]
    },
    "Wardrobe_North": {
        "center": [3.0, 2.2, 1.2],
        "rotation": [0.0, 0.0, 0.0],
        "parts": [
            {"type": "box", "name": "left_side_panel", "loc": [-0.73, 0, 0], "dim": [0.04, 0.6, 2.4], "rot": [0, 0, 0]},
            {"type": "box", "name": "right_side_panel", "loc": [0.73, 0, 0], "dim": [0.04, 0.6, 2.4], "rot": [0, 0, 0]},
            {"type": "box", "name": "back_panel", "loc": [0, 0.285, 0], "dim": [1.5, 0.03, 2.4], "rot": [0, 0, 0]},
            {"type": "box", "name": "top_panel", "loc": [0, 0, 1.18], "dim": [1.5, 0.6, 0.04], "rot": [0, 0, 0]},
            {"type": "box", "name": "bottom_panel", "loc": [0, 0, -1.18], "dim": [1.5, 0.6, 0.04], "rot": [0, 0, 0]},
            {"type": "box", "name": "upper_shelf", "loc": [0, 0, 0.82], "dim": [1.42, 0.54, 0.035], "rot": [0, 0, 0]},
            {"type": "box", "name": "lower_shelf", "loc": [0, 0, -0.82], "dim": [1.42, 0.54, 0.035], "rot": [0, 0, 0]},
            {"type": "cylinder", "name": "hanging_clothes_rail", "loc": [0, -0.03, 0.55], "dim": [0.035, 0.035, 1.3], "rot": [0, 1.5708, 0]},
            {"type": "box", "name": "left_front_stile", "loc": [-0.73, -0.285, 0], "dim": [0.055, 0.03, 2.32], "rot": [0, 0, 0]},
            {"type": "box", "name": "right_front_stile", "loc": [0.73, -0.285, 0], "dim": [0.055, 0.03, 2.32], "rot": [0, 0, 0]},
            {"type": "box", "name": "top_front_rail", "loc": [0, -0.285, 1.13], "dim": [1.46, 0.03, 0.08], "rot": [0, 0, 0]},
            {"type": "box", "name": "bottom_front_rail", "loc": [0, -0.285, -1.13], "dim": [1.46, 0.03, 0.08], "rot": [0, 0, 0]},
        ]
    },
    "Closet_Bench": {
        "center": [2.75, 0.0, 0.225],
        "rotation": [0.0, 0.0, 0.0],
        "parts": [
            {"type": "box", "name": "wooden_bench_top_slab", "loc": [0, 0, 0.195], "dim": [0.4, 1.2, 0.06], "rot": [0, 0, 0]},
            {"type": "box", "name": "front_left_leg", "loc": [-0.15, -0.51, -0.03], "dim": [0.06, 0.06, 0.39], "rot": [0, 0, 0]},
            {"type": "box", "name": "front_right_leg", "loc": [0.15, -0.51, -0.03], "dim": [0.06, 0.06, 0.39], "rot": [0, 0, 0]},
            {"type": "box", "name": "back_left_leg", "loc": [-0.15, 0.51, -0.03], "dim": [0.06, 0.06, 0.39], "rot": [0, 0, 0]},
            {"type": "box", "name": "back_right_leg", "loc": [0.15, 0.51, -0.03], "dim": [0.06, 0.06, 0.39], "rot": [0, 0, 0]},
            {"type": "box", "name": "left_side_support_apron", "loc": [-0.18, 0, 0.11], "dim": [0.04, 1.08, 0.08], "rot": [0, 0, 0]},
            {"type": "box", "name": "right_side_support_apron", "loc": [0.18, 0, 0.11], "dim": [0.04, 1.08, 0.08], "rot": [0, 0, 0]},
            {"type": "box", "name": "front_end_support_apron", "loc": [0, -0.56, 0.11], "dim": [0.32, 0.04, 0.08], "rot": [0, 0, 0]},
            {"type": "box", "name": "back_end_support_apron", "loc": [0, 0.56, 0.11], "dim": [0.32, 0.04, 0.08], "rot": [0, 0, 0]},
        ]
    },
    "Dresser": {
        "center": [3.5, -0.8, 0.45],
        "rotation": [0.0, 0.0, -1.5707963267948966],
        "parts": [
            {"type": "box", "name": "dresser_main_carcass", "loc": [0, 0.015, -0.035], "dim": [1.16, 0.46, 0.79], "rot": [0, 0, 0]},
            {"type": "box", "name": "top_surface_slab", "loc": [0, 0, 0.42], "dim": [1.2, 0.5, 0.06], "rot": [0, 0, 0]},
            {"type": "box", "name": "bottom_plinth_base", "loc": [0, 0.03, -0.425], "dim": [1.12, 0.38, 0.05], "rot": [0, 0, 0]},
            {"type": "box", "name": "front_drawer_top", "loc": [0, -0.235, 0.255], "dim": [1.08, 0.03, 0.16], "rot": [0, 0, 0]},
            {"type": "box", "name": "front_drawer_upper_middle", "loc": [0, -0.235, 0.065], "dim": [1.08, 0.03, 0.16], "rot": [0, 0, 0]},
            {"type": "box", "name": "front_drawer_lower_middle", "loc": [0, -0.235, -0.125], "dim": [1.08, 0.03, 0.16], "rot": [0, 0, 0]},
            {"type": "box", "name": "front_drawer_bottom", "loc": [0, -0.235, -0.315], "dim": [1.08, 0.03, 0.16], "rot": [0, 0, 0]},
            {"type": "box", "name": "drawer_gap_horizontal_1", "loc": [0, -0.252, 0.16], "dim": [1.1, 0.01, 0.015], "rot": [0, 0, 0]},
            {"type": "box", "name": "drawer_gap_horizontal_2", "loc": [0, -0.252, -0.03], "dim": [1.1, 0.01, 0.015], "rot": [0, 0, 0]},
            {"type": "box", "name": "drawer_gap_horizontal_3", "loc": [0, -0.252, -0.22], "dim": [1.1, 0.01, 0.015], "rot": [0, 0, 0]},
            {"type": "cylinder", "name": "top_drawer_handle", "loc": [0, -0.248, 0.255], "dim": [0.025, 0.025, 0.32], "rot": [0, 1.5708, 0]},
            {"type": "cylinder", "name": "upper_middle_drawer_handle", "loc": [0, -0.248, 0.065], "dim": [0.025, 0.025, 0.32], "rot": [0, 1.5708, 0]},
            {"type": "cylinder", "name": "lower_middle_drawer_handle", "loc": [0, -0.248, -0.125], "dim": [0.025, 0.025, 0.32], "rot": [0, 1.5708, 0]},
            {"type": "cylinder", "name": "bottom_drawer_handle", "loc": [0, -0.248, -0.315], "dim": [0.025, 0.025, 0.32], "rot": [0, 1.5708, 0]},
        ]
    },
    "Floor_Mirror": {
        "center": [3.25, -2.0, 0.9],
        "rotation": [0.0, 0.0, -1.5707963267948966],
        "parts": [
            {"type": "box", "name": "left_wood_frame", "loc": [-0.37, 0, 0], "dim": [0.06, 0.08, 1.8], "rot": [0, 0, 0]},
            {"type": "box", "name": "right_wood_frame", "loc": [0.37, 0, 0], "dim": [0.06, 0.08, 1.8], "rot": [0, 0, 0]},
            {"type": "box", "name": "top_wood_frame", "loc": [0, 0, 0.86], "dim": [0.8, 0.08, 0.08], "rot": [0, 0, 0]},
            {"type": "box", "name": "bottom_wood_frame", "loc": [0, 0, -0.86], "dim": [0.8, 0.08, 0.08], "rot": [0, 0, 0]},
            {"type": "box", "name": "front_glass_mirror_panel", "loc": [0, -0.046, 0], "dim": [0.68, 0.008, 1.64], "rot": [0, 0, 0]},
            {"type": "box", "name": "thin_backing_panel", "loc": [0, 0.043, 0], "dim": [0.72, 0.01, 1.68], "rot": [0, 0, 0]},
        ]
    },
    "Orphan_obj_004_Book": {
        "center": [-2.75, -0.08999999999999998, 0.55],
        "rotation": [0.0, 0.0, 0.0],
        "parts": [
            {"type": "box", "name": "page_block", "loc": [0.005, 0, 0], "dim": [0.205, 0.145, 0.076], "rot": [0, 0, 0]},
            {"type": "box", "name": "top_cover", "loc": [0, 0, 0.044], "dim": [0.22, 0.16, 0.012], "rot": [0, 0, 0]},
            {"type": "box", "name": "bottom_cover", "loc": [0, 0, -0.044], "dim": [0.22, 0.16, 0.012], "rot": [0, 0, 0]},
            {"type": "box", "name": "spine_binding", "loc": [-0.105, 0, 0], "dim": [0.01, 0.16, 0.1], "rot": [0, 0, 0]},
            {"type": "box", "name": "front_page_edge", "loc": [0.109, 0, 0], "dim": [0.002, 0.14, 0.07], "rot": [0, 0, 0]},
        ]
    },
    "Orphan_obj_009_Table_Lamp_North": {
        "center": [-1.2, 2.2474999999999996, 0.55],
        "rotation": [0.0, 0.0, 0.0],
        "parts": [
            {"type": "cylinder", "name": "gold_metal_round_base", "loc": [0, 0, -0.044], "dim": [0.09, 0.07, 0.012], "rot": [0, 0, 0]},
            {"type": "cylinder", "name": "gold_metal_stem", "loc": [0, 0, -0.017], "dim": [0.014, 0.014, 0.045], "rot": [0, 0, 0]},
            {"type": "sphere", "name": "warm_glowing_glass_bulb", "loc": [0, 0, 0.011], "dim": [0.04, 0.04, 0.04], "rot": [0, 0, 0]},
            {"type": "cylinder", "name": "warm_yellow_glass_lampshade", "loc": [0, 0, 0.0225], "dim": [0.18, 0.13, 0.055], "rot": [0, 0, 0]},
            {"type": "cylinder", "name": "gold_metal_lower_shade_rim", "loc": [0, 0, -0.004], "dim": [0.185, 0.135, 0.006], "rot": [0, 0, 0]},
            {"type": "cylinder", "name": "gold_metal_upper_shade_rim", "loc": [0, 0, 0.047], "dim": [0.16, 0.115, 0.006], "rot": [0, 0, 0]},
            {"type": "sphere", "name": "gold_metal_top_finial", "loc": [0, 0, 0.041], "dim": [0.018, 0.018, 0.018], "rot": [0, 0, 0]},
        ]
    },
    "Orphan_obj_012_Table_Lamp_South": {
        "center": [1.2, 2.2474999999999996, 0.55],
        "rotation": [0.0, 0.0, 0.0],
        "parts": [
            {"type": "cylinder", "name": "gold_metal_base", "loc": [0, 0, -0.044], "dim": [0.085, 0.065, 0.012], "rot": [0, 0, 0]},
            {"type": "cylinder", "name": "slender_gold_stem", "loc": [0, 0, -0.017], "dim": [0.018, 0.018, 0.055], "rot": [0, 0, 0]},
            {"type": "sphere", "name": "warm_glowing_bulb", "loc": [0, 0, 0.006], "dim": [0.04, 0.04, 0.04], "rot": [0, 0, 0]},
            {"type": "cylinder", "name": "warm_yellow_glass_lampshade", "loc": [0, 0, 0.018], "dim": [0.18, 0.13, 0.05], "rot": [0, 0, 0]},
            {"type": "sphere", "name": "small_gold_finial", "loc": [0, 0, 0.047], "dim": [0.012, 0.012, 0.012], "rot": [0, 0, 0]},
        ]
    },
    "Orphan_obj_017_Decor_Item_Console": {
        "center": [0.592, -2.20375, 0.55],
        "rotation": [0.0, 0.0, 3.141592653589793],
        "parts": [
            {"type": "box", "name": "shallow_rectangular_display_tray_base", "loc": [0, 0, -0.045], "dim": [0.2, 0.14, 0.01], "rot": [0, 0, 0]},
            {"type": "box", "name": "front_tray_lip", "loc": [0, -0.066, -0.037], "dim": [0.2, 0.008, 0.016], "rot": [0, 0, 0]},
            {"type": "box", "name": "back_tray_lip", "loc": [0, 0.066, -0.037], "dim": [0.2, 0.008, 0.016], "rot": [0, 0, 0]},
            {"type": "box", "name": "left_tray_lip", "loc": [-0.096, 0, -0.037], "dim": [0.008, 0.14, 0.016], "rot": [0, 0, 0]},
            {"type": "box", "name": "right_tray_lip", "loc": [0.096, 0, -0.037], "dim": [0.008, 0.14, 0.016], "rot": [0, 0, 0]},
            {"type": "sphere", "name": "rounded_ceramic_vase_body", "loc": [-0.05, -0.005, -0.0125], "dim": [0.06, 0.06, 0.055], "rot": [0, 0, 0]},
            {"type": "cylinder", "name": "narrow_ceramic_vase_neck", "loc": [-0.05, -0.005, 0.0325], "dim": [0.026, 0.026, 0.035], "rot": [0, 0, 0]},
            {"type": "cylinder", "name": "short_metal_candle_holder", "loc": [0.035, 0.025, -0.012], "dim": [0.038, 0.038, 0.056], "rot": [0, 0, 0]},
            {"type": "sphere", "name": "small_rounded_decor_orb", "loc": [0.072, -0.028, -0.018], "dim": [0.042, 0.042, 0.042], "rot": [0, 0, 0]},
            {"type": "cone", "name": "small_tapered_metal_finial", "loc": [0.035, 0.025, 0.029], "dim": [0.026, 0.026, 0.026], "rot": [0, 0, 0]},
        ]
    },
}


# ==============================================================================
# PER-PART MATERIAL DATA (Auto-generated by Stage Material)
# ==============================================================================

PART_MATERIALS = {
    "Orphan_obj_017_Decor_Item_Console": {
        "shallow_rectangular_display_tray_base": {"base_color": (0.350, 0.270, 0.200, 1.0), "roughness": 0.4, "metallic": 0.0, "specular": 0.5, "type": "wood"},
        "front_tray_lip": {"base_color": (0.350, 0.270, 0.200, 1.0), "roughness": 0.4, "metallic": 0.0, "specular": 0.5, "type": "wood"},
        "back_tray_lip": {"base_color": (0.350, 0.270, 0.200, 1.0), "roughness": 0.4, "metallic": 0.0, "specular": 0.5, "type": "wood"},
        "left_tray_lip": {"base_color": (0.350, 0.270, 0.200, 1.0), "roughness": 0.4, "metallic": 0.0, "specular": 0.5, "type": "wood"},
        "right_tray_lip": {"base_color": (0.350, 0.270, 0.200, 1.0), "roughness": 0.4, "metallic": 0.0, "specular": 0.5, "type": "wood"},
        "rounded_ceramic_vase_body": {"base_color": (0.850, 0.820, 0.780, 1.0), "roughness": 0.2, "metallic": 0.0, "specular": 0.5, "type": "ceramic"},
        "narrow_ceramic_vase_neck": {"base_color": (0.850, 0.820, 0.780, 1.0), "roughness": 0.2, "metallic": 0.0, "specular": 0.5, "type": "ceramic"},
        "short_metal_candle_holder": {"base_color": (0.600, 0.450, 0.200, 1.0), "roughness": 0.3, "metallic": 0.85, "specular": 0.45, "type": "anodized_aluminum"},
        "small_rounded_decor_orb": {"base_color": (0.100, 0.100, 0.100, 1.0), "roughness": 0.4, "metallic": 0.2, "specular": 0.5, "type": "painted_metal"},
        "small_tapered_metal_finial": {"base_color": (0.600, 0.450, 0.200, 1.0), "roughness": 0.3, "metallic": 0.85, "specular": 0.45, "type": "anodized_aluminum"},
    },
    "Armchair_North": {
        "plush_seat_cushion": {"base_color": (0.750, 0.720, 0.680, 1.0), "roughness": 0.9, "metallic": 0.0, "specular": 0.3, "type": "fabric"},
        "left_square_arm": {"base_color": (0.750, 0.720, 0.680, 1.0), "roughness": 0.9, "metallic": 0.0, "specular": 0.3, "type": "fabric"},
        "right_square_arm": {"base_color": (0.750, 0.720, 0.680, 1.0), "roughness": 0.9, "metallic": 0.0, "specular": 0.3, "type": "fabric"},
        "tall_padded_back_panel": {"base_color": (0.750, 0.720, 0.680, 1.0), "roughness": 0.9, "metallic": 0.0, "specular": 0.3, "type": "fabric"},
        "inner_back_cushion": {"base_color": (0.750, 0.720, 0.680, 1.0), "roughness": 0.9, "metallic": 0.0, "specular": 0.3, "type": "fabric"},
        "front_lower_apron": {"base_color": (0.750, 0.720, 0.680, 1.0), "roughness": 0.9, "metallic": 0.0, "specular": 0.3, "type": "fabric"},
        "left_front_leg": {"base_color": (0.350, 0.270, 0.200, 1.0), "roughness": 0.4, "metallic": 0.0, "specular": 0.5, "type": "wood"},
        "right_front_leg": {"base_color": (0.350, 0.270, 0.200, 1.0), "roughness": 0.4, "metallic": 0.0, "specular": 0.5, "type": "wood"},
        "left_rear_leg": {"base_color": (0.350, 0.270, 0.200, 1.0), "roughness": 0.4, "metallic": 0.0, "specular": 0.5, "type": "wood"},
        "right_rear_leg": {"base_color": (0.350, 0.270, 0.200, 1.0), "roughness": 0.4, "metallic": 0.0, "specular": 0.5, "type": "wood"},
    },
    "Armchair_South": {
        "plush_seat_base": {"base_color": (0.750, 0.720, 0.680, 1.0), "roughness": 0.9, "metallic": 0.0, "specular": 0.3, "type": "fabric"},
        "soft_seat_cushion": {"base_color": (0.750, 0.720, 0.680, 1.0), "roughness": 0.9, "metallic": 0.0, "specular": 0.3, "type": "fabric"},
        "left_padded_arm": {"base_color": (0.750, 0.720, 0.680, 1.0), "roughness": 0.9, "metallic": 0.0, "specular": 0.3, "type": "fabric"},
        "right_padded_arm": {"base_color": (0.750, 0.720, 0.680, 1.0), "roughness": 0.9, "metallic": 0.0, "specular": 0.3, "type": "fabric"},
        "tall_square_backrest": {"base_color": (0.750, 0.720, 0.680, 1.0), "roughness": 0.9, "metallic": 0.0, "specular": 0.3, "type": "fabric"},
        "inner_back_cushion": {"base_color": (0.750, 0.720, 0.680, 1.0), "roughness": 0.9, "metallic": 0.0, "specular": 0.3, "type": "fabric"},
        "front_plush_apron": {"base_color": (0.750, 0.720, 0.680, 1.0), "roughness": 0.9, "metallic": 0.0, "specular": 0.3, "type": "fabric"},
        "front_left_short_leg": {"base_color": (0.350, 0.270, 0.200, 1.0), "roughness": 0.4, "metallic": 0.0, "specular": 0.5, "type": "wood"},
        "front_right_short_leg": {"base_color": (0.350, 0.270, 0.200, 1.0), "roughness": 0.4, "metallic": 0.0, "specular": 0.5, "type": "wood"},
        "rear_left_short_leg": {"base_color": (0.350, 0.270, 0.200, 1.0), "roughness": 0.4, "metallic": 0.0, "specular": 0.5, "type": "wood"},
        "rear_right_short_leg": {"base_color": (0.350, 0.270, 0.200, 1.0), "roughness": 0.4, "metallic": 0.0, "specular": 0.5, "type": "wood"},
    },
    "Side_Table": {
        "square_tabletop_slab": {"base_color": (0.350, 0.270, 0.200, 1.0), "roughness": 0.4, "metallic": 0.0, "specular": 0.5, "type": "wood"},
        "front_left_leg": {"base_color": (0.350, 0.270, 0.200, 1.0), "roughness": 0.4, "metallic": 0.0, "specular": 0.5, "type": "wood"},
        "front_right_leg": {"base_color": (0.350, 0.270, 0.200, 1.0), "roughness": 0.4, "metallic": 0.0, "specular": 0.5, "type": "wood"},
        "back_left_leg": {"base_color": (0.350, 0.270, 0.200, 1.0), "roughness": 0.4, "metallic": 0.0, "specular": 0.5, "type": "wood"},
        "back_right_leg": {"base_color": (0.350, 0.270, 0.200, 1.0), "roughness": 0.4, "metallic": 0.0, "specular": 0.5, "type": "wood"},
        "front_apron": {"base_color": (0.350, 0.270, 0.200, 1.0), "roughness": 0.4, "metallic": 0.0, "specular": 0.5, "type": "wood"},
        "back_apron": {"base_color": (0.350, 0.270, 0.200, 1.0), "roughness": 0.4, "metallic": 0.0, "specular": 0.5, "type": "wood"},
        "left_side_apron": {"base_color": (0.350, 0.270, 0.200, 1.0), "roughness": 0.4, "metallic": 0.0, "specular": 0.5, "type": "wood"},
        "right_side_apron": {"base_color": (0.350, 0.270, 0.200, 1.0), "roughness": 0.4, "metallic": 0.0, "specular": 0.5, "type": "wood"},
    },
    "Floor_Lamp": {
        "round_brass_base": {"base_color": (0.600, 0.450, 0.200, 1.0), "roughness": 0.3, "metallic": 0.85, "specular": 0.45, "type": "anodized_aluminum"},
        "slender_brass_pole": {"base_color": (0.600, 0.450, 0.200, 1.0), "roughness": 0.3, "metallic": 0.85, "specular": 0.45, "type": "anodized_aluminum"},
        "lower_brass_shade_ring": {"base_color": (0.600, 0.450, 0.200, 1.0), "roughness": 0.3, "metallic": 0.85, "specular": 0.45, "type": "anodized_aluminum"},
        "warm_white_glass_cylindrical_shade": {"base_color": (0.950, 0.950, 0.950, 1.0), "roughness": 0.05, "metallic": 0.0, "specular": 0.5, "type": "glass"},
        "inner_glowing_glass_diffuser": {"base_color": (1.000, 0.900, 0.800, 1.0), "roughness": 0.5, "metallic": 0.0, "specular": 0.3, "type": "emission"},
        "upper_brass_shade_ring": {"base_color": (0.600, 0.450, 0.200, 1.0), "roughness": 0.3, "metallic": 0.85, "specular": 0.45, "type": "anodized_aluminum"},
        "small_brass_top_finial": {"base_color": (0.600, 0.450, 0.200, 1.0), "roughness": 0.3, "metallic": 0.85, "specular": 0.45, "type": "anodized_aluminum"},
    },
    "Bed": {
        "light_beige_upholstered_headboard": {"base_color": (0.750, 0.720, 0.680, 1.0), "roughness": 0.9, "metallic": 0.0, "specular": 0.3, "type": "fabric"},
        "bed_base_platform": {"base_color": (0.750, 0.720, 0.680, 1.0), "roughness": 0.9, "metallic": 0.0, "specular": 0.3, "type": "fabric"},
        "left_side_rail": {"base_color": (0.750, 0.720, 0.680, 1.0), "roughness": 0.9, "metallic": 0.0, "specular": 0.3, "type": "fabric"},
        "right_side_rail": {"base_color": (0.750, 0.720, 0.680, 1.0), "roughness": 0.9, "metallic": 0.0, "specular": 0.3, "type": "fabric"},
        "front_foot_rail": {"base_color": (0.750, 0.720, 0.680, 1.0), "roughness": 0.9, "metallic": 0.0, "specular": 0.3, "type": "fabric"},
        "white_mattress": {"base_color": (0.900, 0.900, 0.900, 1.0), "roughness": 0.9, "metallic": 0.0, "specular": 0.3, "type": "fabric"},
        "white_top_linen_duvet": {"base_color": (0.900, 0.900, 0.900, 1.0), "roughness": 0.9, "metallic": 0.0, "specular": 0.3, "type": "fabric"},
        "taupe_folded_throw_blanket_at_foot": {"base_color": (0.350, 0.310, 0.270, 1.0), "roughness": 0.9, "metallic": 0.0, "specular": 0.3, "type": "fabric"},
        "left_taupe_pillow": {"base_color": (0.350, 0.310, 0.270, 1.0), "roughness": 0.9, "metallic": 0.0, "specular": 0.3, "type": "fabric"},
        "right_taupe_pillow": {"base_color": (0.350, 0.310, 0.270, 1.0), "roughness": 0.9, "metallic": 0.0, "specular": 0.3, "type": "fabric"},
        "center_white_pillow": {"base_color": (0.900, 0.900, 0.900, 1.0), "roughness": 0.9, "metallic": 0.0, "specular": 0.3, "type": "fabric"},
        "front_left_short_bed_leg": {"base_color": (0.350, 0.270, 0.200, 1.0), "roughness": 0.4, "metallic": 0.0, "specular": 0.5, "type": "wood"},
        "front_right_short_bed_leg": {"base_color": (0.350, 0.270, 0.200, 1.0), "roughness": 0.4, "metallic": 0.0, "specular": 0.5, "type": "wood"},
        "back_left_short_bed_leg": {"base_color": (0.350, 0.270, 0.200, 1.0), "roughness": 0.4, "metallic": 0.0, "specular": 0.5, "type": "wood"},
        "back_right_short_bed_leg": {"base_color": (0.350, 0.270, 0.200, 1.0), "roughness": 0.4, "metallic": 0.0, "specular": 0.5, "type": "wood"},
    },
    "Nightstand_North": {
        "top_surface_slab": {"base_color": (0.350, 0.270, 0.200, 1.0), "roughness": 0.3, "metallic": 0.0, "specular": 0.5, "type": "wood"},
        "main_wood_cabinet_body": {"base_color": (0.280, 0.240, 0.200, 1.0), "roughness": 0.4, "metallic": 0.0, "specular": 0.5, "type": "wood"},
        "upper_drawer_front": {"base_color": (0.280, 0.240, 0.200, 1.0), "roughness": 0.4, "metallic": 0.0, "specular": 0.5, "type": "wood"},
        "lower_drawer_front": {"base_color": (0.280, 0.240, 0.200, 1.0), "roughness": 0.4, "metallic": 0.0, "specular": 0.5, "type": "wood"},
        "upper_drawer_handle": {"base_color": (0.600, 0.450, 0.200, 1.0), "roughness": 0.3, "metallic": 0.85, "specular": 0.45, "type": "anodized_aluminum"},
        "lower_drawer_handle": {"base_color": (0.600, 0.450, 0.200, 1.0), "roughness": 0.3, "metallic": 0.85, "specular": 0.45, "type": "anodized_aluminum"},
        "recessed_toe_kick": {"base_color": (0.280, 0.240, 0.200, 1.0), "roughness": 0.4, "metallic": 0.0, "specular": 0.5, "type": "wood"},
    },
    "Closet_Bench": {
        "wooden_bench_top_slab": {"base_color": (0.350, 0.270, 0.200, 1.0), "roughness": 0.4, "metallic": 0.0, "specular": 0.5, "type": "wood"},
        "front_left_leg": {"base_color": (0.350, 0.270, 0.200, 1.0), "roughness": 0.4, "metallic": 0.0, "specular": 0.5, "type": "wood"},
        "front_right_leg": {"base_color": (0.350, 0.270, 0.200, 1.0), "roughness": 0.4, "metallic": 0.0, "specular": 0.5, "type": "wood"},
        "back_left_leg": {"base_color": (0.350, 0.270, 0.200, 1.0), "roughness": 0.4, "metallic": 0.0, "specular": 0.5, "type": "wood"},
        "back_right_leg": {"base_color": (0.350, 0.270, 0.200, 1.0), "roughness": 0.4, "metallic": 0.0, "specular": 0.5, "type": "wood"},
        "left_side_support_apron": {"base_color": (0.350, 0.270, 0.200, 1.0), "roughness": 0.4, "metallic": 0.0, "specular": 0.5, "type": "wood"},
        "right_side_support_apron": {"base_color": (0.350, 0.270, 0.200, 1.0), "roughness": 0.4, "metallic": 0.0, "specular": 0.5, "type": "wood"},
        "front_end_support_apron": {"base_color": (0.350, 0.270, 0.200, 1.0), "roughness": 0.4, "metallic": 0.0, "specular": 0.5, "type": "wood"},
        "back_end_support_apron": {"base_color": (0.350, 0.270, 0.200, 1.0), "roughness": 0.4, "metallic": 0.0, "specular": 0.5, "type": "wood"},
    },
    "Dresser": {
        "dresser_main_carcass": {"base_color": (0.280, 0.240, 0.200, 1.0), "roughness": 0.4, "metallic": 0.0, "specular": 0.5, "type": "wood"},
        "top_surface_slab": {"base_color": (0.280, 0.240, 0.200, 1.0), "roughness": 0.4, "metallic": 0.0, "specular": 0.5, "type": "wood"},
        "bottom_plinth_base": {"base_color": (0.280, 0.240, 0.200, 1.0), "roughness": 0.4, "metallic": 0.0, "specular": 0.5, "type": "wood"},
        "front_drawer_top": {"base_color": (0.280, 0.240, 0.200, 1.0), "roughness": 0.4, "metallic": 0.0, "specular": 0.5, "type": "wood"},
        "front_drawer_upper_middle": {"base_color": (0.280, 0.240, 0.200, 1.0), "roughness": 0.4, "metallic": 0.0, "specular": 0.5, "type": "wood"},
        "front_drawer_lower_middle": {"base_color": (0.280, 0.240, 0.200, 1.0), "roughness": 0.4, "metallic": 0.0, "specular": 0.5, "type": "wood"},
        "front_drawer_bottom": {"base_color": (0.280, 0.240, 0.200, 1.0), "roughness": 0.4, "metallic": 0.0, "specular": 0.5, "type": "wood"},
        "drawer_gap_horizontal_1": {"base_color": (0.200, 0.170, 0.140, 1.0), "roughness": 0.6, "metallic": 0.0, "specular": 0.35, "type": "painted_mdf"},
        "drawer_gap_horizontal_2": {"base_color": (0.200, 0.170, 0.140, 1.0), "roughness": 0.6, "metallic": 0.0, "specular": 0.35, "type": "painted_mdf"},
        "drawer_gap_horizontal_3": {"base_color": (0.200, 0.170, 0.140, 1.0), "roughness": 0.6, "metallic": 0.0, "specular": 0.35, "type": "painted_mdf"},
        "top_drawer_handle": {"base_color": (0.600, 0.450, 0.200, 1.0), "roughness": 0.3, "metallic": 0.85, "specular": 0.45, "type": "anodized_aluminum"},
        "upper_middle_drawer_handle": {"base_color": (0.600, 0.450, 0.200, 1.0), "roughness": 0.3, "metallic": 0.85, "specular": 0.45, "type": "anodized_aluminum"},
        "lower_middle_drawer_handle": {"base_color": (0.600, 0.450, 0.200, 1.0), "roughness": 0.3, "metallic": 0.85, "specular": 0.45, "type": "anodized_aluminum"},
        "bottom_drawer_handle": {"base_color": (0.600, 0.450, 0.200, 1.0), "roughness": 0.3, "metallic": 0.85, "specular": 0.45, "type": "anodized_aluminum"},
    },
    "Floor_Mirror": {
        "left_wood_frame": {"base_color": (0.350, 0.270, 0.200, 1.0), "roughness": 0.4, "metallic": 0.0, "specular": 0.5, "type": "wood"},
        "right_wood_frame": {"base_color": (0.350, 0.270, 0.200, 1.0), "roughness": 0.4, "metallic": 0.0, "specular": 0.5, "type": "wood"},
        "top_wood_frame": {"base_color": (0.350, 0.270, 0.200, 1.0), "roughness": 0.4, "metallic": 0.0, "specular": 0.5, "type": "wood"},
        "bottom_wood_frame": {"base_color": (0.350, 0.270, 0.200, 1.0), "roughness": 0.4, "metallic": 0.0, "specular": 0.5, "type": "wood"},
        "front_glass_mirror_panel": {"base_color": (0.950, 0.960, 0.980, 1.0), "roughness": 0.0, "metallic": 0.0, "specular": 0.5, "type": "glass"},
        "thin_backing_panel": {"base_color": (0.280, 0.240, 0.200, 1.0), "roughness": 0.6, "metallic": 0.0, "specular": 0.35, "type": "painted_mdf"},
    },
    "Orphan_obj_004_Book": {
        "page_block": {"base_color": (0.900, 0.880, 0.850, 1.0), "roughness": 0.9, "metallic": 0.0, "specular": 0.3, "type": "paint"},
        "top_cover": {"base_color": (0.350, 0.310, 0.270, 1.0), "roughness": 0.6, "metallic": 0.0, "specular": 0.4, "type": "leather"},
        "bottom_cover": {"base_color": (0.350, 0.310, 0.270, 1.0), "roughness": 0.6, "metallic": 0.0, "specular": 0.4, "type": "leather"},
        "spine_binding": {"base_color": (0.350, 0.310, 0.270, 1.0), "roughness": 0.6, "metallic": 0.0, "specular": 0.4, "type": "leather"},
        "front_page_edge": {"base_color": (0.900, 0.880, 0.850, 1.0), "roughness": 0.9, "metallic": 0.0, "specular": 0.3, "type": "paint"},
    },
    "Orphan_obj_009_Table_Lamp_North": {
        "gold_metal_round_base": {"base_color": (0.600, 0.450, 0.200, 1.0), "roughness": 0.3, "metallic": 0.85, "specular": 0.45, "type": "anodized_aluminum"},
        "gold_metal_stem": {"base_color": (0.600, 0.450, 0.200, 1.0), "roughness": 0.3, "metallic": 0.85, "specular": 0.45, "type": "anodized_aluminum"},
        "warm_glowing_glass_bulb": {"base_color": (0.950, 0.960, 0.980, 1.0), "roughness": 0.05, "metallic": 0.0, "specular": 0.5, "type": "glass"},
        "warm_yellow_glass_lampshade": {"base_color": (0.950, 0.960, 0.980, 1.0), "roughness": 0.05, "metallic": 0.0, "specular": 0.5, "type": "glass"},
        "gold_metal_lower_shade_rim": {"base_color": (0.600, 0.450, 0.200, 1.0), "roughness": 0.3, "metallic": 0.85, "specular": 0.45, "type": "anodized_aluminum"},
        "gold_metal_upper_shade_rim": {"base_color": (0.600, 0.450, 0.200, 1.0), "roughness": 0.3, "metallic": 0.85, "specular": 0.45, "type": "anodized_aluminum"},
        "gold_metal_top_finial": {"base_color": (0.600, 0.450, 0.200, 1.0), "roughness": 0.3, "metallic": 0.85, "specular": 0.45, "type": "anodized_aluminum"},
    },
    "Orphan_obj_012_Table_Lamp_South": {
        "gold_metal_base": {"base_color": (0.600, 0.450, 0.200, 1.0), "roughness": 0.3, "metallic": 0.85, "specular": 0.45, "type": "anodized_aluminum"},
        "slender_gold_stem": {"base_color": (0.600, 0.450, 0.200, 1.0), "roughness": 0.3, "metallic": 0.85, "specular": 0.45, "type": "anodized_aluminum"},
        "warm_glowing_bulb": {"base_color": (1.000, 0.900, 0.700, 1.0), "roughness": 0.5, "metallic": 0.0, "specular": 0.3, "type": "emission"},
        "warm_yellow_glass_lampshade": {"base_color": (0.950, 0.960, 0.980, 1.0), "roughness": 0.05, "metallic": 0.0, "specular": 0.5, "type": "glass"},
        "small_gold_finial": {"base_color": (0.600, 0.450, 0.200, 1.0), "roughness": 0.3, "metallic": 0.85, "specular": 0.45, "type": "anodized_aluminum"},
    },
    "Nightstand_South": {
        "top_worktop_slab": {"base_color": (0.350, 0.270, 0.200, 1.0), "roughness": 0.3, "metallic": 0.0, "specular": 0.5, "type": "wood"},
        "left_side_panel": {"base_color": (0.280, 0.240, 0.200, 1.0), "roughness": 0.4, "metallic": 0.0, "specular": 0.5, "type": "wood"},
        "right_side_panel": {"base_color": (0.280, 0.240, 0.200, 1.0), "roughness": 0.4, "metallic": 0.0, "specular": 0.5, "type": "wood"},
        "back_panel": {"base_color": (0.280, 0.240, 0.200, 1.0), "roughness": 0.4, "metallic": 0.0, "specular": 0.5, "type": "wood"},
        "bottom_panel": {"base_color": (0.280, 0.240, 0.200, 1.0), "roughness": 0.4, "metallic": 0.0, "specular": 0.5, "type": "wood"},
        "upper_drawer_front": {"base_color": (0.280, 0.240, 0.200, 1.0), "roughness": 0.4, "metallic": 0.0, "specular": 0.5, "type": "wood"},
        "lower_drawer_front": {"base_color": (0.280, 0.240, 0.200, 1.0), "roughness": 0.4, "metallic": 0.0, "specular": 0.5, "type": "wood"},
        "center_drawer_gap": {"base_color": (0.100, 0.100, 0.100, 1.0), "roughness": 0.3, "metallic": 0.2, "specular": 0.5, "type": "painted_metal"},
        "upper_drawer_handle": {"base_color": (0.600, 0.450, 0.200, 1.0), "roughness": 0.3, "metallic": 0.85, "specular": 0.45, "type": "anodized_aluminum"},
        "lower_drawer_handle": {"base_color": (0.600, 0.450, 0.200, 1.0), "roughness": 0.3, "metallic": 0.85, "specular": 0.45, "type": "anodized_aluminum"},
        "front_toe_kick": {"base_color": (0.100, 0.100, 0.100, 1.0), "roughness": 0.3, "metallic": 0.2, "specular": 0.5, "type": "painted_metal"},
    },
    "Bench": {
        "upholstered_seat_cushion": {"base_color": (0.750, 0.720, 0.680, 1.0), "roughness": 0.9, "metallic": 0.0, "specular": 0.3, "type": "fabric"},
        "upholstered_base_panel": {"base_color": (0.750, 0.720, 0.680, 1.0), "roughness": 0.9, "metallic": 0.0, "specular": 0.3, "type": "fabric"},
        "front_apron": {"base_color": (0.750, 0.720, 0.680, 1.0), "roughness": 0.9, "metallic": 0.0, "specular": 0.3, "type": "fabric"},
        "back_apron": {"base_color": (0.750, 0.720, 0.680, 1.0), "roughness": 0.9, "metallic": 0.0, "specular": 0.3, "type": "fabric"},
        "left_apron": {"base_color": (0.750, 0.720, 0.680, 1.0), "roughness": 0.9, "metallic": 0.0, "specular": 0.3, "type": "fabric"},
        "right_apron": {"base_color": (0.750, 0.720, 0.680, 1.0), "roughness": 0.9, "metallic": 0.0, "specular": 0.3, "type": "fabric"},
        "front_left_short_leg": {"base_color": (0.350, 0.270, 0.200, 1.0), "roughness": 0.4, "metallic": 0.0, "specular": 0.5, "type": "wood"},
        "front_right_short_leg": {"base_color": (0.350, 0.270, 0.200, 1.0), "roughness": 0.4, "metallic": 0.0, "specular": 0.5, "type": "wood"},
        "back_left_short_leg": {"base_color": (0.350, 0.270, 0.200, 1.0), "roughness": 0.4, "metallic": 0.0, "specular": 0.5, "type": "wood"},
        "back_right_short_leg": {"base_color": (0.350, 0.270, 0.200, 1.0), "roughness": 0.4, "metallic": 0.0, "specular": 0.5, "type": "wood"},
    },
    "Media_Console": {
        "top_worktop_slab": {"base_color": (0.350, 0.270, 0.200, 1.0), "roughness": 0.3, "metallic": 0.0, "specular": 0.5, "type": "wood"},
        "main_wood_console_body": {"base_color": (0.280, 0.240, 0.200, 1.0), "roughness": 0.4, "metallic": 0.0, "specular": 0.5, "type": "wood"},
        "front_left_door_panel": {"base_color": (0.280, 0.240, 0.200, 1.0), "roughness": 0.4, "metallic": 0.0, "specular": 0.5, "type": "wood"},
        "front_mid_left_drawer_panel": {"base_color": (0.280, 0.240, 0.200, 1.0), "roughness": 0.4, "metallic": 0.0, "specular": 0.5, "type": "wood"},
        "front_mid_right_drawer_panel": {"base_color": (0.280, 0.240, 0.200, 1.0), "roughness": 0.4, "metallic": 0.0, "specular": 0.5, "type": "wood"},
        "front_right_door_panel": {"base_color": (0.280, 0.240, 0.200, 1.0), "roughness": 0.4, "metallic": 0.0, "specular": 0.5, "type": "wood"},
        "thin_shadow_gap_between_front_panels_1": {"base_color": (0.100, 0.100, 0.100, 1.0), "roughness": 0.3, "metallic": 0.2, "specular": 0.5, "type": "painted_metal"},
        "thin_shadow_gap_between_front_panels_2": {"base_color": (0.100, 0.100, 0.100, 1.0), "roughness": 0.3, "metallic": 0.2, "specular": 0.5, "type": "painted_metal"},
        "thin_shadow_gap_between_front_panels_3": {"base_color": (0.100, 0.100, 0.100, 1.0), "roughness": 0.3, "metallic": 0.2, "specular": 0.5, "type": "painted_metal"},
        "left_panel_handle": {"base_color": (0.600, 0.450, 0.200, 1.0), "roughness": 0.3, "metallic": 0.85, "specular": 0.45, "type": "anodized_aluminum"},
        "mid_left_panel_handle": {"base_color": (0.600, 0.450, 0.200, 1.0), "roughness": 0.3, "metallic": 0.85, "specular": 0.45, "type": "anodized_aluminum"},
        "mid_right_panel_handle": {"base_color": (0.600, 0.450, 0.200, 1.0), "roughness": 0.3, "metallic": 0.85, "specular": 0.45, "type": "anodized_aluminum"},
        "right_panel_handle": {"base_color": (0.600, 0.450, 0.200, 1.0), "roughness": 0.3, "metallic": 0.85, "specular": 0.45, "type": "anodized_aluminum"},
        "recessed_toe_kick": {"base_color": (0.100, 0.100, 0.100, 1.0), "roughness": 0.3, "metallic": 0.2, "specular": 0.5, "type": "painted_metal"},
    },
    "Plant": {
        "dark_grey_circular_planter_body": {"base_color": (0.200, 0.200, 0.200, 1.0), "roughness": 0.2, "metallic": 0.0, "specular": 0.5, "type": "ceramic"},
        "dark_grey_planter_top_rim": {"base_color": (0.200, 0.200, 0.200, 1.0), "roughness": 0.2, "metallic": 0.0, "specular": 0.5, "type": "ceramic"},
        "dark_soil_surface": {"base_color": (0.100, 0.080, 0.050, 1.0), "roughness": 0.9, "metallic": 0.0, "specular": 0.3, "type": "concrete"},
        "central_plant_stem": {"base_color": (0.200, 0.250, 0.150, 1.0), "roughness": 0.7, "metallic": 0.0, "specular": 0.3, "type": "wood"},
        "left_plant_stem": {"base_color": (0.200, 0.250, 0.150, 1.0), "roughness": 0.7, "metallic": 0.0, "specular": 0.3, "type": "wood"},
        "right_plant_stem": {"base_color": (0.200, 0.250, 0.150, 1.0), "roughness": 0.7, "metallic": 0.0, "specular": 0.3, "type": "wood"},
        "central_bushy_green_leaves": {"base_color": (0.150, 0.300, 0.150, 1.0), "roughness": 0.5, "metallic": 0.0, "specular": 0.5, "type": "plastic"},
        "top_bushy_green_leaves": {"base_color": (0.150, 0.300, 0.150, 1.0), "roughness": 0.5, "metallic": 0.0, "specular": 0.5, "type": "plastic"},
        "left_bushy_green_leaves": {"base_color": (0.150, 0.300, 0.150, 1.0), "roughness": 0.5, "metallic": 0.0, "specular": 0.5, "type": "plastic"},
        "right_bushy_green_leaves": {"base_color": (0.150, 0.300, 0.150, 1.0), "roughness": 0.5, "metallic": 0.0, "specular": 0.5, "type": "plastic"},
        "front_bushy_green_leaves": {"base_color": (0.150, 0.300, 0.150, 1.0), "roughness": 0.5, "metallic": 0.0, "specular": 0.5, "type": "plastic"},
        "back_bushy_green_leaves": {"base_color": (0.150, 0.300, 0.150, 1.0), "roughness": 0.5, "metallic": 0.0, "specular": 0.5, "type": "plastic"},
    },
    "Wardrobe_East": {
        "left_side_panel": {"base_color": (0.280, 0.240, 0.200, 1.0), "roughness": 0.4, "metallic": 0.0, "specular": 0.5, "type": "wood"},
        "right_side_panel": {"base_color": (0.280, 0.240, 0.200, 1.0), "roughness": 0.4, "metallic": 0.0, "specular": 0.5, "type": "wood"},
        "back_panel": {"base_color": (0.280, 0.240, 0.200, 1.0), "roughness": 0.4, "metallic": 0.0, "specular": 0.5, "type": "wood"},
        "top_panel": {"base_color": (0.280, 0.240, 0.200, 1.0), "roughness": 0.4, "metallic": 0.0, "specular": 0.5, "type": "wood"},
        "bottom_panel": {"base_color": (0.280, 0.240, 0.200, 1.0), "roughness": 0.4, "metallic": 0.0, "specular": 0.5, "type": "wood"},
        "left_vertical_divider": {"base_color": (0.280, 0.240, 0.200, 1.0), "roughness": 0.4, "metallic": 0.0, "specular": 0.5, "type": "wood"},
        "right_vertical_divider": {"base_color": (0.280, 0.240, 0.200, 1.0), "roughness": 0.4, "metallic": 0.0, "specular": 0.5, "type": "wood"},
        "left_shelf_1": {"base_color": (0.280, 0.240, 0.200, 1.0), "roughness": 0.4, "metallic": 0.0, "specular": 0.5, "type": "wood"},
        "left_shelf_2": {"base_color": (0.280, 0.240, 0.200, 1.0), "roughness": 0.4, "metallic": 0.0, "specular": 0.5, "type": "wood"},
        "left_shelf_3": {"base_color": (0.280, 0.240, 0.200, 1.0), "roughness": 0.4, "metallic": 0.0, "specular": 0.5, "type": "wood"},
        "right_shelf_1": {"base_color": (0.280, 0.240, 0.200, 1.0), "roughness": 0.4, "metallic": 0.0, "specular": 0.5, "type": "wood"},
        "right_shelf_2": {"base_color": (0.280, 0.240, 0.200, 1.0), "roughness": 0.4, "metallic": 0.0, "specular": 0.5, "type": "wood"},
        "right_shelf_3": {"base_color": (0.280, 0.240, 0.200, 1.0), "roughness": 0.4, "metallic": 0.0, "specular": 0.5, "type": "wood"},
        "hanging_bay_upper_shelf": {"base_color": (0.280, 0.240, 0.200, 1.0), "roughness": 0.4, "metallic": 0.0, "specular": 0.5, "type": "wood"},
        "hanging_rail": {"base_color": (0.600, 0.450, 0.200, 1.0), "roughness": 0.3, "metallic": 0.85, "specular": 0.45, "type": "anodized_aluminum"},
        "hanging_clothing_left": {"base_color": (0.750, 0.720, 0.680, 1.0), "roughness": 0.9, "metallic": 0.0, "specular": 0.3, "type": "fabric"},
        "hanging_clothing_center": {"base_color": (0.350, 0.310, 0.270, 1.0), "roughness": 0.9, "metallic": 0.0, "specular": 0.3, "type": "fabric"},
        "hanging_clothing_right": {"base_color": (0.750, 0.720, 0.680, 1.0), "roughness": 0.9, "metallic": 0.0, "specular": 0.3, "type": "fabric"},
    },
    "Wardrobe_North": {
        "left_side_panel": {"base_color": (0.280, 0.240, 0.200, 1.0), "roughness": 0.4, "metallic": 0.0, "specular": 0.5, "type": "wood"},
        "right_side_panel": {"base_color": (0.280, 0.240, 0.200, 1.0), "roughness": 0.4, "metallic": 0.0, "specular": 0.5, "type": "wood"},
        "back_panel": {"base_color": (0.280, 0.240, 0.200, 1.0), "roughness": 0.4, "metallic": 0.0, "specular": 0.5, "type": "wood"},
        "top_panel": {"base_color": (0.280, 0.240, 0.200, 1.0), "roughness": 0.4, "metallic": 0.0, "specular": 0.5, "type": "wood"},
        "bottom_panel": {"base_color": (0.280, 0.240, 0.200, 1.0), "roughness": 0.4, "metallic": 0.0, "specular": 0.5, "type": "wood"},
        "upper_shelf": {"base_color": (0.280, 0.240, 0.200, 1.0), "roughness": 0.4, "metallic": 0.0, "specular": 0.5, "type": "wood"},
        "lower_shelf": {"base_color": (0.280, 0.240, 0.200, 1.0), "roughness": 0.4, "metallic": 0.0, "specular": 0.5, "type": "wood"},
        "hanging_clothes_rail": {"base_color": (0.600, 0.450, 0.200, 1.0), "roughness": 0.3, "metallic": 0.85, "specular": 0.45, "type": "anodized_aluminum"},
        "left_front_stile": {"base_color": (0.280, 0.240, 0.200, 1.0), "roughness": 0.4, "metallic": 0.0, "specular": 0.5, "type": "wood"},
        "right_front_stile": {"base_color": (0.280, 0.240, 0.200, 1.0), "roughness": 0.4, "metallic": 0.0, "specular": 0.5, "type": "wood"},
        "top_front_rail": {"base_color": (0.280, 0.240, 0.200, 1.0), "roughness": 0.4, "metallic": 0.0, "specular": 0.5, "type": "wood"},
        "bottom_front_rail": {"base_color": (0.280, 0.240, 0.200, 1.0), "roughness": 0.4, "metallic": 0.0, "specular": 0.5, "type": "wood"},
    },
}

# ==============================================================================
# FLOOR & WALL MATERIAL DATA
# ==============================================================================

FLOOR_MAT_DATA = {
    "material_type": "hardwood",
    "base_color": (0.550, 0.420, 0.280, 1.0),
    "roughness": 0.35,
    "metallic": 0.0,
    "specular": 0.5,
    "pattern": "plank",
    "pattern_scale": 1.5,
    "pattern_color2": (0.450, 0.320, 0.200, 1.0),
    "bump_strength": 0.12,
}

WALL_MAT_DATA = {
    "material_type": "paint",
    "base_color": (0.920, 0.900, 0.880, 1.0),
    "roughness": 0.6,
    "metallic": 0.0,
    "specular": 0.5,
    "finish": "matte",
    "bump_strength": 0.02,
}


def create_floor_material():
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
    return mat


def create_wall_material():
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
    return mat


def create_detailed_object(name, location=None, rotation=None, material=None, collection=None):
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

    return parent


def setup_lighting_and_render():
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

    light_sources = [{'id': 'stage11_light_01', 'blender_type': 'AREA', 'area_class': 'west', 'position': {'x': -3.7, 'y': 0.0, 'z': 1.5}, 'color_rgb': [245, 250, 255], 'energy': 280.0, 'size': 3.0}, {'id': 'stage11_light_02', 'blender_type': 'POINT', 'area_class': 'n/a', 'position': {'x': -3.45, 'y': 2.2, 'z': 1.4}, 'color_rgb': [255, 214, 170], 'energy': 50.0, 'size': 0.1}, {'id': 'stage11_light_03', 'blender_type': 'POINT', 'area_class': 'n/a', 'position': {'x': -1.2, 'y': 2.247, 'z': 0.75}, 'color_rgb': [255, 214, 170], 'energy': 40.0, 'size': 0.1}, {'id': 'stage11_light_04', 'blender_type': 'POINT', 'area_class': 'n/a', 'position': {'x': 1.2, 'y': 2.247, 'z': 0.75}, 'color_rgb': [255, 214, 170], 'energy': 40.0, 'size': 0.1}, {'id': 'stage11_light_05', 'blender_type': 'AREA', 'area_class': 'ceiling', 'position': {'x': 0.0, 'y': 0.0, 'z': 2.75}, 'color_rgb': [255, 245, 235], 'energy': 150.0, 'size': 5.0}, {'id': 'stage11_light_06', 'blender_type': 'AREA', 'area_class': 'ceiling', 'position': {'x': 2.5, 'y': 1.0, 'z': 2.75}, 'color_rgb': [255, 245, 235], 'energy': 100.0, 'size': 3.0}]
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

    portals = [{'id': 'stage11_portal_Window_West', 'location': (-3.7750000000000004, 0.0, 1.4), 'wall': 'west', 'size_x': 2.0, 'size_y': 2.0}]
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
    sun_direction = (0.6, 0.0, -0.8)

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
    use_sky = True
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
                sky.sun_intensity = 0.12
            if hasattr(sky, "air_density"):
                sky.air_density = 1.0
            if hasattr(sky, "dust_density"):
                sky.dust_density = 0.5
            if hasattr(sky, "ozone_density"):
                sky.ozone_density = 1.0
            nt.links.new(sky.outputs["Color"], bg_node.inputs[0])
            bg_node.inputs[1].default_value = 0.25
            sky_ok = True
        except Exception as e:
            print("[stage11] Sky Texture unavailable, using flat ambient: " + str(e))
    if not sky_ok:
        # Either an enclosed interior (no windows + no natural primary)
        # or NISHITA threw — fall back to a soft flat Background so we
        # don't render pitch-black ceilings.
        bg_node.inputs[0].default_value = rgba4([255, 250, 245])
        bg_node.inputs[1].default_value = 0.15

    try:
        scene.view_settings.view_transform = 'AgX'
        scene.view_settings.look = 'Medium High Contrast'
    except Exception:
        scene.view_settings.view_transform = 'Filmic'
        scene.view_settings.look = 'Medium High Contrast'
    if hasattr(scene.view_settings, 'exposure'):
        scene.view_settings.exposure = -0.3
    scene.render.resolution_x = 1024
    scene.render.resolution_y = 1024

# === MAIN LAYOUT ENGINE ===

# ==============================================================================
# WALL ART (Stage Texture)
# ==============================================================================
WALL_ART = {
    "Wall_Art_South": {"path": r"/Users/yangyixuan/Code-as-Room_github/agent_utils/pipeline_output/run_20260521_104358_example1/stage11_texture/images/art_Wall_Art_South.png", "size": (1.200, 0.800), "thickness": 0.050, "location": (0.0000, -2.4750, 1.5000), "rotation": (0.0000, 0.0000, 180.0000)},
    "Wall_Mirror_East": {"path": r"/Users/yangyixuan/Code-as-Room_github/agent_utils/pipeline_output/run_20260521_104358_example1/stage11_texture/images/art_Wall_Mirror_East.png", "size": (0.800, 1.000), "thickness": 0.050, "location": (3.7250, -0.8000, 1.5000), "rotation": (0.0000, 0.0000, -90.0000)},
}

def spawn_wall_arts(collection=None):
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
            bpy.context.scene.collection.objects.link(obj)

def run_layout_engine():
    clear_scene()
    
    # 1. Scene Setup
    SCENE_W = 7.5
    SCENE_D = 5.0
    WALL_H = 2.8
    WALL_T = 0.15
    PARTITION_T = 0.1
    
    # 2. Materials
    mat_wall = create_wall_material()
    mat_floor = create_floor_material()
    mat_fabric = create_pbr_material("FabricMat", (0.850, 0.800, 0.750, 1.0))
    mat_wood = create_pbr_material("WoodMat", (0.400, 0.250, 0.150, 1.0))
    mat_glass = create_material("GlassMat", (0.7, 0.8, 0.9, 0.3), alpha=0.3)
    mat_mirror = create_pbr_material("MirrorMat", (0.900, 0.900, 0.900, 1.0))
    
    # 3. Architectural Shell
    coll_arch = create_collection("Architecture")
    
    # Floor
    create_box("Floor", (0, 0, -0.05), (SCENE_W, SCENE_D, 0.1), material=mat_floor, collection=coll_arch, show_direction=False)
    
    # Boundary Walls
    create_box("Wall_North", (0, SCENE_D/2 + WALL_T/2, WALL_H/2), (SCENE_W + WALL_T*2, WALL_T, WALL_H), material=mat_wall, collection=coll_arch, show_direction=False)
    create_box("Wall_South", (0, -SCENE_D/2 - WALL_T/2, WALL_H/2), (SCENE_W + WALL_T*2, WALL_T, WALL_H), material=mat_wall, collection=coll_arch, show_direction=False)
    create_box("Wall_West", (-SCENE_W/2 - WALL_T/2, 0, WALL_H/2), (WALL_T, SCENE_D, WALL_H), material=mat_wall, collection=coll_arch, show_direction=False)
    create_box("Wall_East", (SCENE_W/2 + WALL_T/2, 0, WALL_H/2), (WALL_T, SCENE_D, WALL_H), material=mat_wall, collection=coll_arch, show_direction=False)
    
    # Window on West Wall
    create_box("Window_West", (-SCENE_W/2 - WALL_T/2, 0, 1.4), (WALL_T + 0.05, 2.0, 2.0), material=mat_glass, collection=coll_arch, show_direction=False)
    
    # Interior Partitions
    # Left Partition (Between Lounge and Bedroom) at x = -1.75
    part1_x = -1.75
    create_box("Partition_Lounge_North", (part1_x, 1.65, WALL_H/2), (PARTITION_T, 1.7, WALL_H), material=mat_wall, collection=coll_arch, show_direction=False)
    create_box("Partition_Lounge_South", (part1_x, -1.65, WALL_H/2), (PARTITION_T, 1.7, WALL_H), material=mat_wall, collection=coll_arch, show_direction=False)
    # Header above the wide opening (gap from y=-0.8 to 0.8)
    create_box("Partition_Lounge_Header", (part1_x, 0, 2.65), (PARTITION_T, 1.6, 0.3), material=mat_wall, collection=coll_arch, show_direction=False)
    
    # Right Partition (Between Bedroom and Closet) at x = 1.75
    part2_x = 1.75
    create_box("Partition_Closet_Main", (part2_x, 0.6, WALL_H/2), (PARTITION_T, 3.8, WALL_H), material=mat_wall, collection=coll_arch, show_direction=False)
    # Header above the closet doorway (gap from y=-2.5 to -1.3)
    create_box("Partition_Closet_Header", (part2_x, -1.9, 2.45), (PARTITION_T, 1.2, 0.7), material=mat_wall, collection=coll_arch, show_direction=False)
    
    # 4. Zone 01: Lounge Area (West)
    coll_zone1 = create_collection("Zone_01_Lounge")
    
    # Armchair North (Faces SE)
    create_detailed_object("Armchair_North", location=(-2.75, 1.0, 0.45), rotation=(0, 0, math.pi/4), material=mat_fabric, collection=coll_zone1)
    
    # Armchair South (Faces NE)
    create_detailed_object("Armchair_South", location=(-2.75, -1.0, 0.45), rotation=(0, 0, 3*math.pi/4), material=mat_fabric, collection=coll_zone1)
    
    # Side Table (Between armchairs)
    create_detailed_object("Side_Table", location=(-2.75, 0.0, 0.25), material=mat_wood, collection=coll_zone1)
    
    # Floor Lamp (Top left corner of lounge)
    create_detailed_object("Floor_Lamp", location=(-3.45, 2.2, 0.8), material=mat_wood, collection=coll_zone1)
    
    # 5. Zone 02: Main Bedroom Area (Center)
    coll_zone2 = create_collection("Zone_02_Bedroom")
    
    # Bed (Against North Wall)
    bed_w, bed_d, bed_h = 1.8, 2.1, 0.6
    bed_y = SCENE_D/2 - bed_d/2
    create_detailed_object("Bed", location=(0.0, bed_y, bed_h/2), rotation=(0, 0, 0), material=mat_fabric, collection=coll_zone2)
    
    # Nightstands (Against North Wall)
    ns_w, ns_d, ns_h = 0.5, 0.4, 0.5
    ns_y = SCENE_D/2 - ns_d/2
    create_detailed_object("Nightstand_North", location=(-1.2, ns_y, ns_h/2), rotation=(0, 0, 0), material=mat_wood, collection=coll_zone2)
    create_detailed_object("Nightstand_South", location=(1.2, ns_y, ns_h/2), rotation=(0, 0, 0), material=mat_wood, collection=coll_zone2)
    
    # Bench (Foot of the bed)
    bench_w, bench_d, bench_h = 1.2, 0.4, 0.45
    bench_y = bed_y - bed_d/2 - bench_d/2
    create_detailed_object("Bench", location=(0.0, bench_y, bench_h/2), rotation=(0, 0, 0), material=mat_fabric, collection=coll_zone2)
    
    # Media Console (Against South Wall)
    mc_w, mc_d, mc_h = 1.8, 0.45, 0.5
    mc_y = -SCENE_D/2 + mc_d/2
    create_detailed_object("Media_Console", location=(0.0, mc_y, mc_h/2), rotation=(0, 0, math.pi), material=mat_wood, collection=coll_zone2)
    
    # Plant (Bottom left corner of bedroom)
    create_detailed_object("Plant", location=(-1.45, -2.2, 0.4), material=mat_fabric, collection=coll_zone2)
    
    # 6. Zone 03: Walk-in Closet / Dressing Area (East)
    coll_zone3 = create_collection("Zone_03_Closet")
    
    # Built-in Wardrobe L-Shape (Constructed from two boxes)
    # Part 1: Along East Wall
    w1_w, w1_d, w1_h = 2.0, 0.6, 2.4
    w1_x = SCENE_W/2 - w1_d/2
    w1_y = 0.9
    create_detailed_object("Wardrobe_East", location=(w1_x, w1_y, w1_h/2), rotation=(0, 0, -math.pi/2), material=mat_wood, collection=coll_zone3)
    
    # Part 2: Along North Wall
    w2_w, w2_d, w2_h = 1.5, 0.6, 2.4
    w2_x = 3.0
    w2_y = SCENE_D/2 - w2_d/2
    create_detailed_object("Wardrobe_North", location=(w2_x, w2_y, w2_h/2), rotation=(0, 0, 0), material=mat_wood, collection=coll_zone3)
    
    # Closet Bench/Island (Center of closet)
    cb_w, cb_d, cb_h = 0.4, 1.2, 0.45
    create_detailed_object("Closet_Bench", location=(2.75, 0.0, cb_h/2), rotation=(0, 0, 0), material=mat_wood, collection=coll_zone3)
    
    # Dresser (Against East Wall)
    dr_w, dr_d, dr_h = 1.2, 0.5, 0.9
    dr_x = SCENE_W/2 - dr_d/2
    dr_y = -0.8
    create_detailed_object("Dresser", location=(dr_x, dr_y, dr_h/2), rotation=(0, 0, -math.pi/2), material=mat_wood, collection=coll_zone3)
    
    # Floor Mirror (Against East Wall)
    fm_w, fm_d, fm_h = 0.8, 0.1, 1.8
    fm_x = 3.25
    fm_y = -2.0
    create_detailed_object("Floor_Mirror", location=(fm_x, fm_y, fm_h/2), rotation=(0, 0, -math.pi/2), material=mat_mirror, collection=coll_zone3)

    # 7. Wall Decorations
    coll_decor = create_collection("Wall_Decorations")
    mat_art = create_pbr_material("ArtMat", (0.200, 0.600, 0.800, 1.0))
    
    # Wall Art (South) - Above Media Console
    # Inner face of South Wall is at y = -2.5. Art thickness = 0.05. Center y = -2.475
    
    # Wall Mirror (East) - Above Dresser
    # Inner face of East Wall is at x = 3.75. Mirror thickness = 0.05. Center x = 3.725


    # Stage 7 semantic inventory orphans promoted into Stage 8 geometry
    create_detailed_object("Orphan_obj_004_Book", location=(-2.75, -0.08999999999999998, 0.55), material=None, collection=None)
    create_detailed_object("Orphan_obj_009_Table_Lamp_North", location=(-1.2, 2.2474999999999996, 0.55), material=None, collection=None)
    create_detailed_object("Orphan_obj_012_Table_Lamp_South", location=(1.2, 2.2474999999999996, 0.55), material=None, collection=None)
    create_detailed_object("Orphan_obj_017_Decor_Item_Console", location=(0.592, -2.20375, 0.55), rotation=(0.0, 0.0, 3.141592653589793), material=None, collection=None)

    spawn_wall_arts()

    # === Lighting & Render ===
    setup_lighting_and_render()
if __name__ == "__main__":
    run_layout_engine()

# ==============================================================================
# Small objects appended by stage7_small_objects.py
# Generated: 2026-05-21T11:18:50
# Total items: 9
# ==============================================================================

_small_objects_collection = create_collection("Small_Objects")

_small_object_materials = {}

def _get_small_material(color_key, color_rgba):
    """Fetch-or-create a cached material for small objects."""
    if color_key in _small_object_materials:
        return _small_object_materials[color_key]
    mat = create_material(f"SmallMat_{color_key}", color_rgba)
    _small_object_materials[color_key] = mat
    return mat

create_box('Media_Console__top__Console_Book_1', (0.36, -2.2075, 0.515), (0.22, 0.15, 0.03), rotation=(0, 0, 3.1416), material=_get_small_material('dark_taupe', (0.7, 0.62, 0.52, 1.0)), collection=_small_objects_collection)
create_box('Media_Console__top__Console_Sphere_1', (0.36, -2.2075, 0.58), (0.1, 0.1, 0.1), rotation=(0, 0, 3.1416), material=_get_small_material('matte_black', (0.1, 0.1, 0.1, 1.0)), collection=_small_objects_collection)
create_cylinder('Closet_Bench__top__Hat_Box_1', (2.75, 0.18, 0.525), (0.25, 0.25, 0.15), rotation=(0, 0, 0.0), material=_get_small_material('beige', (0.88, 0.8, 0.68, 1.0)), collection=_small_objects_collection)
create_box('Closet_Bench__top__Small_Dark_Box_1', (2.75, -0.18, 0.475), (0.2, 0.15, 0.05), rotation=(0, 0, 0.0), material=_get_small_material('dark_brown', (0.42, 0.28, 0.18, 1.0)), collection=_small_objects_collection)
create_box('Nightstand_North__top__Nightstand_Book_1', (-1.025, 2.2476, 0.51), (0.1, 0.15, 0.02), rotation=(0, 0, 0.0), material=_get_small_material('dark_brown', (0.42, 0.28, 0.18, 1.0)), collection=_small_objects_collection)
create_box('Nightstand_South__top__Book_Dark_1', (1.035, 2.24, 0.51), (0.1, 0.15, 0.02), rotation=(0, 0, 0.0), material=_get_small_material('dark_brown', (0.42, 0.28, 0.18, 1.0)), collection=_small_objects_collection)
create_box('Dresser__top__Cosmetic_Box', (3.5, -1.22, 0.94), (0.15, 0.1, 0.08), rotation=(0, 0, -1.5708), material=_get_small_material('taupe', (0.7, 0.62, 0.52, 1.0)), collection=_small_objects_collection)
create_cylinder('Dresser__top__Perfume_Bottle_1', (3.425, -1.064, 0.96), (0.05, 0.05, 0.12), rotation=(0, 0, -1.5708), material=_get_small_material('clear_glass_with_gold_cap', (0.8, 0.86, 0.9, 0.6)), collection=_small_objects_collection)
create_cylinder('Dresser__top__Perfume_Bottle_2', (3.575, -1.064, 0.975), (0.06, 0.06, 0.15), rotation=(0, 0, -1.5708), material=_get_small_material('frosted_glass', (0.8, 0.86, 0.9, 0.6)), collection=_small_objects_collection)
