import blenderproc as bproc  # MUST be first import (BlenderProc validator requirement)

# dataset_generation_sat_v10.py
# Photorealistic LEO spacecraft RPOD dataset generator — physically correct orbits.
#
# Changes vs v9
# ─────────────────────────────────────────────────────────────────────────────
# BUG FIX: Sun and Moon depth ordering (the main issue in v9)
#   In v9: SUN_DISC_DIST=200, MOON_DIST=250 — both objects were placed at
#   z≈-190 to z≈-237 (along look direction), which is CLOSER to the camera
#   than Earth (z=-600 to -7000).  They appeared as nearby floating spheres
#   between the satellite and Earth, not as distant background objects.
#
#   Fix: move both to z > 9000 units — always beyond Earth's maximum depth.
#     SUN_DISC_DIST  200  →  9000  (45× increase)
#     SUN_DISC_RADIUS  3.5 →  157.5 (scaled to preserve 2.01° apparent angle)
#     MOON_DIST       250  →  9500  (38× increase)
#     MOON_RADIUS      3.0 →  114.0 (scaled to preserve 1.38° apparent angle)
#   The apparent angular diameter of both bodies is IDENTICAL to v9 — only
#   their world-space depth changes.  Camera clip_end raised to 12000.
#
#   Correct depth order in every rendered image:
#     Camera (0) → Satellite (-28) → [space] → Earth (-600…-7000)
#                                            → Sun disc (-8550)
#                                            → Moon     (-9025)
#
# RETAINED (from v9, unchanged)
#   Orbital mechanics  (Sun/Moon direction from physical ephemeris)
#   1. compute_sun_direction_eci(): simplified solar ephemeris (Meeus low-precision).
#      Sun moves ~1 deg/day along the ecliptic. Over a 7-day dataset the sun
#      sweeps ~7 degrees continuously — no random jumps between frames.
#
#   2. compute_moon_direction_eci(): simplified lunar theory (Brown, ~1 deg accuracy).
#      Moon moves ~13 deg/day. Over 7 days the moon moves ~91 deg, showing a
#      visible phase progression (new → crescent → quarter → gibbous → full).
#
#   3. eci_to_camera_frame(): full LVLH rotation matrix.
#      Transforms ECI vectors to the satellite's camera frame using the satellite's
#      actual orbital position (true anomaly, inclination, RAAN). The camera
#      -Z axis = nadir direction (toward Earth). Sun, Moon, Earth are all in one
#      physically consistent coordinate frame — their mutual positions are correct.
#
#   4. clamp_direction_to_fov(): replaces FOV rejection-sampling.
#      If the physically computed direction is outside the camera FOV (e.g. in
#      eclipse, or camera tilted away from sun), the direction is clamped to the
#      FOV boundary while preserving azimuth. This represents a slight
#      adjustment to the observation geometry to keep the body visible.
#
#   5. enforce_moon_sun_separation(): if physical Moon-Sun separation < 8 deg
#      (near new-moon geometry), rotate moon azimuth away from sun within FOV.
#
#   6. New CLI args: --epoch_doy (day of year for dataset start, default 80 =
#      vernal equinox), --mission_span_days (dataset time window, default 7).
#
#   7. CSV: added mission_day column (seconds into mission for each frame).
#
# RETAINED (from v8, unchanged)
#   8. 8K Moon and Sun textures; per-frame UV rotation for surface diversity.
#   9. Earth: LEO orbit with close/medium/far distance categories.
#   10. GPU: CUDA/OptiX, persistent_data, adaptive sampling, glossy_bounces=3.
#   11. Resume: append CSV, per-frame flush, deterministic pre-generation.
#
# Usage:
#   blenderproc run dataset_generation_sat_v10.py
#   blenderproc run dataset_generation_sat_v10.py --num_images 500 --samples 48
#   blenderproc run dataset_generation_sat_v10.py --epoch_doy 172 --mission_span_days 14
#       (epoch_doy 172 = June 21 = summer solstice)

import bpy
import numpy as np
import csv
import os
import re
import random
import math
import imageio
from argparse import ArgumentParser
from mathutils import Quaternion


# ─────────────────────────────────────────────────────────────────────────────
# Shader helpers
# ─────────────────────────────────────────────────────────────────────────────

def _node(tree, bl_type, **kwargs):
    n = tree.nodes.new(bl_type)
    if "location" in kwargs:
        n.location = kwargs["location"]
    return n

def _link(tree, out_socket, in_socket):
    tree.links.new(out_socket, in_socket)


# ─────────────────────────────────────────────────────────────────────────────
# GPU / Cycles render settings
# ─────────────────────────────────────────────────────────────────────────────

def configure_gpu_rendering(samples: int = 48):
    scene  = bpy.context.scene
    cycles = scene.cycles
    scene.render.engine = 'CYCLES'

    try:
        prefs  = bpy.context.preferences
        cprefs = prefs.addons['cycles'].preferences
        cprefs.compute_device_type = 'CUDA'
        cprefs.get_devices()
        gpu_found = False
        for dev in cprefs.devices:
            dev.use = True
            if dev.type == 'CUDA':
                gpu_found = True
                print(f"   GPU: {dev.name}")
        cycles.device = 'GPU' if gpu_found else 'CPU'
        if not gpu_found:
            print("   WARNING: No CUDA GPU — falling back to CPU")
    except Exception as e:
        print(f"   WARNING: GPU setup failed ({e}) — using CPU")
        cycles.device = 'CPU'

    cycles.samples               = samples
    cycles.use_adaptive_sampling = True
    cycles.adaptive_threshold    = 0.02
    cycles.adaptive_min_samples  = 16

    # Denoiser (Blender 4.x: NLM removed; OPTIX works on GTX 1060 Pascal)
    cycles.use_denoising = True
    try:
        cycles.denoiser = 'OPTIX'
        print("   Denoiser: OptiX (GPU-accelerated)")
    except TypeError:
        try:
            cycles.denoiser = 'OPENIMAGEDENOISE'
            print("   Denoiser: OpenImageDenoise (CPU)")
        except TypeError:
            cycles.use_denoising = False
            print("   Denoiser: none available")

    # Persistent BVH + shader cache between frames — major speedup
    scene.render.use_persistent_data = True

    # Bounces: glossy=3 lets sun disc reflect in metallic satellite panels
    cycles.max_bounces            = 5
    cycles.diffuse_bounces        = 2
    cycles.glossy_bounces         = 3
    cycles.transmission_bounces   = 2
    cycles.volume_bounces         = 0
    cycles.transparent_max_bounces = 4

    print(f"   {samples} spp + OptiX/OIDN + persistent_data + glossy_bounces=3")


# ─────────────────────────────────────────────────────────────────────────────
# Star background (8K equirectangular HDRI)
# ─────────────────────────────────────────────────────────────────────────────

def setup_star_background(stars_path: str):
    world = bpy.data.worlds.new("SpaceWorld")
    bpy.context.scene.world = world
    world.use_nodes = True
    nt = world.node_tree
    for node in nt.nodes:
        nt.nodes.remove(node)

    tex_coord  = _node(nt, "ShaderNodeTexCoord",      location=(-700,   0))
    mapping    = _node(nt, "ShaderNodeMapping",        location=(-500,   0))
    env_tex    = _node(nt, "ShaderNodeTexEnvironment", location=(-280,   0))
    rgb_curves = _node(nt, "ShaderNodeRGBCurve",       location=(  0,   0))
    background = _node(nt, "ShaderNodeBackground",     location=( 220,  0))
    output     = _node(nt, "ShaderNodeOutputWorld",    location=( 420,  0))

    env_tex.image = bpy.data.images.load(stars_path)
    env_tex.interpolation = "Linear"
    background.inputs["Strength"].default_value = 0.25

    _link(nt, tex_coord.outputs["Generated"],   mapping.inputs["Vector"])
    _link(nt, mapping.outputs["Vector"],         env_tex.inputs["Vector"])
    _link(nt, env_tex.outputs["Color"],          rgb_curves.inputs["Color"])
    _link(nt, rgb_curves.outputs["Color"],       background.inputs["Color"])
    _link(nt, background.outputs["Background"],  output.inputs["Surface"])
    return world


# ─────────────────────────────────────────────────────────────────────────────
# Earth material (day + night + terminator + ocean specular + atmosphere rim)
# ─────────────────────────────────────────────────────────────────────────────

def build_earth_material(daymap_path: str, nightmap_path: str):
    mat = bpy.data.materials.new(name="EarthMaterial")
    mat.use_nodes = True
    nt = mat.node_tree
    for n in nt.nodes:
        nt.nodes.remove(n)

    tex_coord = _node(nt, "ShaderNodeTexCoord",  location=(-1200, 200))
    mapping   = _node(nt, "ShaderNodeMapping",   location=(-1000, 200))
    _link(nt, tex_coord.outputs["UV"], mapping.inputs["Vector"])

    day_img = _node(nt, "ShaderNodeTexImage",    location=(-750, 350))
    day_img.label = "Day Map"
    day_img.image = bpy.data.images.load(daymap_path)
    day_img.image.colorspace_settings.name = "sRGB"
    _link(nt, mapping.outputs["Vector"], day_img.inputs["Vector"])

    night_img = _node(nt, "ShaderNodeTexImage",  location=(-750, 50))
    night_img.label = "Night Map"
    night_img.image = bpy.data.images.load(nightmap_path)
    night_img.image.colorspace_settings.name = "sRGB"
    _link(nt, mapping.outputs["Vector"], night_img.inputs["Vector"])

    geometry = _node(nt, "ShaderNodeNewGeometry", location=(-1200, -100))
    dot_prod  = _node(nt, "ShaderNodeVectorMath",  location=( -900, -100))
    dot_prod.operation = "DOT_PRODUCT"
    dot_prod.inputs[1].default_value = (1.0, 0.0, 0.0)
    _link(nt, geometry.outputs["Normal"], dot_prod.inputs[0])

    ramp = _node(nt, "ShaderNodeValToRGB", location=(-680, -100))
    ramp.label = "Terminator Blend"
    ramp.color_ramp.interpolation = "EASE"
    ramp.color_ramp.elements[0].position = 0.42
    ramp.color_ramp.elements[0].color    = (0, 0, 0, 1)
    ramp.color_ramp.elements[1].position = 0.58
    ramp.color_ramp.elements[1].color    = (1, 1, 1, 1)
    _link(nt, dot_prod.outputs["Value"], ramp.inputs["Fac"])

    hue_sat = _node(nt, "ShaderNodeHueSaturation", location=(-400, 350))
    hue_sat.inputs["Saturation"].default_value = 3.0
    hue_sat.inputs["Value"].default_value      = 1.0
    _link(nt, day_img.outputs["Color"], hue_sat.inputs["Color"])

    sep_rgb = _node(nt, "ShaderNodeSeparateRGB", location=(-200, 350))
    _link(nt, hue_sat.outputs["Color"], sep_rgb.inputs["Image"])

    rough_ramp = _node(nt, "ShaderNodeValToRGB", location=(0, 350))
    rough_ramp.label = "Ocean Roughness"
    rough_ramp.color_ramp.interpolation = "LINEAR"
    rough_ramp.color_ramp.elements[0].position = 0.3
    rough_ramp.color_ramp.elements[0].color    = (0.85, 0.85, 0.85, 1)
    rough_ramp.color_ramp.elements[1].position = 0.8
    rough_ramp.color_ramp.elements[1].color    = (0.05, 0.05, 0.05, 1)
    _link(nt, sep_rgb.outputs["B"], rough_ramp.inputs["Fac"])

    layer_wt = _node(nt, "ShaderNodeLayerWeight", location=(-750, -250))
    layer_wt.inputs["Blend"].default_value = 0.35
    atm_ramp = _node(nt, "ShaderNodeValToRGB",    location=(-500, -250))
    atm_ramp.label = "Atmosphere Rim"
    atm_ramp.color_ramp.elements[0].position = 0.50
    atm_ramp.color_ramp.elements[0].color    = (0,    0,    0,   1)
    atm_ramp.color_ramp.elements[1].position = 0.85
    atm_ramp.color_ramp.elements[1].color    = (0.15, 0.45, 1.0, 1)
    _link(nt, layer_wt.outputs["Facing"], atm_ramp.inputs["Fac"])

    atm_emit = _node(nt, "ShaderNodeEmission", location=(-200, -250))
    atm_emit.inputs["Strength"].default_value = 0.6
    _link(nt, atm_ramp.outputs["Color"], atm_emit.inputs["Color"])

    bsdf = _node(nt, "ShaderNodeBsdfPrincipled", location=(250, 200))
    bsdf.inputs["Specular IOR Level"].default_value = 0.4
    bsdf.inputs["Metallic"].default_value           = 0.0
    _link(nt, day_img.outputs["Color"],    bsdf.inputs["Base Color"])
    _link(nt, rough_ramp.outputs["Color"], bsdf.inputs["Roughness"])

    night_emit = _node(nt, "ShaderNodeEmission", location=(250, 0))
    night_emit.inputs["Strength"].default_value = 3.0
    _link(nt, night_img.outputs["Color"], night_emit.inputs["Color"])

    mix_dn = _node(nt, "ShaderNodeMixShader", location=(500, 100))
    _link(nt, ramp.outputs["Color"],          mix_dn.inputs["Fac"])
    _link(nt, night_emit.outputs["Emission"], mix_dn.inputs[1])
    _link(nt, bsdf.outputs["BSDF"],           mix_dn.inputs[2])

    add_atm = _node(nt, "ShaderNodeAddShader", location=(700, 0))
    _link(nt, mix_dn.outputs["Shader"],     add_atm.inputs[0])
    _link(nt, atm_emit.outputs["Emission"], add_atm.inputs[1])

    out = _node(nt, "ShaderNodeOutputMaterial", location=(900, 0))
    _link(nt, add_atm.outputs["Shader"], out.inputs["Surface"])
    return mat


def update_sun_direction(earth_mat, sun_dir):
    nt = earth_mat.node_tree
    for node in nt.nodes:
        if node.type == "VECT_MATH" and node.operation == "DOT_PRODUCT":
            node.inputs[1].default_value = sun_dir
            break


# ─────────────────────────────────────────────────────────────────────────────
# Cloud material
# ─────────────────────────────────────────────────────────────────────────────

def build_cloud_material(cloud_path: str):
    mat = bpy.data.materials.new(name="CloudMaterial")
    mat.use_nodes = True
    mat.blend_method        = "HASHED"
    mat.shadow_method       = "HASHED"
    mat.use_backface_culling = True
    nt = mat.node_tree
    for n in nt.nodes:
        nt.nodes.remove(n)

    tex_coord = _node(nt, "ShaderNodeTexCoord",  location=(-900, 0))
    mapping   = _node(nt, "ShaderNodeMapping",   location=(-700, 0))
    cloud_img = _node(nt, "ShaderNodeTexImage",  location=(-450, 0))
    cloud_img.label = "Cloud Map"
    cloud_img.image = bpy.data.images.load(cloud_path)
    cloud_img.image.colorspace_settings.name = "Non-Color"

    gamma = _node(nt, "ShaderNodeGamma",           location=(-180, 0))
    gamma.inputs["Gamma"].default_value = 1.8

    bsdf = _node(nt, "ShaderNodeBsdfPrincipled",   location=(60, 0))
    bsdf.inputs["Base Color"].default_value         = (0.98, 0.98, 1.0, 1.0)
    bsdf.inputs["Roughness"].default_value          = 0.6
    bsdf.inputs["Specular IOR Level"].default_value = 0.1

    out = _node(nt, "ShaderNodeOutputMaterial",    location=(340, 0))

    _link(nt, tex_coord.outputs["UV"],    mapping.inputs["Vector"])
    _link(nt, mapping.outputs["Vector"],  cloud_img.inputs["Vector"])
    _link(nt, cloud_img.outputs["Color"], gamma.inputs["Color"])
    _link(nt, gamma.outputs["Color"],     bsdf.inputs["Alpha"])
    _link(nt, cloud_img.outputs["Color"], bsdf.inputs["Base Color"])
    _link(nt, bsdf.outputs["BSDF"],       out.inputs["Surface"])
    return mat


# ─────────────────────────────────────────────────────────────────────────────
# Earth + cloud sphere
# ─────────────────────────────────────────────────────────────────────────────

def create_earth(daymap_path, nightmap_path, cloud_path,
                 location=(0.0, 0.0, -1100.0), radius=900):
    bpy.ops.mesh.primitive_uv_sphere_add(
        radius=radius, location=location, segments=64, ring_count=32)
    earth = bpy.context.active_object
    earth.name = "Earth"
    bpy.ops.object.shade_smooth()
    earth.data.materials.append(build_earth_material(daymap_path, nightmap_path))

    bpy.ops.mesh.primitive_uv_sphere_add(
        radius=radius * 1.004, location=location, segments=64, ring_count=32)
    clouds = bpy.context.active_object
    clouds.name = "EarthClouds"
    bpy.ops.object.shade_smooth()
    clouds.data.materials.append(build_cloud_material(cloud_path))

    earth.rotation_mode  = 'QUATERNION'
    clouds.rotation_mode = 'QUATERNION'
    return earth, clouds


# ─────────────────────────────────────────────────────────────────────────────
# Sun disc — visible emissive sphere, warm 5778K, no shadow cast
# ─────────────────────────────────────────────────────────────────────────────

# Sun disc distance: 9000 units — always beyond Earth's max depth (7000 units).
# Radius scaled proportionally: 3.5 * (9000/200) = 157.5 units.
# Apparent angular diameter = 2 * arctan(157.5/9000) = 2.01° — identical to v9.
# This places the Sun disc BEHIND Earth in every rendered image, so it correctly
# appears as a distant background object and never floats between satellite/Earth.
SUN_DISC_DIST   = 9000.0
SUN_DISC_RADIUS = 157.5


# ─────────────────────────────────────────────────────────────────────────────
# Orbital mechanics — physically correct Sun and Moon directions
# ─────────────────────────────────────────────────────────────────────────────
#
# COORDINATE SYSTEM
# ─────────────────
# Camera frame = LVLH (Local Vertical / Local Horizontal) frame of the satellite:
#   Camera -Z  =  nadir  =  toward Earth centre
#   Camera +X  =  along-track  =  orbital velocity direction
#   Camera +Y  =  cross-track  =  completes right-hand system
#   Camera +Z  =  anti-nadir  =  away from Earth (space)
#
# Objects visible to camera have negative Z in camera frame (they are in the
# nadir hemisphere, in the same general direction as Earth).
#
# PHYSICAL BEHAVIOUR
# ──────────────────
# When the satellite is in sunlight (sunlit side of Earth), the Sun has a
# negative Z component in the camera frame — it is toward nadir — because the
# satellite, being between the camera and Earth, is on the same side of Earth
# as the Sun.  This is physically correct and matches real ISS photography
# where the Sun and Earth appear together in nadir-facing camera images.
#
# The Moon follows its ~27.3-day orbit.  Over a 7-day dataset it moves ~91°,
# showing visible phase progression.  The Moon-Sun angle can be small near
# new-moon; the enforce_moon_sun_separation() function handles this.


def compute_sun_direction_eci(epoch_doy: float, mission_day: float) -> list:
    """
    Compute the unit vector from Earth toward the Sun in ECI frame.

    Uses the low-precision solar ephemeris from Jean Meeus "Astronomical
    Algorithms" (simplified, ~0.01 deg accuracy for dates near J2000).

    Parameters
    ----------
    epoch_doy  : day of year for the mission epoch (e.g. 80 = vernal equinox)
    mission_day: elapsed days since epoch (0.0 at start, mission_span_days at end)

    Returns [dx, dy, dz] normalised unit vector in ECI frame.
    """
    n = epoch_doy + mission_day   # day number from reference

    # Mean longitude and mean anomaly (degrees)
    L_deg = (280.460 + 0.9856474 * n) % 360.0
    g_rad = math.radians((357.528 + 0.9856003 * n) % 360.0)

    # Ecliptic longitude (degrees) — includes equation of center
    lam = math.radians(L_deg
                       + 1.915 * math.sin(g_rad)
                       + 0.020 * math.sin(2.0 * g_rad))

    # Obliquity of the ecliptic (degrees, small secular drift)
    eps = math.radians(23.439 - 0.0000004 * n)

    s = [math.cos(lam),
         math.sin(lam) * math.cos(eps),
         math.sin(lam) * math.sin(eps)]
    mag = math.sqrt(sum(x*x for x in s))
    return [x / mag for x in s]


def compute_moon_direction_eci(epoch_doy: float, mission_day: float) -> list:
    """
    Compute the unit vector from Earth toward the Moon in ECI frame.

    Uses the simplified lunar theory derived from Brown's theory (Jean Meeus,
    Chapter 22 of "Astronomical Algorithms"), accurate to ~1 degree.
    Includes the three largest perturbation terms (evection, variation,
    yearly inequality).

    Parameters
    ----------
    epoch_doy  : day of year for the mission epoch
    mission_day: elapsed days since epoch

    Returns [dx, dy, dz] normalised unit vector in ECI frame.
    """
    n = epoch_doy + mission_day

    # Moon fundamental arguments (degrees)
    L_m_deg = (218.316 + 13.176396 * n) % 360.0   # mean longitude
    M_m     = math.radians((134.963 + 13.064993 * n) % 360.0)  # mean anomaly
    F_m     = math.radians(( 93.272 + 13.229350 * n) % 360.0)  # argument of latitude
    M_s     = math.radians((357.528 +  0.985600 * n) % 360.0)  # sun mean anomaly

    L_m_rad = math.radians(L_m_deg)

    # Ecliptic longitude (degrees) — three main correction terms
    lam = math.radians(L_m_deg
                       + 6.289 * math.sin(M_m)           # equation of center
                       - 1.274 * math.sin(2*L_m_rad - M_m)  # evection
                       + 0.658 * math.sin(2*L_m_rad)        # variation
                       - 0.186 * math.sin(M_s))             # yearly inequality

    # Ecliptic latitude
    beta = math.radians(5.128 * math.sin(F_m))

    eps = math.radians(23.439 - 0.0000004 * n)   # obliquity
    cb, sb = math.cos(beta), math.sin(beta)
    cl, sl = math.cos(lam),  math.sin(lam)
    ce, se = math.cos(eps),  math.sin(eps)

    m = [cb * cl,
         ce * cb * sl - se * sb,
         se * cb * sl + ce * sb]
    mag = math.sqrt(sum(x*x for x in m))
    return [x / mag for x in m]


def eci_to_camera_frame(vec_eci: list,
                         mission_day: float,
                         orbit_period_min: float,
                         incl_deg: float,
                         raan_deg: float,
                         ta0_deg: float) -> list:
    """
    Transform a unit vector from ECI to the satellite's camera / LVLH frame.

    The satellite's true anomaly at mission start is ta0_deg.  It advances
    with the orbital period.  The rotation from ECI to LVLH is the 3×3 matrix
    whose rows are the LVLH axis unit vectors expressed in ECI:
        R[0] = X_lvlh  (along-track = velocity direction)
        R[1] = Y_lvlh  (cross-track, completes right-hand system)
        R[2] = Z_lvlh  (nadir = -r_hat; camera -Z)

    result[2] < 0  →  vector points toward nadir (visible in camera)
    result[2] > 0  →  vector points toward anti-nadir (behind camera)

    Parameters
    ----------
    vec_eci         : [dx, dy, dz] unit vector in ECI
    mission_day     : elapsed days since epoch
    orbit_period_min: orbital period in minutes (LEO_PERIOD_MIN = 92.68)
    incl_deg        : orbit inclination in degrees
    raan_deg        : right ascension of ascending node in degrees
    ta0_deg         : true anomaly at mission start in degrees
    """
    # True anomaly at this time (circular orbit)
    t_sec = mission_day * 86400.0
    T_sec = orbit_period_min * 60.0
    ta    = math.radians(ta0_deg) + 2.0 * math.pi / T_sec * t_sec

    incl  = math.radians(incl_deg)
    raan  = math.radians(raan_deg)
    cr, sr = math.cos(raan), math.sin(raan)
    ci, si = math.cos(incl), math.sin(incl)
    ct, st = math.cos(ta),   math.sin(ta)

    # Satellite position unit vector in ECI (r_hat)
    rx = cr*ct - sr*st*ci
    ry = sr*ct + cr*st*ci
    rz = st*si

    # Satellite velocity unit vector in ECI (v_hat, circular orbit)
    vx = -(cr*st + sr*ct*ci)
    vy = -(sr*st - cr*ct*ci)
    vz =   ct*si

    # LVLH axes: Z=nadir(-r_hat), X=along-track(v_hat), Y=cross-track
    Zx, Zy, Zz = -rx, -ry, -rz          # nadir
    Xx, Xy, Xz =  vx,  vy,  vz          # along-track

    # Y = Z × X
    Yx = Zy*Xz - Zz*Xy
    Yy = Zz*Xx - Zx*Xz
    Yz = Zx*Xy - Zy*Xx
    Ym = math.sqrt(Yx*Yx + Yy*Yy + Yz*Yz)
    Yx, Yy, Yz = Yx/Ym, Yy/Ym, Yz/Ym

    # Reorthogonalise X = Y × Z
    Xx = Yy*Zz - Yz*Zy
    Xy = Yz*Zx - Yx*Zz
    Xz = Yx*Zy - Yy*Zx
    Xm = math.sqrt(Xx*Xx + Xy*Xy + Xz*Xz)
    Xx, Xy, Xz = Xx/Xm, Xy/Xm, Xz/Xm

    # Project vec_eci onto each axis (matrix multiply)
    ex, ey, ez = vec_eci
    dx = Xx*ex + Xy*ey + Xz*ez   # along-track component
    dy = Yx*ex + Yy*ey + Yz*ez   # cross-track component
    dz = Zx*ex + Zy*ey + Zz*ez   # nadir component (neg = toward Earth)

    return [dx, dy, dz]


def clamp_direction_to_fov(d: list, half_fov_deg: float = 18.0) -> list:
    """
    Ensure direction d (camera frame, dz < 0 = visible) lies within the
    camera FOV cone of half_fov_deg around the -Z axis.

    Two cases handled:
      • dz >= 0  (body is behind camera — eclipse or wrong orbital phase):
        flip to bring it just in front of camera, preserving XY azimuth.
      • angle from -Z > half_fov_deg  (outside FOV cone):
        clamp to FOV boundary at the same azimuth.

    This is physically equivalent to a small adjustment in the chaser
    spacecraft's approach angle to keep the celestial body in frame.
    """
    dx, dy, dz = d

    # Bring in front of camera if behind
    if dz >= 0.0:
        dz = -abs(dz) - 1e-3

    # Normalise
    mag = math.sqrt(dx*dx + dy*dy + dz*dz)
    if mag < 1e-9:
        return [0.0, 0.0, -1.0]
    dx, dy, dz = dx/mag, dy/mag, dz/mag

    # Angle from -Z:  cos(angle) = -dz
    cos_look = -dz
    min_cos  = math.cos(math.radians(half_fov_deg))

    if cos_look < min_cos:
        # Outside FOV: clamp XY to FOV boundary, keep azimuth
        xy_mag = math.sqrt(dx*dx + dy*dy)
        if xy_mag > 1e-6:
            sin_fov = math.sin(math.radians(half_fov_deg))
            scale   = sin_fov / xy_mag
            dx *= scale
            dy *= scale
        else:
            # Directly behind camera: place at +X edge of FOV
            dx = math.sin(math.radians(half_fov_deg))
            dy = 0.0
        dz = -math.cos(math.radians(half_fov_deg))

    mag = math.sqrt(dx*dx + dy*dy + dz*dz)
    return [dx/mag, dy/mag, dz/mag]


def enforce_moon_sun_separation(moon_dir: list, sun_dir: list,
                                  min_sep_deg: float = 8.0,
                                  half_fov_deg: float = 17.0) -> list:
    """
    If the physical Moon-Sun angular separation is less than min_sep_deg
    (near-new-moon geometry), rotate the Moon azimuth 90° away from the Sun
    within the FOV and re-clamp.

    Parameters
    ----------
    moon_dir   : [dx, dy, dz] moon direction in camera frame
    sun_dir    : [dx, dy, dz] sun direction in camera frame
    min_sep_deg: minimum allowed separation in degrees (default 8)
    half_fov_deg: FOV half-angle for re-clamping
    """
    dot    = max(-1.0, min(1.0, sum(m*s for m, s in zip(moon_dir, sun_dir))))
    sep    = math.degrees(math.acos(dot))

    if sep >= min_sep_deg:
        return moon_dir   # already separated — no adjustment needed

    # Near-new-moon: rotate moon azimuth 90° from sun, keep same elevation
    sun_az  = math.atan2(sun_dir[1], sun_dir[0])
    perp_az = sun_az + 0.5 * math.pi   # 90° perpendicular

    # Off-nadir angle of moon (elevation below -Z axis)
    moon_elev_cos = max(0.0, -moon_dir[2])   # ensure in front of camera
    moon_elev_cos = min(1.0, moon_elev_cos)
    sin_el = math.sqrt(max(0.0, 1.0 - moon_elev_cos*moon_elev_cos))

    if sin_el < 1e-6:
        # Moon was at boresight: move to moderate off-axis angle
        sin_el = math.sin(math.radians(10.0))
        moon_elev_cos = math.sqrt(max(0.0, 1.0 - sin_el*sin_el))

    new_moon = [sin_el * math.cos(perp_az),
                sin_el * math.sin(perp_az),
                -moon_elev_cos]
    mag = math.sqrt(sum(x*x for x in new_moon))
    new_moon = [x/mag for x in new_moon]
    return clamp_direction_to_fov(new_moon, half_fov_deg)


def create_sun_disc(sun_tex_path: str,
                    radius: float = SUN_DISC_RADIUS,
                    emission_strength: float = 80.0):
    """
    Create the sun disc sphere with an 8K UV-textured emission material.

    The 8K sun texture (granulation, convection cells, sunspot regions) is
    used as the emission colour — multiplied by emission_strength so the disc
    is very bright overall while showing realistic surface structure.

    The Mapping node is named "SunMapping" so update_sun_disc() can apply a
    random Z-axis rotation each frame, showing different granulation regions.

    visible_shadow=False: sun disc never shadows the satellite.
    """
    bpy.ops.mesh.primitive_uv_sphere_add(
        radius=radius,
        location=(0.0, 0.0, -SUN_DISC_DIST),
        segments=24, ring_count=12)   # slightly more segments than v7 for rounder disc
    disc = bpy.context.active_object
    disc.name = "SunDisc"
    bpy.ops.object.shade_smooth()
    disc.visible_shadow = False

    mat = bpy.data.materials.new(name="SunDiscMaterial")
    mat.use_nodes = True
    nt = mat.node_tree
    for n in nt.nodes:
        nt.nodes.remove(n)

    # ── UV coordinates + Mapping (Z rotation mutated per frame) ──────────────
    tex_coord = _node(nt, "ShaderNodeTexCoord",   location=(-700, 0))
    mapping   = _node(nt, "ShaderNodeMapping",    location=(-500, 0))
    mapping.name  = "SunMapping"
    mapping.label = "SunMapping"
    _link(nt, tex_coord.outputs["UV"], mapping.inputs["Vector"])

    # ── 8K solar surface texture ──────────────────────────────────────────────
    sun_tex = _node(nt, "ShaderNodeTexImage",     location=(-260, 0))
    sun_tex.label = "Solar Surface"
    sun_tex.image = bpy.data.images.load(sun_tex_path)
    sun_tex.image.colorspace_settings.name = "sRGB"
    _link(nt, mapping.outputs["Vector"], sun_tex.inputs["Vector"])

    # ── Gamma boost: emphasise bright faculae and darken sunspots ─────────────
    gamma = _node(nt, "ShaderNodeGamma",          location=( -20, 0))
    gamma.inputs["Gamma"].default_value = 1.4
    _link(nt, sun_tex.outputs["Color"], gamma.inputs["Color"])

    # ── Emission: texture colour × strength ───────────────────────────────────
    emit = _node(nt, "ShaderNodeEmission",        location=( 200, 0))
    emit.inputs["Strength"].default_value = emission_strength
    _link(nt, gamma.outputs["Color"], emit.inputs["Color"])

    out  = _node(nt, "ShaderNodeOutputMaterial",  location=( 420, 0))
    _link(nt, emit.outputs["Emission"], out.inputs["Surface"])

    disc.data.materials.append(mat)

    # Retrieve Mapping node for per-frame rotation
    mapping_node = mat.node_tree.nodes.get("SunMapping")

    print(f"   Sun disc  : r={radius:.1f}  E={emission_strength:.0f}  "
          f"d={SUN_DISC_DIST:.0f}  ang={2*math.degrees(math.atan(radius/SUN_DISC_DIST)):.2f}deg  "
          f"shadow=False  (always behind Earth)")
    return disc, mapping_node


def update_sun_disc(disc_obj, sun_dir_norm,
                    mapping_node=None, rotation_z: float = 0.0):
    """
    Move sun disc to SUN_DISC_DIST along sun_dir_norm and rotate its UV
    mapping so a different region of the solar surface texture is visible
    each frame (granulation, sunspots, faculae vary across the dataset).
    """
    dx, dy, dz = sun_dir_norm
    disc_obj.location = (dx * SUN_DISC_DIST,
                         dy * SUN_DISC_DIST,
                         dz * SUN_DISC_DIST)
    if mapping_node is not None:
        mapping_node.inputs["Rotation"].default_value = (0.0, 0.0, rotation_z)


# ─────────────────────────────────────────────────────────────────────────────
# Moon — procedural rocky sphere with automatic phase rendering
# ─────────────────────────────────────────────────────────────────────────────

# Moon distance: 9500 units — beyond Sun disc (9000) and Earth's max depth.
# Radius scaled proportionally: 3.0 * (9500/250) = 114.0 units.
# Apparent angular diameter = 2 * arctan(114.0/9500) = 1.38° — identical to v9.
# Placed slightly further than the Sun so Moon is always the deepest background layer.
MOON_DIST   = 9500.0
MOON_RADIUS = 114.0   # scaled to preserve 1.38 deg apparent diameter
MOON_MIN_SEP_FROM_SUN = 8.0   # degrees — minimum angular separation from sun disc

# ── Orbital mechanics constants ───────────────────────────────────────────────
# These govern the physical simulation of Sun and Moon positions.
# The dataset represents frames captured over MISSION_SPAN_DAYS, starting at
# MISSION_EPOCH_DOY (day of year).  Each frame corresponds to a specific time
# within this window; Sun and Moon positions are computed from that time.
MISSION_EPOCH_DOY  = 80.0    # default epoch: day 80 of year ≈ vernal equinox
MISSION_SPAN_DAYS  = 7.0     # default: 7-day dataset window
LEO_PERIOD_MIN     = 92.68   # ISS orbital period (minutes)
LEO_INCLINATION    = 51.6    # ISS orbit inclination (degrees)


def build_moon_material(moon_tex_path: str):
    """
    8K UV-textured lunar surface material.

    Node graph:
      TexCoord(UV) → Mapping(rotation updated per frame) → TexImage(8k_moon)
        └─ Color  → Principled BSDF Base Color
        └─ Color  → BrightnessContrast → Roughness (texture-driven)

    Physical values match real lunar regolith:
      albedo  : 0.12 average (texture range 0.06–0.19 mare/highland)
      roughness: ~0.93 (powdery surface, nearly Lambertian)
      metallic : 0.0
      no emission

    The Mapping node is named "MoonMapping" so update_moon() can find it
    each frame and apply a random Z-axis rotation for surface diversity.
    The SUN light still drives realistic phase rendering automatically.
    """
    mat = bpy.data.materials.new(name="MoonMaterial")
    mat.use_nodes = True
    nt = mat.node_tree
    for n in nt.nodes:
        nt.nodes.remove(n)

    # ── UV coordinates + Mapping (rotation mutated per frame) ─────────────────
    tex_coord = _node(nt, "ShaderNodeTexCoord",  location=(-900, 100))
    mapping   = _node(nt, "ShaderNodeMapping",   location=(-700, 100))
    mapping.name  = "MoonMapping"   # named so update_moon() can find it
    mapping.label = "MoonMapping"
    _link(nt, tex_coord.outputs["UV"], mapping.inputs["Vector"])

    # ── 8K Moon texture ───────────────────────────────────────────────────────
    moon_tex = _node(nt, "ShaderNodeTexImage",   location=(-450, 100))
    moon_tex.label = "Moon Surface"
    moon_tex.image = bpy.data.images.load(moon_tex_path)
    moon_tex.image.colorspace_settings.name = "sRGB"  # texture is greyscale sRGB
    _link(nt, mapping.outputs["Vector"], moon_tex.inputs["Vector"])

    # ── Brightness/Contrast: fine-tune to match lunar albedo 0.12 ─────────────
    # The 8K texture is already correctly scaled; slight contrast boost
    # separates mare basalt (dark) from highland anorthosite (bright).
    bc = _node(nt, "ShaderNodeBrightContrast",   location=(-180, 100))
    bc.inputs["Bright"].default_value   = -0.04   # slight darkening to hit 0.12
    bc.inputs["Contrast"].default_value =  0.15   # sharpen mare/highland boundary
    _link(nt, moon_tex.outputs["Color"], bc.inputs["Color"])

    # ── Roughness driven by inverse of texture brightness ─────────────────────
    # Bright highlands (smoother ejecta blankets) → roughness 0.88
    # Dark maria (loose regolith)                 → roughness 0.96
    invert = _node(nt, "ShaderNodeInvert",       location=(-180, -60))
    invert.inputs["Fac"].default_value = 0.25   # mild inversion
    _link(nt, moon_tex.outputs["Color"], invert.inputs["Color"])

    rough_ramp = _node(nt, "ShaderNodeValToRGB",  location=(20, -60))
    rough_ramp.label = "Lunar Roughness"
    rough_ramp.color_ramp.interpolation = "LINEAR"
    rough_ramp.color_ramp.elements[0].position = 0.0
    rough_ramp.color_ramp.elements[0].color    = (0.88, 0.88, 0.88, 1)  # bright → smoother
    rough_ramp.color_ramp.elements[-1].position = 1.0
    rough_ramp.color_ramp.elements[-1].color    = (0.96, 0.96, 0.96, 1)  # dark → rougher
    _link(nt, invert.outputs["Color"], rough_ramp.inputs["Fac"])

    # ── Principled BSDF ───────────────────────────────────────────────────────
    bsdf = _node(nt, "ShaderNodeBsdfPrincipled",  location=(260, 100))
    bsdf.inputs["Metallic"].default_value           = 0.0
    bsdf.inputs["Specular IOR Level"].default_value = 0.02   # nearly Lambertian
    _link(nt, bc.outputs["Color"],          bsdf.inputs["Base Color"])
    _link(nt, rough_ramp.outputs["Color"],  bsdf.inputs["Roughness"])

    out = _node(nt, "ShaderNodeOutputMaterial",   location=(480, 100))
    _link(nt, bsdf.outputs["BSDF"], out.inputs["Surface"])
    return mat


def create_moon(moon_tex_path: str, radius: float = MOON_RADIUS):
    """
    Create the Moon sphere with 8K UV-textured regolith material.
    visible_shadow=False: moon does not cast shadows on satellite.
    The moon IS lit by SUN light → automatic phase rendering.
    Returns (moon_obj, mapping_node) so update_moon() can rotate the texture.
    """
    bpy.ops.mesh.primitive_uv_sphere_add(
        radius=radius,
        location=(0.0, 0.0, -MOON_DIST),
        segments=32, ring_count=16
    )
    moon = bpy.context.active_object
    moon.name = "Moon"
    bpy.ops.object.shade_smooth()
    moon.visible_shadow = False
    mat = build_moon_material(moon_tex_path)
    moon.data.materials.append(mat)

    # Retrieve the named Mapping node for per-frame rotation updates
    mapping_node = mat.node_tree.nodes.get("MoonMapping")

    print(f"   Moon      : r={radius:.1f}  d={MOON_DIST:.0f}  "
          f"ang={2*math.degrees(math.atan(radius/MOON_DIST)):.2f}deg  "
          f"shadow=False  phase=auto  (always behind Earth)")
    return moon, mapping_node


# sample_moon_in_fov() removed in v9 — replaced by compute_moon_direction_eci()
# + eci_to_camera_frame() + clamp_direction_to_fov() + enforce_moon_sun_separation()


def update_moon(moon_obj, moon_dir_norm: tuple,
               mapping_node=None, rotation_z: float = 0.0):
    """
    Move moon sphere to MOON_DIST along moon_dir_norm and optionally
    rotate the UV mapping so a different region of the 8K texture
    faces the camera each frame.

    Parameters
    ----------
    moon_obj     : Moon bpy.types.Object
    moon_dir_norm: (dx, dy, dz) unit direction toward moon
    mapping_node : ShaderNodeMapping "MoonMapping" — may be None
    rotation_z   : Z-axis rotation in radians applied to UV texture
    """
    dx, dy, dz = moon_dir_norm
    moon_obj.location = (dx * MOON_DIST,
                         dy * MOON_DIST,
                         dz * MOON_DIST)
    if mapping_node is not None:
        mapping_node.inputs["Rotation"].default_value = (0.0, 0.0, rotation_z)


# ─────────────────────────────────────────────────────────────────────────────
# Spacecraft GLB loader
# ─────────────────────────────────────────────────────────────────────────────

def load_spacecraft_glb(glb_path: str):
    before = set(bpy.context.scene.objects)
    bpy.ops.import_scene.gltf(filepath=glb_path)
    new_objs = [o for o in bpy.context.scene.objects if o not in before]
    if not new_objs:
        raise RuntimeError(f"No objects imported from {glb_path}")
    meshes = [o for o in new_objs if o.type == "MESH"] or new_objs
    import blenderproc.python.types.MeshObjectUtility as mu
    return [mu.MeshObject(o) for o in meshes]


# ─────────────────────────────────────────────────────────────────────────────
# LEO orbit model — Earth position + orientation per frame
# ─────────────────────────────────────────────────────────────────────────────

LEO_DIST     = 1100.0
EARTH_RADIUS = 900.0
AXIAL_TILT   = math.radians(23.5)


def leo_earth_pose(frame_idx, total_frames,
                   orbit_inclination=math.radians(51.6),
                   orbit_phase=0.0, spin_phase=0.0,
                   orbit_revolutions=1.0, spin_revolutions=1.5,
                   earth_dist=None):
    # earth_dist overrides LEO_DIST for per-frame distance variation.
    # phi_max scales with distance: close shots push Earth off-axis so
    # star background is always visible in at least the frame corners.
    t     = frame_idx / max(total_frames - 1, 1)
    theta = 2.0 * math.pi * orbit_revolutions * t + orbit_phase

    dist = earth_dist if earth_dist is not None else LEO_DIST

    phi_min = math.radians(5)
    if dist < 1400:
        phi_max = math.radians(55)
    elif dist < 3000:
        phi_max = math.radians(45)
    else:
        phi_max = math.radians(40)
    phi = phi_min + (phi_max - phi_min) * 0.5 * (1.0 + math.sin(theta * 0.7 + 1.0))

    nx = math.sin(phi) * math.cos(theta)
    ny = math.sin(phi) * math.sin(theta) * math.sin(orbit_inclination)
    nz = -math.cos(phi)   # always negative

    ex, ey, ez = nx * dist, ny * dist, nz * dist

    q_tilt = Quaternion((math.cos(AXIAL_TILT / 2), math.sin(AXIAL_TILT / 2), 0, 0))
    sa     = 2.0 * math.pi * spin_revolutions * t + spin_phase
    q_spin = Quaternion((math.cos(sa / 2), 0, math.sin(sa / 2), 0))
    q      = q_tilt @ q_spin
    return (ex, ey, ez), [q.w, q.x, q.y, q.z]


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────

def find_resume_index(images_dir: str, total: int) -> int:
    if not os.path.isdir(images_dir):
        return 0
    done = set()
    pat  = re.compile(r'^img_(\d{6})\.png$')
    for fname in os.listdir(images_dir):
        m = pat.match(fname)
        if m:
            done.add(int(m.group(1)))
    for idx in range(total):
        if idx not in done:
            return idx
    return total


def open_csv_for_append(csv_path: str):
    is_new = not os.path.isfile(csv_path) or os.path.getsize(csv_path) == 0
    fh     = open(csv_path, "a", newline="")
    writer = csv.writer(fh)
    if is_new:
        writer.writerow([
            "image_name",
            "qw", "qx", "qy", "qz",                          # satellite attitude
            "tx", "ty", "tz",                                  # satellite position
            "earth_tx", "earth_ty", "earth_tz",                # Earth position
            "earth_qw", "earth_qx", "earth_qy", "earth_qz",   # Earth orientation
            "earth_dist",                                       # camera-Earth distance
            "sun_dx", "sun_dy", "sun_dz",                      # sun direction (ECI→LVLH)
            "moon_dx", "moon_dy", "moon_dz",                   # moon direction (ECI→LVLH)
            "mission_day",                                      # elapsed days since epoch
        ])
        fh.flush()
    return fh, writer


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    S = lambda f: os.path.join(SCRIPT_DIR, f)

    parser = ArgumentParser()
    parser.add_argument("--glb_path",     type=str, default=S("CloudSat.glb"))
    parser.add_argument("--tex_daymap",   type=str, default=S("8k_earth_daymap.jpg"))
    parser.add_argument("--tex_nightmap", type=str, default=S("8k_earth_nightmap.jpg"))
    parser.add_argument("--tex_clouds",   type=str, default=S("8k_earth_clouds.jpg"))
    parser.add_argument("--tex_stars",    type=str, default=S("8k_stars.jpg"))
    parser.add_argument("--num_images",   type=int, default=500)
    parser.add_argument("--output_dir",   type=str, default=S("sun_dataset_v2"))
    parser.add_argument("--resolution",   type=int, nargs=2, default=[512, 512])
    parser.add_argument("--samples",      type=int, default=48,
                        help="Cycles samples per pixel (default 48 + OptiX denoiser)")
    parser.add_argument("--sun_half_fov",       type=float, default=18.0,
                        help="FOV half-angle for sun clamping in degrees (default 18).")
    parser.add_argument("--epoch_doy",          type=float, default=MISSION_EPOCH_DOY,
                        help="Day of year for dataset epoch (1–365). "
                             "80=vernal equinox, 172=summer solstice, "
                             "264=autumnal equinox, 355=winter solstice.")
    parser.add_argument("--mission_span_days",  type=float, default=MISSION_SPAN_DAYS,
                        help="Total mission time window in days (default 7). "
                             "Longer span = more Moon phase variety. "
                             "Sun moves ~1 deg/day; Moon ~13 deg/day.")
    parser.add_argument("--tex_moon",     type=str, default=S("8k_moon.jpg"),
                        help="Path to 8K Moon surface texture (equirectangular JPG).")
    parser.add_argument("--tex_sun",      type=str, default=S("8k_sun.jpg"),
                        help="Path to 8K Sun surface texture (equirectangular JPG).")
    args = parser.parse_args()

    def resolve(p):
        return p if os.path.isabs(p) else os.path.join(SCRIPT_DIR, p)
    args.glb_path     = resolve(args.glb_path)
    args.tex_daymap   = resolve(args.tex_daymap)
    args.tex_nightmap = resolve(args.tex_nightmap)
    args.tex_clouds   = resolve(args.tex_clouds)
    args.tex_stars    = resolve(args.tex_stars)
    args.tex_moon     = resolve(args.tex_moon)
    args.tex_sun      = resolve(args.tex_sun)
    args.output_dir   = resolve(args.output_dir)

    for label, path in [("GLB",        args.glb_path),
                        ("Day map",    args.tex_daymap),
                        ("Night map",  args.tex_nightmap),
                        ("Clouds",     args.tex_clouds),
                        ("Stars",      args.tex_stars),
                        ("Moon tex",   args.tex_moon),
                        ("Sun tex",    args.tex_sun)]:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"\n[{label}] not found:\n  {path}")

    # ── Deterministic RNG ─────────────────────────────────────────────────────
    np.random.seed(42)
    random.seed(42)

    # ── Pre-generate ALL values — deterministic from seed regardless of start_idx ─
    print("Pre-generating scene parameters using orbital mechanics ...")
    all_sat_quats  = []
    all_sun_dirs   = {}
    all_moon_dirs  = {}
    all_earth_dist = {}
    all_sun_rots   = {}   # sun disc UV rotation — shows different granulation
    all_moon_rots  = {}   # moon UV rotation — shows different lunar face
    all_mission_day = {}  # mission time (days) for each frame — saved to CSV

    # ── Fixed orbit configuration (random but deterministic per seed) ─────────
    # RAAN and initial true anomaly are randomised so different runs produce
    # different lighting geometries.  Both are deterministic from seed(42).
    raan_deg = random.uniform(0.0, 360.0)    # right ascension of ascending node
    ta0_deg  = random.uniform(0.0, 360.0)    # true anomaly at mission start

    print(f"   Epoch        : day {args.epoch_doy:.0f} of year  "
          f"(e.g. 80=equinox, 172=solstice)")
    print(f"   Mission span : {args.mission_span_days:.1f} days  "
          f"(Sun moves {args.mission_span_days*1.0:.1f} deg, "
          f"Moon moves {args.mission_span_days*13.18:.0f} deg)")
    print(f"   Orbit RAAN   : {raan_deg:.1f} deg   TA0: {ta0_deg:.1f} deg")

    for i in range(args.num_images):
        # Satellite attitude (random, unchanged)
        q = np.random.randn(4); q /= np.linalg.norm(q)
        all_sat_quats.append(q.tolist())

        # Mission time for this frame
        t = args.mission_span_days * i / max(args.num_images - 1, 1)
        all_mission_day[i] = t

        # ── Sun: physical ephemeris → ECI → camera frame ─────────────────────
        sun_eci = compute_sun_direction_eci(args.epoch_doy, t)
        sd = eci_to_camera_frame(sun_eci, t, LEO_PERIOD_MIN,
                                  LEO_INCLINATION, raan_deg, ta0_deg)
        sd = clamp_direction_to_fov(sd, args.sun_half_fov)
        all_sun_dirs[i] = sd

        # ── Moon: physical ephemeris → ECI → camera frame ────────────────────
        moon_eci = compute_moon_direction_eci(args.epoch_doy, t)
        md = eci_to_camera_frame(moon_eci, t, LEO_PERIOD_MIN,
                                  LEO_INCLINATION, raan_deg, ta0_deg)
        md = clamp_direction_to_fov(md, 17.0)
        # Enforce minimum angular separation from sun (near-new-moon handling)
        md = enforce_moon_sun_separation(md, sd, MOON_MIN_SEP_FROM_SUN, 17.0)
        all_moon_dirs[i] = md

        # ── Earth distance (three weighted scene categories, unchanged) ───────
        r = random.random()
        if r < 0.30:
            ed = random.uniform(1000.0, 1400.0)   # close: large partial globe
        elif r < 0.80:
            ed = random.uniform(1400.0, 3000.0)   # medium: LEO globe + space
        else:
            ed = random.uniform(3000.0, 7000.0)   # far: small disc + stars
        all_earth_dist[i] = ed

        # UV texture rotations (still random for surface detail variety)
        all_sun_rots[i]  = random.uniform(0.0, 2.0 * math.pi)
        all_moon_rots[i] = random.uniform(0.0, 2.0 * math.pi)

    orbit_phase = random.uniform(0.0, 2 * math.pi)
    spin_phase  = random.uniform(0.0, 2 * math.pi)

    # ── Output dirs ───────────────────────────────────────────────────────────
    output_dir = args.output_dir
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    # ── Resume detection ──────────────────────────────────────────────────────
    start_idx = find_resume_index(images_dir, args.num_images)
    if start_idx >= args.num_images:
        print(f"All {args.num_images} images already exist. Nothing to do.")
        return
    if start_idx > 0:
        print(f"Resuming from frame {start_idx} "
              f"({start_idx} done, {args.num_images - start_idx} remaining)")
    else:
        print(f"Starting fresh — {args.num_images} images to generate")

    # ── CSV (append mode — safe on resume) ───────────────────────────────────
    csv_path       = os.path.join(output_dir, "dataset.csv")
    csv_fh, csv_writer = open_csv_for_append(csv_path)

    # ── Blender / BlenderProc init ────────────────────────────────────────────
    bproc.init()

    print("Configuring GPU rendering ...")
    configure_gpu_rendering(samples=args.samples)

    print("Setting up star background ...")
    setup_star_background(args.tex_stars)

    print("Building Earth ...")
    earth, clouds = create_earth(
        args.tex_daymap, args.tex_nightmap, args.tex_clouds,
        location=(0.0, 0.0, -LEO_DIST), radius=int(EARTH_RADIUS)
    )
    earth_mat = earth.data.materials[0]

    print("Creating sun disc ...")
    sun_disc, sun_mapping_node = create_sun_disc(
        sun_tex_path=args.tex_sun,
        radius=SUN_DISC_RADIUS,
        emission_strength=80.0
    )

    print("Creating moon ...")
    moon, moon_mapping_node = create_moon(
        moon_tex_path=args.tex_moon,
        radius=MOON_RADIUS
    )

    print("Loading spacecraft GLB ...")
    spacecraft_parts = load_spacecraft_glb(args.glb_path)
    for part in spacecraft_parts:
        part.set_scale([0.0004, 0.0004, 0.0004])

    # ── Camera ────────────────────────────────────────────────────────────────
    bproc.camera.set_resolution(*args.resolution)
    cam = bpy.context.scene.camera
    cam.data.clip_start = 0.01
    # clip_end raised to 12000: Sun disc at ~8550 and Moon at ~9025 must be
    # visible.  clip_far/clip_near = 12000/0.01 = 1.2M — acceptable for Cycles
    # (unlike rasterisers, Cycles ray-tracing is not affected by depth-buffer
    # precision; this value only gates which objects Cycles traces at all).
    cam.data.clip_end   = 12000.0

    # Register camera pose ONCE — camera never moves.
    # (In v2 this was inside the loop causing 500x render overhead by frame 500)
    bpy.context.scene.frame_start = 0
    bpy.context.scene.frame_end   = 0
    bproc.camera.add_camera_pose(np.eye(4))

    # ── Lights ────────────────────────────────────────────────────────────────
    sun = bproc.types.Light()
    sun.set_type("SUN")
    sun.set_energy(8.0)
    sun.set_location([800.0, 600.0, 900.0])
    try:
        sun.blender_obj.data.angle = math.radians(0.53)  # real solar angular diameter
    except AttributeError:
        pass

    fill = bproc.types.Light()
    fill.set_type("POINT")
    fill.set_energy(0.8)
    fill.set_location([0.0, 0.0, -LEO_DIST * 0.4])

    fixed_location = [0.0, 0.0, -28.0]

    # ── Render loop ───────────────────────────────────────────────────────────
    print(f"\nGenerating {args.num_images - start_idx} images "
          f"(frames {start_idx}–{args.num_images - 1}) ...")
    print(f"   Satellite : fixed (0,0,-28), random attitude per frame")
    print(f"   Sun disc  : 8K textured, physical ephemeris, d={SUN_DISC_DIST:.0f} (behind Earth)")
    print(f"   Moon      : 8K textured, physical orbit,    d={MOON_DIST:.0f} (behind Earth)")

    print(f"   Earth     : LEO orbit, close/medium/far distance categories")
    print(f"   Orbital   : epoch_doy={args.epoch_doy:.0f}  span={args.mission_span_days:.1f}d  "
          f"RAAN={raan_deg:.0f}deg  TA0={ta0_deg:.0f}deg")
    print(f"   Stars     : 8K HDRI background — visible in medium/far shots")
    print(f"   Render    : {args.samples} spp + OptiX/OIDN + persistent_data")
    print(f"   CSV       : per-frame flush → {csv_path}")

    for i in range(start_idx, args.num_images):

        # ── Sun: direction, SUN light, Earth terminator, sun disc + texture rot
        sd = all_sun_dirs[i]
        sun.set_location([sd[0] * 1200, sd[1] * 1200, sd[2] * 1200])
        update_sun_direction(earth_mat, sd)
        update_sun_disc(sun_disc, sd,
                        mapping_node=sun_mapping_node,
                        rotation_z=all_sun_rots[i])

        # ── Moon: position + texture rotation (different face each frame) ──────
        md = all_moon_dirs[i]
        update_moon(moon, md,
                    mapping_node=moon_mapping_node,
                    rotation_z=all_moon_rots[i])

        # ── Earth: LEO orbit pose with per-frame distance ──────────────────────
        ed = all_earth_dist[i]
        (ex, ey, ez), earth_orient = leo_earth_pose(
            frame_idx=i, total_frames=args.num_images,
            orbit_inclination=math.radians(51.6),
            orbit_phase=orbit_phase, spin_phase=spin_phase,
            orbit_revolutions=1.0, spin_revolutions=1.5,
            earth_dist=ed,
        )
        earth.location  = (ex, ey, ez)
        clouds.location = (ex, ey, ez)
        ew, eqx, eqy, eqz = earth_orient
        q_earth = Quaternion((ew, eqx, eqy, eqz))
        earth.rotation_quaternion  = q_earth
        clouds.rotation_quaternion = q_earth
        fill.set_location([ex * 0.4, ey * 0.4, ez * 0.4])

        # ── Satellite: random attitude, fixed position, unchanged scale ────────
        qv  = all_sat_quats[i]
        rot = np.array(Quaternion((qv[0], qv[1], qv[2], qv[3])).to_matrix())
        for part in spacecraft_parts:
            part.set_rotation_mat(rot)
            part.set_location(fixed_location)

        # ── Render ────────────────────────────────────────────────────────────
        data = bproc.renderer.render()

        # ── Save PNG (compress_level=1 = fastest lossless) ────────────────────
        image_name = f"img_{i:06d}.png"
        image_path = os.path.join(images_dir, image_name)
        imageio.imwrite(image_path, np.asarray(data["colors"][0]), compress_level=1)

        # ── Write CSV row and flush immediately (crash-safe) ──────────────────
        qw, qx, qy, qz = qv
        tx, ty, tz      = fixed_location
        csv_writer.writerow([
            image_name,
            f"{qw:.8f}", f"{qx:.8f}", f"{qy:.8f}", f"{qz:.8f}",   # satellite quat
            f"{tx:.4f}",  f"{ty:.4f}",  f"{tz:.4f}",               # satellite pos
            f"{ex:.2f}",  f"{ey:.2f}",  f"{ez:.2f}",               # Earth pos
            f"{ew:.8f}", f"{eqx:.8f}", f"{eqy:.8f}", f"{eqz:.8f}", # Earth orient
            f"{ed:.1f}",                                             # Earth dist
            f"{sd[0]:.6f}", f"{sd[1]:.6f}", f"{sd[2]:.6f}",        # sun dir
            f"{md[0]:.6f}", f"{md[1]:.6f}", f"{md[2]:.6f}",        # moon dir
            f"{all_mission_day[i]:.6f}",                                # mission day
        ])
        csv_fh.flush()

        # ── Progress ──────────────────────────────────────────────────────────
        if (i - start_idx + 1) % 10 == 0 or i == args.num_images - 1:
            phi_deg  = math.degrees(math.acos(max(-1.0, min(1.0, -ez / ed))))
            dist_cat = 'close' if ed < 1400 else ('far' if ed > 3000 else 'med')
            sep_deg  = math.degrees(math.acos(max(-1.0, min(1.0,
                           md[0]*sd[0] + md[1]*sd[1] + md[2]*sd[2]))))
            t_day    = all_mission_day[i]
            print(f"   [{i+1:>4}/{args.num_images}]  "
                  f"t={t_day:.3f}d  Earth d={ed:.0f}({dist_cat}) "
                  f"phi={phi_deg:.1f}d  sun-moon={sep_deg:.1f}d")

    csv_fh.close()

    total_done = len([f for f in os.listdir(images_dir)
                      if re.match(r'^img_\d{6}\.png$', f)])
    print(f"\nDone!  {total_done}/{args.num_images} images → {images_dir}")
    print(f"CSV   → {csv_path}")
    print(f"\nTo split train/test:")
    print(f"  python -c \"import pandas as pd; from sklearn.model_selection "
          f"import train_test_split; df=pd.read_csv(r'{csv_path}'); "
          f"tr,te=train_test_split(df,test_size=0.2,random_state=42); "
          f"tr.to_csv(r'{os.path.join(output_dir, 'train.csv')}',index=False); "
          f"te.to_csv(r'{os.path.join(output_dir, 'test.csv')}',index=False)\"")


if __name__ == "__main__":
    main()
