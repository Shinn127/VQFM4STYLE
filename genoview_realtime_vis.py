"""GenoView-based real-time dual-character visualization for 375-dim motion features."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

import numpy as np

from Utils import quat as quat_ops


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
GENOVIEW_DIR = REPO_ROOT / "GenoViewPython-main"
GENOVIEW_RESOURCES_DIR = GENOVIEW_DIR / "resources"

SELECTED_24_NAMES = [
    "Hips",
    "Spine",
    "Spine1",
    "Spine2",
    "Spine3",
    "Neck",
    "Neck1",
    "Head",
    "RightShoulder",
    "RightArm",
    "RightForeArm",
    "RightHand",
    "LeftShoulder",
    "LeftArm",
    "LeftForeArm",
    "LeftHand",
    "RightUpLeg",
    "RightLeg",
    "RightFoot",
    "RightToeBase",
    "LeftUpLeg",
    "LeftLeg",
    "LeftFoot",
    "LeftToeBase",
]


def _as_bytes(path: Path) -> bytes:
    return str(path.expanduser().resolve()).encode("utf-8")


def _load_genoview_module() -> Any:
    if not GENOVIEW_DIR.exists():
        raise FileNotFoundError(f"GenoView project directory not found: {GENOVIEW_DIR}")
    if str(GENOVIEW_DIR) not in sys.path:
        sys.path.insert(0, str(GENOVIEW_DIR))
    try:
        return importlib.import_module("genoview")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Failed to import GenoView runtime. Install the Python raylib bindings first, e.g. `pip install raylib`."
        ) from exc


def _ensure_feature_array(features_375: np.ndarray) -> np.ndarray:
    x = np.asarray(features_375, dtype=np.float32)
    if x.ndim == 1:
        x = x[None, :]
    if x.ndim != 2 or x.shape[1] != 375:
        raise ValueError(f"Expected shape (T, 375) or (375,), got {x.shape}")
    return x


def _decode_selected_state(features_375: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = _ensure_feature_array(features_375)
    state = x[:, 84:]
    root_vel = state[:, :3]

    local_pos = state[:, 3:75].reshape(-1, 24, 3)
    local_rot_x = state[:, 75:147].reshape(-1, 24, 3)
    local_rot_y = state[:, 147:219].reshape(-1, 24, 3)
    local_rot_xy = np.stack([local_rot_x, local_rot_y], axis=-1)
    local_rot = quat_ops.from_xform_xy(local_rot_xy)
    return root_vel.astype(np.float32), local_pos.astype(np.float32), local_rot.astype(np.float32)


def _integrate_root_motion(
    root_vel: np.ndarray,
    *,
    dt: float,
    root_wy_in_degrees: bool,
    use_global: bool,
) -> tuple[np.ndarray, np.ndarray]:
    t_total = root_vel.shape[0]
    root_pos_seq = np.zeros((t_total, 3), dtype=np.float32)
    root_rot_seq = np.zeros((t_total, 4), dtype=np.float32)

    if not use_global:
        root_rot_seq[:] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        return root_pos_seq, root_rot_seq

    root_motion = root_vel * float(dt)
    trans_offset = root_motion.copy()
    trans_offset[:, 1] = 0.0

    yaw_delta = root_motion[:, 1]
    if root_wy_in_degrees:
        yaw_delta = np.radians(yaw_delta)
    yaw_offset = quat_ops.from_angle_axis(yaw_delta, np.array([0.0, 1.0, 0.0], dtype=np.float32))

    root_pos = np.zeros((3,), dtype=np.float32)
    root_rot = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    for frame_idx in range(t_total):
        root_pos += trans_offset[frame_idx]
        root_pos_seq[frame_idx] = root_pos
        root_rot = quat_ops.mul(yaw_offset[frame_idx : frame_idx + 1], root_rot[None, :])[0]
        root_rot_seq[frame_idx] = root_rot

    return root_pos_seq, root_rot_seq


def _reconstruct_selected_globals(
    features_375: np.ndarray,
    *,
    dt: float,
    root_wy_in_degrees: bool,
    use_global: bool,
) -> tuple[np.ndarray, np.ndarray]:
    root_vel, local_pos, local_rot = _decode_selected_state(features_375)
    root_pos_seq, root_rot_seq = _integrate_root_motion(
        root_vel,
        dt=dt,
        root_wy_in_degrees=root_wy_in_degrees,
        use_global=use_global,
    )

    global_pos = quat_ops.mul_vec(root_rot_seq[:, None, :], local_pos) + root_pos_seq[:, None, :]
    global_rot = quat_ops.mul(root_rot_seq[:, None, :], local_rot)
    return global_pos.astype(np.float32), global_rot.astype(np.float32)


def _reduce_parents(full_parents: np.ndarray, selected_indices: list[int]) -> np.ndarray:
    selected_set = set(selected_indices)
    remap = {joint_idx: reduced_idx for reduced_idx, joint_idx in enumerate(selected_indices)}

    reduced: list[int] = []
    for joint_idx in selected_indices:
        parent_idx = int(full_parents[joint_idx])
        while parent_idx != -1 and parent_idx not in selected_set:
            parent_idx = int(full_parents[parent_idx])
        reduced.append(-1 if parent_idx == -1 else remap[parent_idx])
    return np.asarray(reduced, dtype=np.int32)


def _expand_selected_to_full(
    *,
    selected_global_pos: np.ndarray,
    selected_global_rot: np.ndarray,
    full_names: list[str],
    full_parents: np.ndarray,
    bind_local_pos: np.ndarray,
    bind_local_rot: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[int], np.ndarray]:
    if selected_global_pos.ndim != 3 or selected_global_pos.shape[1:] != (24, 3):
        raise ValueError(f"Expected selected_global_pos shape (T, 24, 3), got {selected_global_pos.shape}")
    if selected_global_rot.ndim != 3 or selected_global_rot.shape[1:] != (24, 4):
        raise ValueError(f"Expected selected_global_rot shape (T, 24, 4), got {selected_global_rot.shape}")

    name_to_full_index = {name: idx for idx, name in enumerate(full_names)}
    missing = [name for name in SELECTED_24_NAMES if name not in name_to_full_index]
    if missing:
        raise ValueError(f"Selected 24-joint mapping missing bones in Geno skeleton: {missing}")

    selected_full_indices = [name_to_full_index[name] for name in SELECTED_24_NAMES]
    t_total = selected_global_pos.shape[0]
    full_joint_count = len(full_names)

    full_pos = np.zeros((t_total, full_joint_count, 3), dtype=np.float32)
    full_rot = np.zeros((t_total, full_joint_count, 4), dtype=np.float32)
    full_rot[:] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    selected_mask = np.zeros((full_joint_count,), dtype=bool)
    selected_mask[selected_full_indices] = True

    for local_idx, full_idx in enumerate(selected_full_indices):
        full_pos[:, full_idx] = selected_global_pos[:, local_idx]
        full_rot[:, full_idx] = selected_global_rot[:, local_idx]

    for full_idx in range(full_joint_count):
        if selected_mask[full_idx]:
            continue
        parent_idx = int(full_parents[full_idx])
        if parent_idx == -1:
            full_pos[:, full_idx] = bind_local_pos[full_idx]
            full_rot[:, full_idx] = bind_local_rot[full_idx]
            continue

        parent_rot = full_rot[:, parent_idx : parent_idx + 1, :]
        child_local_rot = bind_local_rot[None, full_idx : full_idx + 1, :]
        child_local_pos = bind_local_pos[None, full_idx : full_idx + 1, :]

        full_rot[:, full_idx] = quat_ops.mul(parent_rot, child_local_rot)[:, 0]
        full_pos[:, full_idx] = quat_ops.mul_vec(parent_rot, child_local_pos)[:, 0] + full_pos[:, parent_idx]

    selected_reduced_parents = _reduce_parents(full_parents, selected_full_indices)
    return full_pos, full_rot, selected_full_indices, selected_reduced_parents


def visualize_dual_genoview_from_375(
    *,
    ref_features_375: np.ndarray,
    pred_features_375: np.ndarray,
    fps: int,
    dt: float,
    use_global: bool,
    root_wy_in_degrees: bool = True,
    title: str | None = None,
    screen_width: int = 1600,
    screen_height: int = 900,
    model_gap: float = 2.4,
) -> None:
    if fps <= 0:
        raise ValueError(f"fps must be > 0, got {fps}")
    if dt <= 0.0:
        raise ValueError(f"dt must be > 0, got {dt}")

    gv = _load_genoview_module()

    ref_selected_pos, ref_selected_rot = _reconstruct_selected_globals(
        ref_features_375,
        dt=dt,
        root_wy_in_degrees=root_wy_in_degrees,
        use_global=use_global,
    )
    pred_selected_pos, pred_selected_rot = _reconstruct_selected_globals(
        pred_features_375,
        dt=dt,
        root_wy_in_degrees=root_wy_in_degrees,
        use_global=use_global,
    )

    t_total = min(ref_selected_pos.shape[0], pred_selected_pos.shape[0])
    if t_total <= 1:
        raise ValueError(f"Need at least 2 frames for GenoView animation, got {t_total}")
    ref_selected_pos = ref_selected_pos[:t_total]
    ref_selected_rot = ref_selected_rot[:t_total]
    pred_selected_pos = pred_selected_pos[:t_total]
    pred_selected_rot = pred_selected_rot[:t_total]

    geno_bind = gv.bvh.load(str(GENOVIEW_RESOURCES_DIR / "Geno_bind.bvh"))
    full_names = list(geno_bind["names"])
    full_parents = np.asarray(geno_bind["parents"], dtype=np.int32)

    shadow_shader = None
    skinned_shadow_shader = None
    skinned_basic_shader = None
    basic_shader = None
    lighting_shader = None
    ssao_shader = None
    blur_shader = None
    fxaa_shader = None
    ground_model = None
    shadow_map = None
    gbuffer = None
    lighted = None
    ssao_front = None
    ssao_back = None
    ref_model = None
    pred_model = None

    gv.SetConfigFlags(gv.FLAG_VSYNC_HINT)
    gv.InitWindow(int(screen_width), int(screen_height), (title or "FlowMatch GenoView Compare").encode("utf-8"))
    gv.SetTargetFPS(int(fps))

    ref_model = gv.LoadGenoModel(_as_bytes(GENOVIEW_RESOURCES_DIR / "Geno.bin"))
    pred_model = gv.LoadGenoModel(_as_bytes(GENOVIEW_RESOURCES_DIR / "Geno.bin"))

    bind_pos, bind_rot = gv.GetModelBindPoseAsNumpyArrays(ref_model)
    bind_pos = np.asarray(bind_pos, dtype=np.float32)
    bind_rot = np.asarray(bind_rot, dtype=np.float32)

    bind_local_rot, bind_local_pos = quat_ops.ik(bind_rot[None, ...], bind_pos[None, ...], full_parents)
    bind_local_rot = np.asarray(bind_local_rot[0], dtype=np.float32)
    bind_local_pos = np.asarray(bind_local_pos[0], dtype=np.float32)

    ref_full_pos, ref_full_rot, selected_full_indices, selected_reduced_parents = _expand_selected_to_full(
        selected_global_pos=ref_selected_pos,
        selected_global_rot=ref_selected_rot,
        full_names=full_names,
        full_parents=full_parents,
        bind_local_pos=bind_local_pos,
        bind_local_rot=bind_local_rot,
    )
    pred_full_pos, pred_full_rot, _, _ = _expand_selected_to_full(
        selected_global_pos=pred_selected_pos,
        selected_global_rot=pred_selected_rot,
        full_names=full_names,
        full_parents=full_parents,
        bind_local_pos=bind_local_pos,
        bind_local_rot=bind_local_rot,
    )

    ref_offset = np.array([-0.5 * float(model_gap), 0.0, 0.0], dtype=np.float32)
    pred_offset = np.array([+0.5 * float(model_gap), 0.0, 0.0], dtype=np.float32)
    ref_full_pos = ref_full_pos + ref_offset[None, None, :]
    pred_full_pos = pred_full_pos + pred_offset[None, None, :]

    ref_selected_overlay_pos = ref_full_pos[:, selected_full_indices]
    ref_selected_overlay_rot = ref_full_rot[:, selected_full_indices]
    pred_selected_overlay_pos = pred_full_pos[:, selected_full_indices]
    pred_selected_overlay_rot = pred_full_rot[:, selected_full_indices]

    shadow_shader = gv.LoadShader(_as_bytes(GENOVIEW_RESOURCES_DIR / "shadow.vs"), _as_bytes(GENOVIEW_RESOURCES_DIR / "shadow.fs"))
    shadow_shader_light_clip_near = gv.GetShaderLocation(shadow_shader, b"lightClipNear")
    shadow_shader_light_clip_far = gv.GetShaderLocation(shadow_shader, b"lightClipFar")

    skinned_shadow_shader = gv.LoadShader(
        _as_bytes(GENOVIEW_RESOURCES_DIR / "skinnedShadow.vs"),
        _as_bytes(GENOVIEW_RESOURCES_DIR / "shadow.fs"),
    )
    skinned_shadow_shader_light_clip_near = gv.GetShaderLocation(skinned_shadow_shader, b"lightClipNear")
    skinned_shadow_shader_light_clip_far = gv.GetShaderLocation(skinned_shadow_shader, b"lightClipFar")

    skinned_basic_shader = gv.LoadShader(
        _as_bytes(GENOVIEW_RESOURCES_DIR / "skinnedBasic.vs"),
        _as_bytes(GENOVIEW_RESOURCES_DIR / "basic.fs"),
    )
    skinned_basic_shader_specularity = gv.GetShaderLocation(skinned_basic_shader, b"specularity")
    skinned_basic_shader_glossiness = gv.GetShaderLocation(skinned_basic_shader, b"glossiness")
    skinned_basic_shader_cam_clip_near = gv.GetShaderLocation(skinned_basic_shader, b"camClipNear")
    skinned_basic_shader_cam_clip_far = gv.GetShaderLocation(skinned_basic_shader, b"camClipFar")

    basic_shader = gv.LoadShader(_as_bytes(GENOVIEW_RESOURCES_DIR / "basic.vs"), _as_bytes(GENOVIEW_RESOURCES_DIR / "basic.fs"))
    basic_shader_specularity = gv.GetShaderLocation(basic_shader, b"specularity")
    basic_shader_glossiness = gv.GetShaderLocation(basic_shader, b"glossiness")
    basic_shader_cam_clip_near = gv.GetShaderLocation(basic_shader, b"camClipNear")
    basic_shader_cam_clip_far = gv.GetShaderLocation(basic_shader, b"camClipFar")

    lighting_shader = gv.LoadShader(_as_bytes(GENOVIEW_RESOURCES_DIR / "post.vs"), _as_bytes(GENOVIEW_RESOURCES_DIR / "lighting.fs"))
    lighting_shader_gbuffer_color = gv.GetShaderLocation(lighting_shader, b"gbufferColor")
    lighting_shader_gbuffer_normal = gv.GetShaderLocation(lighting_shader, b"gbufferNormal")
    lighting_shader_gbuffer_depth = gv.GetShaderLocation(lighting_shader, b"gbufferDepth")
    lighting_shader_ssao = gv.GetShaderLocation(lighting_shader, b"ssao")
    lighting_shader_cam_pos = gv.GetShaderLocation(lighting_shader, b"camPos")
    lighting_shader_cam_inv_view_proj = gv.GetShaderLocation(lighting_shader, b"camInvViewProj")
    lighting_shader_light_dir = gv.GetShaderLocation(lighting_shader, b"lightDir")
    lighting_shader_sun_color = gv.GetShaderLocation(lighting_shader, b"sunColor")
    lighting_shader_sun_strength = gv.GetShaderLocation(lighting_shader, b"sunStrength")
    lighting_shader_sky_color = gv.GetShaderLocation(lighting_shader, b"skyColor")
    lighting_shader_sky_strength = gv.GetShaderLocation(lighting_shader, b"skyStrength")
    lighting_shader_ground_strength = gv.GetShaderLocation(lighting_shader, b"groundStrength")
    lighting_shader_ambient_strength = gv.GetShaderLocation(lighting_shader, b"ambientStrength")
    lighting_shader_exposure = gv.GetShaderLocation(lighting_shader, b"exposure")
    lighting_shader_cam_clip_near = gv.GetShaderLocation(lighting_shader, b"camClipNear")
    lighting_shader_cam_clip_far = gv.GetShaderLocation(lighting_shader, b"camClipFar")

    ssao_shader = gv.LoadShader(_as_bytes(GENOVIEW_RESOURCES_DIR / "post.vs"), _as_bytes(GENOVIEW_RESOURCES_DIR / "ssao.fs"))
    ssao_shader_gbuffer_normal = gv.GetShaderLocation(ssao_shader, b"gbufferNormal")
    ssao_shader_gbuffer_depth = gv.GetShaderLocation(ssao_shader, b"gbufferDepth")
    ssao_shader_cam_view = gv.GetShaderLocation(ssao_shader, b"camView")
    ssao_shader_cam_proj = gv.GetShaderLocation(ssao_shader, b"camProj")
    ssao_shader_cam_inv_proj = gv.GetShaderLocation(ssao_shader, b"camInvProj")
    ssao_shader_cam_inv_view_proj = gv.GetShaderLocation(ssao_shader, b"camInvViewProj")
    ssao_shader_light_view_proj = gv.GetShaderLocation(ssao_shader, b"lightViewProj")
    ssao_shader_shadow_map = gv.GetShaderLocation(ssao_shader, b"shadowMap")
    ssao_shader_shadow_inv_resolution = gv.GetShaderLocation(ssao_shader, b"shadowInvResolution")
    ssao_shader_cam_clip_near = gv.GetShaderLocation(ssao_shader, b"camClipNear")
    ssao_shader_cam_clip_far = gv.GetShaderLocation(ssao_shader, b"camClipFar")
    ssao_shader_light_clip_near = gv.GetShaderLocation(ssao_shader, b"lightClipNear")
    ssao_shader_light_clip_far = gv.GetShaderLocation(ssao_shader, b"lightClipFar")
    ssao_shader_light_dir = gv.GetShaderLocation(ssao_shader, b"lightDir")

    blur_shader = gv.LoadShader(_as_bytes(GENOVIEW_RESOURCES_DIR / "post.vs"), _as_bytes(GENOVIEW_RESOURCES_DIR / "blur.fs"))
    blur_shader_gbuffer_normal = gv.GetShaderLocation(blur_shader, b"gbufferNormal")
    blur_shader_gbuffer_depth = gv.GetShaderLocation(blur_shader, b"gbufferDepth")
    blur_shader_input_texture = gv.GetShaderLocation(blur_shader, b"inputTexture")
    blur_shader_cam_inv_proj = gv.GetShaderLocation(blur_shader, b"camInvProj")
    blur_shader_cam_clip_near = gv.GetShaderLocation(blur_shader, b"camClipNear")
    blur_shader_cam_clip_far = gv.GetShaderLocation(blur_shader, b"camClipFar")
    blur_shader_inv_texture_resolution = gv.GetShaderLocation(blur_shader, b"invTextureResolution")
    blur_shader_blur_direction = gv.GetShaderLocation(blur_shader, b"blurDirection")

    fxaa_shader = gv.LoadShader(_as_bytes(GENOVIEW_RESOURCES_DIR / "post.vs"), _as_bytes(GENOVIEW_RESOURCES_DIR / "fxaa.fs"))
    fxaa_shader_input_texture = gv.GetShaderLocation(fxaa_shader, b"inputTexture")
    fxaa_shader_inv_texture_resolution = gv.GetShaderLocation(fxaa_shader, b"invTextureResolution")

    ground_mesh = gv.GenMeshPlane(40.0, 40.0, 20, 20)
    ground_model = gv.LoadModelFromMesh(ground_mesh)
    ground_position = gv.Vector3(0.0, -0.01, 0.0)
    zero_position = gv.Vector3(0.0, 0.0, 0.0)

    camera = gv.Camera()
    camera.distance = 7.5
    camera.altitude = 0.35

    gv.rlSetClipPlanes(0.01, 50.0)

    light_dir = gv.Vector3Normalize(gv.Vector3(0.35, -1.0, -0.35))
    shadow_light = gv.ShadowLight()
    shadow_light.target = gv.Vector3Zero()
    shadow_light.position = gv.Vector3Scale(light_dir, -6.0)
    shadow_light.up = gv.Vector3(0.0, 1.0, 0.0)
    shadow_light.width = 10.0
    shadow_light.height = 8.0
    shadow_light.near = 0.01
    shadow_light.far = 16.0

    shadow_width = 1024
    shadow_height = 1024
    shadow_inv_resolution = gv.Vector2(1.0 / shadow_width, 1.0 / shadow_height)
    shadow_map = gv.LoadShadowMap(shadow_width, shadow_height)

    gbuffer = gv.LoadGBuffer(int(screen_width), int(screen_height))
    lighted = gv.LoadRenderTexture(int(screen_width), int(screen_height))
    ssao_front = gv.LoadRenderTexture(int(screen_width), int(screen_height))
    ssao_back = gv.LoadRenderTexture(int(screen_width), int(screen_height))

    ref_color = gv.Color(110, 159, 255, 255)
    pred_color = gv.Color(255, 171, 82, 255)
    ref_skeleton_color = gv.Color(70, 105, 180, 255)
    pred_skeleton_color = gv.Color(191, 108, 10, 255)

    animation_frame = 0
    paused = False
    draw_bones = False

    try:
        while not gv.WindowShouldClose():
            if gv.IsKeyPressed(gv.KEY_SPACE):
                paused = not paused
            if gv.IsKeyPressed(gv.KEY_B):
                draw_bones = not draw_bones
            if paused and gv.IsKeyPressed(gv.KEY_RIGHT):
                animation_frame = (animation_frame + 1) % t_total
            if paused and gv.IsKeyPressed(gv.KEY_LEFT):
                animation_frame = (animation_frame - 1) % t_total
            if not paused:
                animation_frame = (animation_frame + 1) % t_total

            ref_frame_pos = ref_full_pos[animation_frame]
            ref_frame_rot = ref_full_rot[animation_frame]
            pred_frame_pos = pred_full_pos[animation_frame]
            pred_frame_rot = pred_full_rot[animation_frame]

            gv.UpdateModelPoseFromNumpyArrays(ref_model, bind_pos, bind_rot, ref_frame_pos, ref_frame_rot)
            gv.UpdateModelPoseFromNumpyArrays(pred_model, bind_pos, bind_rot, pred_frame_pos, pred_frame_rot)

            hip_ref = gv.Vector3(*ref_frame_pos[0])
            hip_pred = gv.Vector3(*pred_frame_pos[0])
            focus_target = gv.Vector3(
                0.5 * (hip_ref.x + hip_pred.x),
                0.75 + 0.5 * (hip_ref.y + hip_pred.y),
                0.5 * (hip_ref.z + hip_pred.z),
            )

            shadow_light.target = gv.Vector3(focus_target.x, 0.0, focus_target.z)
            shadow_light.position = gv.Vector3Add(shadow_light.target, gv.Vector3Scale(light_dir, -6.0))

            mouse_delta = gv.GetMouseDelta()
            camera.update(
                focus_target,
                mouse_delta.x if gv.IsKeyDown(gv.KEY_LEFT_CONTROL) and gv.IsMouseButtonDown(0) else 0.0,
                mouse_delta.y if gv.IsKeyDown(gv.KEY_LEFT_CONTROL) and gv.IsMouseButtonDown(0) else 0.0,
                mouse_delta.x if gv.IsKeyDown(gv.KEY_LEFT_CONTROL) and gv.IsMouseButtonDown(1) else 0.0,
                mouse_delta.y if gv.IsKeyDown(gv.KEY_LEFT_CONTROL) and gv.IsMouseButtonDown(1) else 0.0,
                gv.GetMouseWheelMove(),
                gv.GetFrameTime(),
            )

            gv.rlDisableColorBlend()
            gv.BeginDrawing()

            gv.BeginShadowMap(shadow_map, shadow_light)

            light_view_proj = gv.MatrixMultiply(gv.rlGetMatrixModelview(), gv.rlGetMatrixProjection())
            light_clip_near = gv.rlGetCullDistanceNear()
            light_clip_far = gv.rlGetCullDistanceFar()

            light_clip_near_ptr = gv.ffi.new("float*")
            light_clip_near_ptr[0] = light_clip_near
            light_clip_far_ptr = gv.ffi.new("float*")
            light_clip_far_ptr[0] = light_clip_far

            gv.SetShaderValue(shadow_shader, shadow_shader_light_clip_near, light_clip_near_ptr, gv.SHADER_UNIFORM_FLOAT)
            gv.SetShaderValue(shadow_shader, shadow_shader_light_clip_far, light_clip_far_ptr, gv.SHADER_UNIFORM_FLOAT)
            gv.SetShaderValue(
                skinned_shadow_shader,
                skinned_shadow_shader_light_clip_near,
                light_clip_near_ptr,
                gv.SHADER_UNIFORM_FLOAT,
            )
            gv.SetShaderValue(
                skinned_shadow_shader,
                skinned_shadow_shader_light_clip_far,
                light_clip_far_ptr,
                gv.SHADER_UNIFORM_FLOAT,
            )

            ground_model.materials[0].shader = shadow_shader
            gv.DrawModel(ground_model, ground_position, 1.0, gv.WHITE)

            ref_model.materials[0].shader = skinned_shadow_shader
            pred_model.materials[0].shader = skinned_shadow_shader
            gv.DrawModel(ref_model, zero_position, 1.0, gv.WHITE)
            gv.DrawModel(pred_model, zero_position, 1.0, gv.WHITE)

            gv.EndShadowMap()

            gv.BeginGBuffer(gbuffer, camera.cam3d)

            cam_view = gv.rlGetMatrixModelview()
            cam_proj = gv.rlGetMatrixProjection()
            cam_inv_proj = gv.MatrixInvert(cam_proj)
            cam_inv_view_proj = gv.MatrixInvert(gv.MatrixMultiply(cam_view, cam_proj))
            cam_clip_near = gv.rlGetCullDistanceNear()
            cam_clip_far = gv.rlGetCullDistanceFar()

            cam_clip_near_ptr = gv.ffi.new("float*")
            cam_clip_near_ptr[0] = cam_clip_near
            cam_clip_far_ptr = gv.ffi.new("float*")
            cam_clip_far_ptr[0] = cam_clip_far

            specularity_ptr = gv.ffi.new("float*")
            specularity_ptr[0] = 0.5
            glossiness_ptr = gv.ffi.new("float*")
            glossiness_ptr[0] = 10.0

            gv.SetShaderValue(basic_shader, basic_shader_specularity, specularity_ptr, gv.SHADER_UNIFORM_FLOAT)
            gv.SetShaderValue(basic_shader, basic_shader_glossiness, glossiness_ptr, gv.SHADER_UNIFORM_FLOAT)
            gv.SetShaderValue(basic_shader, basic_shader_cam_clip_near, cam_clip_near_ptr, gv.SHADER_UNIFORM_FLOAT)
            gv.SetShaderValue(basic_shader, basic_shader_cam_clip_far, cam_clip_far_ptr, gv.SHADER_UNIFORM_FLOAT)

            gv.SetShaderValue(
                skinned_basic_shader,
                skinned_basic_shader_specularity,
                specularity_ptr,
                gv.SHADER_UNIFORM_FLOAT,
            )
            gv.SetShaderValue(
                skinned_basic_shader,
                skinned_basic_shader_glossiness,
                glossiness_ptr,
                gv.SHADER_UNIFORM_FLOAT,
            )
            gv.SetShaderValue(
                skinned_basic_shader,
                skinned_basic_shader_cam_clip_near,
                cam_clip_near_ptr,
                gv.SHADER_UNIFORM_FLOAT,
            )
            gv.SetShaderValue(
                skinned_basic_shader,
                skinned_basic_shader_cam_clip_far,
                cam_clip_far_ptr,
                gv.SHADER_UNIFORM_FLOAT,
            )

            ground_model.materials[0].shader = basic_shader
            gv.DrawModel(ground_model, ground_position, 1.0, gv.Color(190, 190, 190, 255))

            ref_model.materials[0].shader = skinned_basic_shader
            pred_model.materials[0].shader = skinned_basic_shader
            gv.DrawModel(ref_model, zero_position, 1.0, ref_color)
            gv.DrawModel(pred_model, zero_position, 1.0, pred_color)

            gv.EndGBuffer(int(screen_width), int(screen_height))

            gv.BeginTextureMode(ssao_front)
            gv.BeginShaderMode(ssao_shader)

            gv.SetShaderValueTexture(ssao_shader, ssao_shader_gbuffer_normal, gbuffer.normal)
            gv.SetShaderValueTexture(ssao_shader, ssao_shader_gbuffer_depth, gbuffer.depth)
            gv.SetShaderValueMatrix(ssao_shader, ssao_shader_cam_view, cam_view)
            gv.SetShaderValueMatrix(ssao_shader, ssao_shader_cam_proj, cam_proj)
            gv.SetShaderValueMatrix(ssao_shader, ssao_shader_cam_inv_proj, cam_inv_proj)
            gv.SetShaderValueMatrix(ssao_shader, ssao_shader_cam_inv_view_proj, cam_inv_view_proj)
            gv.SetShaderValueMatrix(ssao_shader, ssao_shader_light_view_proj, light_view_proj)
            gv.SetShaderValueShadowMap(ssao_shader, ssao_shader_shadow_map, shadow_map)
            gv.SetShaderValue(
                ssao_shader,
                ssao_shader_shadow_inv_resolution,
                gv.ffi.addressof(shadow_inv_resolution),
                gv.SHADER_UNIFORM_VEC2,
            )
            gv.SetShaderValue(ssao_shader, ssao_shader_cam_clip_near, cam_clip_near_ptr, gv.SHADER_UNIFORM_FLOAT)
            gv.SetShaderValue(ssao_shader, ssao_shader_cam_clip_far, cam_clip_far_ptr, gv.SHADER_UNIFORM_FLOAT)
            gv.SetShaderValue(ssao_shader, ssao_shader_light_clip_near, light_clip_near_ptr, gv.SHADER_UNIFORM_FLOAT)
            gv.SetShaderValue(ssao_shader, ssao_shader_light_clip_far, light_clip_far_ptr, gv.SHADER_UNIFORM_FLOAT)
            gv.SetShaderValue(ssao_shader, ssao_shader_light_dir, gv.ffi.addressof(light_dir), gv.SHADER_UNIFORM_VEC3)

            gv.ClearBackground(gv.WHITE)
            gv.DrawTextureRec(
                ssao_front.texture,
                gv.Rectangle(0, 0, ssao_front.texture.width, -ssao_front.texture.height),
                gv.Vector2(0.0, 0.0),
                gv.WHITE,
            )
            gv.EndShaderMode()
            gv.EndTextureMode()

            gv.BeginTextureMode(ssao_back)
            gv.BeginShaderMode(blur_shader)

            blur_direction = gv.Vector2(1.0, 0.0)
            blur_inv_texture_resolution = gv.Vector2(1.0 / ssao_front.texture.width, 1.0 / ssao_front.texture.height)

            gv.SetShaderValueTexture(blur_shader, blur_shader_gbuffer_normal, gbuffer.normal)
            gv.SetShaderValueTexture(blur_shader, blur_shader_gbuffer_depth, gbuffer.depth)
            gv.SetShaderValueTexture(blur_shader, blur_shader_input_texture, ssao_front.texture)
            gv.SetShaderValueMatrix(blur_shader, blur_shader_cam_inv_proj, cam_inv_proj)
            gv.SetShaderValue(blur_shader, blur_shader_cam_clip_near, cam_clip_near_ptr, gv.SHADER_UNIFORM_FLOAT)
            gv.SetShaderValue(blur_shader, blur_shader_cam_clip_far, cam_clip_far_ptr, gv.SHADER_UNIFORM_FLOAT)
            gv.SetShaderValue(
                blur_shader,
                blur_shader_inv_texture_resolution,
                gv.ffi.addressof(blur_inv_texture_resolution),
                gv.SHADER_UNIFORM_VEC2,
            )
            gv.SetShaderValue(blur_shader, blur_shader_blur_direction, gv.ffi.addressof(blur_direction), gv.SHADER_UNIFORM_VEC2)

            gv.DrawTextureRec(
                ssao_back.texture,
                gv.Rectangle(0, 0, ssao_back.texture.width, -ssao_back.texture.height),
                gv.Vector2(0, 0),
                gv.WHITE,
            )
            gv.EndShaderMode()
            gv.EndTextureMode()

            gv.BeginTextureMode(ssao_front)
            gv.BeginShaderMode(blur_shader)

            blur_direction = gv.Vector2(0.0, 1.0)
            gv.SetShaderValueTexture(blur_shader, blur_shader_input_texture, ssao_back.texture)
            gv.SetShaderValue(blur_shader, blur_shader_blur_direction, gv.ffi.addressof(blur_direction), gv.SHADER_UNIFORM_VEC2)

            gv.DrawTextureRec(
                ssao_front.texture,
                gv.Rectangle(0, 0, ssao_front.texture.width, -ssao_front.texture.height),
                gv.Vector2(0, 0),
                gv.WHITE,
            )
            gv.EndShaderMode()
            gv.EndTextureMode()

            gv.BeginTextureMode(lighted)
            gv.BeginShaderMode(lighting_shader)

            sun_color = gv.Vector3(253.0 / 255.0, 255.0 / 255.0, 232.0 / 255.0)
            sun_strength_ptr = gv.ffi.new("float*")
            sun_strength_ptr[0] = 0.25
            sky_color = gv.Vector3(174.0 / 255.0, 183.0 / 255.0, 190.0 / 255.0)
            sky_strength_ptr = gv.ffi.new("float*")
            sky_strength_ptr[0] = 0.15
            ground_strength_ptr = gv.ffi.new("float*")
            ground_strength_ptr[0] = 0.1
            ambient_strength_ptr = gv.ffi.new("float*")
            ambient_strength_ptr[0] = 1.0
            exposure_ptr = gv.ffi.new("float*")
            exposure_ptr[0] = 0.9

            gv.SetShaderValueTexture(lighting_shader, lighting_shader_gbuffer_color, gbuffer.color)
            gv.SetShaderValueTexture(lighting_shader, lighting_shader_gbuffer_normal, gbuffer.normal)
            gv.SetShaderValueTexture(lighting_shader, lighting_shader_gbuffer_depth, gbuffer.depth)
            gv.SetShaderValueTexture(lighting_shader, lighting_shader_ssao, ssao_front.texture)
            gv.SetShaderValue(lighting_shader, lighting_shader_cam_pos, gv.ffi.addressof(camera.cam3d.position), gv.SHADER_UNIFORM_VEC3)
            gv.SetShaderValueMatrix(lighting_shader, lighting_shader_cam_inv_view_proj, cam_inv_view_proj)
            gv.SetShaderValue(lighting_shader, lighting_shader_light_dir, gv.ffi.addressof(light_dir), gv.SHADER_UNIFORM_VEC3)
            gv.SetShaderValue(lighting_shader, lighting_shader_sun_color, gv.ffi.addressof(sun_color), gv.SHADER_UNIFORM_VEC3)
            gv.SetShaderValue(lighting_shader, lighting_shader_sun_strength, sun_strength_ptr, gv.SHADER_UNIFORM_FLOAT)
            gv.SetShaderValue(lighting_shader, lighting_shader_sky_color, gv.ffi.addressof(sky_color), gv.SHADER_UNIFORM_VEC3)
            gv.SetShaderValue(lighting_shader, lighting_shader_sky_strength, sky_strength_ptr, gv.SHADER_UNIFORM_FLOAT)
            gv.SetShaderValue(lighting_shader, lighting_shader_ground_strength, ground_strength_ptr, gv.SHADER_UNIFORM_FLOAT)
            gv.SetShaderValue(lighting_shader, lighting_shader_ambient_strength, ambient_strength_ptr, gv.SHADER_UNIFORM_FLOAT)
            gv.SetShaderValue(lighting_shader, lighting_shader_exposure, exposure_ptr, gv.SHADER_UNIFORM_FLOAT)
            gv.SetShaderValue(lighting_shader, lighting_shader_cam_clip_near, cam_clip_near_ptr, gv.SHADER_UNIFORM_FLOAT)
            gv.SetShaderValue(lighting_shader, lighting_shader_cam_clip_far, cam_clip_far_ptr, gv.SHADER_UNIFORM_FLOAT)

            gv.ClearBackground(gv.RAYWHITE)
            gv.DrawTextureRec(
                gbuffer.color,
                gv.Rectangle(0, 0, gbuffer.color.width, -gbuffer.color.height),
                gv.Vector2(0, 0),
                gv.WHITE,
            )
            gv.EndShaderMode()

            if draw_bones:
                gv.BeginMode3D(camera.cam3d)
                gv.DrawSkeleton(
                    ref_selected_overlay_pos[animation_frame],
                    ref_selected_overlay_rot[animation_frame],
                    selected_reduced_parents,
                    ref_skeleton_color,
                )
                gv.DrawSkeleton(
                    pred_selected_overlay_pos[animation_frame],
                    pred_selected_overlay_rot[animation_frame],
                    selected_reduced_parents,
                    pred_skeleton_color,
                )
                gv.EndMode3D()

            gv.EndTextureMode()

            gv.BeginShaderMode(fxaa_shader)
            fxaa_inv_texture_resolution = gv.Vector2(1.0 / lighted.texture.width, 1.0 / lighted.texture.height)
            gv.SetShaderValueTexture(fxaa_shader, fxaa_shader_input_texture, lighted.texture)
            gv.SetShaderValue(
                fxaa_shader,
                fxaa_shader_inv_texture_resolution,
                gv.ffi.addressof(fxaa_inv_texture_resolution),
                gv.SHADER_UNIFORM_VEC2,
            )
            gv.DrawTextureRec(
                lighted.texture,
                gv.Rectangle(0, 0, lighted.texture.width, -lighted.texture.height),
                gv.Vector2(0, 0),
                gv.WHITE,
            )
            gv.EndShaderMode()

            gv.rlEnableColorBlend()

            gv.DrawText(
                b"Ctrl+Left Rotate | Ctrl+Right Pan | Wheel Zoom | Space Pause | B Bones",
                20,
                18,
                20,
                gv.DARKGRAY,
            )
            gv.DrawText(f"Frame {animation_frame + 1}/{t_total}".encode("utf-8"), 20, 46, 20, gv.BLACK)
            gv.DrawText(b"Reference", 20, 78, 22, ref_color)
            gv.DrawText(b"Prediction", int(screen_width) - 150, 78, 22, pred_color)
            if paused:
                gv.DrawText(b"Paused", 20, 106, 20, gv.MAROON)
            if title:
                gv.DrawText(title.encode("utf-8")[: min(len(title.encode("utf-8")), 180)], 20, int(screen_height) - 30, 18, gv.GRAY)

            gv.EndDrawing()
    finally:
        if lighted is not None:
            gv.UnloadRenderTexture(lighted)
        if ssao_back is not None:
            gv.UnloadRenderTexture(ssao_back)
        if ssao_front is not None:
            gv.UnloadRenderTexture(ssao_front)
        if gbuffer is not None:
            gv.UnloadGBuffer(gbuffer)
        if shadow_map is not None:
            gv.UnloadShadowMap(shadow_map)

        if ref_model is not None:
            gv.UnloadModel(ref_model)
        if pred_model is not None:
            gv.UnloadModel(pred_model)
        if ground_model is not None:
            gv.UnloadModel(ground_model)

        if fxaa_shader is not None:
            gv.UnloadShader(fxaa_shader)
        if blur_shader is not None:
            gv.UnloadShader(blur_shader)
        if ssao_shader is not None:
            gv.UnloadShader(ssao_shader)
        if lighting_shader is not None:
            gv.UnloadShader(lighting_shader)
        if basic_shader is not None:
            gv.UnloadShader(basic_shader)
        if skinned_basic_shader is not None:
            gv.UnloadShader(skinned_basic_shader)
        if skinned_shadow_shader is not None:
            gv.UnloadShader(skinned_shadow_shader)
        if shadow_shader is not None:
            gv.UnloadShader(shadow_shader)

        if gv.IsWindowReady():
            gv.CloseWindow()
