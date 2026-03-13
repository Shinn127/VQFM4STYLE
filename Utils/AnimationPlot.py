import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def _prepare_motion(animations):
    """Normalize floor height and center x/z by root per frame."""
    motion = np.asarray(animations, dtype=np.float32).copy()
    if motion.ndim != 3 or motion.shape[-1] != 3:
        raise ValueError(f"Expected animations shape (T, J, 3), got {motion.shape}")

    # Keep a copy of root trajectory (before centering) for trail rendering.
    root_traj = motion[:, 0, [0, 2]].copy()

    # Move floor to z=0 (visualized vertical axis uses y component from motion).
    floor = float(motion[..., 1].min())
    motion[..., 1] -= floor

    # Root-centered camera space in the horizontal plane.
    motion[..., 0] -= motion[:, 0:1, 0]
    motion[..., 2] -= motion[:, 0:1, 2]
    return motion, root_traj


def _limits_from_motion(motion, pad=0.15):
    x = motion[..., 0]
    y = motion[..., 2]
    z = motion[..., 1]

    radius = max(np.abs(x).max(), np.abs(y).max(), 1.0)
    radius *= (1.0 + pad)
    z_max = max(float(z.max()) * (1.0 + pad), 1.0)
    return (-radius, radius), (-radius, radius), (0.0, z_max)


def _style_axes(ax, xlim, ylim, zlim):
    ax.set_xlim3d(*xlim)
    ax.set_ylim3d(*ylim)
    ax.set_zlim3d(*zlim)
    ax.view_init(elev=20, azim=45)
    ax.grid(False)
    ax.set_axis_off()

    try:
        ax.set_box_aspect((1.0, 1.0, 0.85))
    except Exception:
        pass


def _add_ground_plane(ax, xlim, ylim, z=0.0):
    verts = [[
        [xlim[0], ylim[0], z],
        [xlim[0], ylim[1], z],
        [xlim[1], ylim[1], z],
        [xlim[1], ylim[0], z],
    ]]
    plane = Poly3DCollection(verts, alpha=0.25, facecolor=(0.75, 0.78, 0.84), edgecolor="none")
    ax.add_collection3d(plane)


def _bone_indices(parents):
    return [j for j, p in enumerate(parents) if p != -1]


def _bone_colors(n):
    cmap = plt.get_cmap("tab20")
    return [cmap(i % 20) for i in range(n)]


def animation_plot(animations, parents, interval, save_path=None, title=None, trail_len=90):
    parents = np.asarray(parents, dtype=np.int32)
    motion, root_traj = _prepare_motion(animations)
    if motion.shape[1] != len(parents):
        raise ValueError(f"parents length {len(parents)} does not match joints {motion.shape[1]}")

    xlim, ylim, zlim = _limits_from_motion(motion)
    fig = plt.figure(figsize=(7, 7), dpi=120)
    ax = fig.add_subplot(111, projection="3d")
    _style_axes(ax, xlim, ylim, zlim)
    _add_ground_plane(ax, xlim, ylim, z=0.0)
    if title is not None:
        fig.suptitle(title, fontsize=12)

    bones = _bone_indices(parents)
    colors = _bone_colors(len(bones))
    lines = []
    for k, j in enumerate(bones):
        p = parents[j]
        lw = 2.8 if p == 0 else 2.0
        (line,) = ax.plot([], [], [], color=colors[k], lw=lw, solid_capstyle="round")
        lines.append((j, line))

    (traj_line,) = ax.plot([], [], [], color="#2E6FBB", lw=1.6, alpha=0.9)

    def init():
        for _, line in lines:
            line.set_data_3d([], [], [])
        traj_line.set_data_3d([], [], [])
        return [line for _, line in lines] + [traj_line]

    def animate(i):
        for j, line in lines:
            p = parents[j]
            line.set_data_3d(
                [motion[i, j, 0], motion[i, p, 0]],
                [motion[i, j, 2], motion[i, p, 2]],
                [motion[i, j, 1], motion[i, p, 1]],
            )

        if i > 1:
            start = max(0, i - trail_len)
            trail = root_traj[start:i + 1] - root_traj[i:i + 1]
            traj_line.set_data_3d(trail[:, 0], trail[:, 1], np.zeros(trail.shape[0], dtype=np.float32))
        else:
            traj_line.set_data_3d([], [], [])

        return [line for _, line in lines] + [traj_line]

    ani = animation.FuncAnimation(
        fig=fig,
        func=animate,
        init_func=init,
        frames=motion.shape[0],
        interval=interval,
        blit=True,
    )

    if save_path is not None:
        fps = max(1, int(round(1000.0 / max(interval, 1))))
        ani.save(save_path, writer="pillow", fps=fps)
        plt.close(fig)
    else:
        plt.show()

    return ani


def animation_plot_frame(animations, parents, save_path=None, frame_idx=0, title=None):
    parents = np.asarray(parents, dtype=np.int32)
    motion, root_traj = _prepare_motion(animations)
    if motion.shape[1] != len(parents):
        raise ValueError(f"parents length {len(parents)} does not match joints {motion.shape[1]}")

    frame_idx = int(np.clip(frame_idx, 0, motion.shape[0] - 1))
    xlim, ylim, zlim = _limits_from_motion(motion)

    fig = plt.figure(figsize=(7, 7), dpi=140)
    ax = fig.add_subplot(111, projection="3d")
    _style_axes(ax, xlim, ylim, zlim)
    _add_ground_plane(ax, xlim, ylim, z=0.0)
    if title is not None:
        fig.suptitle(title, fontsize=12)

    bones = _bone_indices(parents)
    colors = _bone_colors(len(bones))

    for k, j in enumerate(bones):
        p = parents[j]
        lw = 3.0 if p == 0 else 2.2
        ax.plot(
            [motion[frame_idx, j, 0], motion[frame_idx, p, 0]],
            [motion[frame_idx, j, 2], motion[frame_idx, p, 2]],
            [motion[frame_idx, j, 1], motion[frame_idx, p, 1]],
            color=colors[k],
            lw=lw,
            solid_capstyle="round",
        )

    # Light trajectory context on floor.
    if frame_idx > 1:
        trail = root_traj[: frame_idx + 1] - root_traj[frame_idx : frame_idx + 1]
        ax.plot(trail[:, 0], trail[:, 1], np.zeros(trail.shape[0], dtype=np.float32), color="#2E6FBB", lw=1.5)

    # Subtle joint markers improve depth perception.
    ax.scatter(
        motion[frame_idx, :, 0],
        motion[frame_idx, :, 2],
        motion[frame_idx, :, 1],
        s=9,
        color="#1f1f1f",
        alpha=0.85,
    )

    if save_path is not None:
        plt.savefig(save_path, dpi=140, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
    else:
        plt.show()


def _to_numpy(array_like):
    """Convert numpy/torch-like tensors to float32 numpy arrays."""
    if hasattr(array_like, "detach"):
        array_like = array_like.detach()
    if hasattr(array_like, "cpu"):
        array_like = array_like.cpu()
    return np.asarray(array_like, dtype=np.float32)


def _decode_375_state(features_375):
    """
    Decode 375-dim traj+motion features from generate_database.py layout.

    Layout per frame:
      [0:84]   trajectory context (unused here)
      [84:87]  root velocity (vx, wyaw, vz)
      [87:159] local positions (24 * 3)
      [159:231] local rot x-axis proxy (24 * 3)
      [231:303] local rot y-axis proxy (24 * 3)
      [303:375] local velocities (24 * 3)
    """
    x = _to_numpy(features_375)
    if x.ndim == 1:
        x = x[None, :]
    if x.ndim != 2 or x.shape[1] != 375:
        raise ValueError(f"Expected shape (T, 375) or (375,), got {x.shape}")

    state = x[:, 84:]  # (T, 291)
    root_vel = state[:, :3]  # (T, 3)

    n_joints = 24
    lp_start = 3
    lp_end = lp_start + n_joints * 3
    local_pos = state[:, lp_start:lp_end].reshape(-1, n_joints, 3)
    return root_vel, local_pos


def _reconstruct_global_positions_from_375(features_375, dt=1.0 / 60.0, root_wy_in_degrees=True):
    """
    Reconstruct global joint positions from 375 features.

    Uses the same root integration idea as eval_4.py:
    - integrate root x/z displacement from velocity * dt
    - integrate yaw from wyaw * dt
    - rotate local positions by accumulated root rotation and add root translation

    Note:
    - In this project pipeline, wy is typically stored in degrees/second.
      Set root_wy_in_degrees=True (default) to convert back to radians.
    """
    try:
        from . import quat as quat_ops
    except Exception:
        from Utils import quat as quat_ops

    root_vel, local_pos = _decode_375_state(features_375)
    T = local_pos.shape[0]

    root_motion = root_vel * float(dt)
    trans_offset = root_motion.copy()
    trans_offset[:, 1] = 0.0  # keep floor vertical offset from local joints

    yaw_delta = root_motion[:, 1]
    if root_wy_in_degrees:
        yaw_delta = np.radians(yaw_delta)
    yaw_offset = quat_ops.from_angle_axis(yaw_delta, np.array([0.0, 1.0, 0.0], dtype=np.float32))

    root_pos = np.zeros((3,), dtype=np.float32)
    root_rot = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    root_pos_seq = np.zeros((T, 3), dtype=np.float32)
    root_rot_seq = np.zeros((T, 4), dtype=np.float32)

    for i in range(T):
        root_pos += trans_offset[i]
        root_pos_seq[i] = root_pos
        root_rot = quat_ops.mul(yaw_offset[i : i + 1], root_rot[None, :])[0]
        root_rot_seq[i] = root_rot

    global_pos = quat_ops.mul_vec(root_rot_seq[:, None, :], local_pos) + root_pos_seq[:, None, :]
    return global_pos


def animation_plot_from_375(
    features_375,
    parents,
    interval=33,
    save_path=None,
    title=None,
    dt=1.0 / 60.0,
    root_wy_in_degrees=True,
    use_global=True,
    frame_idx=0,
    single_frame=False,
    trail_len=90,
):
    """
    Visualize 375-dim traj+motion features.

    Args:
      features_375: (T, 375) or (375,)
      parents: parent list for the 24 selected joints
      interval: ms between frames for animation mode
      save_path: optional output path (.gif recommended for animation)
      title: optional figure title
      dt: frame delta time used for root integration
      root_wy_in_degrees: whether wy is stored in degrees/second (default True)
      use_global: True reconstruct global positions, False use local positions directly
      frame_idx: frame index when single_frame=True
      single_frame: render one frame instead of full animation
      trail_len: root trail length for animation mode
    """
    parents = np.asarray(parents, dtype=np.int32)
    root_vel, local_pos = _decode_375_state(features_375)
    if local_pos.shape[1] != len(parents):
        raise ValueError(
            f"parents length {len(parents)} does not match joints decoded from 375 ({local_pos.shape[1]})."
        )

    motion = (
        _reconstruct_global_positions_from_375(
            features_375,
            dt=dt,
            root_wy_in_degrees=root_wy_in_degrees,
        )
        if use_global
        else local_pos
    )

    if single_frame:
        return animation_plot_frame(
            motion,
            parents,
            save_path=save_path,
            frame_idx=frame_idx,
            title=title,
        )

    return animation_plot(
        motion,
        parents,
        interval=interval,
        save_path=save_path,
        title=title,
        trail_len=trail_len,
    )
