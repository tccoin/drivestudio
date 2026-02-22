# Camera pose manipulation and trajectory generation.
import os
import torch
import numpy as np
from typing import Dict, List

from scipy.spatial.transform import Slerp
from scipy.spatial.transform import Rotation as R


def elevate_and_tilt_trajectory(
    poses: torch.Tensor,
    elevation_m: float = 2.0,
    tilt_deg: float = -15.0,
    world_up_axis: int = 2,
) -> torch.Tensor:
    """
    Transform camera-to-world poses: elevate by elevation_m along world up, then tilt down by |tilt_deg|.

    Args:
        poses: (N, 4, 4) camera-to-world matrices.
        elevation_m: meters to add along world up (default 2m).
        tilt_deg: degrees to rotate camera view (negative = look down, default -15°).
        world_up_axis: 0=X, 1=Y, 2=Z (default 2 for Z-up).

    Returns:
        (N, 4, 4) transformed camera-to-world poses.
    """
    out = poses.clone()
    # Elevate: translate along world up
    up = torch.zeros(3, device=poses.device, dtype=poses.dtype)
    up[world_up_axis] = elevation_m
    out[:, :3, 3] = poses[:, :3, 3] + up
    # Tilt: in camera space, rotate around camera X (right); negative tilt_deg = look down
    deg = np.deg2rad(tilt_deg)
    c, s = np.cos(deg), np.sin(deg)
    Rx = torch.tensor([[1, 0, 0], [0, c, -s], [0, s, c]], device=poses.device, dtype=poses.dtype)
    out[:, :3, :3] = poses[:, :3, :3] @ Rx
    return out

def interpolate_poses(key_poses: torch.Tensor, target_frames: int) -> torch.Tensor:
    """
    Interpolate between key poses to generate a smooth trajectory.
    
    Args:
        key_poses (torch.Tensor): Tensor of shape (N, 4, 4) containing key camera poses.
        target_frames (int): Number of frames to interpolate.
    
    Returns:
        torch.Tensor: Interpolated poses of shape (target_frames, 4, 4).
    """
    device = key_poses.device
    key_poses = key_poses.cpu().numpy()
    
    # Separate translation and rotation
    translations = key_poses[:, :3, 3]
    rotations = key_poses[:, :3, :3]
    
    # Create time array
    times = np.linspace(0, 1, len(key_poses))
    target_times = np.linspace(0, 1, target_frames)
    
    # Interpolate translations
    interp_translations = np.stack([
        np.interp(target_times, times, translations[:, i])
        for i in range(3)
    ], axis=-1)
    
    # Interpolate rotations using Slerp
    key_rots = R.from_matrix(rotations)
    slerp = Slerp(times, key_rots)
    interp_rotations = slerp(target_times).as_matrix()
    
    # Combine interpolated translations and rotations
    interp_poses = np.eye(4)[None].repeat(target_frames, axis=0)
    interp_poses[:, :3, :3] = interp_rotations
    interp_poses[:, :3, 3] = interp_translations
    
    return torch.tensor(interp_poses, dtype=torch.float32, device=device)

def look_at_rotation(direction: torch.Tensor, up: torch.Tensor = torch.tensor([0., 0., 1.])) -> torch.Tensor:
    """Calculate rotation matrix to look at a specific direction."""
    front = torch.nn.functional.normalize(direction, dim=-1)
    right = torch.nn.functional.normalize(torch.cross(front, up), dim=-1)
    up = torch.cross(right, front)
    rotation_matrix = torch.stack([right, up, -front], dim=-1)
    return rotation_matrix

def elevated_2m_tilt15(
    dataset_type: str,
    per_cam_poses: Dict[int, torch.Tensor],
    original_frames: int,
    target_frames: int,
    elevation_m: float = 2.0,
    tilt_deg: float = -15.0,
) -> torch.Tensor:
    """Same as original front-camera trajectory but elevated by 2m and tilted down 15° (tilt_deg=-15)."""
    assert 0 in per_cam_poses.keys(), "Front center camera (ID 0) is required for elevated_2m_tilt15"
    traj = per_cam_poses[0]  # (original_frames, 4, 4)
    out = elevate_and_tilt_trajectory(traj, elevation_m=elevation_m, tilt_deg=tilt_deg)
    if target_frames != original_frames:
        out = interpolate_poses(out, target_frames)
    return out


def get_elevated_tilt_trajectories_multi_cam(
    per_cam_poses: Dict[int, torch.Tensor],
    elevation_m: float = 2.0,
    tilt_deg: float = -15.0,
) -> List[torch.Tensor]:
    """
    Apply elevate + tilt to each camera's trajectory. Returns one (T, 4, 4) tensor per camera,
    in ascending cam_id order (e.g. [traj_cam0, traj_cam1, traj_cam2]).
    """
    cam_ids = sorted(per_cam_poses.keys())
    return [
        elevate_and_tilt_trajectory(per_cam_poses[cid], elevation_m=elevation_m, tilt_deg=tilt_deg)
        for cid in cam_ids
    ]


def get_interp_novel_trajectories(
    dataset_type: str,
    scene_idx: str,
    per_cam_poses: Dict[int, torch.Tensor],
    traj_type: str = "front_center_interp",
    target_frames: int = 100
) -> torch.Tensor:
    original_frames = per_cam_poses[list(per_cam_poses.keys())[0]].shape[0]
    
    trajectory_generators = {
        "front_center_interp": front_center_interp,
        "s_curve": s_curve,
        "three_key_poses": three_key_poses_trajectory,
        "elevated_2m_tilt15": elevated_2m_tilt15,
    }
    
    if traj_type not in trajectory_generators:
        raise ValueError(f"Unknown trajectory type: {traj_type}")
    
    return trajectory_generators[traj_type](dataset_type, per_cam_poses, original_frames, target_frames)

def front_center_interp(
    dataset_type: str, per_cam_poses: Dict[int, torch.Tensor], original_frames: int, target_frames: int, num_loops: int = 1
) -> torch.Tensor:
    """Interpolate key frames from the front center camera."""
    assert 0 in per_cam_poses.keys(), "Front center camera (ID 0) is required for front_center_interp"
    key_poses = per_cam_poses[0][::original_frames//4]  # Select every 4th frame as key frame
    return interpolate_poses(key_poses, target_frames)

def s_curve(
    dataset_type: str, per_cam_poses: Dict[int, torch.Tensor], original_frames: int, target_frames: int
) -> torch.Tensor:
    """Create an S-shaped trajectory using the front three cameras."""
    assert all(cam in per_cam_poses.keys() for cam in [0, 1, 2]), "Front three cameras (IDs 0, 1, 2) are required for s_curve"
    key_poses = torch.cat([
        per_cam_poses[0][0:1],
        per_cam_poses[1][original_frames//4:original_frames//4+1],
        per_cam_poses[0][original_frames//2:original_frames//2+1],
        per_cam_poses[2][3*original_frames//4:3*original_frames//4+1],
        per_cam_poses[0][-1:]
    ], dim=0)
    return interpolate_poses(key_poses, target_frames)

def three_key_poses_trajectory(
    dataset_type: str,
    per_cam_poses: Dict[int, torch.Tensor],
    original_frames: int,
    target_frames: int
) -> torch.Tensor:
    """
    Create a trajectory using three key poses:
    1. First frame of front center camera
    2. Middle frame with interpolated rotation and position from camera 1 or 2
    3. Last frame of front center camera

    The rotation of the middle pose is calculated using Slerp between
    the start frame and the middle frame of camera 1 or 2.

    Args:
        dataset_type (str): Type of the dataset (e.g., "waymo", "pandaset", etc.).
        per_cam_poses (Dict[int, torch.Tensor]): Dictionary of camera poses.
        original_frames (int): Number of original frames.
        target_frames (int): Number of frames in the output trajectory.

    Returns:
        torch.Tensor: Trajectory of shape (target_frames, 4, 4).
    """
    assert 0 in per_cam_poses.keys(), "Front center camera (ID 0) is required"
    assert 1 in per_cam_poses.keys() or 2 in per_cam_poses.keys(), "Either camera 1 or camera 2 is required"

    # First key pose: First frame of front center camera
    start_pose = per_cam_poses[0][0]
    key_poses = [start_pose]

    # Select camera for middle frame
    middle_frame = int(original_frames // 2)
    chosen_cam = np.random.choice([1, 2])

    middle_pose = per_cam_poses[chosen_cam][middle_frame]

    # Calculate interpolated rotation for middle pose
    start_rotation = R.from_matrix(start_pose[:3, :3].cpu().numpy())
    middle_rotation = R.from_matrix(middle_pose[:3, :3].cpu().numpy())
    slerp = Slerp([0, 1], R.from_quat([start_rotation.as_quat(), middle_rotation.as_quat()]))
    interpolated_rotation = slerp(0.5).as_matrix()

    # Create middle key pose with interpolated rotation and original translation
    middle_key_pose = torch.eye(4, device=start_pose.device)
    middle_key_pose[:3, :3] = torch.tensor(interpolated_rotation, device=start_pose.device)
    middle_key_pose[:3, 3] = middle_pose[:3, 3]  # Keep the original translation
    key_poses.append(middle_key_pose)

    # Third key pose: Last frame of front center camera
    key_poses.append(per_cam_poses[0][-1])

    # Stack the key poses and interpolate
    key_poses = torch.stack(key_poses)
    return interpolate_poses(key_poses, target_frames)