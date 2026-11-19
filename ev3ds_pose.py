"""Read the frame-specific Blender camera from EV3DS, matching the EvGGS loader."""
from functools import lru_cache
import numpy as np

OPENGL_TO_OPENCV = np.diag([1.,-1.,-1.,1.])


@lru_cache(maxsize=8192)
def camera(root,scene,frame):
    with np.load(root/"Scenes"/scene/f"{frame:04}.npz") as archive:
        objects = archive["object_poses"]
        poses = objects["pose"][objects["name"]=="Camera"]
    if poses.shape != (1,4,4):
        raise ValueError(f"Expected one camera in {scene}/{frame:04}")
    return poses[0].astype(np.float64) @ OPENGL_TO_OPENCV
