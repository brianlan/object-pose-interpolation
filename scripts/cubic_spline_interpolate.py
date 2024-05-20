import argparse
from typing import List, Dict, Tuple
from pathlib import Path

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize
from loguru import logger

from copious.io.fs import read_json


def main(args):
    obj = read_object(args.label_frames_dir, 1)
    kf_ts = sorted(range(1692759644664, 1692759677564 + 1, 2000))
    all_ts = sorted(range(1692759644664, 1692759677564 + 1, 100))
    kf_pos = np.array(
        [
            [obj[ts]["psr"]["position"]["x"], obj[ts]["psr"]["position"]["y"], obj[ts]["psr"]["position"]["z"]]
            for ts in kf_ts
        ]
    )

    # orientations = np.array([0, np.pi/4, np.pi/2, np.pi/3])
    # velocities = np.array([1, 1, 1, 1])
    # steering_angles = np.array([0, 0.1, 0.2, 0.1])
    # L = 2.5  # 车辆轴距

    # 初始样条插值
    cs_x = CubicSpline(kf_ts, kf_pos[:, 0])
    cs_y = CubicSpline(kf_ts, kf_pos[:, 1])
    cs_z = CubicSpline(kf_ts, kf_pos[:, 2])

    interp_x = cs_x(all_ts)
    interp_y = cs_y(all_ts)
    interp_z = cs_z(all_ts)


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(interp_x, interp_y, interp_z, c=interp_z, cmap='viridis')

    # Set labels and title
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.title('3D Scatter Plot')

    a = 100


def read_object(label_dir: Path, obj_track_id: int) -> Dict[int, Dict]:
    obj = {}
    for p in sorted(label_dir.glob("*.json")):
        objects = read_json(p)
        for o in objects:
            if int(o["obj_track_id"]) == obj_track_id:
                o["ts"] = int(p.stem)
                obj[o["ts"]] = o
                break
    return obj


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--label-frames-dir", type=Path, required=True)
    main(parser.parse_args())
