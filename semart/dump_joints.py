"""Recreate <model>/train/ours_<it>/pred_joints.pkl from a trained model (cfg_args + deform weights).

For runs rendered before render.py wrote pred_joints.pkl. Needs only deform/iteration_<it>/deform.pth.
    python -m semart.dump_joints --model_path runs/<RUN>/dta/sapien/storage_47254/sem_k_s0 --iteration 2000
"""
import argparse
import os
import pickle
from argparse import Namespace

import torch

from scene.artgs import ArtGS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--iteration", required=True)
    a = ap.parse_args()
    cfg = eval(open(os.path.join(a.model_path, "cfg_args")).read(), {"Namespace": Namespace})
    model = ArtGS(cfg).cuda()
    sd = torch.load(os.path.join(a.model_path, "deform", f"iteration_{a.iteration}", "deform.pth"), map_location="cuda")
    model.load_state_dict(sd)
    types = model.joint_types[1:]
    joints = model.get_joint_param(types)
    out = os.path.join(a.model_path, "train", f"ours_{a.iteration}")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "pred_joints.pkl"), "wb") as f:
        pickle.dump({"types": types, "joints": joints}, f)
    print(f"{len(types)} joints {','.join(types)} -> {out}/pred_joints.pkl")


if __name__ == "__main__":
    main()
