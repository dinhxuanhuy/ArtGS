"""CLI flags for SemArt, shared by train.py and train_predict.py.

All defaults reproduce the original ArtGS behaviour, so the baseline is unchanged unless a flag
is passed.
"""
import json
import os

from arguments import ParamGroup


class SemParams(ParamGroup):
    def __init__(self, parser):
        # dir written by semart/lift.py (center_info.npy + slots.json); '' = original ArtGS init
        self.slot_init = ""
        # where joint types come from: 'json' = arguments/joint_types_<center_init>.json (ArtGS),
        # 'vlm' = slots.json from the lift stage
        self.joint_source = "json"
        # train_predict.py: write predicted joint types here instead of the shared json
        self.joint_types_out = ""
        # train.py: read joint types from this file (output of train_predict --joint_types_out)
        self.joint_types_file = ""
        # semantic consistency loss
        self.sem_mask_dir = ""          # .../masks (contains start/, end/)
        self.lambda_sem = 0.0
        self.sem_every = 1
        self.sem_from_iter = 500
        self.sem_until_iter = 15000
        self.sem_w_intra = 1.0
        self.sem_w_inter = 1.0
        self.sem_w_static = 1.0
        self.sem_min_pixels = 64
        # protocol: do not pick/save a "best" checkpoint by GT joint error
        self.no_gt_select = False
        super().__init__(parser, "SemArt Parameters")


def apply_slot_overrides(args):
    """Override num_slots / joint_types / center file from the lift stage output.

    Call after the script has filled args.joint_types / args.num_slots the ArtGS way.
    Returns the center_info path to use (or None to keep the ArtGS default).
    """
    if not getattr(args, "slot_init", ""):
        if getattr(args, "joint_source", "json") != "json":
            raise ValueError("--joint_source vlm needs --slot_init")
        if getattr(args, "joint_types_file", ""):
            # ArtGS slot init, but predicted types from a per-run file instead of the shared json
            # (several scenes in parallel would otherwise race on arguments/joint_types_cgs.json)
            args.joint_types = read_joint_types(args.joint_types_file)
            args.num_slots = len(args.joint_types.split(","))
        return None
    slots = json.load(open(os.path.join(args.slot_init, "slots.json")))
    args.num_slots = int(slots["num_slots"])
    if getattr(args, "joint_types_file", ""):
        args.joint_types = read_joint_types(args.joint_types_file)
        assert len(args.joint_types.split(",")) == args.num_slots, (args.joint_types, args.num_slots)
    elif args.joint_source == "vlm":
        args.joint_types = slots["joint_types"]
    else:
        # the json holds types for the ArtGS slot order, not ours: use the no-prior layout
        # (train_predict.py ignores types beyond the count; train.py must get --joint_types_file)
        args.joint_types = ",".join(["s"] + ["r"] * (args.num_slots - 1))
    return os.path.join(args.slot_init, "center_info.npy")


def read_joint_types(path):
    with open(path) as f:
        return f.read().strip()
