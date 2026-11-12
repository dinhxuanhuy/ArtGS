"""Content-addressed cache for intermediate results.

Layout:  <ARTGS_CACHE or repo/cache>/<stage>/<object_id>/<config_hash>/
           meta.json   {stage, object_id, config, config_hash, created, git_sha, complete}
           ...         stage outputs

`config` must contain EVERY parameter that affects the stage output (model name +
version/checkpoint, prompt text, thresholds, resolution, iterations, upstream cache
hashes, code version that matters...). The hash is sha256 over canonical JSON, so key
order does not matter but any value change gives a new entry. An entry is only
reused if it is complete AND its stored config equals the requested one exactly.

    from tools.cache import CacheEntry
    e = CacheEntry("sam_masks", "paris/sapien/washer_103776",
                   {"model": "sam2.1_hiera_l", "points_per_side": 32, "iou_thresh": 0.88})
    if not e.complete:
        run_sam(out_dir=e.path)          # write outputs into e.path
        e.mark_complete(n_masks=123)     # only after outputs are fully written
    use(e.path)

CLI:  python tools/cache.py hash <stage> <object_id> '<json config>'
      python tools/cache.py ls [stage]
"""
from __future__ import annotations

import hashlib, json, os, shutil, subprocess, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
HASH_LEN = 12


def cache_root() -> Path:
    return Path(os.environ.get("ARTGS_CACHE", REPO / "cache"))


def canonical(config: dict) -> str:
    return json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def config_hash(config: dict) -> str:
    return hashlib.sha256(canonical(config).encode()).hexdigest()[:HASH_LEN]


def _git_sha() -> str | None:
    try:
        return subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return None


class CacheEntry:
    def __init__(self, stage: str, object_id: str, config: dict, root: Path | None = None):
        for name, v in (("stage", stage), ("object_id", object_id)):
            if not v or ".." in Path(v).parts or Path(v).is_absolute():
                raise ValueError(f"bad {name}: {v!r}")
        self.stage, self.object_id, self.config = stage, object_id, config
        self.hash = config_hash(config)
        self.path = (root or cache_root()) / stage / object_id / self.hash
        self.meta_path = self.path / "meta.json"

    @property
    def meta(self) -> dict | None:
        try:
            return json.loads(self.meta_path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            return None

    @property
    def complete(self) -> bool:
        m = self.meta
        if not m or not m.get("complete"):
            return False
        if canonical(m.get("config", {})) != canonical(self.config):
            raise RuntimeError(f"cache collision at {self.path}: stored config differs; refusing to reuse")
        return True

    def prepare(self, clean: bool = True) -> Path:
        """Create the entry dir (wiping an incomplete one) and write an incomplete meta."""
        if self.complete:
            return self.path
        if clean and self.path.exists():
            shutil.rmtree(self.path)
        self.path.mkdir(parents=True, exist_ok=True)
        self._write_meta(complete=False)
        return self.path

    def mark_complete(self, **extra) -> None:
        self.path.mkdir(parents=True, exist_ok=True)
        self._write_meta(complete=True, **extra)

    def _write_meta(self, complete: bool, **extra) -> None:
        m = {"stage": self.stage, "object_id": self.object_id, "config_hash": self.hash,
             "config": self.config, "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
             "git_sha": _git_sha(), "complete": complete, **extra}
        tmp = self.meta_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(m, indent=2, sort_keys=True, default=str))
        tmp.replace(self.meta_path)


def _ls(stage: str | None) -> None:
    root = cache_root()
    for meta in sorted(root.glob(f"{stage or '*'}/**/meta.json")):
        m = json.loads(meta.read_text())
        print(f"{'OK ' if m.get('complete') else '.. '} {m['stage']:<16} {m['object_id']:<40} {m['config_hash']}  {m.get('created', '')}")


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[:1] == ["hash"] and len(a) == 4:
        e = CacheEntry(a[1], a[2], json.loads(a[3]))
        print(e.path, "complete" if e.complete else "missing")
    elif a[:1] == ["ls"]:
        _ls(a[1] if len(a) > 1 else None)
    else:
        print(__doc__)
        sys.exit(2)
