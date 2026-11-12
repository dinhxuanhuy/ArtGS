#!/usr/bin/env python3
"""Fetch + verify + extract ArtGS datasets. Idempotent.

    python scripts/load_data.py paris                       # -> <root>/paris
    python scripts/load_data.py paris dta --root /content/data \
        --persist /content/drive/MyDrive/artgs_colab/archives

Archive lookup order: --downloads dir -> --persist dir (copied to local disk
first) -> HTTP download (then copied into --persist so the next VM skips it).
Every archive is sha256-checked against configs/datasets.json. A dataset is
skipped when <root>/<dest>/.artgs_complete.json matches the archive hash.
Archives over 5 GB need --yes.
"""
import argparse, hashlib, json, os, shutil, subprocess, sys, tarfile, time, zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MANIFEST = REPO / "configs" / "datasets.json"
MARKER = ".artgs_complete.json"
BIG = 5 * 1024**3


def log(*a):
    print("[load_data]", *a, flush=True)


def sha256(path, archive):
    """Hash with a sidecar cache keyed on size+mtime (re-hashing 3 GB costs ~10 s)."""
    side = path.with_name(path.name + ".sha256")
    st = path.stat()
    key = f"{st.st_size}:{int(st.st_mtime)}"
    if side.exists():
        k, _, h = side.read_text().strip().partition(" ")
        if k == key:
            return h
    t = time.time()
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(16 << 20), b""):
            h.update(chunk)
    h = h.hexdigest()
    try:
        side.write_text(f"{key} {h}\n")
    except OSError:
        pass
    log(f"sha256 {path.name}: {time.time() - t:.1f}s")
    return h


def verified(path, meta):
    return path.exists() and path.stat().st_size == meta["size"] and sha256(path, meta) == meta["sha256"]


def download(url, out):
    out.parent.mkdir(parents=True, exist_ok=True)
    part = out.with_name(out.name + ".part")
    t = time.time()
    if shutil.which("curl"):
        r = subprocess.run(["curl", "-fL", "--retry", "5", "--retry-delay", "3", "-C", "-",
                            "-sS", "-o", str(part), url])
        if r.returncode:
            sys.exit(f"download failed ({r.returncode}): {url}")
    else:
        import urllib.request
        with urllib.request.urlopen(url) as r, open(part, "wb") as f:
            shutil.copyfileobj(r, f, 16 << 20)
    part.rename(out)
    log(f"downloaded {out.name} ({out.stat().st_size / 1e9:.2f} GB) in {time.time() - t:.0f}s")


def get_archive(name, meta, downloads, persist):
    local = downloads / name
    if verified(local, meta):
        return local
    if persist and verified(persist / name, meta):
        t = time.time()
        shutil.copyfile(persist / name, local.with_name(name + ".part"))
        local.with_name(name + ".part").rename(local)
        log(f"copied {name} from persistent storage in {time.time() - t:.0f}s")
    elif meta.get("url"):
        download(meta["url"], local)
    else:
        sys.exit(f"{name}: not in {local.parent} or --persist, and it has no download URL ({meta.get('note', '')})")
    if not verified(local, meta):
        local.unlink(missing_ok=True)
        sys.exit(f"checksum mismatch for {name}; deleted, re-run to retry")
    if persist and not (persist / name).exists():
        t = time.time()
        persist.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(local, persist / (name + ".part"))
        (persist / (name + ".part")).rename(persist / name)
        log(f"saved {name} to persistent storage in {time.time() - t:.0f}s")
    return local


def extract(archive, prefix, dest):
    t = time.time()
    tmp = dest.with_name(dest.name + ".extracting")
    shutil.rmtree(tmp, ignore_errors=True)
    n = 0

    def write(name, opener):
        nonlocal n
        rel = Path(name[len(prefix):])
        if rel.is_absolute() or ".." in rel.parts:
            sys.exit(f"unsafe path in archive: {name}")
        target = tmp / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        with opener() as src, open(target, "wb") as dst:
            shutil.copyfileobj(src, dst, 4 << 20)
        n += 1

    if archive.name.endswith(".zip"):
        with zipfile.ZipFile(archive) as z:
            for m in z.infolist():
                if m.filename.startswith(prefix) and not m.is_dir():
                    write(m.filename, lambda m=m: z.open(m))
    else:
        with tarfile.open(archive) as tf:
            for m in tf:
                if m.name.startswith(prefix) and m.isfile():
                    write(m.name, lambda m=m: tf.extractfile(m))
    if n == 0:
        sys.exit(f"no members under '{prefix}' in {archive.name}")
    shutil.rmtree(dest, ignore_errors=True)
    tmp.rename(dest)
    log(f"extracted {n} files -> {dest} in {time.time() - t:.0f}s")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("datasets", nargs="+")
    ap.add_argument("--root", type=Path, default=Path(os.environ.get("ARTGS_DATA", REPO / "data")))
    ap.add_argument("--downloads", type=Path, default=None, help="local archive dir (default: <root>/_archives)")
    ap.add_argument("--persist", type=Path, default=os.environ.get("ARTGS_PERSIST_ARCHIVES"),
                    help="persistent archive cache (e.g. a Drive folder)")
    ap.add_argument("--yes", action="store_true", help="allow archives > 5 GB")
    ap.add_argument("--adopt-existing", action="store_true",
                    help="mark an existing, unmarked dataset dir as complete without re-extracting")
    ap.add_argument("--delete-archive", action="store_true", help="remove local archive after extraction")
    a = ap.parse_args()

    man = json.loads(MANIFEST.read_text())
    downloads = a.downloads or a.root / "_archives"
    downloads.mkdir(parents=True, exist_ok=True)
    persist = Path(a.persist) if a.persist else None
    used = set()
    for ds in a.datasets:
        if ds not in man["datasets"]:
            sys.exit(f"unknown dataset '{ds}'. known: {', '.join(man['datasets'])}")
        spec = man["datasets"][ds]
        aname = spec["archive"]
        meta = man["archives"][aname]
        dest = a.root / spec["dest"]
        want = {"archive": aname, "sha256": meta["sha256"], "prefix": meta["provides"][ds]}
        marker = dest / MARKER
        if marker.exists() and json.loads(marker.read_text()) == want:
            log(f"{ds}: present at {dest}, skipping")
            continue
        if dest.exists() and any(dest.iterdir()) and not marker.exists():
            if a.adopt_existing:
                marker.write_text(json.dumps(want, indent=1))
                log(f"{ds}: adopted existing {dest}")
                continue
            sys.exit(f"{dest} exists without a completion marker; use --adopt-existing or remove it")
        if meta["size"] > BIG and not a.yes:
            log(f"{aname} is {meta['size'] / 1e9:.1f} GB (> 5 GB). Re-run with --yes to confirm.")
            sys.exit(3)
        archive = get_archive(aname, meta, downloads, persist)
        used.add(archive)
        extract(archive, want["prefix"], dest)
        marker.write_text(json.dumps(want, indent=1))
    if a.delete_archive:
        for p in used:
            p.unlink(missing_ok=True)
            p.with_name(p.name + ".sha256").unlink(missing_ok=True)


if __name__ == "__main__":
    main()
