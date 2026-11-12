#!/usr/bin/env python3
"""Claude Code PreToolUse(Bash) hook: block committing data, checkpoints, caches or secrets.
Denies `git add -f/--force` of protected paths, explicit `git add` of protected paths, and
`git commit` while protected files are staged."""
import json, re, subprocess, sys

PROTECTED = re.compile(r"(^|/)(data|outputs|cache|runs|downloads|envs|logs)(/|$)|\.whl$|\.zip$|\.tar$|\.ckpt$|\.pth?$"
                       r"|(^|/)\.env$|token|credential|secret|rclone\.conf|\.pem$|id_ed25519|id_rsa", re.I)

def deny(reason):
    print(json.dumps({"hookSpecificOutput": {"hookEventName": "PreToolUse",
          "permissionDecision": "deny", "permissionDecisionReason": reason}}))
    sys.exit(0)

cmd = (json.load(sys.stdin).get("tool_input") or {}).get("command", "")
for part in re.split(r"&&|\|\||;|\n", cmd):
    toks = part.split()
    if "git" not in toks:
        continue
    g = toks.index("git")
    sub = next((t for t in toks[g + 1:] if not t.startswith("-") and "=" not in t), "")
    args = toks[toks.index(sub) + 1:] if sub in toks else []
    if sub == "add":
        paths = [a for a in args if not a.startswith("-")]
        bad = [p for p in paths if PROTECTED.search(p)]
        if bad:
            deny(f"Refusing to stage protected paths {bad}: datasets, checkpoints, caches and secrets must never be committed (CLAUDE.md rules).")
        if any(a in ("-f", "--force") for a in args):
            deny("Refusing `git add --force`: it bypasses .gitignore, which protects data/outputs/cache/secrets.")
    if sub == "commit":
        staged = subprocess.run(["git", "diff", "--cached", "--name-only"], capture_output=True, text=True).stdout.split()
        if "-a" in args or "--all" in args or any(re.fullmatch(r"-\w*a\w*", a) for a in args):
            staged += subprocess.run(["git", "diff", "--name-only"], capture_output=True, text=True).stdout.split()
        bad = [p for p in staged if PROTECTED.search(p)]
        if bad:
            deny(f"Refusing to commit protected files {bad[:10]}: unstage them first (git restore --staged <path>).")
sys.exit(0)
