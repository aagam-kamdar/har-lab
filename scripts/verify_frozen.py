import hashlib
import os

frozen_md = "FROZEN.md"
total = 0
missing = 0
mismatches = 0

with open(frozen_md, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line.startswith("|") or "---" in line or "File" in line:
            continue
        parts = [p.strip() for p in line.split("|") if p.strip()]
        if len(parts) >= 2:
            rel_path, expected_sha = parts[0], parts[1]
            total += 1
            if not os.path.exists(rel_path):
                print(f"MISSING: {rel_path}")
                missing += 1
                continue
            h = hashlib.sha256()
            with open(rel_path, "rb") as fp:
                while chunk := fp.read(65536):
                    h.update(chunk)
            actual_sha = h.hexdigest()
            if actual_sha != expected_sha:
                print(f"MISMATCH: {rel_path} (exp: {expected_sha[:8]}, got: {actual_sha[:8]})")
                mismatches += 1

print(f"Verified {total} files: {missing} missing, {mismatches} mismatches")
