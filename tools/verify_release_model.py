from pathlib import Path
import hashlib
import sys

EXPECTED = "fd43c5f83463f00d189292b4d4034bec21f3147c453232c4fbf8336cfd2047f9"

if len(sys.argv) != 2:
    raise SystemExit("Usage: python tools/verify_release_model.py PATH_TO_BEST_PT")

path = Path(sys.argv[1])
digest = hashlib.sha256(path.read_bytes()).hexdigest()
print("file   :", path)
print("sha256 :", digest)
print("expect :", EXPECTED)
raise SystemExit(0 if digest == EXPECTED else 1)
