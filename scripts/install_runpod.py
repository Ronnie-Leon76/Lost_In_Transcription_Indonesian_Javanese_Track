"""Install ``requirements-runpod.txt`` on a RunPod image, repairing Debian packages.

RunPod's Debian-based images ship some Python packages (``blinker``,
``pyparsing``, ``six``, ...) through apt, without pip's ``RECORD`` metadata.
``pip install --upgrade-strategy eager`` then aborts with
``uninstall-no-record-file`` / ``Cannot uninstall <pkg>`` as soon as it needs to
replace one of them. Which packages are affected varies by image, so instead
of hard-coding a list this script runs the install, and whenever pip names a
package it cannot uninstall, it installs a pip-managed copy of it with
``--ignore-installed --no-deps`` (which shadows the apt copy in
``/usr/local``) and retries.

Usage (from the repository root, or with any cwd):

    python scripts/install_runpod.py            # install + verify
    python scripts/install_runpod.py --no-verify
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
#: Known offenders, repaired up front so the common case needs no retry.
KNOWN_DEBIAN_PACKAGES = ("blinker",)
CANNOT_UNINSTALL = re.compile(r"Cannot uninstall '?([A-Za-z0-9][A-Za-z0-9._\-]*)'?")
MAX_REPAIRS = 25


def pip(*args: str) -> tuple[int, str]:
    """Run pip, streaming its output, and return (exit code, combined output)."""
    process = subprocess.Popen(
        [sys.executable, "-m", "pip", *args],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
    )
    lines = []
    assert process.stdout is not None
    for line in process.stdout:
        sys.stdout.write(line)
        lines.append(line)
    sys.stdout.flush()
    return process.wait(), "".join(lines)


def repair(package: str) -> None:
    print(f"\n[install_runpod] {package} is apt-managed without a RECORD file; "
          "installing a pip-managed copy over it\n", flush=True)
    code, _ = pip("install", "--upgrade", "--ignore-installed", "--no-deps", package)
    if code:
        raise SystemExit(f"could not repair {package}; install it manually with "
                         f"`pip install --ignore-installed {package}`")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--requirements", type=Path, default=ROOT / "requirements-runpod.txt")
    parser.add_argument("--no-verify", action="store_true", help="skip scripts/verify_runpod.py")
    args = parser.parse_args()

    for package in KNOWN_DEBIAN_PACKAGES:
        repair(package)

    repaired: set[str] = set(KNOWN_DEBIAN_PACKAGES)
    for _ in range(MAX_REPAIRS + 1):
        code, output = pip("install", "--upgrade", "--upgrade-strategy", "eager", "-r", str(args.requirements))
        if code == 0:
            break
        blocked = [name for name in dict.fromkeys(CANNOT_UNINSTALL.findall(output)) if name.lower() not in repaired]
        if not blocked:
            raise SystemExit(f"pip install failed (exit {code}) for a reason other than an apt-managed package; see the output above")
        for package in blocked:
            repair(package)
            repaired.add(package.lower())
    else:
        raise SystemExit(f"gave up after repairing {len(repaired)} apt-managed packages: {sorted(repaired)}")

    extra = sorted(repaired - set(KNOWN_DEBIAN_PACKAGES))
    if extra:
        print(f"[install_runpod] repaired apt-managed packages: {extra}")
    if not args.no_verify:
        subprocess.run([sys.executable, str(ROOT / "scripts" / "verify_runpod.py")], check=True)
    print("[install_runpod] done. Restart the kernel if pip upgraded a package this kernel already imported.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
