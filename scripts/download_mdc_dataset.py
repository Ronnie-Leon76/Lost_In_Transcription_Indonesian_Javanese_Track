"""Download a Mozilla Data Collective dataset via the presigned-URL API.

This is for **development boxes only**. The competition container has no
network: test clips are already at ``/code_execution/data/clips/``. Do not put
``MDC_API_KEY`` in ``submission.zip``.

    export MDC_API_KEY=mdc_...
    python scripts/download_mdc_dataset.py
    python scripts/download_mdc_dataset.py --id cmhkl8z2a007rnr07p9bm5kmz --out nahuatl_dev

The Tetelancingo Nahuatl id is the default. Override ``--id`` for other MDC
corpora. After extract, point ``LIT_NAHUATL_DIR`` at the output folder.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tarfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lit_jv.data import load_dotenv  # noqa: E402

DEFAULT_DATASET_ID = "cmhkl8z2a007rnr07p9bm5kmz"
API_TEMPLATE = "https://mozilladatacollective.com/api/datasets/{dataset_id}/download"


def _request_download_url(dataset_id: str, token: str, timeout: int = 60) -> str:
    url = API_TEMPLATE.format(dataset_id=dataset_id)
    req = urllib.request.Request(
        url,
        method="POST",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        data=b"{}",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"MDC download request failed ({exc.code}): {body}") from exc
    download_url = payload.get("downloadUrl") or payload.get("download_url") or payload.get("url")
    if not download_url:
        raise SystemExit(f"MDC response had no downloadUrl: {payload!r}")
    return str(download_url)


def _download(url: str, dest: Path, timeout: int = 600) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading -> {dest}")
    urllib.request.urlretrieve(url, dest)
    return dest


def _extract(archive: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"extracting {archive.name} -> {out_dir}")
    name = archive.name.lower()
    if name.endswith(".zip"):
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(out_dir)
    else:
        with tarfile.open(archive, "r:*") as tf:
            tf.extractall(out_dir)
    return out_dir


def main() -> int:
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--id",
        default=os.environ.get("LIT_MDC_DATASET_ID", DEFAULT_DATASET_ID),
        help="MDC dataset id (default: Tetelancingo Nahuatl).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "nahuatl_dev",
        help="Directory to extract into.",
    )
    parser.add_argument(
        "--archive",
        type=Path,
        default=None,
        help="Where to store the downloaded archive. Defaults to <out>.tar.gz.",
    )
    parser.add_argument(
        "--keep-archive",
        action="store_true",
        help="Do not delete the tarball after extract.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Request a presigned URL, but never download or write an archive.",
    )
    args = parser.parse_args()

    token = os.environ.get("MDC_API_KEY") or os.environ.get("LIT_MDC_API_KEY")
    if not token:
        print(
            "MDC_API_KEY is not set. Put it in .env (development only).\n"
            "The competition container must not receive this key: it has no network "
            "and already mounts test clips at data/clips/.",
            file=sys.stderr,
        )
        return 2

    archive = args.archive or args.out.with_suffix(".tar.gz")
    url = _request_download_url(args.id, token)
    if args.check_only:
        # A presigned URL is intentionally neither printed nor persisted: it is
        # temporary capability data, not a diagnostic artifact.
        print(f"MDC access check succeeded for dataset {args.id}; no download started.")
        return 0

    _download(url, archive)
    _extract(archive, args.out)
    if not args.keep_archive and archive.exists():
        archive.unlink()
        print(f"removed {archive}")
    print(f"ready: {args.out}")
    print("set LIT_NAHUATL_DIR to that path, then re-run training.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
