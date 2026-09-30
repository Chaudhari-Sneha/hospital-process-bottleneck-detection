"""
Downloads the Sepsis Cases event log from 4TU.ResearchData.

Why this file exists:
The raw dataset is NOT committed to git. Anyone who clones this
repo runs this script to fetch it. That keeps the repo small and respects 4TU's
terms of use, while still making the project fully reproducible.

The md5 checksum is published by 4TU. Verifying it proves the file
analysed is byte-for-byte the official one - not a corrupted or edited copy.
"""

import hashlib
import urllib.request
from pathlib import Path

# 4TU article 12707639 = "Sepsis Cases - Event Log"
# Resolved from DOI 10.4121/uuid:915d2bfb-7e84-49ad-a286-dc35f063a460
BASE = "https://data.4tu.nl/file/33632f3c-5c48-40cf-8d8f-2db57f5a6ce7"

FILES = [
    # (filename, file uuid on 4TU, published md5)
    ("Sepsis Cases - Event Log.xes.gz", "643dccf2-985a-459e-835c-a82bce1c0339",
     "b5671166ac71eb20680d3c74616c43d2"),
    ("readme.txt", "c3729698-8f67-4c12-8fda-20983a0c2a65",
     "d990049d11124aeb5d68f2db6a2981d4"),
]

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def md5_of(path: Path) -> str:
    """Return the md5 hex digest of a file, reading it in chunks."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    for name, uuid, expected_md5 in FILES:
        target = RAW_DIR / name

        # Skip the download if we already hold a verified copy.
        if target.exists() and md5_of(target) == expected_md5:
            print(f"[ok, cached]  {name}")
            continue

        print(f"[downloading] {name} ...")
        urllib.request.urlretrieve(f"{BASE}/{uuid}", target)

        actual_md5 = md5_of(target)
        if actual_md5 != expected_md5:
            raise SystemExit(
                f"CHECKSUM MISMATCH for {name}\n"
                f"  expected {expected_md5}\n"
                f"  got      {actual_md5}\n"
                "The download may be corrupted. Delete the file and retry."
            )
        print(f"[verified]    {name}  (md5 {actual_md5})")

    print(f"\nData ready in: {RAW_DIR}")


if __name__ == "__main__":
    main()
