"""
ingest_recording.py

Pushes one night's raw CSV into a running SomniGuard backend, exactly the way the
mobile app will eventually do automatically each morning once it has Wi-Fi. This is
NOT part of the doctor's UI on purpose -- by the time the doctor opens the dashboard,
the platform should already have the data. Use this script (or call POST /api/upload
directly) to get data into the platform for now, until that mobile integration exists.

Usage:
    python scripts/ingest_recording.py path/to/night1.csv
    python scripts/ingest_recording.py path/to/*.csv                 # ingest several at once
    python scripts/ingest_recording.py --base-url http://host:8000 night1.csv

Each CSV should have a "Date" column (dd/mm/yy, e.g. 15/09/26) constant across all its
rows -- that's what the doctor's day picker filters on. If it's missing, the backend
falls back to today's date and returns a warning (printed below).
"""

import sys
import argparse
import requests


def ingest_one(base_url: str, csv_path: str) -> None:
    print(f"Ingesting {csv_path} ...")
    with open(csv_path, "rb") as f:
        r = requests.post(f"{base_url}/api/upload", files={"file": (csv_path, f)})
    if not r.ok:
        print(f"  FAILED ({r.status_code}): {r.text}")
        return
    data = r.json()
    session = data["session"]
    print(
        f"  OK: session #{session['id']} | date={session['recording_date']} | "
        f"{session['n_steps']} steps | {session['n_segments']} segment(s)"
    )
    for w in data.get("warnings", []):
        print(f"  WARNING: {w}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_paths", nargs="+", help="One or more raw night CSVs to ingest")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="Backend base URL")
    args = parser.parse_args()

    try:
        requests.get(f"{args.base_url}/api/health", timeout=5).raise_for_status()
    except Exception as e:
        print(f"Cannot reach backend at {args.base_url}: {e}")
        sys.exit(1)

    for path in args.csv_paths:
        ingest_one(args.base_url, path)


if __name__ == "__main__":
    main()
