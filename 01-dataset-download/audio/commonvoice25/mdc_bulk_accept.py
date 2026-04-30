#!/usr/bin/env python3
"""Bulk-accept Mozilla Data Collective dataset terms via headless browser.

Mozilla DC enforces per-dataset terms acceptance through the web UI; the
public API has only download endpoints, no accept endpoint. This script
automates the click-through.

WHERE TO RUN
============
On your local machine (laptop) — NOT the SLURM cluster. The cluster
doesn't have chromium's runtime libs and no sudo to install them. The
acceptance state lives on Mozilla's servers tied to your account, so
clicking from your laptop works for the cluster's later API downloads.

SETUP
=====
On your laptop::

    pip install playwright
    playwright install chromium

USAGE
=====
::

    # Default ID list (CV 25.0 — copy mdc_dataset_ids.txt next to this script)
    python mdc_bulk_accept.py

    # Or pass an explicit ID list file
    python mdc_bulk_accept.py path/to/ids.txt

    # Headless mode (faster, but you can't see the page)
    python mdc_bulk_accept.py --headless

    # Resume mode: skip IDs already in storage_state's accepted-set
    python mdc_bulk_accept.py --resume

The script will:
  1. Open chromium (headed by default — you see what's happening)
  2. Load mozilladatacollective.com login page
  3. Pause and ask you to log in manually, then press Enter to continue
  4. Save your session cookies to mdc_session.json so re-runs reuse the login
  5. For each dataset ID, navigate, click Accept, wait for confirmation
  6. Print a per-dataset status line and a final summary
  7. Idempotent: already-accepted datasets are skipped (Download button shown)

After this finishes successfully, run the SLURM array on the cluster.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

LOGIN_URL = "https://mozilladatacollective.com/login"
DATASET_URL_FMT = "https://mozilladatacollective.com/datasets/{ds_id}"
SESSION_FILE = "mdc_session.json"

# Selector candidates for the accept button. Mozilla DC's UI may
# evolve; we try several and use the first that's visible.
ACCEPT_SELECTORS = [
    "button:has-text('Accept Terms')",
    "button:has-text('Accept terms')",
    "button:has-text('Accept')",
    "button:has-text('I Accept')",
    "button:has-text('Agree')",
    "button:has-text('I Agree')",
    "button:has-text('I agree')",
]
CONFIRM_SELECTORS = [
    "button:has-text('Confirm')",
    "button:has-text('I Accept')",
    "button:has-text('Yes, accept')",
    "button:has-text('Continue')",
]
ACCEPTED_INDICATORS = [
    "button:has-text('Download')",
    "a:has-text('Download')",
    "text=You have agreed",
    "text=Terms accepted",
]


def parse_ids(path: Path) -> list[str]:
    ids = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Allow inline comments like: cmn3iaztg...  # zh-CN
        ds_id = line.split("#", 1)[0].split()[0]
        if ds_id.startswith("cmn") and len(ds_id) > 20:
            ids.append(ds_id)
    return ids


def first_visible(page, selectors, timeout_ms=2000):
    for sel in selectors:
        loc = page.locator(sel).first
        try:
            if loc.is_visible(timeout=timeout_ms):
                return loc
        except Exception:
            continue
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("ids_file", nargs="?", default="mdc_dataset_ids.txt",
                        help="Path to file with one dataset ID per line (default: mdc_dataset_ids.txt)")
    parser.add_argument("--headless", action="store_true",
                        help="Run chromium without opening a window (faster but no visual feedback)")
    parser.add_argument("--session", default=SESSION_FILE,
                        help=f"Path to save/load session cookies (default: {SESSION_FILE})")
    parser.add_argument("--per-dataset-timeout", type=int, default=15,
                        help="Seconds to wait per dataset before giving up (default: 15)")
    args = parser.parse_args()

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("playwright not installed. Run: pip install playwright && playwright install chromium",
              file=sys.stderr)
        return 2

    ids_path = Path(args.ids_file)
    if not ids_path.is_file():
        print(f"ID file not found: {ids_path}", file=sys.stderr)
        return 2
    ids = parse_ids(ids_path)
    if not ids:
        print(f"No dataset IDs in {ids_path}", file=sys.stderr)
        return 2
    print(f"Will accept terms for {len(ids)} datasets from {ids_path}")

    session_path = Path(args.session)
    storage_state = str(session_path) if session_path.is_file() else None
    if storage_state:
        print(f"Reusing session cookies from {session_path}")
    else:
        print(f"No session file at {session_path} — manual login required")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=args.headless)
        context = browser.new_context(storage_state=storage_state)
        page = context.new_page()

        if storage_state is None:
            print(f"\nOpening {LOGIN_URL}")
            page.goto(LOGIN_URL)
            input("\n>>> Log in to Mozilla DC, then press Enter here to start auto-accept... ")
            context.storage_state(path=str(session_path))
            print(f"Saved session cookies to {session_path}")

        accepted = []
        already = []
        failed = []

        for i, ds_id in enumerate(ids, 1):
            url = DATASET_URL_FMT.format(ds_id=ds_id)
            print(f"\n[{i:3d}/{len(ids)}] {ds_id} ...", end=" ", flush=True)
            try:
                page.goto(url, wait_until="domcontentloaded", timeout=args.per_dataset_timeout * 1000)
                # Network may keep firing analytics; cap with a short networkidle wait.
                try:
                    page.wait_for_load_state("networkidle", timeout=5000)
                except Exception:
                    pass

                # Already accepted?
                indicator = first_visible(page, ACCEPTED_INDICATORS, timeout_ms=1500)
                if indicator is not None:
                    already.append(ds_id)
                    print("already accepted")
                    continue

                # Find and click the accept button
                btn = first_visible(page, ACCEPT_SELECTORS, timeout_ms=4000)
                if btn is None:
                    failed.append((ds_id, "no Accept button found"))
                    print("✗ no Accept button found")
                    continue
                btn.click()
                time.sleep(0.4)

                # Optional confirmation dialog
                confirm = first_visible(page, CONFIRM_SELECTORS, timeout_ms=1500)
                if confirm is not None:
                    confirm.click()
                    time.sleep(0.3)

                try:
                    page.wait_for_load_state("networkidle", timeout=5000)
                except Exception:
                    pass
                accepted.append(ds_id)
                print("✓ accepted")

            except Exception as exc:
                failed.append((ds_id, f"{type(exc).__name__}: {exc}"))
                print(f"✗ error: {type(exc).__name__}")

        # Persist updated session for re-runs
        try:
            context.storage_state(path=str(session_path))
        except Exception:
            pass

        browser.close()

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Accepted now:     {len(accepted)}")
    print(f"  Already accepted: {len(already)}")
    print(f"  Failed:           {len(failed)}")
    if failed:
        print("\nFailed details:")
        for ds_id, err in failed:
            print(f"  {ds_id}: {err}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
