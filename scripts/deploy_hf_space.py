"""
Deploy TRAVELIQ to Hugging Face Spaces (Docker SDK, Free Tier: 2 vCPU / 16 GB RAM).

Usage:
    set HF_TOKEN=hf_...
    python scripts/deploy_hf_space.py --space-name traveliq-platform

Only uploads safe, clean production runtime files (no secrets, no mlflow.db, no backups).
"""

import argparse
import os
import shutil
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

from huggingface_hub import HfApi

PROJECT_ROOT = Path(__file__).resolve().parent.parent

STAGE_FILES = [
    ".streamlit/config.toml",
    "app/__init__.py",
    "app/api_client.py",
    "app/app.py",
    "app/config.py",
    "app/gender_api.py",
    "app/gender_features.py",
    "app/recommender.py",
    "assets/traveliq_logo.svg",
    "deploy/cloud/requirements-cloud.txt",
    "deploy/cloud/start_traveliq.sh",
    "models/flight_price_pipeline.joblib",
    "models/gender_classifier.joblib",
    "travel_capstone_dataset/flights.csv",
    "travel_capstone_dataset/hotels.csv",
    "travel_capstone_dataset/users.csv",
    "streamlit_app.py",
    "ATTRIBUTION.md",
]


def stage_clean_bundle(stage_dir: Path) -> None:
    """Copy strictly necessary runtime files plus root Dockerfile & HF README."""
    for rel in STAGE_FILES:
        src = PROJECT_ROOT / rel
        if not src.exists():
            raise FileNotFoundError(f"Missing required runtime file: {src}")
        dst = stage_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    # Place cloud Dockerfile & HF Spaces YAML README at Space repository root
    shutil.copy2(PROJECT_ROOT / "deploy/cloud/Dockerfile", stage_dir / "Dockerfile")
    shutil.copy2(PROJECT_ROOT / "deploy/cloud/README_HF_SPACE.md", stage_dir / "README.md")


def wait_for_space_running(api: HfApi, repo_id: str, timeout_s: int = 600) -> str:
    """Poll Hugging Face Space runtime until RUNNING or terminal failure."""
    start = time.time()
    last_stage = None
    while time.time() - start < timeout_s:
        rt = api.get_space_runtime(repo_id=repo_id)
        stage = getattr(rt, "stage", "UNKNOWN")
        if stage != last_stage:
            print(f"[HF Space] Stage: {stage} (elapsed {int(time.time() - start)}s)", flush=True)
            last_stage = stage
        if stage == "RUNNING":
            return stage
        if stage in {"BUILD_ERROR", "RUNTIME_ERROR", "CONFIG_ERROR"}:
            raise RuntimeError(f"Hugging Face Space failed in stage: {stage}")
        time.sleep(10)
    raise TimeoutError(f"Timed out waiting for Space {repo_id} to reach RUNNING (last stage: {last_stage})")


def main() -> int:
    parser = argparse.ArgumentParser(description="Deploy TRAVELIQ to Hugging Face Spaces (Docker SDK)")
    parser.add_argument("--space-name", default="traveliq", help="Target Space name (default: traveliq)")
    parser.add_argument("--private", action="store_true", help="Create Space as private (default: public)")
    parser.add_argument("--dry-run", action="store_true", help="Stage bundle locally without uploading")
    args = parser.parse_args()

    token = os.getenv("HF_TOKEN") or None

    with tempfile.TemporaryDirectory(prefix="traveliq_hf_stage_") as tmp:
        stage_dir = Path(tmp)
        stage_clean_bundle(stage_dir)
        staged = sorted(str(p.relative_to(stage_dir)) for p in stage_dir.rglob("*") if p.is_file())
        print(f"Staged {len(staged)} clean production files for HF Docker Space.")
        if args.dry_run:
            for item in staged:
                print(f"  - {item}")
            return 0

        api = HfApi(token=token)
        who = api.whoami()
        username = who["name"]
        repo_id = f"{username}/{args.space_name}"
        print(f"Authenticated as Hugging Face user '{username}'. Deploying to Space: {repo_id}")

        api.create_repo(
            repo_id=repo_id,
            repo_type="space",
            space_sdk="docker",
            exist_ok=True,
            private=args.private,
        )
        api.upload_folder(
            folder_path=str(stage_dir),
            repo_id=repo_id,
            repo_type="space",
            commit_message="Deploy TRAVELIQ v2.0 (internal Flask Flight & Gender APIs + Streamlit UI)",
            delete_patterns="*",
        )
        print(f"Uploaded clean bundle to https://huggingface.co/spaces/{repo_id}")

        wait_for_space_running(api, repo_id)
        space_slug = f"{username}-{args.space_name}".lower().replace("_", "-")
        public_url = f"https://{space_slug}.hf.space"
        print(f"PUBLIC_URL: {public_url}")

        # Verify HTTPS health probe on public URL
        health_url = f"{public_url}/_stcore/health"
        with urllib.request.urlopen(health_url, timeout=20) as resp:
            print(f"HEALTH_CHECK ({health_url}): {resp.status} {resp.read().decode('utf-8').strip()}")
        return 0


if __name__ == "__main__":
    sys.exit(main())
