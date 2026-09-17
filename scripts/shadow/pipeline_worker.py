#!/usr/bin/env python3
"""Internal paid replay worker; normally launched by scripts/shadow/run.py."""
import argparse
import asyncio
import os

import _bootstrap  # noqa: F401

os.environ["NEWS_SHADOW_SKIP_DOTENV"] = "1"
from shadow.pipeline_runner import pipeline_worker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--branch", required=True)
    args = parser.parse_args()
    result = asyncio.run(pipeline_worker(args.bundle, args.branch))
    return 0 if result["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
