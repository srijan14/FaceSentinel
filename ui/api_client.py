"""Thin HTTP client for the Face De-Duplication API (used by the Streamlit console)."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Optional

import requests


class DedupClient:
    def __init__(self, base_url: str, api_key: str, timeout: int = 60):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    @property
    def _auth(self) -> dict:
        # NOTE: the API compares the raw Authorization header (no "Bearer " prefix).
        return {"Authorization": self.api_key}

    def health(self) -> dict:
        r = requests.get(f"{self.base_url}/health", timeout=10)
        return r.json()

    def stats(self) -> dict:
        r = requests.get(f"{self.base_url}/v1/dedup/face/stats", timeout=10)
        return r.json()

    def store(self, image_bytes: bytes, filename: str, transaction_id: str, identity: dict) -> requests.Response:
        meta = {k: v for k, v in identity.items() if v not in (None, "")}
        meta.setdefault("created_on", datetime.now(timezone.utc).isoformat())
        meta.setdefault("image_path", "")
        files = {"image": (filename, image_bytes, "image/jpeg")}
        data = {"transaction_id": transaction_id, "metadata": json.dumps(meta)}
        return requests.post(f"{self.base_url}/v1/dedup/face/store",
                             headers=self._auth, files=files, data=data, timeout=self.timeout)

    def check(self, image_bytes: bytes, filename: str, transaction_id: str, identity: dict,
              threshold: Optional[float] = None, limit: int = 10) -> requests.Response:
        meta = {k: v for k, v in identity.items() if v not in (None, "")}
        files = {"image": (filename, image_bytes, "image/jpeg")}
        data = {"transaction_id": transaction_id, "metadata": json.dumps(meta), "limit": str(limit)}
        if threshold is not None:
            data["threshold"] = str(threshold)
        return requests.post(f"{self.base_url}/v1/dedup/face/check",
                             headers=self._auth, files=files, data=data, timeout=self.timeout)

    def purge(self, transaction_id: str) -> requests.Response:
        return requests.post(f"{self.base_url}/v1/dedup/face/purge",
                             headers={**self._auth, "Content-Type": "application/json"},
                             json={"transaction_id": transaction_id}, timeout=30)

    def seed_demo(self, reset: bool = False) -> requests.Response:
        """Enrol the fictional sample-KYC gallery into the active vector DB."""
        return requests.post(f"{self.base_url}/v1/dedup/face/demo/seed",
                             headers=self._auth,
                             data={"reset": "true" if reset else "false"}, timeout=180)

    def demo_probes(self) -> dict:
        """Fetch the planted onboarding probes (base64 avatars) for replay."""
        r = requests.get(f"{self.base_url}/v1/dedup/face/demo/probes", timeout=30)
        return r.json()
