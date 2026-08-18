"""Replay manually authored narration through the real AIKP engine pipeline."""

from __future__ import annotations

import json
from pathlib import Path


class ReplayNarrationProvider:
    """Record every prompt and return the matching response from a JSON file."""

    def __init__(self, responses_path: Path, requests_dir: Path):
        payload = json.loads(responses_path.read_text(encoding="utf-8"))
        responses = payload.get("responses", payload)
        if not isinstance(responses, list):
            raise ValueError("manual responses must be a list or {responses: [...]}")
        self.responses = responses
        self.requests_dir = requests_dir
        self.requests_dir.mkdir(parents=True, exist_ok=True)
        self.index = 0

    def __call__(self, request: dict) -> str:
        self.index += 1
        path = self.requests_dir / f"request-{self.index:03d}.json"
        path.write_text(
            json.dumps(request, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if self.index > len(self.responses):
            raise RuntimeError(
                f"manual response {self.index} missing; prompt saved to {path}")
        response = self.responses[self.index - 1]
        if not isinstance(response, str):
            raise ValueError(f"manual response {self.index} must be a string")
        return response
