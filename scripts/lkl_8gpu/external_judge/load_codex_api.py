#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tomllib


DEEPSEEK_CHAT_COMPLETIONS_ENDPOINT = "https://api.deepseek.com/chat/completions"


def responses_endpoint(base_url: str) -> str:
    endpoint = base_url.rstrip("/")
    if endpoint.endswith("/responses"):
        return endpoint
    if endpoint.endswith("/chat/completions"):
        raise ValueError("Chat Completions endpoint cannot be used for Responses API evaluation")
    return f"{endpoint}/responses"


def load_codex_environment(config_path: Path, auth_path: Path) -> dict[str, str]:
    with config_path.open("rb") as handle:
        config = tomllib.load(handle)
    provider_name = config["model_provider"]
    provider = config["model_providers"][provider_name]
    if provider["wire_api"] != "responses":
        raise ValueError(f"Codex provider {provider_name!r} is not configured for Responses API")

    auth = json.loads(auth_path.read_text(encoding="utf-8"))
    key = auth["OPENAI_API_KEY"]
    if not isinstance(key, str) or not key:
        raise ValueError("OPENAI_API_KEY must be a nonempty string")

    return {
        "OPENAI_API_KEY": key,
        "OPENAI_API_BASE": responses_endpoint(provider["base_url"]),
    }


def load_deepseek_environment() -> dict[str, str]:
    key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not key.startswith("sk-"):
        raise ValueError("DEEPSEEK_API_KEY must be a nonempty DeepSeek API key")
    return {
        "OPENAI_API_KEY": key,
        "OPENAI_API_BASE": DEEPSEEK_CHAT_COMPLETIONS_ENDPOINT,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inject an external judge API provider into a child process without printing credentials."
    )
    parser.add_argument("--provider", choices=("codex", "deepseek"), default="deepseek")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--auth", type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a child command is required after --")

    if args.provider == "codex":
        if args.config is None or args.auth is None:
            parser.error("--config and --auth are required for the Codex provider")
        provider_environment = load_codex_environment(args.config, args.auth)
    else:
        provider_environment = load_deepseek_environment()
    child_environment = os.environ.copy()
    child_environment.update(provider_environment)
    os.execvpe(command[0], command, child_environment)


if __name__ == "__main__":
    main()
