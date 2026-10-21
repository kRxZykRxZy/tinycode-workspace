"""The command line interface of the inference engine."""

from __future__ import annotations

import argparse
from typing import Sequence

from .commands import command_bench, command_doctor, command_models, command_serve, command_tokenize

__all__ = ["main", "build_parser"]


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""

    parser = argparse.ArgumentParser(
        prog="tinycode-engine",
        description="Serve TinyCode models through an OpenAI compatible API.",
    )
    parser.add_argument("--version", action="store_true", help="Print the version and exit.")
    parser.add_argument(
        "--config",
        dest="config_path",
        default=None,
        help="Path to a configuration directory or file.",
    )

    subparsers = parser.add_subparsers(dest="command")

    serve = subparsers.add_parser("serve", help="Start the HTTP server (default).")
    serve.add_argument("--model", default=None, help="Path of the model artifact directory.")
    serve.add_argument("--host", default=None, help="Bind address (default 127.0.0.1).")
    serve.add_argument("--port", type=int, default=None, help="Port (default 8000).")
    serve.add_argument("--api-key", default=None, help="The API key the server accepts.")
    serve.add_argument("--workers", type=int, default=1, help="Number of workers.")

    doctor = subparsers.add_parser("doctor", help="Print the diagnostics of the environment.")
    models = subparsers.add_parser("models", help="List the configured models.")

    tokenize = subparsers.add_parser("tokenize", help="Tokenise a text.")
    tokenize.add_argument("text", help="The text to tokenise.")
    tokenize.add_argument("--model", default=None, help="Path of the model artifact directory.")

    bench = subparsers.add_parser("bench", help="Run a small local benchmark.")
    bench.add_argument("--model", default=None, help="Path of the model artifact directory.")
    bench.add_argument("--prompt", default="Write a Python function that adds two numbers.")
    bench.add_argument("--max-tokens", type=int, default=64)
    bench.add_argument("--iterations", type=int, default=1)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point of the inference engine."""

    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.version:
        from .. import __version__

        print(f"tinycode-inference-engine {__version__}")
        return 0

    command = args.command or "serve"
    if command == "serve":
        return command_serve(
            model=getattr(args, "model", None),
            host=getattr(args, "host", None),
            port=getattr(args, "port", None),
            config_path=args.config_path,
            api_key=getattr(args, "api_key", None),
            workers=getattr(args, "workers", 1),
        )
    if command == "doctor":
        return command_doctor(args.config_path)
    if command == "models":
        return command_models(args.config_path)
    if command == "tokenize":
        return command_tokenize(
            args.text, model=getattr(args, "model", None), config_path=args.config_path
        )
    if command == "bench":
        return command_bench(
            model=getattr(args, "model", None),
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            iterations=args.iterations,
            config_path=args.config_path,
        )

    parser.print_help()
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
