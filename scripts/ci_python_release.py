#!/usr/bin/env python3
"""Helpers for the automated Python package release workflow."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "bindings" / "python" / "pyproject.toml"
CARGO_TOML = ROOT / "bindings" / "python" / "Cargo.toml"
VERSION_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")


Version = tuple[int, int, int]


def parse_version(value: str) -> Version | None:
    match = VERSION_RE.match(value)
    if not match:
        return None
    major, minor, patch = match.groups()
    return int(major), int(minor), int(patch)


def format_version(version: Version) -> str:
    return ".".join(str(part) for part in version)


def project_version() -> Version:
    with PYPROJECT.open("rb") as pyproject_file:
        project = tomllib.load(pyproject_file)["project"]
    version = parse_version(project["version"])
    if version is None:
        raise SystemExit(f"Unsupported project version: {project['version']}")
    return version


def release_tag_versions() -> set[Version]:
    output = subprocess.check_output(
        ["git", "tag", "--list", "v*"],
        cwd=ROOT,
        text=True,
    )
    versions: set[Version] = set()
    for tag in output.splitlines():
        version = parse_version(tag.removeprefix("v"))
        if version is not None:
            versions.add(version)
    return versions


def next_version() -> Version:
    base_version = project_version()
    released_versions = release_tag_versions()
    if not released_versions:
        return base_version

    latest_version = max(released_versions)
    if base_version > latest_version:
        return base_version

    major, minor, patch = latest_version
    return major, minor, patch + 1


def replace_section_version(path: Path, section: str, version: str) -> None:
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    in_section = False
    replaced = False
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped == f"[{section}]":
            in_section = True
            continue
        if stripped.startswith("[") and stripped.endswith("]"):
            in_section = False
        if in_section and stripped.startswith("version = "):
            newline = "\n" if line.endswith("\n") else ""
            lines[index] = f'version = "{version}"{newline}'
            replaced = True
            break

    if not replaced:
        raise SystemExit(f"Could not find version in [{section}] of {path}")
    path.write_text("".join(lines), encoding="utf-8")


def set_version(version: str) -> None:
    if parse_version(version) is None:
        raise SystemExit(f"Unsupported release version: {version}")
    replace_section_version(PYPROJECT, "project", version)
    replace_section_version(CARGO_TOML, "package", version)


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("next-version")
    set_version_parser = subparsers.add_parser("set-version")
    set_version_parser.add_argument("version")
    args = parser.parse_args()

    if args.command == "next-version":
        print(format_version(next_version()))
    elif args.command == "set-version":
        set_version(args.version)
    else:
        parser.error(f"Unknown command: {args.command}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
