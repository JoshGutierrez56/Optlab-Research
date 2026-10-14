"""Locate defaults in an installed wheel or an editable source checkout."""

from pathlib import Path


def default_config_path(filename: str) -> Path:
    package_root = Path(__file__).resolve().parent
    bundled = package_root / "_config" / filename
    if bundled.is_file():
        return bundled
    return package_root.parent / "config" / filename
