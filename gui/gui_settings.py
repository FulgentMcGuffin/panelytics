from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from gui.plot_settings_dialog import GRAPH_SETTINGS_DEFAULTS

SETTINGS_FILENAME = "panelytics.json"

DEFAULT_DOWNLOAD_SETTINGS: dict[str, Any] = {
    "format": "csv",
    "headers": True,
    "directory": None,
}

DEFAULT_SETTINGS: dict[str, Any] = {
    "ui_theme": "Fluent Dark",
    "plot_settings": dict(GRAPH_SETTINGS_DEFAULTS),
    "download_settings": dict(DEFAULT_DOWNLOAD_SETTINGS),
    "data_sources": [],
    "formula_rows": [{"is_regime": False, "formula": ""}],
    "user_functions": {
        "pythonpath": [],
        "imports": [],
    },
}


def settings_path() -> Path:
    """Return the path to the persistent JSON settings file."""
    return Path.home() / SETTINGS_FILENAME


def load_settings(path: Path | None = None) -> dict[str, Any]:
    """Load persisted settings from panelytics.json, merging with defaults."""
    target = path or settings_path()
    settings = deepcopy(DEFAULT_SETTINGS)
    if not target.is_file():
        return settings

    try:
        with open(target, encoding="utf-8") as handle:
            data = json.load(handle) or {}
    except Exception:
        return settings

    if not isinstance(data, dict):
        return settings

    if isinstance(data.get("ui_theme"), str):
        settings["ui_theme"] = data["ui_theme"]

    if isinstance(data.get("plot_settings"), dict):
        settings["plot_settings"] = {
            **GRAPH_SETTINGS_DEFAULTS,
            **data["plot_settings"],
        }

    if isinstance(data.get("download_settings"), dict):
        settings["download_settings"] = {
            **DEFAULT_DOWNLOAD_SETTINGS,
            **data["download_settings"],
        }

    if isinstance(data.get("data_sources"), list):
        settings["data_sources"] = [
            dict(item) for item in data["data_sources"] if isinstance(item, dict)
        ]

    if isinstance(data.get("formula_rows"), list) and data["formula_rows"]:
        rows = [
            {
                "is_regime": bool(item.get("is_regime", False)),
                "formula": str(item.get("formula", "")),
            }
            for item in data["formula_rows"]
            if isinstance(item, dict)
        ]
        if rows:
            settings["formula_rows"] = rows[:10]

    if isinstance(data.get("user_functions"), dict):
        uf = data["user_functions"]
        pythonpath = uf.get("pythonpath")
        imports = uf.get("imports")
        settings["user_functions"] = {
            "pythonpath": (
                [str(p) for p in pythonpath] if isinstance(pythonpath, list) else []
            ),
            "imports": (
                [str(s) for s in imports] if isinstance(imports, list) else []
            ),
        }

    return settings


def save_settings(settings: dict[str, Any], path: Path | None = None) -> None:
    """Persist settings to panelytics.json."""
    target = path or settings_path()
    uf = settings.get("user_functions") or {}
    payload: dict[str, Any] = {
        "ui_theme": settings.get("ui_theme", DEFAULT_SETTINGS["ui_theme"]),
        "plot_settings": {
            **GRAPH_SETTINGS_DEFAULTS,
            **(settings.get("plot_settings") or {}),
        },
        "download_settings": {
            **DEFAULT_DOWNLOAD_SETTINGS,
            **(settings.get("download_settings") or {}),
        },
        "data_sources": list(settings.get("data_sources") or []),
        "formula_rows": list(settings.get("formula_rows") or [])[:10],
        "user_functions": {
            "pythonpath": list(uf.get("pythonpath") or []),
            "imports": list(uf.get("imports") or []),
        },
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
