"""Re-point a module's mira.paths constants at a test root (the pre-paths.py
tests patched `<module>.ROOT`; path constants are bound at import time)."""
import importlib

_LAYOUT = {
    "REPORT_FILES": ("report-files",),
    "LOCAL_CACHE": ("cache",),
    "TEMP_DIR": ("scripts", "temp"),
    "CONFIG_DIR": ("configs",),
    "CRAWLERS_DIR": ("Micron",),
    "LIGHTRAG_DIR": ("lightrag",),
    "REALTIME_DIR": ("realtime-state",),
}


def repoint(monkeypatch, module: str, root) -> None:
    mod = importlib.import_module(module)
    monkeypatch.setattr(mod, "ROOT", root, raising=False)
    for name, parts in _LAYOUT.items():
        if hasattr(mod, name):
            monkeypatch.setattr(mod, name, root.joinpath(*parts))
