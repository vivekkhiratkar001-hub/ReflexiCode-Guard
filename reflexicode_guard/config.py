"""6-7 Oct: configuration + plugin loader so M1/M2/M3 modules plug in without touching M4 code.

Analyzer spec format: "package.module:function"  e.g. "m2_ast.analyzer:ast_analyzer"
Sources, in priority order:
  1. env  RFG_ANALYZERS="pkg.a:fn1,pkg.b:fn2"
  2. file reflexicode.json  {"analyzers": [...], "max_findings": 25, "llm_provider": "mock"}
  3. built-in defaults (demo static analyzer + mock LLM stand-in)
"""
from __future__ import annotations

import importlib
import json
import os
from dataclasses import dataclass, field
from typing import Callable

CONFIG_FILE = "reflexicode.json"
BUILTIN = {
    "demo_static": "reflexicode_guard.pipeline:demo_static_analyzer",
    "mock_llm": "reflexicode_guard.stubs:mock_llm_analyzer",
}


@dataclass
class Config:
    analyzers: list[str] = field(default_factory=lambda: ["demo_static"])
    max_findings: int = 25
    llm_provider: str = "mock"          # "mock" | "ollama" (ollama is M3's job)
    protected_bases: list[str] = field(default_factory=lambda: ["main", "master", "develop"])


def load_config(path: str | None = None, env=os.environ) -> Config:
    cfg = Config()
    path = path or CONFIG_FILE
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        for k, v in data.items():
            if k.startswith("_"):
                continue                 # "_comment" etc.
            if not hasattr(cfg, k):
                raise ValueError(f"Unknown config key '{k}' in {path}")
            setattr(cfg, k, v)
    if env.get("RFG_ANALYZERS"):
        cfg.analyzers = [s.strip() for s in env["RFG_ANALYZERS"].split(",") if s.strip()]
    if env.get("LLM_PROVIDER"):
        cfg.llm_provider = env["LLM_PROVIDER"]
    if env.get("RFG_MAX_FINDINGS"):
        cfg.max_findings = int(env["RFG_MAX_FINDINGS"])
    return cfg


def resolve_analyzer(spec: str) -> Callable:
    spec = BUILTIN.get(spec, spec)
    if ":" not in spec:
        raise ValueError(f"Bad analyzer spec '{spec}' (expected 'module:function' or a built-in name {list(BUILTIN)})")
    mod, fn = spec.split(":", 1)
    try:
        return getattr(importlib.import_module(mod), fn)
    except (ImportError, AttributeError) as e:
        raise ValueError(f"Cannot load analyzer '{spec}': {e}") from e


def load_analyzers(cfg: Config) -> list[Callable]:
    return [resolve_analyzer(s) for s in cfg.analyzers]
