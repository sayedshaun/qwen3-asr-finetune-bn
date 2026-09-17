"""Loads a section of config.yaml into a Namespace."""

import re
from argparse import Namespace

import yaml

_FLOAT_LOADER = yaml.SafeLoader
_FLOAT_LOADER.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(
        r"""^(?:
         [-+]?(?:[0-9][0-9_]*)\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\.[0-9_]*
        |[-+]?\.(?:inf|Inf|INF)
        |\.(?:nan|NaN|NAN))$""",
        re.VERBOSE,
    ),
    list("-+0123456789."),
)


def apply_overrides(section_cfg: dict, overrides: list = ()) -> None:
    """Applies CLI `key=value` overrides (as collected by an argparse `--set`
    option) onto a loaded config section in place. Values are parsed as YAML
    so `16` becomes an int, `true` a bool, etc., matching how config.yaml
    itself is read."""
    for pair in overrides:
        key, sep, value = pair.partition("=")
        if not sep:
            raise SystemExit(f"Invalid override {pair!r}, expected key=value")
        section_cfg[key] = yaml.safe_load(value)


def load_section(
    config_path: str, section: str, required: list = (), overrides: list = ()
) -> Namespace:
    with open(config_path) as f:
        cfg = yaml.load(f, Loader=_FLOAT_LOADER)
    section_cfg = cfg.get(section) or {}
    apply_overrides(section_cfg, overrides)

    missing = [key for key in required if section_cfg.get(key) is None]
    if missing:
        raise SystemExit(
            f"Missing required config values: {', '.join(missing)} "
            f"(set them in {config_path} under '{section}:')"
        )

    return Namespace(**section_cfg)
