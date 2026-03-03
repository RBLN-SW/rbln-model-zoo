#!/usr/bin/env python3
"""
Count models per framework. Each compile.py/main.py = 1 base.
If entry has model_name/model_id as argparse with choices=[...], count each choice as 1 variant.
Supports preset.yaml (configs/preset.yaml) for compile presets.
"""
import re
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None

ROOT = Path(__file__).resolve().parent.parent
EXCLUDE = {".venv", "serving"}

# (framework_dir, entry_filename)
FRAMEWORKS = [
    ("huggingface", "compile.py"),
    ("pytorch", "compile.py"),
    ("tensorflow", "compile.py"),
    ("cpp", "compile.py"),
    ("pytorch_dynamo", "main.py"),
]


def count_choices_in_content(content: str) -> int:
    """Extract model choices from argparse. Returns 0 if no choices found."""
    max_choices = 0
    for m in re.finditer(r"choices\s*=\s*\[(.*?)\]", content, re.DOTALL):
        inner = m.group(1)
        strings = re.findall(r'["\']([^"\']*)["\']', inner)
        if strings:
            start = max(0, m.start() - 500)
            block = content[start : m.start()]
            if "model_name" in block or "model_id" in block or 'dest="model_name"' in block:
                max_choices = max(max_choices, len(strings))
    return max_choices


def count_preset_yaml(compile_path: Path) -> int | None:
    """If configs/preset.yaml exists alongside compile.py, return compile preset count."""
    preset_path = compile_path.parent / "configs" / "preset.yaml"
    if not preset_path.is_file():
        return None
    try:
        if yaml:
            data = yaml.safe_load(preset_path.read_text()) or {}
            compile_group = data.get("compile", {})
            return len(compile_group) if isinstance(compile_group, dict) else None
        # Fallback: count top-level keys under "compile:" section
        text = preset_path.read_text()
        if m := re.search(r"^compile:\s*\n(.*?)(?=\n\w|\Z)", text, re.DOTALL):
            block = m.group(1)
            keys = re.findall(r"^\s{2}(\w+):", block, re.MULTILINE)
            return len(keys) if keys else None
    except Exception:
        pass
    return None


def count_variants_in_file(path: Path, entry_name: str) -> int:
    """Extract model choices from entry file. Returns 1 if no choices found."""
    content = path.read_text(errors="ignore")

    # preset.yaml overrides (for cosmos_transfer1 etc.)
    if entry_name == "compile.py":
        preset_count = count_preset_yaml(path)
        if preset_count is not None:
            return preset_count

    n = count_choices_in_content(content)
    return n if n > 0 else 1


def count_framework(fw_dir: str, entry_name: str) -> int:
    """Return total model variants (including choices)."""
    path = ROOT / fw_dir
    if not path.exists():
        return 0
    total = 0
    for entry in path.rglob(entry_name):
        if any(ex in str(entry) for ex in EXCLUDE):
            continue
        total += count_variants_in_file(entry, entry_name)
    return total


def main():
    counts = {}
    for fw_dir, entry_name in FRAMEWORKS:
        n = count_framework(fw_dir, entry_name)
        # Merge pytorch_dynamo into pytorch for display
        key = "pytorch" if fw_dir == "pytorch_dynamo" else fw_dir
        counts[key] = counts.get(key, 0) + n

    for fw, n in counts.items():
        print(f"{fw}: {n}")
    if counts:
        print(f"total: {sum(counts.values())}")
    return counts


if __name__ == "__main__":
    main()
