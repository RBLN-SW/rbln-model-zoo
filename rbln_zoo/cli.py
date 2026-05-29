#!/usr/bin/env python3
"""RBLN Model Zoo CLI — manage and run models across card types."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import asdict
from difflib import get_close_matches
from enum import Enum
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from rbln_zoo import __version__ as _pkg_version
from rbln_zoo import config as _config
from rbln_zoo.args_inspector import ArgSpec, extract_arg_specs
from rbln_zoo.registry import (
    CLI_SEARCH_ROOTS,
    COMPILE_ONLY_ROOTS,
    COMPILE_PY,
    INFERENCE_PY,
    MAIN_PY,
    REPO_ROOT,
    REQ_FILE,
    RK_CARDS,
    RK_DEFAULT_CARDS,
    STEM_COMPILE,
    STEM_INFERENCE,
    STEM_MAIN,
    ModelEntry,
    discover_models_cached,
    filter_models,
    load_registry,
    resolve_card,
)

console = Console()

_GLOBAL_CTX = {"help_option_names": ["-h", "--help"]}


class _CaseInsensitiveGroup(typer.core.TyperGroup):
    def get_command(self, ctx, name):
        cmd = super().get_command(ctx, name)
        if cmd is not None:
            return cmd
        for cmd_name in self.list_commands(ctx):
            if cmd_name.lower() == name.lower():
                return super().get_command(ctx, cmd_name)
        return None


app = typer.Typer(
    name="rbln-zoo",
    help="RBLN Model Zoo CLI — manage and run models across card types.",
    rich_markup_mode="rich",
    no_args_is_help=True,
    context_settings=_GLOBAL_CTX,
    cls=_CaseInsensitiveGroup,
)


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"rbln-zoo {_pkg_version}")
        raise typer.Exit()


@app.callback()
def _root(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            "-V",
            callback=_version_callback,
            is_eager=True,
            help="Show the rbln-zoo version and exit",
        ),
    ] = False,
) -> None:
    pass


def _suggest(name: str, candidates: list[str]) -> str:
    matches = get_close_matches(
        name.lower(), [c.lower() for c in candidates], n=3, cutoff=0.5
    )
    if not matches:
        return ""
    return f"Did you mean: {', '.join(matches)}?"


_CARD_COLORS: dict[str, str] = {
    "RBLN-CA22": "cyan",
    "RBLN-CA25": "yellow",
}

RefreshOption = Annotated[
    bool,
    typer.Option(
        "--refresh",
        "-r",
        help="Re-scan the tree and refresh the model index cache",
    ),
]


class Step(str, Enum):
    all = "all"
    compile = STEM_COMPILE
    inference = STEM_INFERENCE


def _card_badge(card: str) -> str:
    return f"[bold {_CARD_COLORS.get(card, 'white')}]{card}[/]"


def _models_for_cli(
    *,
    refresh: bool,
    registry: dict | None = None,
    cli_only: bool = True,
) -> list[ModelEntry]:
    if registry is None:
        registry = load_registry()
    msg = (
        "[bold cyan]Scanning repository for models…[/]"
        if refresh
        else "[bold cyan]Loading model index…[/]"
    )
    with console.status(msg, spinner="dots"):
        models = discover_models_cached(registry, refresh=refresh)
    if cli_only:
        models = [m for m in models if m.path.split("/", 1)[0] in CLI_SEARCH_ROOTS]
    return models


def _scripts_label(m: ModelEntry) -> str:
    """User-facing label describing which entry scripts the model ships."""
    if m.has_main:
        return "[green]main[/]"
    if m.has_compile and m.has_inference:
        return "[cyan]compile + inference[/]"
    if m.has_compile:
        return "[yellow]compile[/]"
    return ""


def _model_table(models: list[ModelEntry], title: str = "Models") -> Table:
    table = Table(
        title=title,
        title_style="bold bright_white",
        border_style="dim",
        pad_edge=False,
        show_lines=False,
    )
    table.add_column("#", width=4, justify="right")
    table.add_column("Path", style="bright_white", ratio=3)
    table.add_column("Cards", justify="center", ratio=1)
    table.add_column("Task", style="green", ratio=1)
    table.add_column("Scripts", justify="center", ratio=1)
    for i, m in enumerate(models, 1):
        table.add_row(
            str(i),
            m.path,
            " ".join(_card_badge(c) for c in m.cards),
            m.task,
            _scripts_label(m),
        )
    return table


def _check_card(model_path: str, card: str, models: list[ModelEntry]) -> None:
    if not any(m.path == model_path and card in m.cards for m in models):
        console.print(
            f"[bold red]Error:[/] [cyan]{model_path}[/] "
            f"does not support card [magenta]{card}[/]"
        )
        raise typer.Exit(code=1)


def _env_for_subprocess(*, verbose: bool) -> dict[str, str]:
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    if verbose:
        env.setdefault("RBLN_VERBOSITY", "debug")
        env.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "0")
        env.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    return env


def _run(
    model_path: str,
    script: str,
    *,
    card: str | None = None,
    extra_args: list[str] | None = None,
    verbose: bool = False,
) -> int:
    path = REPO_ROOT / model_path / f"{script}.py"
    if not path.exists():
        console.print(f"[bold red]Error:[/] {path.relative_to(REPO_ROOT)} not found")
        return 1

    cmd = [sys.executable, "-u", str(path)]
    if card:
        cmd += ["--card", card]
    if extra_args:
        cmd += extra_args

    console.print(
        Panel(
            f"[bold]{' '.join(cmd)}[/]\n[dim]cwd: {path.parent}[/]",
            title=f"[yellow]{script}[/]",
            border_style="yellow",
        )
    )
    return subprocess.run(
        cmd,
        cwd=path.parent,
        env=_env_for_subprocess(verbose=verbose),
    ).returncode


def _scripts_for(model_path: str, step: Step) -> list[str]:
    d = REPO_ROOT / model_path

    if (d / MAIN_PY).exists():
        if step in (Step.compile, Step.inference):
            console.print(
                f"[bold yellow]Warning:[/] [cyan]{model_path}[/] uses {MAIN_PY} "
                f"(not separable). Running {MAIN_PY} instead."
            )
        return [STEM_MAIN]

    if step != Step.all:
        if not (d / f"{step.value}.py").exists():
            console.print(f"[bold red]Error:[/] {model_path}/{step.value}.py not found")
            raise typer.Exit(code=1)
        return [step.value]

    scripts = [
        s
        for s, fn in ((STEM_COMPILE, COMPILE_PY), (STEM_INFERENCE, INFERENCE_PY))
        if (d / fn).exists()
    ]
    if not scripts:
        console.print(f"[bold red]Error:[/] No scripts in [cyan]{model_path}[/]")
        raise typer.Exit(code=1)
    return scripts


@app.command("list")
def list_models(
    card: Annotated[
        str | None, typer.Option("--card", "-c", help="Filter by card type")
    ] = None,
    framework: Annotated[
        str | None, typer.Option("--framework", "-f", help="Filter by framework")
    ] = None,
    task: Annotated[
        str | None, typer.Option("--task", "-t", help="Filter by task type")
    ] = None,
    search: Annotated[
        str | None, typer.Option("--search", "-s", help="Search model paths")
    ] = None,
    json_out: Annotated[
        bool, typer.Option("--json", help="Emit results as JSON")
    ] = False,
    refresh: RefreshOption = False,
) -> None:
    """List available models with filtering."""
    registry = load_registry()
    if card:
        # Resolve aliases (e.g. RBLN-CA12 -> CA). Unknown values pass
        # through so the existing difflib typo-hint can suggest a match.
        card = resolve_card(registry, card) or card
    all_models = _models_for_cli(refresh=refresh, registry=registry)
    models = filter_models(
        all_models, card=card, framework=framework, task=task, search=search
    )
    if not models:
        if framework:
            hint = _suggest(framework, sorted({m.framework for m in all_models}))
            if hint:
                console.print(hint)
        if card:
            defined = sorted(load_registry()[RK_CARDS])
            hint = _suggest(card, defined)
            if hint:
                console.print(hint)
        console.print("No models found matching the given filters.")
        raise typer.Exit(code=1)

    if json_out:
        print(json.dumps([asdict(m) for m in models], indent=2))
        return

    filters = {
        k: v
        for k, v in {
            "card": card,
            "framework": framework,
            "task": task,
            "search": search,
        }.items()
        if v
    }
    title = "Models"
    if filters:
        title += f" ({', '.join(f'{k}={v}' for k, v in filters.items())})"

    console.print(_model_table(models, title=title))
    console.print(f"\n  [bold]{len(models)}[/] model(s)")


@app.command()
def cards(
    refresh: RefreshOption = False,
) -> None:
    """Show available card types and model counts."""
    registry = load_registry()
    card_defs, default, models = (
        registry[RK_CARDS],
        registry[RK_DEFAULT_CARDS],
        _models_for_cli(refresh=refresh, registry=registry),
    )
    if not card_defs:
        console.print("[dim]No cards defined in model_registry.yaml[/]")
        return

    table = Table(
        title="Card Types",
        title_style="bold bright_white",
        border_style="dim",
        show_lines=False,
    )
    table.add_column("Card", style="bold")
    table.add_column("Description")
    table.add_column("Default", justify="center")
    table.add_column("Models", justify="right", style="bold")
    for name, info in card_defs.items():
        desc = info.get("description", "") if isinstance(info, dict) else str(info)
        table.add_row(
            _card_badge(name),
            desc,
            "[green]Yes[/]" if name in default else "[dim]—[/]",
            str(sum(1 for m in models if name in m.cards)),
        )
    console.print(table)


@app.command("validate", hidden=True)
def validate(
    card: Annotated[
        str | None,
        typer.Option("--card", "-c", help="Only validate models for this card"),
    ] = None,
    refresh: RefreshOption = False,
) -> None:
    """Internal maintenance audit (hidden from --help).

    Scans every model directory (including compile-only roots like
    cpp/serving/vllm) and reports NO SCRIPTS / AMBIGUOUS errors plus
    NO INFER / NO REQS warnings. Same checks run automatically inside
    ``pytest`` via ``tests/test_model_directory_audit.py`` — this
    command exists for ad-hoc maintainer use with formatted output.
    """
    registry = load_registry()
    if card:
        card = resolve_card(registry, card) or card
        defined = {k.lower() for k in registry[RK_CARDS]}
        if card.lower() not in defined:
            console.print(
                f"[bold red]Error:[/] unknown card [cyan]{card}[/]; "
                f"defined cards: {sorted(defined) or 'none'}"
            )
            raise typer.Exit(code=1)
    models = filter_models(_models_for_cli(refresh=refresh, cli_only=False), card=card)
    errors, warnings = _audit_models(models)

    for label, items in [("Warnings", warnings), ("Errors", errors)]:
        if items:
            color = "yellow" if label == "Warnings" else "red"
            console.print(f"\n[bold {color}]{label} ({len(items)}):[/]")
            for item in items:
                console.print(f"  {item}")

    if errors:
        raise typer.Exit(code=1)
    console.print(
        f"\n[bold green]{len(models)} model(s) checked, {len(warnings)} warning(s).[/]"
    )


def _audit_models(models: list[ModelEntry]) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    for m in models:
        if not (m.has_compile or m.has_inference or m.has_main):
            errors.append(f"[red]NO SCRIPTS[/]  {m.path}")
            continue
        if m.has_main and m.has_compile:
            errors.append(
                f"[red]AMBIGUOUS[/]   {m.path}  (both {MAIN_PY} and {COMPILE_PY})"
            )
            continue
        compile_only = m.has_compile and not m.has_inference and not m.has_main
        if compile_only and not any(
            m.path.startswith(f"{r}/") for r in COMPILE_ONLY_ROOTS
        ):
            warnings.append(f"[yellow]NO INFER[/]    {m.path}")
        if not (REPO_ROOT / m.path / REQ_FILE).exists():
            warnings.append(f"[yellow]NO REQS[/]     {m.path}")
    return errors, warnings


def _resolve_model(
    model: str | None,
    search: str | None,
    card: str | None,
    *,
    models: list[ModelEntry],
) -> str:
    if model:
        return model
    if not search:
        console.print("[bold red]Error:[/] Provide a model path or use -s")
        raise typer.Exit(code=1)

    hits = filter_models(models, card=card, search=search)
    if not hits:
        console.print(f"[bold red]Error:[/] No model matching [cyan]{search}[/]")
        raise typer.Exit(code=1)
    if len(hits) == 1:
        console.print(f"  [dim]resolved →[/] [bold]{hits[0].path}[/]")
        return hits[0].path

    console.print(f"[bold yellow]{len(hits)} matches for[/] [cyan]{search}[/]:\n")
    for i, m in enumerate(hits, 1):
        console.print(f"  [dim]{i}.[/] {m.path}")
    console.print("\n  [dim]Refine with a more specific name.[/]")
    raise typer.Exit(code=1)


def _render_args_table(script: str, specs: list[ArgSpec]) -> Table:
    table = Table(
        title=f"[cyan]{script}.py[/] flags",
        title_style="bold",
        border_style="dim",
    )
    table.add_column("Flag", style="bright_white")
    table.add_column("Type", style="green")
    table.add_column("Default", style="yellow")
    table.add_column("Help")
    for spec in specs:
        flag = ", ".join(spec.flags) + (" [red]*[/]" if spec.required else "")
        type_ = spec.type or spec.action or ""
        default = "" if spec.default is None else repr(spec.default)
        help_ = spec.help or ""
        if spec.choices:
            help_ += f"\n[dim]choices: {', '.join(map(str, spec.choices))}[/]"
        table.add_row(flag, type_, default, help_)
    return table


def _run_live_help(model_path: str, script: str, path: Path) -> None:
    console.print(
        Panel(
            f"[bold]{model_path}/{script}.py --help[/]",
            border_style="cyan",
            title=f"[cyan]{script}[/]",
        )
    )
    subprocess.run(
        [sys.executable, str(path), "--help"],
        cwd=path.parent,
        env=_env_for_subprocess(verbose=False),
    )


@app.command(
    "args",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def show_args(
    ctx: typer.Context,
    model: Annotated[str | None, typer.Argument(help="Model path (or use -s)")] = None,
    search: Annotated[
        str | None, typer.Option("--search", "-s", help="Find model by keyword")
    ] = None,
    step: Annotated[
        Step, typer.Option("--step", "-S", help="Which script(s) to inspect")
    ] = Step.all,
    live: Annotated[
        bool,
        typer.Option(
            "--live",
            help="Run the script with --help instead of AST parsing (slow but exact)",
        ),
    ] = False,
    refresh: RefreshOption = False,
) -> None:
    """Show argparse flags accepted by a model's compile.py / inference.py.

    Extra tokens after ` -- ` are ignored (they are only meaningful for ``exec``).
    """
    if model and model.startswith("-"):
        model = None
    models = _models_for_cli(refresh=refresh)
    resolved = _resolve_model(model, search, None, models=models)

    for script in _scripts_for(resolved, step):
        path = REPO_ROOT / resolved / f"{script}.py"
        if live:
            _run_live_help(resolved, script, path)
            continue

        specs = extract_arg_specs(path)
        if not specs:
            console.print(
                f"[yellow]No argparse flags found in[/] [cyan]{resolved}/{script}.py[/]"
                " — it may not use argparse. Try [bold]--live[/] for exact help."
            )
            continue
        console.print(_render_args_table(script, specs))


@app.command(
    "exec",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def exec_model(
    ctx: typer.Context,
    model: Annotated[str | None, typer.Argument(help="Model path (or use -s)")] = None,
    search: Annotated[
        str | None, typer.Option("--search", "-s", help="Find model by keyword")
    ] = None,
    step: Annotated[
        Step, typer.Option("--step", "-S", help="Which step to run")
    ] = Step.all,
    card: Annotated[
        str | None, typer.Option("--card", "-c", help="Target card")
    ] = None,
    refresh: RefreshOption = False,
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            "-v",
            help="Enable RBLN_VERBOSITY=debug and HF progress bars",
        ),
    ] = False,
) -> None:
    """Execute a model — compile, inference, or both.

    Anything after ` -- ` is forwarded to the underlying compile.py / inference.py:

        rbln-zoo exec -s yolov10 -- --model_name yolov10n
    """
    extra_args = list(ctx.args)
    if model and model.startswith("-"):
        extra_args = [model, *extra_args]
        model = None

    if model and search:
        console.print("[bold red]Error:[/] use either MODEL or -s/--search, not both")
        raise typer.Exit(code=1)

    registry = load_registry()
    if card:
        card = resolve_card(registry, card) or card
    all_models = _models_for_cli(refresh=refresh, registry=registry)
    resolved = _resolve_model(model, search, card, models=all_models)
    if card and model:
        _check_card(resolved, card, all_models)
    for script in _scripts_for(resolved, step):
        ret = _run(
            resolved,
            script,
            card=card,
            extra_args=extra_args,
            verbose=verbose,
        )
        if ret != 0:
            if script == STEM_COMPILE:
                console.print("[bold red]Compile failed, skipping inference.[/]")
            raise typer.Exit(code=ret)


config_app = typer.Typer(
    name="config",
    help="Read/write persistent CLI defaults (e.g. rbln_home).",
    no_args_is_help=True,
)
app.add_typer(config_app, name="config")


def _validate_key(key: str) -> str:
    if key not in _config.ALLOWED_KEYS:
        console.print(
            f"[bold red]Error:[/] unknown key [cyan]{key}[/]; "
            f"allowed: {sorted(_config.ALLOWED_KEYS)}"
        )
        raise typer.Exit(code=1)
    return key


@config_app.command("set")
def config_set(
    key: Annotated[str, typer.Argument(help="Config key (e.g. rbln_home)")],
    value: Annotated[str, typer.Argument(help="Value to store")],
) -> None:
    _validate_key(key)
    _config.set_(key, value)
    console.print(f"[green]set[/] {key} = {value}  [dim]({_config.config_path()})[/]")
    if key == "rbln_home":
        console.print(f'[dim]To apply: [/]export RBLN_HOME="{value}"')


@config_app.command("get")
def config_get(
    key: Annotated[str, typer.Argument(help="Config key")],
) -> None:
    _validate_key(key)
    value = _config.get(key)
    if value is None:
        raise typer.Exit(code=1)
    console.print(value)


@config_app.command("unset")
def config_unset(
    key: Annotated[str, typer.Argument(help="Config key")],
) -> None:
    _validate_key(key)
    if _config.unset(key):
        console.print(f"[green]unset[/] {key}")
    else:
        console.print(f"[yellow]{key}[/] was not set")
        raise typer.Exit(code=1)


@config_app.command("list")
def config_list() -> None:
    data = _config.load()
    for k in sorted(_config.ALLOWED_KEYS):
        if k in data:
            console.print(f"{k} = {data[k]}")
        else:
            console.print(f"{k} = [dim](unset)[/]")


@config_app.command("path")
def config_path_cmd() -> None:
    console.print(str(_config.config_path()))
