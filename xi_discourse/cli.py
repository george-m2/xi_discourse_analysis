"""Command line entry point: `uv run xi-discourse <command>`."""
from __future__ import annotations

import argparse
import sys

from . import config


def _corpus(force: bool = False):
    from .preprocess import build_corpus
    return build_corpus(force=force)


def cmd_preprocess(args) -> int:
    from dataclasses import asdict
    _, _, stats = _corpus(force=args.force)
    for k, v in asdict(stats).items():
        print(f"{k}: {v}")
    return 0


def cmd_train(args) -> int:
    from .train import train_all
    sentences, _, _ = _corpus()
    models = train_all(sentences, seeds=tuple(args.seeds), force=args.force)
    for s, m in models.items():
        print(f"seed {s}: vocab {len(m.wv):,}")
    return 0


def cmd_diachronic(args) -> int:
    from .diachronic import train_slices, year_slices
    sentences, meta, _ = _corpus()
    slices = year_slices(sentences, meta)
    print(f"{len(slices)} year slices: {sorted(slices)}")
    models = train_slices(slices, force=args.force)
    for y, m in sorted(models.items()):
        print(f"{y}: {len(slices[y]):,} sentences, vocab {len(m.wv):,}")
    return 0


def cmd_analyse(args) -> int:
    from .report import run
    from .train import load_all
    sentences, meta, stats = _corpus()
    models = load_all(seeds=tuple(args.seeds))
    print(f"Loaded {len(models)} model(s): {sorted(models)}")

    year_models = None
    if args.with_diachronic:
        from gensim.models import Word2Vec
        from .train import model_path
        year_models = {}
        for y in config.DIACHRONIC_YEARS:
            p = model_path(config.PRIMARY_SEED, f"year{y}")
            if p.exists():
                year_models[y] = Word2Vec.load(str(p))
        print(f"Loaded {len(year_models)} year models.")
        if not year_models:
            print("  (none found -- run `xi-discourse diachronic` first)")
            year_models = None

    R = run(models, sentences, stats, meta=meta, year_models=year_models)
    print(f"\nWrote {config.RESULTS_DIR / 'FINDINGS.md'}")
    print(f"Tables: {config.TABLES_DIR}")
    print(f"Figures: {config.FIGURES_DIR}")
    return 0


def cmd_all(args) -> int:
    for fn in (cmd_preprocess, cmd_train, cmd_diachronic):
        rc = fn(args)
        if rc:
            return rc
    args.with_diachronic = True
    return cmd_analyse(args)


def main(argv: list[str] | None = None) -> int:
    # Shared options live on a parent parser so they are accepted on both sides
    # of the subcommand: `xi-discourse --force train` and `xi-discourse train
    # --force` both work. Declared only on the top-level parser, the trailing
    # form -- the one anyone types first -- is rejected outright.
    #
    # default=SUPPRESS is load-bearing: a subparser sharing a dest writes its
    # own default over whatever the top-level parser already parsed, so
    # `--force train` would silently come back force=False. Suppressed
    # defaults leave the attribute unset instead, and it is filled in below.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--force", action="store_true",
                        default=argparse.SUPPRESS,
                        help="rebuild cached artefacts")
    # One comma-separated token rather than nargs="+". A variadic option can
    # never precede a subcommand: argparse greedily swallows the subcommand
    # name as another value, leaving `command` unset. A single token is
    # unambiguous in either position.
    common.add_argument("--seeds", metavar="SEEDS",
                        default=argparse.SUPPRESS,
                        help=f"comma-separated model seeds "
                             f"(default: {','.join(map(str, config.SEEDS))})")

    p = argparse.ArgumentParser(prog="xi-discourse", description=__doc__,
                                parents=[common])
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("preprocess", parents=[common], help="build the sentence corpus")
    sub.add_parser("train", parents=[common], help="train one model per seed")
    sub.add_parser("diachronic", parents=[common], help="train per-year models")
    a = sub.add_parser("analyse", parents=[common],
                       help="run analyses, write tables and figures")
    a.add_argument("--with-diachronic", action="store_true")
    sub.add_parser("all", parents=[common],
                   help="preprocess, train, diachronic, analyse")

    args = p.parse_args(argv)
    args.force = getattr(args, "force", False)
    args.with_diachronic = getattr(args, "with_diachronic", False)

    raw = getattr(args, "seeds", None)
    if raw:
        try:
            args.seeds = [int(s) for s in str(raw).split(",") if s]
        except ValueError:
            p.error(f"--seeds expects comma-separated integers, got: {raw!r}")
        if not args.seeds:
            p.error("--seeds was empty")
    else:
        args.seeds = list(config.SEEDS)
    return {
        "preprocess": cmd_preprocess, "train": cmd_train,
        "diachronic": cmd_diachronic, "analyse": cmd_analyse, "all": cmd_all,
    }[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
