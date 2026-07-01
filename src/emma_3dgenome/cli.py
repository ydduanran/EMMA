from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .config import PRESETS
from .io import get_chrom_info, load_contact_matrix
from .logging_utils import write_json
from .masks import detect_missing_bins
from .restore import NO_MASK_MESSAGE, EmmaRestorer


def _path_or_none(value: str | None) -> Path | None:
    return None if value is None else Path(value)


def _build_restorer(args: argparse.Namespace) -> EmmaRestorer:
    if args.preset not in PRESETS:
        valid = ", ".join(sorted(PRESETS))
        raise ValueError(f"Unknown preset '{args.preset}'. Available presets: {valid}.")
    overrides = {
        "device": args.device,
        "seed": args.seed,
        "max_diag": args.max_diag,
        "min_diag": args.min_diag,
        "max_imfs": args.max_imfs,
        "residual_weight": args.residual_weight,
        "diag_calib_strength": args.diag_calib_strength,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "patch_len": args.patch_len,
        "stride": args.stride,
        "n_samples": args.n_samples,
        "pseudo_mask_ratio": args.pseudo_mask_ratio,
        "base_channels": args.base_channels,
        "depth": args.depth,
        "dropout": args.dropout,
        "num_workers": args.num_workers,
        "prefetch_factor": args.prefetch_factor,
        "inference_batch_size": args.inference_batch_size,
        "recompute_pseudo_emd": args.recompute_pseudo_emd,
        "verbose": args.verbose,
    }
    if args.imf_weights is not None:
        overrides["imf_weights"] = tuple(args.imf_weights)
    return EmmaRestorer(preset=args.preset, **overrides)


def _add_common_matrix_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("path", type=Path, help="Input .cool, .mcool, .npy, or .npz file.")
    parser.add_argument("--chrom", default=None, help="Chromosome for .cool/.mcool input.")
    parser.add_argument("--resolution", type=int, default=None, help="Resolution for .mcool input.")
    parser.add_argument("--start-bin", type=int, default=None, help="Inclusive local bin start for windowed analysis.")
    parser.add_argument("--end-bin", type=int, default=None, help="Exclusive local bin end for windowed analysis.")
    parser.add_argument("--balance", action="store_true", help="Use balanced cooler matrix if available.")
    parser.add_argument("--key", default=None, help="Array key for .npz input.")


def _add_config_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--preset", default="default", choices=sorted(PRESETS))
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-diag", type=int, default=500)
    parser.add_argument("--min-diag", type=int, default=1)
    parser.add_argument("--max-imfs", type=int, default=5)
    parser.add_argument("--imf-weights", type=float, nargs="+", default=None)
    parser.add_argument("--residual-weight", type=float, default=1.0)
    parser.add_argument("--diag-calib-strength", type=float, default=0.20)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patch-len", type=int, default=256)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--n-samples", type=int, default=30000)
    parser.add_argument("--pseudo-mask-ratio", type=float, default=0.15)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--inference-batch-size", type=int, default=256)
    parser.add_argument("--recompute-pseudo-emd", action="store_true")
    parser.add_argument("--verbose", action="store_true")


def _cmd_info(args: argparse.Namespace) -> None:
    payload = get_chrom_info(args.path, resolution=args.resolution)
    if args.chrom is not None and args.path.suffix.lower() in {".cool", ".mcool"}:
        matrix = load_contact_matrix(
            args.path,
            chrom=args.chrom,
            resolution=args.resolution,
            start_bin=args.start_bin,
            end_bin=args.end_bin,
        )
        payload["selected_chrom"] = args.chrom
        payload["selected_shape"] = list(matrix.shape)
    elif args.path.suffix.lower() in {".npy", ".npz"}:
        matrix = load_contact_matrix(args.path, key=args.key, start_bin=args.start_bin, end_bin=args.end_bin)
        payload["shape"] = list(matrix.shape)
    payload["start_bin"] = args.start_bin
    payload["end_bin"] = args.end_bin
    print(json.dumps(payload, indent=2, default=str))


def _cmd_detect(args: argparse.Namespace) -> None:
    matrix = load_contact_matrix(
        args.path,
        chrom=args.chrom,
        resolution=args.resolution,
        balance=args.balance,
        key=args.key,
        start_bin=args.start_bin,
        end_bin=args.end_bin,
    )
    bin_offset = 0 if args.start_bin is None else int(args.start_bin)
    mask_info = detect_missing_bins(
        matrix,
        chrom=args.chrom,
        resolution=args.resolution,
        mode=args.auto_mask_mode,
        max_diag=args.max_diag,
        exclude_bed=_path_or_none(args.exclude_bed),
        bin_offset=bin_offset,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    mask_info.save(args.output, chrom=args.chrom, resolution=args.resolution, bin_offset=bin_offset)
    write_json(
        args.output / "report.json",
        {
            "mode": "detect",
            "input_path": str(args.path),
            "chrom": args.chrom,
            "resolution": args.resolution,
            "start_bin": args.start_bin,
            "end_bin": args.end_bin,
            "window_shape": list(matrix.shape),
            "auto_mask_mode": args.auto_mask_mode,
            "missing_bins": len(mask_info.missing_bins),
            "excluded_bins": len(mask_info.excluded_bins or []),
            "mask_points": int(mask_info.mask.sum()),
        },
    )
    print(f"Saved detection outputs to {args.output}")


def _cmd_restore(args: argparse.Namespace) -> None:
    if args.mask is None and args.mask_regions is None and not args.auto_mask:
        raise ValueError(NO_MASK_MESSAGE)
    restorer = _build_restorer(args)
    restorer.restore_from_file(
        args.path,
        chrom=args.chrom,
        resolution=args.resolution,
        mask=_path_or_none(args.mask),
        mask_regions=_path_or_none(args.mask_regions),
        mask_region_format=args.mask_region_format,
        auto_mask=args.auto_mask,
        exclude_bed=_path_or_none(args.exclude_bed),
        auto_mask_mode=args.auto_mask_mode,
        output_dir=args.output,
        balance=args.balance,
        key=args.key,
        start_bin=args.start_bin,
        end_bin=args.end_bin,
    )
    print(f"Saved restoration outputs to {args.output}")


def _cmd_reconstruct(args: argparse.Namespace) -> None:
    matrix = load_contact_matrix(
        args.path,
        chrom=args.chrom,
        resolution=args.resolution,
        balance=args.balance,
        key=args.key,
        start_bin=args.start_bin,
        end_bin=args.end_bin,
    )
    restorer = _build_restorer(args)
    result = restorer.reconstruct(matrix, mode=args.mode, blend=args.blend)
    if result.report is not None:
        result.report.update({"start_bin": args.start_bin, "end_bin": args.end_bin, "window_shape": list(matrix.shape)})
    result.save(args.output)
    print(f"Saved reconstruction outputs to {args.output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="emma", description="EMMA 3D genome contact-map restoration toolkit.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    info_parser = subparsers.add_parser("info", help="Print input file metadata.")
    _add_common_matrix_args(info_parser)
    info_parser.set_defaults(func=_cmd_info)

    detect_parser = subparsers.add_parser("detect", help="Detect low-coverage or missing bins.")
    _add_common_matrix_args(detect_parser)
    detect_parser.add_argument("--output", "-o", type=Path, required=True)
    detect_parser.add_argument("--auto-mask-mode", default="balanced", choices=["conservative", "balanced", "aggressive"])
    detect_parser.add_argument("--exclude-bed", default=None)
    detect_parser.add_argument("--max-diag", type=int, default=500)
    detect_parser.set_defaults(func=_cmd_detect)

    restore_parser = subparsers.add_parser("restore", help="Restore user-provided or auto-detected missing bins.")
    _add_common_matrix_args(restore_parser)
    restore_parser.add_argument("--output", "-o", type=Path, required=True)
    restore_parser.add_argument("--mask", default=None)
    restore_parser.add_argument("--mask-regions", default=None)
    restore_parser.add_argument("--mask-region-format", default="auto", choices=["auto", "bed", "bin"])
    restore_parser.add_argument("--auto-mask", action="store_true")
    restore_parser.add_argument("--auto-mask-mode", default="balanced", choices=["conservative", "balanced", "aggressive"])
    restore_parser.add_argument("--exclude-bed", default=None)
    _add_config_args(restore_parser)
    restore_parser.set_defaults(func=_cmd_restore)

    reconstruct_parser = subparsers.add_parser("reconstruct", help="Run EMMA-style matrix reconstruction.")
    _add_common_matrix_args(reconstruct_parser)
    reconstruct_parser.add_argument("--output", "-o", type=Path, required=True)
    reconstruct_parser.add_argument("--mode", default="conservative", choices=["conservative", "full"])
    reconstruct_parser.add_argument("--blend", type=float, default=None)
    _add_config_args(reconstruct_parser)
    reconstruct_parser.set_defaults(func=_cmd_reconstruct)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except Exception as exc:
        parser.exit(2, f"emma: error: {exc}\n")
    return 0


def app() -> None:
    raise SystemExit(main())


if __name__ == "__main__":
    app()
