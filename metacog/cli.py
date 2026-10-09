"""Small stable CLI for registries, preparation, and contract validation."""

from __future__ import annotations

import argparse
import json
from typing import Sequence

from metacog.datasets import get_dataset, list_datasets
from metacog.datasets.mathqa import prepare_mathqa
from metacog.datasets.ultrachat import prepare_ultrachat
from metacog.models import get_model, list_models


def _model_command(args: argparse.Namespace) -> None:
    if args.model_action == "list":
        print(
            json.dumps(
                [
                    {
                        "name": model.name,
                        "path": model.resolved_path,
                        "dtype": model.dtype,
                        "extraction_batch_size": model.extraction_batch_size,
                        "trust_remote_code": model.trust_remote_code,
                    }
                    for model in list_models(args.registry)
                ],
                indent=2,
            )
        )
        return
    print(get_model(args.profile, args.registry).field(args.field))


def _dataset_command(args: argparse.Namespace) -> None:
    if args.dataset_action == "list":
        print(
            json.dumps(
                [
                    {
                        "name": dataset.name,
                        "metric_profile": dataset.metric_profile,
                        "answer_extraction": dataset.answer_extraction,
                    }
                    for dataset in list_datasets()
                ],
                indent=2,
            )
        )
        return
    if args.dataset_action == "validate":
        report = get_dataset(args.profile).validate(args.path, args.min_samples)
        print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
        return
    if args.dataset_action == "get":
        dataset = get_dataset(args.profile)
        print(getattr(dataset, args.field))
        return
    if args.profile == "mathqa":
        if len(args.input) != 1:
            raise ValueError("MathQA preparation accepts exactly one input directory.")
        manifest = prepare_mathqa(
            input_dir=args.input[0],
            output_dir=args.output,
            split_mode=args.split_mode,
            train_size=args.train_size,
            validation_size=args.validation_size,
            seed=args.seed,
        )
    else:
        manifest = prepare_ultrachat(
            inputs=args.input,
            output=args.output,
            max_samples=args.max_samples,
            seed=args.seed,
        )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="metacog")
    parser.add_argument("--version", action="version", version="%(prog)s 0.1.0")
    subparsers = parser.add_subparsers(dest="command", required=True)

    model = subparsers.add_parser("model", help="Inspect declarative model profiles.")
    model_sub = model.add_subparsers(dest="model_action", required=True)
    model_list = model_sub.add_parser("list")
    model_list.add_argument("--registry", default=None)
    model_list.set_defaults(handler=_model_command)
    model_get = model_sub.add_parser("get")
    model_get.add_argument("profile")
    model_get.add_argument(
        "field",
        choices=[
            "path",
            "dtype",
            "extraction_batch_size",
            "trust_remote_code",
            "chat_template_enable_thinking",
        ],
    )
    model_get.add_argument("--registry", default=None)
    model_get.set_defaults(handler=_model_command)

    dataset = subparsers.add_parser("dataset", help="Prepare or validate datasets.")
    dataset_sub = dataset.add_subparsers(dest="dataset_action", required=True)
    dataset_list = dataset_sub.add_parser("list")
    dataset_list.set_defaults(handler=_dataset_command)
    validate = dataset_sub.add_parser("validate")
    validate.add_argument("profile", choices=[item.name for item in list_datasets()])
    validate.add_argument("path")
    validate.add_argument("--min-samples", type=int, default=0)
    validate.set_defaults(handler=_dataset_command)
    dataset_get = dataset_sub.add_parser("get")
    dataset_get.add_argument("profile", choices=[item.name for item in list_datasets()])
    dataset_get.add_argument("field", choices=["metric_profile", "answer_extraction"])
    dataset_get.set_defaults(handler=_dataset_command)
    prepare = dataset_sub.add_parser("prepare")
    prepare.add_argument("profile", choices=["mathqa", "ultrachat"])
    prepare.add_argument("--input", nargs="+", required=True)
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--max-samples", type=int, default=50000)
    prepare.add_argument("--seed", type=int, default=42)
    prepare.add_argument("--split-mode", choices=["official", "resplit"], default="resplit")
    prepare.add_argument("--train-size", type=int, default=22000)
    prepare.add_argument("--validation-size", type=int, default=7000)
    prepare.set_defaults(handler=_dataset_command)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.handler(args)


if __name__ == "__main__":
    main()
