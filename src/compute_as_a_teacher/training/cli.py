"""CLI for planning and launching the MATH-500 CaT training run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from compute_as_a_teacher.data.math500 import DatasetPreparationError, load_locked_questions
from compute_as_a_teacher.evaluation.errors import EvaluationError
from compute_as_a_teacher.evaluation.prompts import load_prompt

from .config import load_training_config
from .checkpoints import register_final_checkpoint
from .eval_handoff import write_eval_handoff
from .errors import TrainingError
from .experiment_registry import (
    verify_final_experiment_registry,
    write_experiment_preregistration,
    write_final_experiment_registry,
)
from .launch_approval import (
    build_launch_evidence,
    verify_launch_approval,
    write_launch_approval,
)
from .planning import (
    TRAINING_DATA_NAME,
    build_training_rows,
    dry_run_summary,
    load_training_plan,
    write_training_plan,
)
from .preflight import (
    discover_model_identity,
    discover_runtime_identity,
    hash_model_snapshot_tree,
    run_preflight,
    write_preflight_receipt,
)
from .qualification import (
    QUALIFICATION_PROFILES,
    launch_qualification,
    load_qualification_plan,
    write_qualification_plan,
)
from .verl_adapter import (
    checkpointed_step,
    exclusive_launch,
    exclusive_launches,
    launch_verl,
    merge_command,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def _repo_path(path: Path) -> Path:
    return (path if path.is_absolute() else REPOSITORY_ROOT / path).resolve()


def _print(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def _require_operational_preflight(receipt: Any) -> None:
    if (
        not isinstance(receipt, dict)
        or receipt.get("operationally_ready_to_launch") is not True
        or receipt.get("missing_gates") != []
    ):
        raise TrainingError("Complete operational preflight gates are required")


def _prepare(args: argparse.Namespace) -> dict[str, Any]:
    config = load_training_config(
        _repo_path(args.config),
        allow_unresolved=args.dry_run,
    )
    questions_path = _repo_path(Path(config.questions_path))
    lock_path = _repo_path(Path(config.dataset_lock_path))
    questions = load_locked_questions(questions_path, lock_path=lock_path)
    raw_prompt = load_prompt(REPOSITORY_ROOT, config.rollouts.prompt)
    synthesis_prompt = load_prompt(REPOSITORY_ROOT, config.synthesis.prompt)
    if args.dry_run:
        build_training_rows(questions, config, raw_prompt)
        return dry_run_summary(config, len(questions))
    run_dir = _repo_path(args.run_dir)
    manifest = write_training_plan(
        run_dir,
        questions,
        config,
        raw_prompt,
        synthesis_prompt,
        questions_path=questions_path,
        dataset_lock_path=lock_path,
        repository_root=REPOSITORY_ROOT,
        force=args.force,
    )
    return {
        "mode": "plan_written",
        "run_dir": str(run_dir),
        "plan_fingerprint": manifest["plan_fingerprint"],
        "counts": manifest["counts"],
        "training_data": str(run_dir / TRAINING_DATA_NAME),
        "model_loaded": False,
        "labels_loaded": False,
        "framework_imported": False,
    }


def _inspect(args: argparse.Namespace) -> dict[str, Any]:
    manifest, command = load_training_plan(_repo_path(args.run_dir))
    return {
        "run_name": manifest["run_name"],
        "protocol_version": manifest["protocol_version"],
        "plan_fingerprint": manifest["plan_fingerprint"],
        "counts": manifest["counts"],
        "label_firewall": manifest["label_firewall"],
        "framework_revision": command.framework_revision,
        "command_fingerprint": command.fingerprint,
    }


def _launch(args: argparse.Namespace) -> dict[str, Any]:
    config = load_training_config(_repo_path(args.config))
    run_dir = _repo_path(args.run_dir)
    manifest, command = load_training_plan(run_dir)
    if manifest["config_fingerprint"] != config.fingerprint:
        raise TrainingError("Config no longer matches the prepared training plan")
    completed_step = checkpointed_step(
        run_dir,
        config.grpo.max_steps,
        expected_world_size=config.runtime.nodes * config.runtime.gpus_per_node,
    )
    if completed_step == config.grpo.max_steps:
        return {
            "mode": "training_already_complete",
            "completed_step": completed_step,
            "would_execute": False,
        }
    if not args.execute:
        return {
            "mode": "launch_preflight",
            "would_execute": False,
            "argv": list(command.argv),
            "cwd": command.cwd,
            "environment": dict(command.environment),
            "checkpointed_step": completed_step,
            "note": "Add --execute only in the prepared verl/GPU environment.",
        }
    if args.preregistration is None or args.launch_approval is None:
        raise TrainingError(
            "Canonical execution requires --preregistration and --launch-approval"
        )
    with exclusive_launch(run_dir) as lease:
        manifest, command = load_training_plan(run_dir)
        if manifest["config_fingerprint"] != config.fingerprint:
            raise TrainingError("Config no longer matches the prepared training plan")
        locked_step = checkpointed_step(
            run_dir,
            config.grpo.max_steps,
            expected_world_size=config.runtime.nodes * config.runtime.gpus_per_node,
        )
        if locked_step == config.grpo.max_steps:
            return {
                "mode": "training_already_complete",
                "completed_step": locked_step,
                "would_execute": False,
            }
        approval = verify_launch_approval(
            _repo_path(args.launch_approval),
            _repo_path(args.preregistration),
            run_dir,
        )
        receipt = run_preflight(
            config,
            manifest,
            command,
            run_dir,
            REPOSITORY_ROOT,
            hash_model=True,
            check_anchor=True,
        )
        _require_operational_preflight(receipt)
        write_preflight_receipt(run_dir, receipt, force=True)
        return_code = launch_verl(command, config, run_dir, lease=lease)
        if return_code:
            raise TrainingError(f"verl exited with status {return_code}")
        terminal_step = checkpointed_step(
            run_dir,
            config.grpo.max_steps,
            expected_world_size=config.runtime.nodes * config.runtime.gpus_per_node,
        )
        if terminal_step != config.grpo.max_steps:
            raise TrainingError(
                "verl exited successfully without the fixed terminal checkpoint"
            )
    return {
        "mode": "training_process_finished",
        "return_code": return_code,
        "terminal_checkpoint_verified": True,
        "launch_approval_fingerprint": approval["launch_approval_fingerprint"],
    }


def _merge(args: argparse.Namespace) -> dict[str, Any]:
    config = load_training_config(_repo_path(args.config))
    manifest, _ = load_training_plan(_repo_path(args.run_dir))
    if manifest["config_fingerprint"] != config.fingerprint:
        raise TrainingError("Config no longer matches the prepared training plan")
    argv = merge_command(
        config,
        run_dir=_repo_path(args.run_dir),
        export_directory=_repo_path(args.export_dir),
    )
    return {
        "mode": "merge_command",
        "would_execute": False,
        "argv": list(argv),
        "note": "Run this after fixed-step training; it is not executed by this command.",
    }


def _register_checkpoint(args: argparse.Namespace) -> dict[str, Any]:
    manifest, completion = register_final_checkpoint(
        _repo_path(args.run_dir),
        _repo_path(args.export_dir),
        force=args.force,
    )
    return {
        "mode": "checkpoint_registered",
        "step": manifest["step"],
        "export_tree_sha256": manifest["export"]["tree_sha256"],
        "reportable": completion["reportable"],
        "non_reportable_reasons": completion["non_reportable_reasons"],
    }


def _plan_eval(args: argparse.Namespace) -> dict[str, Any]:
    handoff = write_eval_handoff(
        _repo_path(args.run_dir),
        _repo_path(args.output_dir),
        args.served_model,
        force=args.force,
    )
    return {
        "mode": "trained_eval_ready",
        "checkpoint_tree_sha256": handoff["checkpoint_tree_sha256"],
        "raw_config": handoff["raw_config"]["path"],
        "synthesis_config": handoff["synthesis_config"]["path"],
        "labels_loaded": False,
        "reportable": handoff["reportable"],
    }


def _prepare_qualification(args: argparse.Namespace) -> dict[str, Any]:
    manifest = write_qualification_plan(
        _repo_path(args.run_dir),
        _repo_path(args.qualification_dir),
        args.profile,
        force=args.force,
    )
    return {
        "mode": "qualification_plan_written",
        "qualification_dir": str(_repo_path(args.qualification_dir)),
        "profile": manifest["profile"],
        "counts": manifest["counts"],
        "qualification_fingerprint": manifest["qualification_fingerprint"],
        "reportable": False,
        "would_execute": False,
    }


def _inspect_qualification(args: argparse.Namespace) -> dict[str, Any]:
    manifest, command = load_qualification_plan(_repo_path(args.qualification_dir))
    return {
        "mode": "qualification_plan_verified",
        "profile": manifest["profile"],
        "counts": manifest["counts"],
        "source_plan_fingerprint": manifest["source"]["plan_fingerprint"],
        "qualification_fingerprint": manifest["qualification_fingerprint"],
        "command_fingerprint": command.fingerprint,
        "reportable": False,
    }


def _launch_qualification(args: argparse.Namespace) -> dict[str, Any]:
    config = load_training_config(_repo_path(args.config))
    qualification_dir = _repo_path(args.qualification_dir)
    manifest, command = load_qualification_plan(qualification_dir)
    if manifest["source"]["config_fingerprint"] != config.fingerprint:
        raise TrainingError("Config no longer matches the qualification source plan")
    profile = QUALIFICATION_PROFILES[manifest["profile"]["name"]]
    completed_step = checkpointed_step(
        qualification_dir,
        profile.max_steps,
        expected_world_size=config.runtime.nodes * config.runtime.gpus_per_node,
    )
    if completed_step == profile.max_steps:
        return {
            "mode": "qualification_already_complete",
            "profile": profile.name,
            "completed_step": completed_step,
            "reportable": False,
            "would_execute": False,
        }
    if not args.execute:
        return {
            "mode": "qualification_launch_preflight",
            "profile": profile.name,
            "argv": list(command.argv),
            "cwd": command.cwd,
            "environment": dict(command.environment),
            "checkpointed_step": completed_step,
            "reportable": False,
            "would_execute": False,
        }
    source_run_dir = Path(manifest["source"]["run_dir"]).resolve()
    with exclusive_launches((source_run_dir, qualification_dir)) as leases:
        manifest, command = load_qualification_plan(qualification_dir)
        locked_source_run_dir = Path(manifest["source"]["run_dir"]).resolve()
        if locked_source_run_dir != source_run_dir:
            raise TrainingError("Qualification source changed during lock acquisition")
        if manifest["source"]["config_fingerprint"] != config.fingerprint:
            raise TrainingError("Config no longer matches the qualification source plan")
        profile = QUALIFICATION_PROFILES[manifest["profile"]["name"]]
        locked_step = checkpointed_step(
            qualification_dir,
            profile.max_steps,
            expected_world_size=config.runtime.nodes * config.runtime.gpus_per_node,
        )
        if locked_step == profile.max_steps:
            return {
                "mode": "qualification_already_complete",
                "profile": profile.name,
                "completed_step": locked_step,
                "reportable": False,
                "would_execute": False,
            }
        source_manifest, _ = load_training_plan(source_run_dir)
        receipt = run_preflight(
            config,
            source_manifest,
            command,
            qualification_dir,
            REPOSITORY_ROOT,
            hash_model=True,
            check_anchor=True,
            qualification_lineage={
                "profile": manifest["profile"],
                "qualification_fingerprint": manifest["qualification_fingerprint"],
                "training_data_sha256": manifest["artifacts"]["training_data"]["sha256"],
                "command_sha256": manifest["artifacts"]["verl_command"]["sha256"],
            },
        )
        _require_operational_preflight(receipt)
        write_preflight_receipt(qualification_dir, receipt, force=True)
        return_code = launch_qualification(
            qualification_dir,
            command,
            config,
            lease=leases[qualification_dir],
            _locked_source_run_dir=source_run_dir,
        )
        if return_code:
            raise TrainingError(
                f"verl qualification run exited with status {return_code}"
            )
        terminal_step = checkpointed_step(
            qualification_dir,
            profile.max_steps,
            expected_world_size=config.runtime.nodes * config.runtime.gpus_per_node,
        )
        if terminal_step != profile.max_steps:
            raise TrainingError(
                "verl qualification exited successfully without its terminal checkpoint"
            )
    return {
        "mode": "qualification_process_finished",
        "profile": profile.name,
        "return_code": return_code,
        "terminal_checkpoint_verified": True,
        "qualified": False,
        "manual_gates_required": [
            "finite loss, KL, and gradient metrics",
            "valid reward and extraction distribution",
            "anchor latency and error rate",
            "throughput, memory, checkpoint time, and projected cost",
        ],
        "reportable": False,
    }


def _preflight(args: argparse.Namespace) -> dict[str, Any]:
    config = load_training_config(_repo_path(args.config))
    run_dir = _repo_path(args.run_dir)
    manifest, command = load_training_plan(run_dir)
    if manifest["config_fingerprint"] != config.fingerprint:
        raise TrainingError("Config no longer matches the prepared training plan")
    with exclusive_launch(run_dir):
        manifest, command = load_training_plan(run_dir)
        if manifest["config_fingerprint"] != config.fingerprint:
            raise TrainingError("Config no longer matches the prepared training plan")
        receipt = run_preflight(
            config,
            manifest,
            command,
            run_dir,
            REPOSITORY_ROOT,
            hash_model=args.hash_model,
            check_anchor=args.check_anchor,
        )
        output = None
        if args.write:
            output = str(
                write_preflight_receipt(run_dir, receipt, force=args.force)
            )
    return {
        "mode": "training_preflight_complete",
        "operationally_ready_to_launch": receipt[
            "operationally_ready_to_launch"
        ],
        "scientifically_attested": receipt["scientifically_attested"],
        "missing_gates": receipt["missing_gates"],
        "preflight_fingerprint": receipt["preflight_fingerprint"],
        "receipt": output,
        "model_loaded": False,
        "anchor_called": args.check_anchor,
    }


def _model_identity(args: argparse.Namespace) -> dict[str, Any]:
    identity = discover_model_identity(args.model_path)
    return {
        "mode": "model_identity_discovered",
        "path": identity["path"],
        "files": identity["files"],
        "bytes": identity["bytes"],
        "snapshot_revision": identity["snapshot_revision"],
        "chat_template_sha256": identity["chat_template_sha256"],
        "model_snapshot_tree_sha256": identity["tree_sha256"],
        "model_loaded": False,
    }


def _runtime_identity(args: argparse.Namespace) -> dict[str, Any]:
    identity = discover_runtime_identity(
        args.python,
        args.verl_source,
        REPOSITORY_ROOT,
    )
    return {
        "mode": "runtime_identity_discovered",
        "python": identity.get("python"),
        "executable": identity.get("executable"),
        "platform": identity.get("platform"),
        "packages": identity.get("packages"),
        "package_count": identity.get("package_count"),
        "package_inventory_sha256": identity.get("package_inventory_sha256"),
        "trainer_image_digest": identity.get("trainer_image_digest"),
        "torch_cuda_version": identity.get("torch_cuda_version"),
        "cudnn_version": identity.get("cudnn_version"),
        "nccl_version": identity.get("nccl_version"),
        "gpus": identity.get("gpus"),
        "actor_optimizer": identity.get("actor_optimizer"),
        "custom_modules": identity.get("custom_modules"),
        "model_loaded": False,
    }


def _preregister_experiment(args: argparse.Namespace) -> dict[str, Any]:
    value = write_experiment_preregistration(
        _repo_path(args.output),
        _repo_path(args.initial_raw_run_dir),
        _repo_path(args.initial_synthesis_config),
        _repo_path(args.training_run_dir),
        force=args.force,
    )
    return {
        "mode": "experiment_preregistered",
        "output": str(_repo_path(args.output)),
        "preregistration_fingerprint": value["preregistration_fingerprint"],
        "results_included": value["results_included"],
        "labels_loaded": value["labels_loaded"],
        "scientifically_attested": value["scientifically_attested"],
    }


def _finalize_experiment(args: argparse.Namespace) -> dict[str, Any]:
    value = write_final_experiment_registry(
        _repo_path(args.output),
        _repo_path(args.preregistration),
        _repo_path(args.initial_raw_run_dir),
        _repo_path(args.initial_synthesis_run_dir),
        _repo_path(args.training_run_dir),
        _repo_path(args.trained_raw_run_dir),
        _repo_path(args.trained_synthesis_run_dir),
        force=args.force,
    )
    return {
        "mode": "experiment_registry_finalized",
        "output": str(_repo_path(args.output)),
        "registry_fingerprint": value["registry_fingerprint"],
        "results_included": value["results_included"],
        "labels_loaded": value["labels_loaded"],
        "scientifically_attested": value["scientifically_attested"],
    }


def _verify_experiment(args: argparse.Namespace) -> dict[str, Any]:
    value = verify_final_experiment_registry(
        _repo_path(args.registry),
        _repo_path(args.preregistration),
        _repo_path(args.initial_raw_run_dir),
        _repo_path(args.initial_synthesis_run_dir),
        _repo_path(args.training_run_dir),
        _repo_path(args.trained_raw_run_dir),
        _repo_path(args.trained_synthesis_run_dir),
    )
    return {
        "mode": "experiment_registry_verified",
        "registry_fingerprint": value["registry_fingerprint"],
        "results_included": value["results_included"],
        "labels_loaded": value["labels_loaded"],
        "scientifically_attested": value["scientifically_attested"],
    }


def _write_launch_approval(args: argparse.Namespace) -> dict[str, Any]:
    value = write_launch_approval(
        _repo_path(args.output),
        _repo_path(args.preregistration),
        _repo_path(args.training_run_dir),
        _repo_path(args.one_step_dir),
        _repo_path(args.resume_three_step_dir),
        _repo_path(args.full_shape_five_step_dir),
        _repo_path(args.manual_attestation),
        force=args.force,
    )
    return {
        "mode": "canonical_launch_approval_written",
        "output": str(_repo_path(args.output)),
        "launch_approval_fingerprint": value["launch_approval_fingerprint"],
        "approved_for_canonical_launch": value["approved_for_canonical_launch"],
        "scientifically_attested": value["scientifically_attested"],
    }


def _inspect_launch_evidence(args: argparse.Namespace) -> dict[str, Any]:
    training_run_dir = _repo_path(args.training_run_dir)
    with exclusive_launch(training_run_dir):
        value = build_launch_evidence(
            _repo_path(args.preregistration),
            training_run_dir,
            _repo_path(args.one_step_dir),
            _repo_path(args.resume_three_step_dir),
            _repo_path(args.full_shape_five_step_dir),
        )
    return {
        "mode": "canonical_launch_evidence_verified",
        "reviewed_evidence_fingerprint": value[
            "reviewed_evidence_fingerprint"
        ],
        "qualification_profiles": sorted(value["qualifications"]),
    }


def _inspect_launch_approval(args: argparse.Namespace) -> dict[str, Any]:
    training_run_dir = _repo_path(args.training_run_dir)
    with exclusive_launch(training_run_dir):
        value = verify_launch_approval(
            _repo_path(args.launch_approval),
            _repo_path(args.preregistration),
            training_run_dir,
        )
    return {
        "mode": "canonical_launch_approval_verified",
        "launch_approval_fingerprint": value["launch_approval_fingerprint"],
        "approved_for_canonical_launch": value["approved_for_canonical_launch"],
        "scientifically_attested": value["scientifically_attested"],
        "qualification_profiles": sorted(value["qualifications"]),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare and launch reference-free CaT GRPO training on MATH-500."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Build the locked label-free verl input and launch plan.")
    prepare.add_argument("--config", type=Path, required=True)
    prepare.add_argument("--run-dir", type=Path, required=True)
    prepare.add_argument("--dry-run", action="store_true")
    prepare.add_argument("--force", action="store_true")

    inspect = commands.add_parser("inspect", help="Verify and summarize a prepared training plan.")
    inspect.add_argument("--run-dir", type=Path, required=True)

    launch = commands.add_parser("launch", help="Show or explicitly execute the pinned verl command.")
    launch.add_argument("--config", type=Path, required=True)
    launch.add_argument("--run-dir", type=Path, required=True)
    launch.add_argument("--execute", action="store_true")
    launch.add_argument("--preregistration", type=Path)
    launch.add_argument("--launch-approval", type=Path)

    merge = commands.add_parser("merge-command", help="Print the pinned verl final-checkpoint export command.")
    merge.add_argument("--config", type=Path, required=True)
    merge.add_argument("--run-dir", type=Path, required=True)
    merge.add_argument("--export-dir", type=Path, required=True)

    register = commands.add_parser(
        "register-checkpoint",
        help="Content-address the merged fixed final checkpoint.",
    )
    register.add_argument("--run-dir", type=Path, required=True)
    register.add_argument("--export-dir", type=Path, required=True)
    register.add_argument("--force", action="store_true")

    plan_eval = commands.add_parser(
        "plan-trained-eval",
        help="Create raw and synthesis eval configs for the registered checkpoint.",
    )
    plan_eval.add_argument("--run-dir", type=Path, required=True)
    plan_eval.add_argument("--output-dir", type=Path, required=True)
    plan_eval.add_argument("--served-model", required=True)
    plan_eval.add_argument("--force", action="store_true")

    prepare_qualification = commands.add_parser(
        "prepare-qualification",
        help="Derive a locked, non-reportable smoke or scale qualification plan.",
    )
    prepare_qualification.add_argument("--run-dir", type=Path, required=True)
    prepare_qualification.add_argument(
        "--qualification-dir", type=Path, required=True
    )
    prepare_qualification.add_argument(
        "--profile", choices=sorted(QUALIFICATION_PROFILES), required=True
    )
    prepare_qualification.add_argument("--force", action="store_true")

    inspect_qualification = commands.add_parser(
        "inspect-qualification",
        help="Verify a derived qualification plan without loading a model.",
    )
    inspect_qualification.add_argument(
        "--qualification-dir", type=Path, required=True
    )

    launch_qualification_parser = commands.add_parser(
        "launch-qualification",
        help="Show or explicitly execute a non-reportable qualification command.",
    )
    launch_qualification_parser.add_argument("--config", type=Path, required=True)
    launch_qualification_parser.add_argument(
        "--qualification-dir", type=Path, required=True
    )
    launch_qualification_parser.add_argument("--execute", action="store_true")

    preflight = commands.add_parser(
        "preflight",
        help="Check the target model, tokenizer, Verl composition, runtime, and anchor.",
    )
    preflight.add_argument("--config", type=Path, required=True)
    preflight.add_argument("--run-dir", type=Path, required=True)
    preflight.add_argument("--hash-model", action="store_true")
    preflight.add_argument("--check-anchor", action="store_true")
    preflight.add_argument("--write", action="store_true")
    preflight.add_argument("--force", action="store_true")

    model_identity = commands.add_parser(
        "model-identity",
        help="Hash a local model snapshot without loading its tensors.",
    )
    model_identity.add_argument("--model-path", type=Path, required=True)

    runtime_identity = commands.add_parser(
        "runtime-identity",
        help="Discover target package, CUDA, and GPU identity before pinning config.",
    )
    runtime_identity.add_argument("--python", type=Path, required=True)
    runtime_identity.add_argument("--verl-source", type=Path, required=True)

    preregister = commands.add_parser(
        "preregister-experiment",
        help="Freeze initial evaluation and canonical training lineage before results.",
    )
    preregister.add_argument("--output", type=Path, required=True)
    preregister.add_argument("--initial-raw-run-dir", type=Path, required=True)
    preregister.add_argument("--initial-synthesis-config", type=Path, required=True)
    preregister.add_argument("--training-run-dir", type=Path, required=True)
    preregister.add_argument("--force", action="store_true")

    def add_registry_inputs(command: argparse.ArgumentParser) -> None:
        command.add_argument("--preregistration", type=Path, required=True)
        command.add_argument("--initial-raw-run-dir", type=Path, required=True)
        command.add_argument("--initial-synthesis-run-dir", type=Path, required=True)
        command.add_argument("--training-run-dir", type=Path, required=True)
        command.add_argument("--trained-raw-run-dir", type=Path, required=True)
        command.add_argument("--trained-synthesis-run-dir", type=Path, required=True)

    finalize = commands.add_parser(
        "finalize-experiment",
        help="Join the fixed checkpoint and final evaluation plans to preregistration.",
    )
    finalize.add_argument("--output", type=Path, required=True)
    add_registry_inputs(finalize)
    finalize.add_argument("--force", action="store_true")

    verify = commands.add_parser(
        "verify-experiment",
        help="Revalidate a finalized experiment registry against every stage.",
    )
    verify.add_argument("--registry", type=Path, required=True)
    add_registry_inputs(verify)

    def add_launch_evidence_inputs(command: argparse.ArgumentParser) -> None:
        command.add_argument("--preregistration", type=Path, required=True)
        command.add_argument("--training-run-dir", type=Path, required=True)
        command.add_argument("--one-step-dir", type=Path, required=True)
        command.add_argument("--resume-three-step-dir", type=Path, required=True)
        command.add_argument("--full-shape-five-step-dir", type=Path, required=True)

    evidence = commands.add_parser(
        "inspect-launch-evidence",
        help="Verify qualification evidence and print its review fingerprint.",
    )
    add_launch_evidence_inputs(evidence)

    approve = commands.add_parser(
        "write-launch-approval",
        help="Bind completed qualifications and manual attestations for launch.",
    )
    approve.add_argument("--output", type=Path, required=True)
    add_launch_evidence_inputs(approve)
    approve.add_argument("--manual-attestation", type=Path, required=True)
    approve.add_argument("--force", action="store_true")

    inspect_approval = commands.add_parser(
        "inspect-launch-approval",
        help="Reverify a canonical launch approval and all bound evidence.",
    )
    inspect_approval.add_argument("--launch-approval", type=Path, required=True)
    inspect_approval.add_argument("--preregistration", type=Path, required=True)
    inspect_approval.add_argument("--training-run-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            result = _prepare(args)
        elif args.command == "inspect":
            result = _inspect(args)
        elif args.command == "launch":
            result = _launch(args)
        elif args.command == "merge-command":
            result = _merge(args)
        elif args.command == "register-checkpoint":
            result = _register_checkpoint(args)
        elif args.command == "plan-trained-eval":
            result = _plan_eval(args)
        elif args.command == "prepare-qualification":
            result = _prepare_qualification(args)
        elif args.command == "inspect-qualification":
            result = _inspect_qualification(args)
        elif args.command == "launch-qualification":
            result = _launch_qualification(args)
        elif args.command == "preflight":
            result = _preflight(args)
        elif args.command == "model-identity":
            result = _model_identity(args)
        elif args.command == "runtime-identity":
            result = _runtime_identity(args)
        elif args.command == "preregister-experiment":
            result = _preregister_experiment(args)
        elif args.command == "finalize-experiment":
            result = _finalize_experiment(args)
        elif args.command == "verify-experiment":
            result = _verify_experiment(args)
        elif args.command == "inspect-launch-evidence":
            result = _inspect_launch_evidence(args)
        elif args.command == "write-launch-approval":
            result = _write_launch_approval(args)
        elif args.command == "inspect-launch-approval":
            result = _inspect_launch_approval(args)
        else:
            parser.error(f"Unsupported command: {args.command}")
    except (TrainingError, EvaluationError, DatasetPreparationError) as exc:
        parser.error(str(exc))
    _print(result)
    return 0
