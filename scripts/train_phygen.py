#!/usr/bin/env python3
"""Train PhyGen residual-field model with a task adapter."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phygen.adapters.base_adapter import load_records, load_success_demo_keys
from phygen.adapters.registry import ADAPTER_REGISTRY, get_adapter
from phygen.core.selector import build_candidate_plan, offline_selector_report
from phygen.core.trainer import PhyGenTrainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PhyGen residual-field model with a task adapter")
    parser.add_argument("--task", default="stack_three", choices=sorted(ADAPTER_REGISTRY.keys()))
    parser.add_argument("--feedback-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--budget", type=int, default=5)
    parser.add_argument("--pool-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=9701)
    parser.add_argument("--start-index", type=int, default=100)
    parser.add_argument("--boundary-weight", type=float, default=1.5)
    parser.add_argument("--uncertainty-weight", type=float, default=0.3)
    parser.add_argument("--include-repaired", action="store_true")
    parser.add_argument("--target-failed-hdf5", default=None)
    parser.add_argument("--target-max-failed-demos", type=int, default=0)
    parser.add_argument("--target-exclude-success-jsonl", action="append", default=None)
    parser.add_argument("--candidate-mode", choices=["default", "safe"], default="default")
    parser.add_argument("--use-component-loss", action="store_true")
    parser.add_argument("--component-weight", type=float, default=0.45)
    parser.add_argument("--true-pinn", action="store_true")
    parser.add_argument("--pinn-weight", type=float, default=0.35)
    parser.add_argument("--standard-pinn", action="store_true")
    parser.add_argument("--standard-pinn-weight", type=float, default=0.35)
    args = parser.parse_args()

    adapter = get_adapter(args.task)
    spec = adapter.task_spec()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = load_records(args.feedback_jsonl)
    if not records:
        raise RuntimeError("No usable feedback records found")

    trainer = PhyGenTrainer(adapter)
    result = trainer.train_model(
        records,
        epochs=args.epochs,
        lr=args.lr,
        use_component_loss=args.use_component_loss or args.true_pinn or args.standard_pinn,
        component_weight=args.component_weight,
        use_true_pinn=args.true_pinn,
        pinn_weight=args.pinn_weight,
        use_standard_pinn=args.standard_pinn,
        standard_pinn_weight=args.standard_pinn_weight,
    )

    offline = offline_selector_report(
        adapter,
        records,
        result.model,
        budget=args.budget,
        boundary_weight=args.boundary_weight,
        uncertainty_weight=args.uncertainty_weight,
    )

    plan_records = records
    if args.target_failed_hdf5:
        target_contexts = adapter.load_failed_contexts(
            args.target_failed_hdf5,
            args.target_max_failed_demos if args.target_max_failed_demos > 0 else None,
        )
        exclude_success = load_success_demo_keys(args.target_exclude_success_jsonl)
        target_contexts = [ctx for ctx in target_contexts if ctx["demo_key"] not in exclude_success]
        plan_records = [
            {
                "task_name": spec.task_name,
                "demo_key": ctx["demo_key"],
                "context_metrics": ctx["context_metrics"],
                "success": False,
            }
            for ctx in target_contexts
        ]

    plan_info = build_candidate_plan(
        adapter=adapter,
        records=plan_records,
        model=result.model,
        out_path=out_dir / spec.output_plan_name,
        pool_size=args.pool_size,
        budget=args.budget,
        seed=args.seed,
        start_index=args.start_index,
        boundary_weight=args.boundary_weight,
        include_repaired=args.include_repaired,
        candidate_mode=args.candidate_mode,
        uncertainty_weight=args.uncertainty_weight,
    )

    import torch

    model_family = "repair_parameter_residual_field_pinn" if (args.use_component_loss or args.true_pinn or args.standard_pinn) else "mlp_selector"
    checkpoint = {
        "state_dict": result.model.state_dict(),
        "input_dim": result.input_dim,
        "layout": asdict(result.layout),
        "task_spec": asdict(spec),
        "context_keys": spec.context_keys,
        "component_keys": spec.component_keys if (args.use_component_loss or args.true_pinn or args.standard_pinn) else [],
        "continuous_theta_keys": spec.theta_cont_keys,
        "independent_continuous_theta_keys": spec.theta_cont_independent_keys,
        "derived_continuous_theta_keys": spec.theta_cont_derived_keys,
        "discrete_theta_keys": spec.theta_disc_keys,
        "pde_applied_to": "independent_theta_cont_only",
        "collocation_manifold_projection": "adapter.project_theta_cont_manifold() is applied after independent-theta perturbation",
        "model_family": model_family,
        "residual_groups": spec.residual_groups,
        "optional_mujoco_residual_keys": spec.optional_residual_keys,
        "residual_field_definition": {
            "context": "failed-trajectory context c supplied by TaskAdapter",
            "discrete_repair_mode": "discrete repair choices d supplied by ThetaAdapter",
            "continuous_repair_parameters": "continuous repair vector z; PDEs apply only to independent z axes",
            "value_field": "V(c,d,z): normalized residual energy / cost-to-success field",
            "physics_source": "q(c,d,z): weighted physical residual source from TaskResidualAdapter",
            "success_probability": "p(c,d,z)=sigmoid(beta*(tau-V)+small_correction)",
        },
        "use_component_loss": args.use_component_loss or args.true_pinn or args.standard_pinn,
        "true_pinn": args.true_pinn,
        "standard_pinn": args.standard_pinn,
    }
    torch.save(checkpoint, out_dir / spec.output_model_name)

    summary = {
        "method": (
            "standard_pinn_hjb_repair_field"
            if args.standard_pinn
            else (
                "true_pinn_utility_boundary_union"
                if args.true_pinn
                else ("component_pinn_utility_boundary_union" if args.use_component_loss else "pinn_utility_boundary_union")
            )
        ),
        "task": spec.task_name,
        "architecture": "phygen_core_plus_task_adapter",
        "adapter": adapter.__class__.__name__,
        "true_pinn": args.true_pinn,
        "standard_pinn": args.standard_pinn,
        "use_component_loss": args.use_component_loss or args.true_pinn or args.standard_pinn,
        "component_keys": spec.component_keys if (args.use_component_loss or args.true_pinn or args.standard_pinn) else [],
        "component_weight": args.component_weight,
        "pinn_weight": args.pinn_weight,
        "standard_pinn_weight": args.standard_pinn_weight,
        "boundary_weight": args.boundary_weight,
        "uncertainty_weight": args.uncertainty_weight,
        "model_family": model_family,
        "task_spec": asdict(spec),
        "paper_method_name": "PhyGen / Repair-Parameter Residual Field PINN",
        "residual_field_definition": {
            "c": "failed-trajectory context supplied by task adapter",
            "d": "discrete repair mode / conditional embedding supplied by theta adapter",
            "z": "independent continuous repair parameters; derived features are projected by adapter",
            "V(c,d,z)": "residual energy / cost-to-success value field",
            "q(c,d,z)": "weighted physical residual source from adapter-defined residual components",
            "p(c,d,z)": "success probability induced mainly by V through sigmoid(beta*(tau-V)) with a small regularized correction",
        },
        "continuous_theta_keys": spec.theta_cont_keys,
        "independent_continuous_theta_keys": spec.theta_cont_independent_keys,
        "derived_continuous_theta_keys": spec.theta_cont_derived_keys,
        "discrete_theta_keys": spec.theta_disc_keys,
        "pde_applied_to": "independent_theta_cont_only",
        "collocation_manifold_projection": "collocation perturbs only independent continuous theta and calls adapter.project_theta_cont_manifold()",
        "backward_compatible_outputs": True,
        "governing_equations": (
            [
                "V(c,d,z): residual cost-to-success field over independent continuous repair parameters",
                "q(c,d,z)=weighted_sum(adapter_residual_components)",
                "p(c,d,z)=sigmoid(beta*(tau - V(c,d,z)) + epsilon(c,d,z))",
                "0.5 * ||grad_z V||^2 - q(c,d,z) = 0",
                "grad_z p + beta * p * (1-p) * grad_z V = 0",
            ]
            if args.standard_pinn
            else []
        ),
        "loss_terms": [
            "supervised_energy",
            "success_bce",
            "energy_success_consistency",
            "energy_margin",
            "regularized_success_correction",
            "component_residual_supervision" if (args.use_component_loss or args.true_pinn or args.standard_pinn) else "disabled_component_residual_supervision",
            "energy_component_consistency" if (args.true_pinn or args.standard_pinn) else "disabled_energy_component_consistency",
            "adapter_physical_component_relations" if (args.true_pinn or args.standard_pinn) else "disabled_adapter_physical_component_relations",
            "boundary_conditions" if (args.true_pinn or args.standard_pinn) else "disabled_boundary_conditions",
            "independent_theta_cont_differential_consistency" if (args.true_pinn or args.standard_pinn) else "disabled_independent_theta_cont_differential_consistency",
            "independent_theta_cont_hjb_eikonal_collocation" if args.standard_pinn else "disabled_independent_theta_cont_hjb_eikonal_collocation",
            "independent_theta_cont_success_transport_collocation" if args.standard_pinn else "disabled_independent_theta_cont_success_transport_collocation",
            "acquisition_uncertainty_bonus",
        ],
        "num_feedback_records": len(records),
        "num_feedback_success": int(sum(1 for r in records if r["success"])),
        "num_feedback_demos": len({r["demo_key"] for r in records}),
        "offline_budget": args.budget,
        "offline_oracle_demo_success": int(sum(1 for r in offline if r["oracle_success"])),
        "offline_selector_demo_success": int(sum(1 for r in offline if r["selector_success"])),
        "offline_per_demo": offline,
        "target_failed_hdf5": args.target_failed_hdf5,
        "loss_history": result.history,
        **plan_info,
    }
    (out_dir / spec.output_summary_name).write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
