"""训练产物追溯字段：从 train_config / model_type_definition 提取统一元数据。"""

from __future__ import annotations

from typing import Any, Optional


def resolve_model_type_definition(train_config: dict[str, Any]) -> Optional[dict[str, Any]]:
    defn = train_config.get("_modelTypeDefinition")
    if isinstance(defn, dict):
        return defn
    snapshot = train_config.get("adaptationSnapshot") or {}
    if isinstance(snapshot.get("modelTypeDefinition"), dict):
        return snapshot["modelTypeDefinition"]
    return None


def build_model_type_traceability_fields(
    train_config: dict[str, Any],
    *,
    model_type_definition: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """生成写入 train_config.json / model_manifest.json 的追溯字段。"""
    defn = model_type_definition or resolve_model_type_definition(train_config) or {}
    model_params = dict(train_config.get("modelParams") or {})
    training_defaults = dict(defn.get("trainingDefaults") or train_config.get("trainingDefaults") or {})
    structure_config = dict(defn.get("structureConfig") or train_config.get("structureConfig") or {})

    fields: dict[str, Any] = {
        "modelTypeId": train_config.get("modelTypeId") or defn.get("modelTypeId"),
        "modelTypeName": defn.get("name") or train_config.get("modelTypeName") or train_config.get("taskName"),
        "baseAlgorithm": defn.get("baseAlgorithm") or train_config.get("baseAlgorithm"),
        "adapterId": defn.get("adapterKey") or train_config.get("adapterId"),
        "structureConfig": structure_config,
        "trainingDefaults": training_defaults,
        "resolvedModelParams": model_params,
    }

    if str(train_config.get("trainingBackend") or "").lower() == "pi0":
        try:
            from app.services.pi0_training_runner import get_pi0_env

            pi0_env = get_pi0_env()
            fields["openpiEnvironment"] = {
                "enabled": pi0_env.get("enabled"),
                "openpiRoot": pi0_env.get("openpi_root"),
                "openpiPython": pi0_env.get("openpi_python"),
                "openpiBaseConfig": pi0_env.get("openpi_base_config"),
                "openpiTrainScript": pi0_env.get("openpi_train_script"),
            }
            pi0_config = train_config.get("pi0Config") or {}
            if isinstance(pi0_config, dict):
                fields["openpiPlatformConfig"] = {
                    "openpiBaseConfig": pi0_config.get("openpi_base_config"),
                    "structure": pi0_config.get("structure"),
                    "dataset": pi0_config.get("dataset"),
                }
        except Exception:
            pass

    return {k: v for k, v in fields.items() if v is not None}
