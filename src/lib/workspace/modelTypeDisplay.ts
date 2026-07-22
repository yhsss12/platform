import type { ModelTypeDefinition } from '@/types/modelType';
import { BASE_ALGORITHM_OPTIONS } from '@/types/modelType';

export function baseAlgorithmLabel(baseAlgorithm: string): string {
  return BASE_ALGORITHM_OPTIONS.find((item) => item.value === baseAlgorithm)?.label ?? baseAlgorithm;
}

export function simulatorLabel(value?: string | null): string {
  if (!value) return '—';
  const map: Record<string, string> = { mujoco: 'MuJoCo', isaac: 'Isaac', general: '通用' };
  return map[value] ?? value;
}

export function robotTypeLabel(value?: string | null): string {
  if (!value) return '—';
  const map: Record<string, string> = { panda: 'Panda', dual_arm: 'Dual Arm', general: '通用' };
  return map[value] ?? value;
}

export function structureConfigSummary(defn: ModelTypeDefinition): string {
  const config = defn.structureConfig ?? {};
  const algo = defn.baseAlgorithm;

  if (algo === 'robomimic_bc') {
    const dims = config.actor_hidden_dims ?? config.hidden_dims ?? '—';
    const reg = config.l2_regularization ?? config.weight_decay ?? 0;
    return `hidden=${dims}, l2=${reg}`;
  }

  if (algo === 'act') {
    return [
      `hidden=${config.hidden_dim ?? '—'}`,
      `chunk=${config.chunk_size ?? '—'}`,
      `enc/dec=${config.enc_layers ?? '—'}/${config.dec_layers ?? '—'}`,
    ].join(', ');
  }

  if (algo === 'diffusion_policy') {
    return [
      `horizon=${config.horizon ?? '—'}`,
      `obs=${config.n_obs_steps ?? '—'}`,
      `action=${config.n_action_steps ?? '—'}`,
      `infer=${config.num_inference_steps ?? '—'}`,
    ].join(', ');
  }

  if (algo === 'pi0') {
    return [
      `ctx=${config.context_window ?? '—'}`,
      `action_h=${config.action_horizon ?? '—'}`,
      `vision=${config.vision_encoder ?? '—'}`,
      `head=${config.action_head ?? '—'}`,
    ].join(', ');
  }

  return Object.keys(config).length ? JSON.stringify(config) : '—';
}

export function defaultStructureConfigForAlgorithm(baseAlgorithm: string): Record<string, unknown> {
  if (baseAlgorithm === 'robomimic_bc') {
    return {
      actor_hidden_dims: '512,512',
      l2_regularization: 0,
      encoder_type: 'low_dim',
      activation: 'relu',
    };
  }
  if (baseAlgorithm === 'act') {
    return {
      hidden_dim: 512,
      dim_feedforward: 2048,
      chunk_size: 100,
      n_action_steps: 100,
      kl_weight: 10,
      latent_dim: 32,
      enc_layers: 4,
      dec_layers: 4,
      nheads: 8,
      dropout: 0.1,
    };
  }
  if (baseAlgorithm === 'diffusion_policy') {
    return {
      horizon: 16,
      n_obs_steps: 2,
      n_action_steps: 8,
      num_inference_steps: 20,
      weight_decay: 0.0001,
      vision_encoder: 'resnet18',
      noise_scheduler: 'ddpm',
    };
  }
  if (baseAlgorithm === 'pi0') {
    return {
      context_window: 256,
      action_horizon: 16,
      vision_encoder: 'siglip',
      language_conditioning: true,
      action_head: 'flow_matching',
      tokenizer_or_processor: 'default',
    };
  }
  return {};
}

export function adapterKeyForBaseAlgorithm(baseAlgorithm: string): string {
  const map: Record<string, string> = {
    robomimic_bc: 'robomimic_bc_adapter',
    act: 'act_adapter',
    diffusion_policy: 'diffusion_policy_adapter',
    pi0: 'pi0_adapter',
  };
  return map[baseAlgorithm] ?? `${baseAlgorithm}_adapter`;
}

export function trainingBackendForModelType(defn: ModelTypeDefinition): string {
  const map: Record<string, string> = {
    robomimic_bc_adapter: 'robomimic_bc',
    act_adapter: 'act',
    diffusion_policy_adapter: 'diffusion_policy',
    pi0_adapter: 'pi0',
  };
  return map[defn.adapterKey] ?? String(defn.baseAlgorithm);
}

export function downstreamModelTypeForModelType(defn: ModelTypeDefinition): string {
  const map: Record<string, string> = {
    robomimic_bc_adapter: 'Robomimic',
    act_adapter: 'ACT',
    diffusion_policy_adapter: 'Diffusion Policy',
    pi0_adapter: 'pi0',
  };
  return map[defn.adapterKey] ?? defn.name;
}
