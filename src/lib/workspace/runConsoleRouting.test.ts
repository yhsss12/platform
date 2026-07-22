
import {
  isValidDataGenJobId,
  isValidIsaacGenerateJobId,
  isValidNutAssemblyGenerateJobId,
} from './backendJobIds';
import { buildIsaacBlockStackingConsoleHref } from './isaacBlockStacking';
import {
  getRunConsoleKindDisplayName,
  isValidRunConsoleJobId,
  resolveRunConsoleKind,
} from './runConsoleAdapters';

describe('Nut Assembly run console routing', () => {
  const jobId = 'na_gen_20260703_141635_6967';

  it('resolveRunConsoleKind maps na_gen_ to nut_assembly', () => {
    assert.equal(resolveRunConsoleKind(jobId), 'nut_assembly');
    assert.equal(resolveRunConsoleKind(jobId, 'cable_threading'), 'nut_assembly');
    assert.equal(isValidNutAssemblyGenerateJobId(jobId), true);
  });

  it('isValidRunConsoleJobId accepts nut assembly job ids', () => {
    assert.equal(isValidRunConsoleJobId('nut_assembly', jobId), true);
    assert.equal(isValidRunConsoleJobId('cable_threading', jobId), false);
  });

  it('getRunConsoleKindDisplayName returns 螺母装配', () => {
    assert.equal(getRunConsoleKindDisplayName('nut_assembly'), '螺母装配');
  });

  it('resolveRunConsoleKind falls back to taskType nut_assembly', () => {
    assert.equal(resolveRunConsoleKind('unknown_job', 'nut_assembly'), 'nut_assembly');
    assert.equal(resolveRunConsoleKind('unknown_job', 'nut_assembly_single_arm'), 'nut_assembly');
  });
});

describe('Isaac block stacking run console routing', () => {
  const jobId = 'isaac_gen_20260721_120000_abcd';

  it('routes isaac_gen jobs to the dedicated Isaac console', () => {
    assert.equal(isValidIsaacGenerateJobId(jobId), true);
    assert.equal(resolveRunConsoleKind(jobId), 'isaac_block_stacking');
    assert.equal(isValidRunConsoleJobId('isaac_block_stacking', jobId), true);
  });

  it('builds a data-generation URL with the Isaac task type', () => {
    assert.equal(
      buildIsaacBlockStackingConsoleHref({ jobId }),
      `/workspace/simulation/console?mode=data-generation&taskType=isaac_block_stacking&jobId=${jobId}`
    );
  });
});
