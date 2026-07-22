// Workspace replay adapter acceptance entrypoint.
process.env.NEXT_PUBLIC_API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL || 'http://127.0.0.1:8000/api';

const token = process.env.ACCEPT_TOKEN;
if (!token) {
  console.error('ACCEPT_TOKEN required');
  process.exit(1);
}

globalThis.window = {
  sessionStorage: {
    getItem: (k: string) => (k === 'auth.access_token' ? token : null),
    setItem: () => {},
    removeItem: () => {},
  },
  localStorage: {
    getItem: () => null,
    setItem: () => {},
    removeItem: () => {},
  },
  location: { href: 'http://127.0.0.1:3000' },
} as unknown as Window & typeof globalThis;

import { resolveCableThreadingReplay } from '../src/lib/workspace/replayAdapters.ts';
import { replayVideoSourceUserLabel } from '../src/lib/workspace/replayAdapters.ts';
import { buildTaskTemplateAssetRow } from '../src/lib/workspace/taskTemplatePresentation.ts';

async function main() {
  const ctGen = await resolveCableThreadingReplay({ jobId: 'ct_gen_20260617_202234_4507' });
  const ctEval = await resolveCableThreadingReplay({
    jobId: 'ct_eval_20260627_222156_3c30',
    evalId: 'ct_eval_20260627_222156_3c30',
  });

  console.log(
    JSON.stringify(
      {
        ctGen: {
          display: ctGen.videoSourceDisplay,
          tag: ctGen.videoTag,
          source: ctGen.videoSource,
          error: ctGen.error,
        },
        ctEval: {
          display: ctEval.videoSourceDisplay,
          tag: ctEval.videoTag,
          source: ctEval.videoSource,
          error: ctEval.error,
        },
        dualArmStabilityLabel: replayVideoSourceUserLabel(
          'evaluation',
          'evaluation',
          'episode_stability'
        ),
      },
      null,
      2
    )
  );
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
