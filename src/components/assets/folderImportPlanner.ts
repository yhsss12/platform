/**
 * 选择文件夹（webkitRelativePath）后的资产单元识别与导入任务拆分。
 * LeRobot 根 / 多子目录 → directory；其余中的 HDF5/MCAP → single/multi；其它文件忽略不阻断。
 */

export type FolderImportJob =
  | { kind: 'directory'; rootDirName: string; files: File[] }
  | { kind: 'multi_file'; files: File[] }
  | { kind: 'single_file'; file: File };

/** 成功：可导入任务 + 分类明细（供 UI / 调试） */
export type FolderImportPlanSuccess = {
  ok: true;
  jobs: FolderImportJob[];
  summary: string;
  /** LeRobot 目录型任务数量 */
  recognized_dataset_units: number;
  /** 将走 single_file / multi_file 的文件 */
  recognized_data_files: File[];
  /** 非 LeRobot 场景下不参与导入、也不阻断的文件 */
  ignored_files: File[];
  /** 成功时为空；语义上「无阻断未识别」 */
  blocking_unrecognized_files: [];
};

export type FolderImportPlanFailure = {
  ok: false;
  message: string;
  /** 无可导入目标时，列出本批路径便于提示（可选展示） */
  blocking_unrecognized_files: string[];
};

export type FolderImportPlan = FolderImportPlanSuccess | FolderImportPlanFailure;

/** 弹窗列表：一条对应一个将入库的资产单元 */
export type ImportAssetRow = {
  id: string;
  displayName: string;
  assetType: string;
  sizeBytes: number;
  /** 从 pending 中移除该资产时需删除的所有底层 File 键 */
  fileKeysToRemove: string[];
};

const DATA_EXT = /\.(mcap|hdf5|h5)$/i;

function assetTypeLabelForFile(f: File): string {
  const n = (f.name || '').toLowerCase();
  if (n.endsWith('.mcap')) return 'MCAP';
  return 'HDF5';
}

/** 由 folder jobs 生成待展示的资产行（不展开 LeRobot 内部文件） */
export function jobsToImportAssetRows(jobs: FolderImportJob[]): ImportAssetRow[] {
  const rows: ImportAssetRow[] = [];
  let dirSeq = 0;
  for (const job of jobs) {
    if (job.kind === 'directory') {
      const keys = job.files.map((f) => fileKey(f));
      rows.push({
        id: `dir:${dirSeq++}:${job.rootDirName}`,
        displayName: job.rootDirName,
        assetType: 'LeRobot',
        sizeBytes: job.files.reduce((s, f) => s + f.size, 0),
        fileKeysToRemove: keys,
      });
    } else if (job.kind === 'multi_file') {
      for (const f of job.files) {
        rows.push({
          id: `file:${fileKey(f)}`,
          displayName: f.name,
          assetType: assetTypeLabelForFile(f),
          sizeBytes: f.size,
          fileKeysToRemove: [fileKey(f)],
        });
      }
    } else {
      const f = job.file;
      rows.push({
        id: `file:${fileKey(f)}`,
        displayName: f.name,
        assetType: assetTypeLabelForFile(f),
        sizeBytes: f.size,
        fileKeysToRemove: [fileKey(f)],
      });
    }
  }
  return rows;
}

/** 列表下方绿条：资产视角摘要 */
export function buildAssetPerspectiveSummary(plan: FolderImportPlanSuccess): string {
  const lr = plan.recognized_dataset_units;
  const dataFiles = plan.recognized_data_files;
  const ig = plan.ignored_files.length;
  const mcaps = dataFiles.filter((f) => /\.mcap$/i.test(f.name)).length;
  const hdf = dataFiles.length - mcaps;
  const parts: string[] = [];
  if (lr) {
    parts.push(`已识别 ${lr} 条 LeRobot 资产`);
  }
  if (dataFiles.length) {
    if (mcaps > 0 && hdf > 0) {
      parts.push(`已识别 ${dataFiles.length} 条 HDF5/MCAP 资产`);
    } else if (mcaps > 0) {
      parts.push(`已识别 ${mcaps} 条 MCAP 资产`);
    } else {
      parts.push(`已识别 ${hdf} 条 HDF5 资产`);
    }
  }
  let s = parts.join('，');
  if (ig > 0) {
    s += (s ? '；' : '') + `忽略 ${ig} 个非目标文件`;
  }
  return s;
}

export type FlatImportPlanResult = {
  rows: ImportAssetRow[];
  summary: string;
  blocked: boolean;
  blockMessage: string;
};

/** 非文件夹多选 / 单选：仅目标后缀成为资产行 */
export function planFlatFilesImportRows(files: File[]): FlatImportPlanResult {
  if (!files.length) {
    return { rows: [], summary: '', blocked: false, blockMessage: '' };
  }
  const dataFiles = files.filter((f) => DATA_EXT.test(f.name || ''));
  const ignored = files.length - dataFiles.length;
  const rows: ImportAssetRow[] = dataFiles.map((f) => ({
    id: `file:${fileKey(f)}`,
    displayName: f.name,
    assetType: assetTypeLabelForFile(f),
    sizeBytes: f.size,
    fileKeysToRemove: [fileKey(f)],
  }));
  if (dataFiles.length === 0 && files.length > 0) {
    return {
      rows: [],
      summary: '',
      blocked: true,
      blockMessage: '当前所选文件中未发现可导入的 HDF5 或 MCAP。',
    };
  }
  const mcaps = dataFiles.filter((f) => /\.mcap$/i.test(f.name)).length;
  const hdf = dataFiles.length - mcaps;
  let summary = '';
  if (dataFiles.length === 1) {
    summary = `已识别 1 条 ${rows[0].assetType} 资产`;
  } else if (mcaps > 0 && hdf > 0) {
    summary = `已识别 ${dataFiles.length} 条 HDF5/MCAP 资产`;
  } else if (mcaps > 0) {
    summary = `已识别 ${mcaps} 条 MCAP 资产`;
  } else {
    summary = `已识别 ${hdf} 条 HDF5 资产`;
  }
  if (ignored > 0) {
    summary += `；忽略 ${ignored} 个非目标文件`;
  }
  return { rows, summary, blocked: false, blockMessage: '' };
}

export function normalizeWebkitRelativePath(f: File): string {
  const rel = ((f as File & { webkitRelativePath?: string }).webkitRelativePath || f.name || '').replace(
    /\\/g,
    '/',
  );
  return rel.replace(/^\/+/, '');
}

function directChildNames(allRels: string[]): string[] {
  const dirs = new Set<string>();
  for (const rel of allRels) {
    const i = rel.indexOf('/');
    if (i > 0) dirs.add(rel.slice(0, i));
  }
  return [...dirs];
}

/** 仅看 prefix 下文件的「去掉 prefix 后」路径，是否同时出现 data、meta、videos 三个顶层目录 */
function isLeRobotRootAtPrefix(files: File[], prefix: string): boolean {
  const pref = prefix.replace(/\/+$/, '');
  const stripped: string[] = [];
  for (const f of files) {
    const rel = normalizeWebkitRelativePath(f);
    if (pref) {
      if (rel === pref || rel.startsWith(pref + '/')) {
        stripped.push(rel === pref ? '' : rel.slice(pref.length + 1));
      }
    } else {
      stripped.push(rel);
    }
  }
  const firstSegs = new Set<string>();
  for (const s of stripped) {
    if (!s) continue;
    firstSegs.add(s.split('/')[0].toLowerCase());
  }
  return firstSegs.has('data') && firstSegs.has('meta') && firstSegs.has('videos');
}

function syntheticRootDirName(): string {
  return `import_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 8)}`;
}

function isDataAssetFile(f: File): boolean {
  return DATA_EXT.test(f.name || '');
}

export function fileKey(f: File): string {
  const rel = normalizeWebkitRelativePath(f);
  return `${rel}|${f.name}|${f.size}`;
}

/**
 * 目录直传 / webkitdirectory：浏览器可能混入「子目录」占位条目（size=0），或惰性计算 size。
 * 在发起 upload-init 前调用：轻触 Blob 以尽量拿到真实 size，并过滤仍 ≤0 的项（仅用于直传清单）。
 */
export async function resolveFilesForDirectUploadItems(files: File[]): Promise<File[]> {
  const out: File[] = [];
  for (const f of files) {
    if (!(f instanceof File)) continue;
    let n = f.size;
    if (!Number.isFinite(n) || n <= 0) {
      try {
        const probe = Math.min(1024 * 1024, Math.max(1, Number.isFinite(n) && n > 0 ? n : 1));
        await f.slice(0, probe).arrayBuffer();
        n = f.size;
      } catch {
        /* 无法读取则保持 */
      }
    }
    if (!Number.isFinite(n) || n <= 0) continue;
    out.push(f);
  }
  return out;
}

function countDataFilesInJobs(jobs: FolderImportJob[]): number {
  let n = 0;
  for (const j of jobs) {
    if (j.kind === 'multi_file') n += j.files.length;
    else if (j.kind === 'single_file') n += 1;
  }
  return n;
}

function buildSuccessSummary(
  jobs: FolderImportJob[],
  ignoredCount: number,
): string {
  const lr = jobs.filter((j) => j.kind === 'directory').length;
  const dataCount = countDataFilesInJobs(jobs);
  const parts: string[] = [];
  if (lr) {
    parts.push(`${lr} 个 LeRobot 目录资产`);
  }
  if (dataCount > 0 && ignoredCount > 0) {
    parts.push(`将导入 ${dataCount} 个 HDF5/MCAP 文件，忽略 ${ignoredCount} 个非目标文件`);
  } else if (dataCount > 0) {
    parts.push(`将导入 ${dataCount} 个 HDF5/MCAP 文件`);
  } else if (lr && ignoredCount > 0) {
    parts.push(`忽略 ${ignoredCount} 个非目标文件（不纳入上述目录包）`);
  }
  if (!parts.length) return '';
  return `${parts.join('；')}（将按顺序导入）`;
}

/**
 * 为 directory 直传构造 items：保证 relative_path 以 rootDirName/ 开头（顶层 LeRobot 需补合成前缀）
 */
export function buildDirectoryUploadItems(files: File[], rootDirName: string): { relative_path: string }[] {
  const r = rootDirName.replace(/\/+$/, '');
  return files.map((f) => {
    const raw = normalizeWebkitRelativePath(f);
    let rp = raw;
    if (!raw.startsWith(r + '/') && raw !== r) {
      rp = `${r}/${raw}`;
    }
    return { relative_path: rp };
  });
}

export function planFolderTreeImport(files: File[]): FolderImportPlan {
  if (!files.length) {
    return {
      ok: false,
      message: '没有选择任何文件',
      blocking_unrecognized_files: [],
    };
  }

  const rels = files.map((f) => normalizeWebkitRelativePath(f));

  const jobs: FolderImportJob[] = [];
  const assigned = new Set<string>();

  const tryAssignDirectory = (rootDirName: string, subset: File[]) => {
    if (subset.length === 0) return;
    jobs.push({ kind: 'directory', rootDirName, files: subset });
    subset.forEach((f) => assigned.add(fileKey(f)));
  };

  // 规则 1：顶层即 LeRobot 根（整包上传，内部 yaml/json 等一并纳入，不算「忽略」）
  if (isLeRobotRootAtPrefix(files, '')) {
    const rootDirName = syntheticRootDirName();
    tryAssignDirectory(rootDirName, [...files]);
    const summary =
      buildSuccessSummary(jobs, 0) ||
      `已识别 1 个 LeRobot 目录（将按顺序导入）`;
    return {
      ok: true,
      jobs,
      summary,
      recognized_dataset_units: 1,
      recognized_data_files: [],
      ignored_files: [],
      blocking_unrecognized_files: [],
    };
  }

  // 规则 2：直接子目录中的多个 LeRobot 根
  const children = directChildNames(rels);
  const lrChildNames = children.filter((c) => isLeRobotRootAtPrefix(files, c));
  lrChildNames.sort();

  for (const c of lrChildNames) {
    const subset = files.filter((f) => {
      const rel = normalizeWebkitRelativePath(f);
      return rel === c || rel.startsWith(c + '/');
    });
    tryAssignDirectory(c, subset);
  }

  const remaining = files.filter((f) => !assigned.has(fileKey(f)));
  const dataFiles = remaining.filter(isDataAssetFile);
  const ignoredFiles = remaining.filter((f) => !isDataAssetFile(f));

  // 规则 3：普通容器 — 只导入目标后缀，其余一律忽略，不阻断
  if (dataFiles.length >= 2) {
    jobs.push({ kind: 'multi_file', files: dataFiles });
  } else if (dataFiles.length === 1) {
    jobs.push({ kind: 'single_file', file: dataFiles[0] });
  }

  // 规则 4：既无 LeRobot 任务也无目标数据文件 → 拒绝
  if (jobs.length === 0) {
    return {
      ok: false,
      message:
        '当前目录未识别为 LeRobot 数据集（需含 data、meta、videos），且未发现可导入的 HDF5/MCAP 文件。',
      blocking_unrecognized_files: rels,
    };
  }

  const summary = buildSuccessSummary(jobs, ignoredFiles.length) || '将按顺序导入';

  return {
    ok: true,
    jobs,
    summary,
    recognized_dataset_units: jobs.filter((j) => j.kind === 'directory').length,
    recognized_data_files: dataFiles,
    ignored_files: ignoredFiles,
    blocking_unrecognized_files: [],
  };
}
