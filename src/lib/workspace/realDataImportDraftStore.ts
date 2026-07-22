import type { RealDataImportDraft } from '@/types/benchmark';

const STORAGE_KEY = 'workspace_real_data_import_drafts';

function readAll(): RealDataImportDraft[] {
  if (typeof window === 'undefined') return [];
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw) as RealDataImportDraft[];
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

function writeAll(items: RealDataImportDraft[]): void {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(items));
}

export function listRealDataImportDrafts(): RealDataImportDraft[] {
  return readAll();
}

export function saveRealDataImportDraft(draft: RealDataImportDraft): void {
  const items = readAll();
  const next = [draft, ...items.filter((d) => d.id !== draft.id)];
  writeAll(next.slice(0, 50));
}

export function getRealDataImportDraft(id: string): RealDataImportDraft | null {
  return readAll().find((d) => d.id === id) ?? null;
}

export function makeRealDataImportDraftId(): string {
  const ts = Date.now().toString(36);
  const rand = Math.random().toString(36).slice(2, 6);
  return `rdid_${ts}_${rand}`;
}
