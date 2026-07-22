/**
 * 时间桶聚合（事件节奏、资产曲线）
 */
import type { ProjectActivity } from '@/lib/projects/projectActivity';
import type { TimeRangeKey } from './types';

export interface TimeBucket {
  label: string;
  ts: number;
  count: number;
}

/** 活动按时间桶聚合 */
export function aggregateActivitiesByTime(
  activities: ProjectActivity[],
  range: TimeRangeKey
): TimeBucket[] {
  const now = Date.now();
  let bucketMs: number;
  let bucketCount: number;
  let formatLabel: (d: Date) => string;

  if (range === '30m') {
    bucketMs = 60 * 1000;
    bucketCount = 30;
    formatLabel = (d) => `${d.getMinutes()}分`;
  } else if (range === 'today') {
    bucketMs = 60 * 60 * 1000;
    bucketCount = 24;
    formatLabel = (d) => `${d.getHours()}时`;
  } else {
    bucketMs = 24 * 60 * 60 * 1000;
    bucketCount = 7;
    formatLabel = (d) => `${d.getMonth() + 1}/${d.getDate()}`;
  }

  const start = now - bucketCount * bucketMs;
  const buckets: TimeBucket[] = [];
  for (let i = 0; i < bucketCount; i++) {
    const ts = start + i * bucketMs;
    const d = new Date(ts);
    buckets.push({ label: formatLabel(d), ts, count: 0 });
  }

  activities.forEach((a) => {
    const t = new Date(a.createdAt).getTime();
    if (t < start || t > now) return;
    const idx = Math.floor((t - start) / bucketMs);
    if (idx >= 0 && idx < bucketCount) buckets[idx].count += 1;
  });

  return buckets;
}

/** 数据资产按时间累计（用于增量曲线） */
export function aggregateAssetGrowth(
  assets: Array<{ created_at: string }>,
  range: TimeRangeKey
): Array<{ label: string; ts: number; cumulative: number }> {
  const now = Date.now();
  let bucketMs: number;
  let bucketCount: number;
  let formatLabel: (d: Date) => string;

  if (range === '30m') {
    bucketMs = 60 * 1000;
    bucketCount = 30;
    formatLabel = (d) => `${d.getMinutes()}分`;
  } else if (range === 'today') {
    bucketMs = 60 * 60 * 1000;
    bucketCount = 24;
    formatLabel = (d) => `${d.getHours()}时`;
  } else {
    bucketMs = 24 * 60 * 60 * 1000;
    bucketCount = 7;
    formatLabel = (d) => `${d.getMonth() + 1}/${d.getDate()}`;
  }

  const start = now - bucketCount * bucketMs;
  const buckets: Array<{ label: string; ts: number; cumulative: number }> = [];
  for (let i = 0; i < bucketCount; i++) {
    const ts = start + i * bucketMs;
    buckets.push({ label: formatLabel(new Date(ts)), ts, cumulative: 0 });
  }

  const sorted = [...assets].map((a) => new Date(a.created_at).getTime()).sort((a, b) => a - b);
  let cum = 0;
  sorted.forEach((t) => {
    if (t < start) {
      cum += 1;
      return;
    }
    if (t > now) return;
    const idx = Math.floor((t - start) / bucketMs);
    if (idx >= 0 && idx < bucketCount) {
      cum += 1;
      buckets[idx].cumulative = cum;
    }
  });
  for (let i = 0; i < bucketCount; i++) {
    if (buckets[i].cumulative === 0 && i > 0) buckets[i].cumulative = buckets[i - 1].cumulative;
  }
  return buckets;
}

/** 近 7 天按日：数据资产累计 + 当日新增 */
export function aggregateAssetByDay(
  assets: Array<{ created_at: string }>,
  days = 7
): Array<{ date: string; label: string; cumulative: number; new: number }> {
  const now = new Date();
  now.setHours(23, 59, 59, 999);
  const result: Array<{ date: string; label: string; cumulative: number; new: number }> = [];
  const dayMs = 24 * 60 * 60 * 1000;
  const times = assets.map((a) => new Date(a.created_at).getTime()).filter((t) => t <= now.getTime());
  for (let i = days - 1; i >= 0; i--) {
    const d = new Date(now);
    d.setDate(d.getDate() - i);
    d.setHours(0, 0, 0, 0);
    const start = d.getTime();
    const end = start + dayMs;
    const dayNew = times.filter((t) => t >= start && t < end).length;
    const cumulative = times.filter((t) => t < end).length;
    result.push({
      date: d.toISOString().slice(0, 10),
      label: `${d.getMonth() + 1}/${d.getDate()}`,
      cumulative,
      new: dayNew,
    });
  }
  return result;
}

/** 近 7 天按日：任务新增数、完成数（采集+标注） */
export function aggregateTaskTrendByDay(
  collectionTasks: Array<{ createdAt: string }>,
  labelTasks: Array<{ createdAt: string; completed?: boolean; updatedAt?: string }>,
  days = 7
): Array<{ date: string; label: string; created: number; completed: number }> {
  const now = new Date();
  const dayMs = 24 * 60 * 60 * 1000;
  const result: Array<{ date: string; label: string; created: number; completed: number }> = [];
  for (let i = days - 1; i >= 0; i--) {
    const d = new Date(now);
    d.setDate(d.getDate() - i);
    d.setHours(0, 0, 0, 0);
    const start = d.getTime();
    const end = start + dayMs;
    const collCreated = collectionTasks.filter((t) => {
      const t0 = new Date(t.createdAt).getTime();
      return t0 >= start && t0 < end;
    }).length;
    const labelCreated = labelTasks.filter((t) => {
      const t0 = new Date(t.createdAt).getTime();
      return t0 >= start && t0 < end;
    }).length;
    const labelCompleted = labelTasks.filter((t) => {
      if (!t.completed || !t.updatedAt) return false;
      const t0 = new Date(t.updatedAt).getTime();
      return t0 >= start && t0 < end;
    }).length;
    result.push({
      date: d.toISOString().slice(0, 10),
      label: `${d.getMonth() + 1}/${d.getDate()}`,
      created: collCreated + labelCreated,
      completed: labelCompleted,
    });
  }
  return result;
}
