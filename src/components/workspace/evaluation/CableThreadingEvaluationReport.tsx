'use client';

import type { ParsedEvaluationReport } from '@/lib/workspace/evaluationReport';

function boolLabel(value: boolean | null): string {
  if (value === true) return '是';
  if (value === false) return '否';
  return '—';
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section style={{ marginBottom: 28 }}>
      <h3
        style={{
          margin: '0 0 12px',
          fontSize: 14,
          fontWeight: 600,
          color: '#374151',
          borderBottom: '1px solid #f3f4f6',
          paddingBottom: 8,
        }}
      >
        {title}
      </h3>
      {children}
    </section>
  );
}

function InfoGrid({ items }: { items: { label: string; value: string }[] }) {
  return (
    <div
      style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fill, minmax(220px, 1fr))',
        gap: '12px 24px',
      }}
    >
      {items.map((item) => (
        <div key={item.label}>
          <div style={{ fontSize: 12, color: '#6b7280', marginBottom: 4 }}>{item.label}</div>
          <div style={{ fontSize: 14, color: '#111827', wordBreak: 'break-word' }}>{item.value}</div>
        </div>
      ))}
    </div>
  );
}

export function CableThreadingEvaluationReport({
  report,
  loading,
  error,
}: {
  report: ParsedEvaluationReport | null;
  loading?: boolean;
  error?: string | null;
}) {
  if (loading) {
    return (
      <p style={{ margin: 0, fontSize: 13, color: '#6b7280' }}>正在加载评测结果…</p>
    );
  }

  if (error) {
    return (
      <div
        style={{
          padding: '14px 16px',
          borderRadius: 8,
          backgroundColor: '#fff7ed',
          border: '1px solid #fed7aa',
          color: '#9a3412',
          fontSize: 13,
          lineHeight: 1.6,
        }}
      >
        {error}
      </div>
    );
  }

  if (!report) return null;

  if (!report.hasCoreMetrics) {
    return (
      <>
        <div
          style={{
            padding: '16px 18px',
            borderRadius: 10,
            backgroundColor: '#f8fafc',
            border: '1px solid #e2e8f0',
            marginBottom: 20,
          }}
        >
          <div style={{ fontSize: 15, fontWeight: 600, color: '#0f172a', marginBottom: 8 }}>
            评测结果尚未生成
          </div>
          <p style={{ margin: 0, fontSize: 13, color: '#475569', lineHeight: 1.65 }}>
            当前任务已生成配置与日志，但未检测到有效的评测聚合结果。请确认评测任务已完成，并等待结果文件写入。
          </p>
        </div>
        <details style={{ fontSize: 13, color: '#64748b' }}>
          <summary style={{ cursor: 'pointer', userSelect: 'none' }}>高级信息</summary>
          <ul style={{ margin: '10px 0 0', paddingLeft: 18, lineHeight: 1.7 }}>
            <li>aggregate_result.json / eval.results.json：{report.fileChecks.aggregateResult ? '已检测到' : '未检测到'}</li>
            <li>per_episode_results.json：{report.fileChecks.perEpisodeResults ? '已检测到' : '未检测到'}</li>
            <li>status.json 状态：{report.fileChecks.statusCompleted ? 'completed' : report.status}</li>
            <li>results 目录：{report.fileChecks.resultsDirectory ? '存在' : '缺失或为空'}</li>
          </ul>
        </details>
      </>
    );
  }

  const successCount = report.successEpisodes ?? 0;
  const totalCount = report.totalEpisodes ?? report.episodes.length;
  const successRateText =
    report.finalSuccessRate != null
      ? `${successCount}/${totalCount}（${(report.finalSuccessRate * 100).toFixed(1)}%）`
      : totalCount > 0
        ? `${successCount}/${totalCount}`
        : '—';

  return (
    <>
      <Section title="评测结论">
        <InfoGrid
          items={[
            { label: '综合结果', value: report.conclusion },
            { label: '成功率', value: successRateText },
            {
              label: 'Ever Success Rate',
              value:
                report.everSuccessRate != null
                  ? `${(report.everSuccessRate * 100).toFixed(1)}%`
                  : '—',
            },
            {
              label: '平均耗时',
              value:
                report.meanDurationSec != null ? `${report.meanDurationSec.toFixed(1)} s` : '—',
            },
            {
              label: '主要失败原因',
              value: report.primaryFailureReason ?? '—',
            },
          ]}
        />
      </Section>

      <Section title="核心指标">
        <InfoGrid items={report.coreMetrics} />
      </Section>

      {report.episodes.length > 0 ? (
        <Section title="Episode 明细">
          <div style={{ overflowX: 'auto' }}>
            <table
              style={{
                width: '100%',
                borderCollapse: 'collapse',
                fontSize: 13,
                minWidth: 720,
              }}
            >
              <thead>
                <tr style={{ backgroundColor: '#f9fafb', textAlign: 'left' }}>
                  {['Episode', 'Seed', 'Success', 'Final Success', 'Thread Completion', 'Failure Reason', 'Video'].map(
                    (head) => (
                      <th
                        key={head}
                        style={{
                          padding: '8px 10px',
                          borderBottom: '1px solid #e5e7eb',
                          color: '#6b7280',
                          fontWeight: 600,
                        }}
                      >
                        {head}
                      </th>
                    )
                  )}
                </tr>
              </thead>
              <tbody>
                {report.episodes.map((row) => (
                  <tr key={`${row.episode}-${row.seed}`}>
                    <td style={{ padding: '8px 10px', borderBottom: '1px solid #f3f4f6' }}>{row.episode}</td>
                    <td style={{ padding: '8px 10px', borderBottom: '1px solid #f3f4f6' }}>{row.seed}</td>
                    <td style={{ padding: '8px 10px', borderBottom: '1px solid #f3f4f6' }}>
                      {boolLabel(row.success)}
                    </td>
                    <td style={{ padding: '8px 10px', borderBottom: '1px solid #f3f4f6' }}>
                      {boolLabel(row.finalSuccess)}
                    </td>
                    <td style={{ padding: '8px 10px', borderBottom: '1px solid #f3f4f6' }}>
                      {row.threadCompletion}
                    </td>
                    <td style={{ padding: '8px 10px', borderBottom: '1px solid #f3f4f6' }}>
                      {row.failureReason}
                    </td>
                    <td
                      style={{
                        padding: '8px 10px',
                        borderBottom: '1px solid #f3f4f6',
                        fontFamily: 'ui-monospace, monospace',
                        fontSize: 11,
                      }}
                    >
                      {row.videoPath}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Section>
      ) : null}

      <Section title="失败案例与结果文件">
        <InfoGrid
          items={[
            {
              label: 'aggregate_result.json',
              value: report.artifacts.aggregateResultPath ?? report.artifacts.resultsJsonPath ?? '—',
            },
            {
              label: 'per_episode_results.json',
              value: report.artifacts.perEpisodeResultsPath ?? '—',
            },
            { label: 'eval.results.json', value: report.artifacts.resultsJsonPath ?? '—' },
            { label: 'eval.csv', value: report.artifacts.evalCsvPath ?? '—' },
            { label: 'run.log', value: report.artifacts.logPath ?? '—' },
            { label: 'eval.failures.json', value: report.artifacts.failuresPath ?? '—' },
            { label: '评测视频', value: report.artifacts.videoPath ?? '—' },
          ]}
        />
        {report.failureReasonsText ? (
          <p style={{ margin: '12px 0 0', fontSize: 13, color: '#6b7280', lineHeight: 1.6 }}>
            失败原因汇总：{report.failureReasonsText}
          </p>
        ) : null}
      </Section>
    </>
  );
}
