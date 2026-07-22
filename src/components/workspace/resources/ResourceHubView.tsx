'use client';

import Link from 'next/link';
import { ArrowRight, type LucideIcon } from 'lucide-react';
import type {
  ResourceHubOverviewItemConfig,
  ResourceHubSectionConfig,
} from '@/lib/workspace/resourceHubSections';
import { resolveResourceHubOverviewCount } from '@/lib/workspace/resourceHubSections';
import type { ResourceHubCountKey } from '@/lib/workspace/resourceHubSections';
import './resourceHubView.css';

export interface ResourceHubEntryViewModel {
  id: string;
  title: string;
  description: string;
  href: string;
  icon: LucideIcon;
  count: number | null;
}

export interface ResourceHubSectionViewModel {
  id: string;
  title: string;
  entries: ResourceHubEntryViewModel[];
}

export interface ResourceHubOverviewItemViewModel {
  id: string;
  title: string;
  href?: string;
  count: number | null;
}

interface ResourceHubViewProps {
  overviewItems: ResourceHubOverviewItemViewModel[];
  overviewTitle: string;
  sections: ResourceHubSectionViewModel[];
  enterLabel: string;
  loading?: boolean;
}

function formatItemCount(count: number | null | undefined, loading: boolean): string {
  if (loading) return '…';
  if (count === null || count === undefined) return '--';
  return `${count} 项`;
}

function formatOverviewCount(count: number | null | undefined, loading: boolean): string {
  if (loading) return '…';
  if (count === null || count === undefined) return '--';
  return String(count);
}

function CountBadge({
  count,
  loading = false,
  compact = false,
}: {
  count: number | null | undefined;
  loading?: boolean;
  compact?: boolean;
}) {
  const muted = !loading && (count === null || count === undefined);
  const label = compact ? formatOverviewCount(count, loading) : formatItemCount(count, loading);
  return (
    <span className={`resource-hub-count-badge${muted ? ' resource-hub-count-badge-muted' : ''}`}>
      {label}
    </span>
  );
}

export function ResourceCard({
  entry,
  enterLabel,
  loading = false,
}: {
  entry: ResourceHubEntryViewModel;
  enterLabel: string;
  loading?: boolean;
}) {
  const Icon = entry.icon;

  return (
    <Link href={entry.href} className="resource-hub-card">
      <div className="resource-hub-card-top">
        <div className="resource-hub-card-icon">
          <Icon size={18} strokeWidth={1.65} />
        </div>
        <CountBadge count={entry.count} loading={loading} />
      </div>

      <div className="resource-hub-card-body">
        <h3 className="resource-hub-card-title">{entry.title}</h3>
        <p className="resource-hub-card-caption">{entry.description}</p>
      </div>

      <span className="resource-hub-card-enter">
        {enterLabel}
        <ArrowRight size={12} strokeWidth={1.75} />
      </span>
    </Link>
  );
}

function ResourceOverviewStrip({
  items,
  title,
  loading = false,
}: {
  items: ResourceHubOverviewItemViewModel[];
  title: string;
  loading?: boolean;
}) {
  return (
    <section className="resource-hub-overview-section">
      <h2 className="resource-hub-overview-title">{title}</h2>
      <div className="resource-hub-overview-grid">
        {items.map((item) => {
          const content = (
            <>
              <span className="resource-hub-overview-label">{item.title}</span>
              <CountBadge count={item.count} loading={loading} compact />
            </>
          );

          if (!item.href) {
            return (
              <div key={item.id} className="resource-hub-overview-card">
                {content}
              </div>
            );
          }

          return (
            <Link
              key={item.id}
              href={item.href}
              className="resource-hub-overview-card resource-hub-overview-card-link"
            >
              {content}
            </Link>
          );
        })}
      </div>
    </section>
  );
}

function ResourceSection({
  section,
  enterLabel,
  loading = false,
}: {
  section: ResourceHubSectionViewModel;
  enterLabel: string;
  loading?: boolean;
}) {
  return (
    <section className="resource-hub-section">
      <div className="resource-hub-section-heading">
        <h2 className="resource-hub-section-title">{section.title}</h2>
      </div>

      <div className="resource-hub-grid">
        {section.entries.map((entry) => (
          <ResourceCard key={entry.id} entry={entry} enterLabel={enterLabel} loading={loading} />
        ))}
      </div>
    </section>
  );
}

export function ResourceHubView({
  overviewItems,
  overviewTitle,
  sections,
  enterLabel,
  loading = false,
}: ResourceHubViewProps) {
  return (
    <div className="resource-hub-page">
      <ResourceOverviewStrip items={overviewItems} title={overviewTitle} loading={loading} />
      <div className="resource-hub-sections">
        {sections.map((section) => (
          <ResourceSection
            key={section.id}
            section={section}
            enterLabel={enterLabel}
            loading={loading}
          />
        ))}
      </div>
    </div>
  );
}

export function buildResourceHubOverviewItems(
  items: ResourceHubOverviewItemConfig[],
  counts: Partial<Record<ResourceHubCountKey, number | null>>
): ResourceHubOverviewItemViewModel[] {
  return items.map((item) => ({
    id: item.id,
    title: item.title,
    href: item.href,
    count: resolveResourceHubOverviewCount(item.countKey, counts),
  }));
}

export function buildResourceHubSections(
  sections: ResourceHubSectionConfig[],
  counts: Partial<Record<string, number | null>>
): ResourceHubSectionViewModel[] {
  return sections.map((section) => ({
    id: section.id,
    title: section.title,
    entries: section.entries.map((entry) => ({
      id: entry.id,
      title: entry.title,
      description: entry.description,
      href: entry.href,
      icon: entry.icon,
      count: entry.countKey in counts ? (counts[entry.countKey] ?? null) : 0,
    })),
  }));
}

/** @deprecated 使用 buildResourceHubSections */
export function buildResourceHubEntries(
  entries: ResourceHubSectionConfig['entries'],
  counts: Partial<Record<string, number | null>>
): ResourceHubEntryViewModel[] {
  return entries.map((entry) => ({
    id: entry.id,
    title: entry.title,
    description: entry.description,
    href: entry.href,
    icon: entry.icon,
    count: entry.countKey in counts ? (counts[entry.countKey] ?? null) : 0,
  }));
}
