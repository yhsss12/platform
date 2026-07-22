'use client';

import { usePathname } from 'next/navigation';
import TaskCenterLauncher from './TaskCenterLauncher';
import TaskCenterPanel from './TaskCenterPanel';
import TaskCenterTooltip from './TaskCenterTooltip';

function shouldShowTaskCenter(pathname: string): boolean {
  // 仅在「数据资产页」显示（上传/导出进度由 TaskCenter 承接）
  return pathname === '/data' || pathname.startsWith('/data/');
}

export default function TaskCenterMount() {
  const pathname = usePathname() || '';
  if (!shouldShowTaskCenter(pathname)) return null;
  return (
    <div className="tc-root">
      <div className="tc-launcher-layer">
        <TaskCenterLauncher />
      </div>
      <div className="tc-tooltip-layer">
        <TaskCenterTooltip />
      </div>
      <div className="tc-panel-layer">
        <TaskCenterPanel />
      </div>
      <style jsx>{`
        .tc-root {
          position: fixed;
          inset: 0;
          pointer-events: none;
          z-index: 1000;
        }
        .tc-launcher-layer,
        .tc-tooltip-layer,
        .tc-panel-layer {
          position: fixed;
          inset: 0;
          pointer-events: none;
        }
        .tc-launcher-layer {
          z-index: 1020;
        }
        .tc-tooltip-layer {
          z-index: 1010;
        }
        .tc-panel-layer {
          z-index: 1000;
        }
      `}</style>
    </div>
  );
}

