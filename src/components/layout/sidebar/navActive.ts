export const WORKSPACE_HOME_PATHS = ['/workspace', '/workspace/dashboard', '/workspace/overview'] as const;

export const normalizePath = (path: string) => {
  if (!path) return '/';
  const normalized = path.replace(/\/+$/, '');
  return normalized || '/';
};

export const isExactActive = (pathname: string, targets: readonly string[]) => {
  const current = normalizePath(pathname);
  return targets.map(normalizePath).includes(current);
};

export const isSectionActive = (pathname: string, sectionRoot: string) => {
  const current = normalizePath(pathname);
  const root = normalizePath(sectionRoot);
  return current === root || current.startsWith(`${root}/`);
};

export const isWorkspaceHomeActive = (pathname: string) =>
  isExactActive(pathname, WORKSPACE_HOME_PATHS);
