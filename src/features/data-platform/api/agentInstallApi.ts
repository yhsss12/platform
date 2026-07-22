import { apiGet, apiPost, ApiResponse } from './client';

export interface ZeroInstallStartData {
  token: string;
  script_url: string;
  status_url: string;
  command: string;
}

export async function startZeroInstall(): Promise<ApiResponse<ZeroInstallStartData>> {
  return apiPost('/api/agent/installer/start', {});
}

export async function getZeroInstallStatus(token: string): Promise<ApiResponse<any>> {
  return apiGet(`/api/agent/installer/status/${encodeURIComponent(token)}`);
}
