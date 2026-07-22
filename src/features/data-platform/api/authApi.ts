import { apiPost, apiGet, setAuthToken, type ApiResponse } from './client';

export interface LoginRequest {
  username: string;
  password: string;
}

export interface TokenResponse {
  access_token: string;
  token_type: string;
}

export interface UserResponse {
  id: string;
  username: string;
  role: string;
}

/**
 * 用户登录
 */
export async function login(
  credentials: LoginRequest
): Promise<ApiResponse<TokenResponse>> {
  const response = await apiPost<TokenResponse>('/api/auth/login', credentials);
  if (response.ok && response.data) {
    setAuthToken(response.data.access_token);
  }
  return response;
}

/**
 * 获取当前用户信息
 */
export async function getCurrentUser(): Promise<ApiResponse<UserResponse>> {
  return apiGet<UserResponse>('/api/auth/me');
}

/**
 * 用户登出
 */
export function logout(): void {
  setAuthToken('');
}


