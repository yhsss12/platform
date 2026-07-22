// 平台统一四层角色（与后端 users.role / JWT 一致）
export type Role = 'SUPER_ADMIN' | 'ADMIN' | 'OWNER' | 'USER';

export interface LoginResponse {
  access_token: string;
  // 改造说明：refresh_token 不再使用 cookie；前端需显式保存到 sessionStorage
  refresh_token?: string;
  token_type: string;
  role: Role;
  /** 登录账号（与 JWT sub 一致） */
  account_id: string;
  /** 展示名称 */
  username: string;
}

export interface AccessTokenOnlyResponse {
  access_token: string;
  token_type: string;
  /** POST /auth/refresh 轮换刷新令牌时由后端返回 */
  refresh_token?: string;
  session_id?: string;
}

export interface MeResponse {
  id: string;
  account_id: string;
  username: string;
  role: Role;
  is_active: boolean;
  created_at: string;
  last_login_at?: string | null;
}
