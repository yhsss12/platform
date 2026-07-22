'use client';

import { create } from 'zustand';
import type { MeResponse } from '@/lib/api/types';
import { clearAuthState } from '@/lib/auth/session';
import { logoutOnServer } from '@/lib/api/authClient';

interface AuthState {
  accessToken: string | null;
  user: MeResponse | null;
  isHydrated: boolean;
  setAccessToken: (token: string | null) => void;
  setUser: (user: MeResponse | null) => void;
  logout: () => Promise<void>;
}

export const useAuthStore = create<AuthState>((set) => ({
  accessToken: null,
  user: null,
  isHydrated: false,
  setAccessToken: (token) =>
    set((state) => {
      const prev = state.accessToken;
      const prevMask = prev ? `${prev.slice(0, 8)}...${prev.slice(-6)}` : 'null';
      const nextMask = token ? `${token.slice(0, 8)}...${token.slice(-6)}` : 'null';
      console.info('[AUTH-TRACE][STORE] setAccessToken', {
        prev: prevMask,
        next: nextMask,
      });
      return { accessToken: token };
    }),
  setUser: (user) =>
    set((state) => {
      console.info('[AUTH-TRACE][STORE] setUser', {
        prev: state.user ? { id: state.user.id, username: state.user.username } : null,
        next: user ? { id: user.id, username: user.username } : null,
      });
      return { user };
    }),
  logout: async () => {
    console.info('[AUTH-TRACE][STORE] logout start', {
      ts: new Date().toISOString(),
      pathname: typeof window !== 'undefined' ? window.location.pathname : '',
    });
    await logoutOnServer();
    clearAuthState({ preserveRememberedUsername: true });
    set({ isHydrated: true });
    console.info('[AUTH-TRACE][STORE] logout done');
  },
}));
















