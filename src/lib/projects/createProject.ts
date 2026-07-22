import type { Project } from './types';

/** 创建项目时的输入，由创建弹窗/页面提交。Owner 由 create 时根据 ownerId/ownerName 自动写入 members。 */
export interface CreateProjectInput {
  name: string;
  description?: string;
  tags?: string[];
  /** 绑定团队（如团队项目页创建时传入当前 teamId） */
  teamId?: string | null;
  /** 当前用户 ID，用于 members[0].id 且 ownerId */
  ownerId: string;
  /** 当前用户名称，用于 members[0].name 且展示 Owner */
  ownerName: string;
}
