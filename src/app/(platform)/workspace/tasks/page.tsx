import { redirect } from 'next/navigation';

export default function LegacyTasksRedirectPage() {
  redirect('/workspace/resources/task-templates');
}
