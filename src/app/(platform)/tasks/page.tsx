import { redirect } from 'next/navigation';

export default function LegacyRouteRedirectPage() {
  redirect('/workspace/resources/task-templates');
}
