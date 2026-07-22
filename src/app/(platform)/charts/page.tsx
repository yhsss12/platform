import { redirect } from 'next/navigation';

export default function LegacyRouteRedirectPage() {
  redirect('/workspace/data');
}
