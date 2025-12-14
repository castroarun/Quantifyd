import ProfileClient from './ProfileClient'

// Required for static export with dynamic routes
export async function generateStaticParams(): Promise<{ id: string }[]> {
  return []
}

interface ProfileDetailPageProps {
  params: Promise<{ id: string }>
}

export default function ProfileDetailPage({ params }: ProfileDetailPageProps) {
  return <ProfileClient params={params} />
}
