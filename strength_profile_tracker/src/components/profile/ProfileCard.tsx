'use client'

import Link from 'next/link'
import { Profile, SEX_INFO } from '@/types'
import { Card } from '@/components/ui'
import { useUnit } from '@/contexts'
import { formatWeightValue, getProfileColor } from '@/lib/utils/units'

interface ProfileCardProps {
  profile: Profile
}

export default function ProfileCard({ profile }: ProfileCardProps) {
  const { unit } = useUnit()

  // Generate initials from name
  const initials = profile.name
    .split(' ')
    .map(n => n[0])
    .join('')
    .toUpperCase()
    .slice(0, 2)

  return (
    <Link href={`/profile/${profile.id}`}>
      <Card className="hover:border-[#3498DB] transition-colors cursor-pointer">
        <div className="flex items-center gap-4">
          {/* Avatar with Sex indicator */}
          <div className="relative">
            <div className="w-12 h-12 rounded-full overflow-hidden">
              {profile.avatarUrl ? (
                <img
                  src={profile.avatarUrl}
                  alt={profile.name}
                  className="w-full h-full object-cover"
                />
              ) : (
                <div
                  className="w-full h-full flex items-center justify-center text-white font-semibold text-lg"
                  style={{ backgroundColor: getProfileColor(profile.id) }}
                >
                  {initials}
                </div>
              )}
            </div>
            {profile.sex && (
              <span className={`absolute -bottom-1 -right-1 w-5 h-5 rounded-full flex items-center justify-center text-xs ${
                profile.sex === 'male' ? 'bg-blue-500' : 'bg-pink-500'
              } text-white border-2 border-white dark:border-gray-800`}>
                {SEX_INFO[profile.sex].icon}
              </span>
            )}
          </div>

          {/* Profile Info */}
          <div className="flex-1">
            <h3 className="font-semibold text-[#2C3E50] dark:text-gray-100 text-base">
              {profile.name}
            </h3>
            <div className="text-sm text-gray-500 dark:text-gray-400 mt-1">
              <span>{profile.age} yrs</span>
              <span className="mx-2">•</span>
              <span>{profile.height} cm</span>
              <span className="mx-2">•</span>
              <span>{formatWeightValue(profile.weight, unit)} {unit}</span>
            </div>
          </div>

          {/* Arrow */}
          <svg
            className="w-5 h-5 text-gray-400 dark:text-gray-500"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M9 5l7 7-7 7"
            />
          </svg>
        </div>
      </Card>
    </Link>
  )
}
