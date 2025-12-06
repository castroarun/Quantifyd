'use client'

import Link from 'next/link'
import { Card } from '@/components/ui'

interface EmptyProfileSlotProps {
  disabled?: boolean
}

export default function EmptyProfileSlot({ disabled = false }: EmptyProfileSlotProps) {
  const content = (
    <Card className={`border-dashed ${!disabled ? 'hover:border-[#3498DB] cursor-pointer' : 'opacity-50'}`}>
      <div className="flex items-center gap-4 py-2">
        {/* Empty Avatar */}
        <div className="w-12 h-12 rounded-full bg-gray-100 dark:bg-gray-700 flex items-center justify-center">
          <svg
            className="w-6 h-6 text-gray-400 dark:text-gray-500"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M12 4v16m8-8H4"
            />
          </svg>
        </div>

        {/* Text */}
        <div className="flex-1">
          <p className="text-gray-400 dark:text-gray-500 font-medium">Add New Profile</p>
        </div>
      </div>
    </Card>
  )

  if (disabled) {
    return content
  }

  return <Link href="/profile/new">{content}</Link>
}
