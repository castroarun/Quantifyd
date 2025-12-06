'use client'

import { Level, LEVEL_COLORS, LEVEL_NAMES, BODY_PART_NAMES, CalculatedStrength } from '@/types'
import { Card } from '@/components/ui'

interface StrengthCardProps {
  strength: CalculatedStrength
  onLevelSelect: (level: Level) => void
  showBodyPart?: boolean
}

export default function StrengthCard({ strength, onLevelSelect, showBodyPart = false }: StrengthCardProps) {
  const isRated = strength.levels.some(l => l.isSelected)

  return (
    <Card padding="sm" className="mb-3">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <h3 className="font-semibold text-[#2C3E50] dark:text-gray-100 text-sm">
            {strength.exerciseName}
          </h3>
          {strength.isDumbbell && (
            <span className="text-xs text-gray-400 dark:text-gray-500">(per hand)</span>
          )}
        </div>
        {showBodyPart && (
          <span className="text-xs px-2 py-1 bg-gray-100 dark:bg-gray-700 text-gray-500 dark:text-gray-400 rounded">
            {BODY_PART_NAMES[strength.bodyPart]}
          </span>
        )}
      </div>

      <div className="grid grid-cols-4 gap-2">
        {strength.levels.map(({ level, weight, isSelected }) => (
          <button
            key={level}
            onClick={() => onLevelSelect(level)}
            className={`
              relative p-2 rounded-lg border-2 transition-all
              flex flex-col items-center justify-center
              min-h-[70px]
              ${isSelected
                ? 'border-[#2C3E50] dark:border-gray-400 bg-gray-50 dark:bg-gray-700'
                : 'border-transparent bg-gray-50 dark:bg-gray-700 hover:border-gray-300 dark:hover:border-gray-500'
              }
            `}
            style={{
              backgroundColor: isSelected ? `${LEVEL_COLORS[level]}15` : undefined
            }}
          >
            {/* Checkmark for selected */}
            {isSelected && (
              <div
                className="absolute top-1 right-1 w-4 h-4 rounded-full flex items-center justify-center"
                style={{ backgroundColor: LEVEL_COLORS[level] }}
              >
                <svg
                  className="w-3 h-3 text-white"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={3}
                    d="M5 13l4 4L19 7"
                  />
                </svg>
              </div>
            )}

            {/* Weight */}
            <span
              className="text-lg font-bold"
              style={{ color: LEVEL_COLORS[level] }}
            >
              {weight}
            </span>
            <span className="text-xs text-gray-500 dark:text-gray-400">kg</span>

            {/* Level label */}
            <span
              className="text-xs font-medium mt-1"
              style={{ color: LEVEL_COLORS[level] }}
            >
              {LEVEL_NAMES[level]}
            </span>
          </button>
        ))}
      </div>
    </Card>
  )
}
