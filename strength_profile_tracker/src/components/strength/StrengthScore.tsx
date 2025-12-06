'use client'

interface StrengthScoreProps {
  score: number  // 0-100
}

export default function StrengthScore({ score }: StrengthScoreProps) {
  // Calculate circle properties
  const size = 120
  const strokeWidth = 8
  const radius = (size - strokeWidth) / 2
  const circumference = 2 * Math.PI * radius
  const progress = (score / 100) * circumference
  const offset = circumference - progress

  // Determine color based on score
  const getScoreColor = () => {
    if (score >= 88) return '#E74C3C'      // Advanced (88-100)
    if (score >= 63) return '#F39C12'      // Intermediate (63-87)
    if (score >= 38) return '#3498DB'      // Novice (38-62)
    return '#2ECC71'                        // Beginner (0-37)
  }

  const getScoreLabel = () => {
    if (score >= 88) return 'Elite'
    if (score >= 63) return 'Strong'
    if (score >= 38) return 'Building'
    if (score > 0) return 'Starting'
    return 'Unrated'
  }

  const color = getScoreColor()

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg border border-[#E0E0E0] dark:border-gray-700 p-4">
      <h3 className="text-sm font-medium text-gray-500 dark:text-gray-400 mb-3 text-center">
        Strength Score
      </h3>

      <div className="flex justify-center">
        <div className="relative" style={{ width: size, height: size }}>
          {/* Background circle */}
          <svg className="transform -rotate-90" width={size} height={size}>
            <circle
              cx={size / 2}
              cy={size / 2}
              r={radius}
              fill="none"
              stroke="currentColor"
              strokeWidth={strokeWidth}
              className="text-gray-100 dark:text-gray-700"
            />
            {/* Progress circle */}
            <circle
              cx={size / 2}
              cy={size / 2}
              r={radius}
              fill="none"
              stroke={color}
              strokeWidth={strokeWidth}
              strokeLinecap="round"
              strokeDasharray={circumference}
              strokeDashoffset={offset}
              className="transition-all duration-500 ease-out"
            />
          </svg>

          {/* Score text in center */}
          <div className="absolute inset-0 flex flex-col items-center justify-center">
            <span
              className="text-3xl font-bold"
              style={{ color }}
            >
              {score}
            </span>
            <span className="text-xs text-gray-500 dark:text-gray-400">
              {getScoreLabel()}
            </span>
          </div>
        </div>
      </div>

      {/* Score range indicator */}
      <div className="mt-4 flex justify-between text-xs text-gray-400 dark:text-gray-500">
        <span>0</span>
        <span>25</span>
        <span>50</span>
        <span>75</span>
        <span>100</span>
      </div>
      <div className="h-1.5 rounded-full flex overflow-hidden mt-1">
        <div className="flex-1 bg-[#2ECC71]" />
        <div className="flex-1 bg-[#3498DB]" />
        <div className="flex-1 bg-[#F39C12]" />
        <div className="flex-1 bg-[#E74C3C]" />
      </div>
    </div>
  )
}
