'use client'

import { use, useEffect, useState } from 'react'
import Link from 'next/link'
import { ThemeToggle } from '@/components/ui'
import { Profile, BodyPart, Exercise } from '@/types'
import { getProfileById } from '@/lib/storage/profiles'
import { calculateStrengthScore, EXERCISES } from '@/lib/calculations/strength'
import { getAllWorkouts, formatSessionDate } from '@/lib/storage/workouts'
import { getScoreHistory, ScoreHistoryEntry } from '@/lib/storage/seedData'

interface ProgressPageProps {
  params: Promise<{ id: string }>
}

interface BodyPartData {
  name: string
  value: number
}

interface PRData {
  exercise: string
  weight: number
  date: string
  reps: number
}

interface WorkoutDay {
  date: string
  count: number
}

export default function ProgressPage({ params }: ProgressPageProps) {
  const { id } = use(params)
  const [profile, setProfile] = useState<Profile | null>(null)
  const [bodyPartData, setBodyPartData] = useState<BodyPartData[]>([])
  const [prs, setPrs] = useState<PRData[]>([])
  const [workoutDays, setWorkoutDays] = useState<WorkoutDay[]>([])
  const [currentScore, setCurrentScore] = useState(0)
  const [scoreHistory, setScoreHistory] = useState<ScoreHistoryEntry[]>([])
  const [exerciseProgress, setExerciseProgress] = useState<{exercise: string, data: {date: string, weight: number}[]}[]>([])
  const [profileColor, setProfileColor] = useState('#3B82F6')
  const [isLoading, setIsLoading] = useState(true)

  useEffect(() => {
    const loadedProfile = getProfileById(id)
    if (!loadedProfile) {
      setIsLoading(false)
      return
    }

    setProfile(loadedProfile)
    const score = calculateStrengthScore(loadedProfile.exerciseRatings)
    setCurrentScore(score)

    // Generate profile color based on ID
    const colors = ['#3B82F6', '#8B5CF6', '#EC4899', '#F59E0B', '#10B981', '#06B6D4']
    const colorIndex = id.split('').reduce((acc, char) => acc + char.charCodeAt(0), 0) % colors.length
    setProfileColor(colors[colorIndex])

    // Load score history - generate from workout data if empty
    let history = getScoreHistory(id)

    // If no history but has workouts, generate from workout dates
    if (history.length < 2) {
      const allWorkouts = getAllWorkouts().filter(w => w.profileId === id)
      if (allWorkouts.length > 0) {
        // Get unique workout dates sorted
        const workoutDates = [...new Set(allWorkouts.map(w => w.date))].sort()

        // Generate score progression (simulating gradual improvement)
        const baseScore = Math.max(20, score - workoutDates.length * 2)
        history = workoutDates.slice(-8).map((date, i, arr) => ({
          date,
          score: Math.round(baseScore + ((score - baseScore) * (i + 1) / arr.length))
        }))
      }
    }
    setScoreHistory(history)

    // Calculate body part balance from ratings
    const bodyPartLevels: Record<BodyPart, number[]> = {
      chest: [], back: [], shoulders: [], legs: [], arms: [], core: []
    }
    const levelValues = { beginner: 25, novice: 50, intermediate: 75, advanced: 100 }

    Object.entries(loadedProfile.exerciseRatings).forEach(([exerciseId, level]) => {
      if (level) {
        const exercise = EXERCISES.find(e => e.id === exerciseId)
        if (exercise) {
          bodyPartLevels[exercise.bodyPart].push(levelValues[level])
        }
      }
    })

    const bodyParts: BodyPartData[] = [
      { name: 'Chest', value: avg(bodyPartLevels.chest) },
      { name: 'Back', value: avg(bodyPartLevels.back) },
      { name: 'Shoulders', value: avg(bodyPartLevels.shoulders) },
      { name: 'Legs', value: avg(bodyPartLevels.legs) },
      { name: 'Arms', value: avg(bodyPartLevels.arms) },
    ]
    setBodyPartData(bodyParts)

    // Load workout data
    const allWorkouts = getAllWorkouts().filter(w => w.profileId === id)

    // Calculate PRs (max weight per exercise)
    const prMap: Record<string, PRData> = {}
    allWorkouts.forEach(session => {
      const exercise = EXERCISES.find(e => e.id === session.exerciseId)
      if (!exercise) return

      session.sets.forEach(set => {
        if (set.weight && set.reps) {
          const key = session.exerciseId
          if (!prMap[key] || set.weight > prMap[key].weight) {
            prMap[key] = {
              exercise: exercise.name,
              weight: set.weight,
              date: formatSessionDate(session.date),
              reps: set.reps
            }
          }
        }
      })
    })

    const sortedPrs = Object.values(prMap)
      .sort((a, b) => b.weight - a.weight)
      .slice(0, 3)
    setPrs(sortedPrs)

    // Calculate workout frequency (sessions per day)
    const dayCount: Record<string, number> = {}
    allWorkouts.forEach(session => {
      dayCount[session.date] = (dayCount[session.date] || 0) + 1
    })

    const days = Object.entries(dayCount)
      .map(([date, count]) => ({ date, count }))
      .sort((a, b) => a.date.localeCompare(b.date))
      .slice(-28) // Last 4 weeks
    setWorkoutDays(days)

    // Calculate exercise progression (group by exercise, sort by date)
    const progressMap: Record<string, {date: string, weight: number}[]> = {}
    allWorkouts.forEach(session => {
      const weightsWithValues = session.sets.filter(s => s.weight && s.weight > 0).map(s => s.weight!)
      if (weightsWithValues.length > 0) {
        const maxWeight = Math.max(...weightsWithValues)
        if (!progressMap[session.exerciseId]) {
          progressMap[session.exerciseId] = []
        }
        progressMap[session.exerciseId].push({
          date: session.date,
          weight: maxWeight
        })
      }
    })

    const progress = Object.entries(progressMap)
      .map(([exerciseId, data]) => {
        const exercise = EXERCISES.find(e => e.id === exerciseId)
        return {
          exercise: exercise?.name || exerciseId,
          data: data.sort((a, b) => a.date.localeCompare(b.date)).slice(-5)
        }
      })
      .filter(p => p.data.length >= 2)

    setExerciseProgress(progress)
    setIsLoading(false)
  }, [id])

  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-gray-500">Loading...</div>
      </div>
    )
  }

  if (!profile) {
    return (
      <div className="min-h-screen flex flex-col items-center justify-center p-4">
        <p className="text-gray-500 mb-4">Profile not found</p>
        <Link href="/" className="text-blue-500 hover:underline">Go back</Link>
      </div>
    )
  }

  return (
    <div className="min-h-screen pb-20 bg-gray-50 dark:bg-gray-900">
      {/* Header */}
      <header className="bg-[#2C3E50] text-white px-4 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Link href={`/profile/${id}`} className="text-white hover:text-gray-300">
              <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
              </svg>
            </Link>
            <h1 className="text-lg font-semibold">Progress</h1>
          </div>
          <ThemeToggle />
        </div>
      </header>

      <main className="p-4 max-w-lg mx-auto space-y-4">
        {/* 1. Score History Chart */}
        <ScoreHistoryChart history={scoreHistory} currentScore={currentScore} color={profileColor} />

        {/* 2. Personal Records */}
        <PersonalRecords prs={prs} />

        {/* 3. Body Part Balance */}
        <BodyPartRadarChart data={bodyPartData} color={profileColor} />

        {/* 4. Workout Frequency */}
        <CalendarHeatmap days={workoutDays} />

        {/* 5. Exercise Progression */}
        {exerciseProgress.map((prog, index) => (
          <ExerciseProgressionChart
            key={index}
            exercise={prog.exercise}
            color={profileColor}
            data={prog.data}
          />
        ))}
      </main>
    </div>
  )
}

// Helper function
function avg(arr: number[]): number {
  if (arr.length === 0) return 0
  return Math.round(arr.reduce((a, b) => a + b, 0) / arr.length)
}

// 1. Score History Chart
function ScoreHistoryChart({ history, currentScore, color }: { history: ScoreHistoryEntry[], currentScore: number, color: string }) {
  const getLabel = (score: number) => {
    if (score >= 88) return 'Elite'
    if (score >= 63) return 'Strong'
    if (score >= 38) return 'Building'
    if (score > 0) return 'Starting'
    return 'Unrated'
  }

  // If no history, show just current score
  if (history.length < 2) {
    return (
      <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-4">
        <h3 className="text-sm font-semibold text-gray-700 dark:text-gray-200 mb-3">
          Strength Score History
        </h3>
        <div className="text-center py-4">
          <p className="text-5xl font-bold" style={{ color }}>{currentScore}</p>
          <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">{getLabel(currentScore)}</p>
        </div>
        <div className="mt-3 h-2 bg-gray-100 dark:bg-gray-700 rounded-full overflow-hidden">
          <div
            className="h-full transition-all"
            style={{ width: `${currentScore}%`, backgroundColor: color }}
          />
        </div>
        <p className="text-xs text-gray-400 text-center mt-3">
          Keep training to see your score progression!
        </p>
      </div>
    )
  }

  // Chart dimensions
  const chartWidth = 300
  const chartHeight = 140
  const paddingLeft = 35
  const paddingRight = 15
  const paddingTop = 20
  const paddingBottom = 30

  const plotWidth = chartWidth - paddingLeft - paddingRight
  const plotHeight = chartHeight - paddingTop - paddingBottom

  const scores = history.map(h => h.score)
  const minScore = Math.min(...scores) - 5
  const maxScore = Math.max(...scores) + 5
  const scoreRange = maxScore - minScore

  const points = history.map((entry, i) => ({
    x: paddingLeft + (i / (history.length - 1)) * plotWidth,
    y: paddingTop + plotHeight - ((entry.score - minScore) / scoreRange) * plotHeight,
    score: entry.score,
    date: formatSessionDate(entry.date)
  }))

  const linePath = points.map((p, i) => `${i === 0 ? 'M' : 'L'} ${p.x} ${p.y}`).join(' ')
  const areaPath = `${linePath} L ${points[points.length - 1].x} ${paddingTop + plotHeight} L ${points[0].x} ${paddingTop + plotHeight} Z`

  const firstScore = history[0].score
  const lastScore = history[history.length - 1].score
  const improvement = lastScore - firstScore

  // Generate unique gradient ID based on color
  const gradientId = `scoreGradient-${color.replace('#', '')}`

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-4">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-semibold text-gray-700 dark:text-gray-200">
          Strength Score History
        </h3>
        <div className="flex items-center gap-2">
          <span className="text-2xl font-bold" style={{ color }}>{currentScore}</span>
          <span className="text-xs text-gray-400">{getLabel(currentScore)}</span>
        </div>
      </div>

      <svg width="100%" viewBox={`0 0 ${chartWidth} ${chartHeight}`} className="overflow-visible">
        <defs>
          <linearGradient id={gradientId} x1="0%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" stopColor={color} stopOpacity="0.3" />
            <stop offset="100%" stopColor={color} stopOpacity="0.05" />
          </linearGradient>
        </defs>

        {/* Grid lines */}
        {[0, 0.5, 1].map((ratio, i) => {
          const y = paddingTop + plotHeight * (1 - ratio)
          const value = Math.round(minScore + scoreRange * ratio)
          return (
            <g key={i}>
              <line
                x1={paddingLeft}
                y1={y}
                x2={chartWidth - paddingRight}
                y2={y}
                stroke="currentColor"
                strokeWidth="1"
                strokeDasharray="3,3"
                className="text-gray-200 dark:text-gray-700"
              />
              <text
                x={paddingLeft - 5}
                y={y}
                textAnchor="end"
                dominantBaseline="middle"
                className="text-[9px] fill-gray-400"
              >
                {value}
              </text>
            </g>
          )
        })}

        {/* Area fill */}
        <path d={areaPath} fill={`url(#${gradientId})`} />

        {/* Line */}
        <path
          d={linePath}
          fill="none"
          stroke={color}
          strokeWidth="2.5"
          strokeLinecap="round"
          strokeLinejoin="round"
        />

        {/* Points */}
        {points.map((p, i) => {
          const isLast = i === points.length - 1
          return (
            <g key={i}>
              <circle
                cx={p.x}
                cy={p.y}
                r={isLast ? 5 : 3}
                fill={isLast ? '#22C55E' : color}
                stroke="white"
                strokeWidth="2"
              />
              {/* Show score on first and last points */}
              {(i === 0 || isLast) && (
                <text
                  x={p.x}
                  y={p.y - 10}
                  textAnchor="middle"
                  className={`text-[9px] font-medium ${isLast ? 'fill-green-600 dark:fill-green-400' : 'fill-gray-500 dark:fill-gray-400'}`}
                >
                  {p.score}
                </text>
              )}
              {/* Date labels */}
              {(i === 0 || isLast || i === Math.floor(points.length / 2)) && (
                <text
                  x={p.x}
                  y={chartHeight - 5}
                  textAnchor="middle"
                  className="text-[9px] fill-gray-400"
                >
                  {p.date}
                </text>
              )}
            </g>
          )
        })}
      </svg>

      {/* Progress indicator */}
      <div className="mt-3 pt-3 border-t border-gray-100 dark:border-gray-700 flex items-center justify-center gap-2">
        <span className={`text-sm font-medium ${improvement >= 0 ? 'text-green-500' : 'text-red-500'}`}>
          {improvement >= 0 ? '↑' : '↓'} {improvement >= 0 ? '+' : ''}{improvement} pts
        </span>
        <span className="text-xs text-gray-400">since tracking started</span>
      </div>
    </div>
  )
}

// 2. Personal Records
function PersonalRecords({ prs }: { prs: PRData[] }) {
  if (prs.length === 0) {
    return (
      <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-4">
        <div className="flex items-center gap-2 mb-3">
          <span className="text-lg">🏆</span>
          <h3 className="text-sm font-semibold text-gray-700 dark:text-gray-200">
            Personal Records
          </h3>
        </div>
        <p className="text-sm text-gray-400 text-center py-4">
          Log workouts to see your PRs here!
        </p>
      </div>
    )
  }

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-4">
      <div className="flex items-center gap-2 mb-3">
        <span className="text-lg">🏆</span>
        <h3 className="text-sm font-semibold text-gray-700 dark:text-gray-200">
          Personal Records
        </h3>
      </div>

      <div className="space-y-2">
        {prs.map((pr, i) => (
          <div
            key={i}
            className="flex items-center justify-between p-2 bg-gradient-to-r from-amber-50 to-yellow-50 dark:from-amber-900/20 dark:to-yellow-900/20 rounded-lg border border-amber-200 dark:border-amber-800"
          >
            <div className="flex items-center gap-2">
              <span className="text-xl">
                {i === 0 ? '🥇' : i === 1 ? '🥈' : '🥉'}
              </span>
              <div>
                <p className="text-sm font-medium text-gray-700 dark:text-gray-200">
                  {pr.exercise}
                </p>
                <p className="text-[10px] text-gray-400">{pr.date}</p>
              </div>
            </div>
            <div className="text-right">
              <p className="text-lg font-bold text-amber-600 dark:text-amber-400">
                {pr.weight}kg
              </p>
              <p className="text-[10px] text-gray-500 dark:text-gray-400">
                × {pr.reps} reps
              </p>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

// 3. Body Part Radar Chart
function BodyPartRadarChart({ data, color }: { data: BodyPartData[], color: string }) {
  const centerX = 80
  const centerY = 80
  const maxRadius = 60

  const hasData = data.some(d => d.value > 0)

  // Convert hex to rgba for fill
  const hexToRgba = (hex: string, alpha: number) => {
    const r = parseInt(hex.slice(1, 3), 16)
    const g = parseInt(hex.slice(3, 5), 16)
    const b = parseInt(hex.slice(5, 7), 16)
    return `rgba(${r}, ${g}, ${b}, ${alpha})`
  }

  const getPoint = (index: number, value: number) => {
    const angle = (index * 72 - 90) * (Math.PI / 180)
    const radius = (value / 100) * maxRadius
    return {
      x: centerX + radius * Math.cos(angle),
      y: centerY + radius * Math.sin(angle)
    }
  }

  const points = data.map((part, i) => getPoint(i, part.value))
  const pathD = points.map((p, i) => `${i === 0 ? 'M' : 'L'} ${p.x} ${p.y}`).join(' ') + ' Z'
  const gridLevels = [25, 50, 75, 100]

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-4">
      <h3 className="text-sm font-semibold text-gray-700 dark:text-gray-200 mb-3">
        Body Part Balance
      </h3>

      {!hasData ? (
        <p className="text-sm text-gray-400 text-center py-8">
          Rate exercises to see your balance chart
        </p>
      ) : (
        <>
          <div className="flex justify-center">
            <svg width="160" height="160" viewBox="0 0 160 160">
              {gridLevels.map(level => {
                const gridPoints = data.map((_, i) => getPoint(i, level))
                const gridPath = gridPoints.map((p, i) => `${i === 0 ? 'M' : 'L'} ${p.x} ${p.y}`).join(' ') + ' Z'
                return (
                  <path
                    key={level}
                    d={gridPath}
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="1"
                    className="text-gray-200 dark:text-gray-700"
                  />
                )
              })}

              {data.map((_, i) => {
                const outer = getPoint(i, 100)
                return (
                  <line
                    key={i}
                    x1={centerX}
                    y1={centerY}
                    x2={outer.x}
                    y2={outer.y}
                    stroke="currentColor"
                    strokeWidth="1"
                    className="text-gray-200 dark:text-gray-700"
                  />
                )
              })}

              <path
                d={pathD}
                fill={hexToRgba(color, 0.3)}
                stroke={color}
                strokeWidth="2"
              />

              {points.map((p, i) => (
                <circle key={i} cx={p.x} cy={p.y} r="4" fill={color} />
              ))}

              {data.map((part, i) => {
                const labelPoint = getPoint(i, 125)
                return (
                  <text
                    key={i}
                    x={labelPoint.x}
                    y={labelPoint.y}
                    textAnchor="middle"
                    dominantBaseline="middle"
                    className="text-[10px] fill-gray-500 dark:fill-gray-400"
                  >
                    {part.name}
                  </text>
                )
              })}
            </svg>
          </div>

          <div className="mt-3 flex flex-wrap justify-center gap-2">
            {data.filter(p => p.value > 0).map(part => (
              <div key={part.name} className="flex items-center gap-1">
                <div
                  className="w-2 h-2 rounded-full"
                  style={{
                    backgroundColor: part.value >= 75 ? '#22C55E' : part.value >= 50 ? '#3B82F6' : '#F59E0B'
                  }}
                />
                <span className="text-[10px] text-gray-500 dark:text-gray-400">
                  {part.name}: {part.value}%
                </span>
              </div>
            ))}
          </div>
        </>
      )}
    </div>
  )
}

// 4. Calendar Heatmap
function CalendarHeatmap({ days }: { days: WorkoutDay[] }) {
  // Build 4 weeks of data
  const today = new Date()
  const weeks: number[][] = []

  for (let w = 3; w >= 0; w--) {
    const week: number[] = []
    for (let d = 0; d < 7; d++) {
      const date = new Date(today)
      date.setDate(date.getDate() - (w * 7 + (6 - d)))
      const dateStr = date.toISOString().split('T')[0]
      const dayData = days.find(day => day.date === dateStr)
      week.push(dayData?.count || 0)
    }
    weeks.push(week)
  }

  const totalWorkouts = days.reduce((sum, d) => sum + d.count, 0)
  const thisWeekWorkouts = weeks[3]?.reduce((sum, d) => sum + d, 0) || 0

  const dayLabels = ['S', 'M', 'T', 'W', 'T', 'F', 'S']

  const getColor = (value: number) => {
    if (value === 0) return 'bg-gray-100 dark:bg-gray-700'
    if (value === 1) return 'bg-green-200 dark:bg-green-900'
    if (value === 2) return 'bg-green-400 dark:bg-green-700'
    return 'bg-green-600 dark:bg-green-500'
  }

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-4">
      <h3 className="text-sm font-semibold text-gray-700 dark:text-gray-200 mb-3">
        Workout Frequency
      </h3>

      <div className="flex gap-1 mb-1">
        <div className="w-8"></div>
        {dayLabels.map((day, i) => (
          <div key={i} className="w-6 text-center text-[10px] text-gray-400">
            {day}
          </div>
        ))}
      </div>

      {weeks.map((week, weekIndex) => (
        <div key={weekIndex} className="flex gap-1 mb-1">
          <div className="w-8 text-[10px] text-gray-400 flex items-center">
            W{weekIndex + 1}
          </div>
          {week.map((value, dayIndex) => (
            <div
              key={dayIndex}
              className={`w-6 h-6 rounded ${getColor(value)}`}
              title={`${value} exercises logged`}
            />
          ))}
        </div>
      ))}

      <div className="flex items-center justify-center gap-2 mt-3 pt-3 border-t border-gray-100 dark:border-gray-700">
        <span className="text-[10px] text-gray-400">Less</span>
        <div className="w-4 h-4 rounded bg-gray-100 dark:bg-gray-700" />
        <div className="w-4 h-4 rounded bg-green-200 dark:bg-green-900" />
        <div className="w-4 h-4 rounded bg-green-400 dark:bg-green-700" />
        <div className="w-4 h-4 rounded bg-green-600 dark:bg-green-500" />
        <span className="text-[10px] text-gray-400">More</span>
      </div>

      <div className="flex justify-around mt-3 text-center">
        <div>
          <p className="text-lg font-bold text-green-600 dark:text-green-400">{totalWorkouts}</p>
          <p className="text-[10px] text-gray-400">Total Logged</p>
        </div>
        <div>
          <p className="text-lg font-bold text-blue-600 dark:text-blue-400">{thisWeekWorkouts}</p>
          <p className="text-[10px] text-gray-400">This Week</p>
        </div>
      </div>
    </div>
  )
}

// 5. Exercise Progression Chart
function ExerciseProgressionChart({ exercise, data, color }: { exercise: string, data: {date: string, weight: number}[], color: string }) {
  if (data.length < 2) return null

  const maxWeight = Math.max(...data.map(d => d.weight))
  const minWeight = Math.min(...data.map(d => d.weight))
  const padding = (maxWeight - minWeight) * 0.15 || 5
  const yMin = minWeight - padding
  const yMax = maxWeight + padding
  const yRange = yMax - yMin

  const chartWidth = 280
  const chartHeight = 120
  const paddingLeft = 35
  const paddingRight = 10
  const paddingTop = 15
  const paddingBottom = 25

  const plotWidth = chartWidth - paddingLeft - paddingRight
  const plotHeight = chartHeight - paddingTop - paddingBottom

  const points = data.map((point, i) => ({
    x: paddingLeft + (i / (data.length - 1)) * plotWidth,
    y: paddingTop + plotHeight - ((point.weight - yMin) / yRange) * plotHeight,
    weight: point.weight,
    date: formatSessionDate(point.date)
  }))

  const linePath = points.map((p, i) => `${i === 0 ? 'M' : 'L'} ${p.x} ${p.y}`).join(' ')
  const areaPath = `${linePath} L ${points[points.length - 1].x} ${paddingTop + plotHeight} L ${points[0].x} ${paddingTop + plotHeight} Z`

  const improvement = data[data.length - 1].weight - data[0].weight

  // Generate unique gradient ID for this exercise
  const gradientId = `progressGradient-${exercise.replace(/\s+/g, '-')}`

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-4">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-semibold text-gray-700 dark:text-gray-200">
          Exercise Progression
        </h3>
        <span className="text-xs text-gray-400 px-2 py-1 bg-gray-100 dark:bg-gray-700 rounded">
          {exercise}
        </span>
      </div>

      <svg width="100%" viewBox={`0 0 ${chartWidth} ${chartHeight}`} className="overflow-visible">
        <defs>
          <linearGradient id={gradientId} x1="0%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" stopColor={color} stopOpacity="0.3" />
            <stop offset="100%" stopColor={color} stopOpacity="0.05" />
          </linearGradient>
        </defs>

        {[0, 0.5, 1].map((ratio, i) => {
          const y = paddingTop + plotHeight * (1 - ratio)
          const value = Math.round(yMin + yRange * ratio)
          return (
            <g key={i}>
              <line
                x1={paddingLeft}
                y1={y}
                x2={chartWidth - paddingRight}
                y2={y}
                stroke="currentColor"
                strokeWidth="1"
                strokeDasharray="3,3"
                className="text-gray-200 dark:text-gray-700"
              />
              <text
                x={paddingLeft - 5}
                y={y}
                textAnchor="end"
                dominantBaseline="middle"
                className="text-[9px] fill-gray-400"
              >
                {value}
              </text>
            </g>
          )
        })}

        <path d={areaPath} fill={`url(#${gradientId})`} />
        <path
          d={linePath}
          fill="none"
          stroke={color}
          strokeWidth="2.5"
          strokeLinecap="round"
          strokeLinejoin="round"
        />

        {points.map((p, i) => {
          const isLast = i === points.length - 1
          return (
            <g key={i}>
              <circle
                cx={p.x}
                cy={p.y}
                r={isLast ? 5 : 4}
                fill={isLast ? '#22C55E' : color}
                stroke="white"
                strokeWidth="2"
              />
              <text
                x={p.x}
                y={p.y - 10}
                textAnchor="middle"
                className={`text-[9px] font-medium ${isLast ? 'fill-green-600 dark:fill-green-400' : 'fill-gray-500 dark:fill-gray-400'}`}
              >
                {p.weight}kg
              </text>
              <text
                x={p.x}
                y={chartHeight - 5}
                textAnchor="middle"
                className="text-[9px] fill-gray-400"
              >
                {p.date}
              </text>
            </g>
          )
        })}
      </svg>

      <div className="mt-3 pt-3 border-t border-gray-100 dark:border-gray-700 flex items-center justify-center gap-2">
        <span className={`text-sm ${improvement >= 0 ? 'text-green-500' : 'text-red-500'}`}>
          {improvement >= 0 ? '↑' : '↓'} {improvement >= 0 ? '+' : ''}{improvement}kg
        </span>
        <span className="text-xs text-gray-400">progress</span>
      </div>
    </div>
  )
}
