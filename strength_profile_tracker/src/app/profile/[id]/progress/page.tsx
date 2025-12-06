'use client'

import { use } from 'react'
import Link from 'next/link'
import { ThemeToggle } from '@/components/ui'

// Mock data for visualization demos
const MOCK_WEIGHT_PROGRESS = [
  { date: 'Nov 20', weight: 65 },
  { date: 'Nov 25', weight: 67.5 },
  { date: 'Nov 30', weight: 70 },
  { date: 'Dec 3', weight: 72.5 },
  { date: 'Dec 6', weight: 75 },
]

const MOCK_BODY_PARTS = [
  { name: 'Chest', value: 75 },
  { name: 'Back', value: 60 },
  { name: 'Shoulders', value: 50 },
  { name: 'Legs', value: 40 },
  { name: 'Arms', value: 85 },
]

const MOCK_CALENDAR = [
  // Week 1
  [0, 1, 0, 2, 0, 3, 0],
  // Week 2
  [1, 0, 2, 0, 1, 0, 0],
  // Week 3
  [0, 2, 0, 3, 0, 2, 1],
  // Week 4
  [2, 0, 1, 0, 3, 0, 0],
]

const MOCK_PRS = [
  { exercise: 'Bench Press', weight: 80, date: 'Dec 5', improvement: '+5kg' },
  { exercise: 'Squat', weight: 100, date: 'Dec 3', improvement: '+10kg' },
  { exercise: 'Deadlift', weight: 120, date: 'Nov 28', improvement: '+7.5kg' },
]

const MOCK_SCORE_HISTORY = [
  { date: 'Nov 1', score: 35 },
  { date: 'Nov 15', score: 42 },
  { date: 'Dec 1', score: 55 },
  { date: 'Dec 6', score: 63 },
]

interface ProgressPageProps {
  params: Promise<{ id: string }>
}

export default function ProgressPage({ params }: ProgressPageProps) {
  const { id } = use(params)

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
            <h1 className="text-lg font-semibold">Progress Visualizations</h1>
          </div>
          <ThemeToggle />
        </div>
      </header>

      <main className="p-4 max-w-lg mx-auto space-y-4">
        {/* Banner */}
        <div className="bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-800 rounded-lg p-3 text-center">
          <p className="text-xs text-amber-700 dark:text-amber-300">
            These are mockup visualizations showing potential features
          </p>
        </div>

        {/* 1. Weight Progression Chart */}
        <WeightProgressionChart />

        {/* 2. Body Part Radar Chart */}
        <BodyPartRadarChart />

        {/* 3. Calendar Heatmap */}
        <CalendarHeatmap />

        {/* 4. Personal Records */}
        <PersonalRecords />

        {/* 5. Score History */}
        <ScoreHistory />
      </main>
    </div>
  )
}

// 1. Weight Progression Chart
function WeightProgressionChart() {
  const maxWeight = Math.max(...MOCK_WEIGHT_PROGRESS.map(d => d.weight))
  const minWeight = Math.min(...MOCK_WEIGHT_PROGRESS.map(d => d.weight))
  const range = maxWeight - minWeight || 1

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-4">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-semibold text-gray-700 dark:text-gray-200">
          Weight Progression
        </h3>
        <span className="text-xs text-gray-400 px-2 py-1 bg-gray-100 dark:bg-gray-700 rounded">
          Bench Press
        </span>
      </div>

      {/* Chart */}
      <div className="h-32 flex items-end justify-between gap-1 mb-2">
        {MOCK_WEIGHT_PROGRESS.map((point, i) => {
          const height = ((point.weight - minWeight) / range) * 100
          const isLast = i === MOCK_WEIGHT_PROGRESS.length - 1

          return (
            <div key={i} className="flex-1 flex flex-col items-center">
              <span className="text-[10px] text-gray-500 dark:text-gray-400 mb-1">
                {point.weight}kg
              </span>
              <div
                className={`w-full rounded-t transition-all ${
                  isLast ? 'bg-green-500' : 'bg-blue-400 dark:bg-blue-500'
                }`}
                style={{ height: `${Math.max(height, 10)}%` }}
              />
            </div>
          )
        })}
      </div>

      {/* X-axis labels */}
      <div className="flex justify-between">
        {MOCK_WEIGHT_PROGRESS.map((point, i) => (
          <span key={i} className="text-[10px] text-gray-400 flex-1 text-center">
            {point.date}
          </span>
        ))}
      </div>

      {/* Trend indicator */}
      <div className="mt-3 pt-3 border-t border-gray-100 dark:border-gray-700 flex items-center justify-center gap-2">
        <span className="text-green-500 text-sm">↑ +10kg</span>
        <span className="text-xs text-gray-400">in last 3 weeks</span>
      </div>
    </div>
  )
}

// 2. Body Part Radar Chart
function BodyPartRadarChart() {
  const centerX = 80
  const centerY = 80
  const maxRadius = 60

  // Calculate points for pentagon
  const getPoint = (index: number, value: number) => {
    const angle = (index * 72 - 90) * (Math.PI / 180)
    const radius = (value / 100) * maxRadius
    return {
      x: centerX + radius * Math.cos(angle),
      y: centerY + radius * Math.sin(angle)
    }
  }

  const points = MOCK_BODY_PARTS.map((part, i) => getPoint(i, part.value))
  const pathD = points.map((p, i) => `${i === 0 ? 'M' : 'L'} ${p.x} ${p.y}`).join(' ') + ' Z'

  // Grid lines (25%, 50%, 75%, 100%)
  const gridLevels = [25, 50, 75, 100]

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-4">
      <h3 className="text-sm font-semibold text-gray-700 dark:text-gray-200 mb-3">
        Body Part Balance
      </h3>

      <div className="flex justify-center">
        <svg width="160" height="160" viewBox="0 0 160 160">
          {/* Grid circles */}
          {gridLevels.map(level => {
            const gridPoints = MOCK_BODY_PARTS.map((_, i) => getPoint(i, level))
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

          {/* Axis lines */}
          {MOCK_BODY_PARTS.map((_, i) => {
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

          {/* Data polygon */}
          <path
            d={pathD}
            fill="rgba(59, 130, 246, 0.3)"
            stroke="#3B82F6"
            strokeWidth="2"
          />

          {/* Data points */}
          {points.map((p, i) => (
            <circle key={i} cx={p.x} cy={p.y} r="4" fill="#3B82F6" />
          ))}

          {/* Labels */}
          {MOCK_BODY_PARTS.map((part, i) => {
            const labelPoint = getPoint(i, 120)
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

      {/* Legend */}
      <div className="mt-3 flex flex-wrap justify-center gap-2">
        {MOCK_BODY_PARTS.map(part => (
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
    </div>
  )
}

// 3. Calendar Heatmap
function CalendarHeatmap() {
  const days = ['S', 'M', 'T', 'W', 'T', 'F', 'S']
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

      {/* Day labels */}
      <div className="flex gap-1 mb-1">
        <div className="w-8"></div>
        {days.map((day, i) => (
          <div key={i} className="w-6 text-center text-[10px] text-gray-400">
            {day}
          </div>
        ))}
      </div>

      {/* Weeks */}
      {MOCK_CALENDAR.map((week, weekIndex) => (
        <div key={weekIndex} className="flex gap-1 mb-1">
          <div className="w-8 text-[10px] text-gray-400 flex items-center">
            W{weekIndex + 1}
          </div>
          {week.map((value, dayIndex) => (
            <div
              key={dayIndex}
              className={`w-6 h-6 rounded ${getColor(value)}`}
              title={`${value} workouts`}
            />
          ))}
        </div>
      ))}

      {/* Legend */}
      <div className="flex items-center justify-center gap-2 mt-3 pt-3 border-t border-gray-100 dark:border-gray-700">
        <span className="text-[10px] text-gray-400">Less</span>
        <div className="w-4 h-4 rounded bg-gray-100 dark:bg-gray-700" />
        <div className="w-4 h-4 rounded bg-green-200 dark:bg-green-900" />
        <div className="w-4 h-4 rounded bg-green-400 dark:bg-green-700" />
        <div className="w-4 h-4 rounded bg-green-600 dark:bg-green-500" />
        <span className="text-[10px] text-gray-400">More</span>
      </div>

      {/* Stats */}
      <div className="flex justify-around mt-3 text-center">
        <div>
          <p className="text-lg font-bold text-green-600 dark:text-green-400">12</p>
          <p className="text-[10px] text-gray-400">Total Workouts</p>
        </div>
        <div>
          <p className="text-lg font-bold text-blue-600 dark:text-blue-400">3</p>
          <p className="text-[10px] text-gray-400">This Week</p>
        </div>
        <div>
          <p className="text-lg font-bold text-amber-600 dark:text-amber-400">5</p>
          <p className="text-[10px] text-gray-400">Day Streak</p>
        </div>
      </div>
    </div>
  )
}

// 4. Personal Records
function PersonalRecords() {
  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-4">
      <div className="flex items-center gap-2 mb-3">
        <span className="text-lg">🏆</span>
        <h3 className="text-sm font-semibold text-gray-700 dark:text-gray-200">
          Personal Records
        </h3>
      </div>

      <div className="space-y-2">
        {MOCK_PRS.map((pr, i) => (
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
              <p className="text-[10px] text-green-600 dark:text-green-400">
                {pr.improvement}
              </p>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

// 5. Score History
function ScoreHistory() {
  const maxScore = 100
  const chartHeight = 80

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-4">
      <h3 className="text-sm font-semibold text-gray-700 dark:text-gray-200 mb-3">
        Strength Score History
      </h3>

      {/* Line Chart */}
      <div className="relative h-24 mb-2">
        {/* Y-axis labels */}
        <div className="absolute left-0 top-0 bottom-0 w-8 flex flex-col justify-between text-[10px] text-gray-400">
          <span>100</span>
          <span>50</span>
          <span>0</span>
        </div>

        {/* Chart area */}
        <div className="ml-8 h-full relative">
          {/* Grid lines */}
          <div className="absolute inset-0 flex flex-col justify-between">
            <div className="border-b border-gray-100 dark:border-gray-700" />
            <div className="border-b border-gray-100 dark:border-gray-700" />
            <div className="border-b border-gray-100 dark:border-gray-700" />
          </div>

          {/* SVG Line */}
          <svg className="absolute inset-0 w-full h-full overflow-visible">
            {/* Line path */}
            <polyline
              fill="none"
              stroke="#3B82F6"
              strokeWidth="2"
              points={MOCK_SCORE_HISTORY.map((point, i) => {
                const x = (i / (MOCK_SCORE_HISTORY.length - 1)) * 100 + '%'
                const y = chartHeight - (point.score / maxScore) * chartHeight
                return `${(i / (MOCK_SCORE_HISTORY.length - 1)) * 100}%,${y}`
              }).join(' ')}
            />

            {/* Data points */}
            {MOCK_SCORE_HISTORY.map((point, i) => {
              const x = (i / (MOCK_SCORE_HISTORY.length - 1)) * 100
              const y = chartHeight - (point.score / maxScore) * chartHeight
              const isLast = i === MOCK_SCORE_HISTORY.length - 1

              return (
                <g key={i}>
                  <circle
                    cx={`${x}%`}
                    cy={y}
                    r={isLast ? 6 : 4}
                    fill={isLast ? '#22C55E' : '#3B82F6'}
                  />
                  {isLast && (
                    <text
                      x={`${x}%`}
                      y={y - 12}
                      textAnchor="middle"
                      className="text-xs font-bold fill-green-600 dark:fill-green-400"
                    >
                      {point.score}
                    </text>
                  )}
                </g>
              )
            })}
          </svg>
        </div>
      </div>

      {/* X-axis labels */}
      <div className="ml-8 flex justify-between">
        {MOCK_SCORE_HISTORY.map((point, i) => (
          <span key={i} className="text-[10px] text-gray-400">
            {point.date}
          </span>
        ))}
      </div>

      {/* Progress summary */}
      <div className="mt-3 pt-3 border-t border-gray-100 dark:border-gray-700 flex items-center justify-center gap-4">
        <div className="text-center">
          <p className="text-xs text-gray-400">Started</p>
          <p className="font-semibold text-gray-600 dark:text-gray-300">35</p>
        </div>
        <div className="text-2xl">→</div>
        <div className="text-center">
          <p className="text-xs text-gray-400">Current</p>
          <p className="font-semibold text-green-600 dark:text-green-400">63</p>
        </div>
        <div className="text-center bg-green-100 dark:bg-green-900/30 px-2 py-1 rounded">
          <p className="text-xs text-green-600 dark:text-green-400 font-medium">+28 pts</p>
        </div>
      </div>
    </div>
  )
}
