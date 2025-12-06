'use client'

import { useState, useEffect } from 'react'
import { Quote } from '@/types'
import { getDailyQuote, getRandomQuote } from '@/data/quotes'

interface MotivationalQuoteProps {
  variant?: 'daily' | 'random'
}

export default function MotivationalQuote({ variant = 'daily' }: MotivationalQuoteProps) {
  const [quote, setQuote] = useState<Quote | null>(null)
  const [isRefreshing, setIsRefreshing] = useState(false)

  useEffect(() => {
    setQuote(variant === 'daily' ? getDailyQuote() : getRandomQuote())
  }, [variant])

  const handleRefresh = () => {
    setIsRefreshing(true)
    setTimeout(() => {
      setQuote(getRandomQuote())
      setIsRefreshing(false)
    }, 300)
  }

  if (!quote) return null

  const categoryColors = {
    motivation: 'bg-purple-50 dark:bg-purple-900/30 border-purple-200 dark:border-purple-800',
    science: 'bg-blue-50 dark:bg-blue-900/30 border-blue-200 dark:border-blue-800',
    benefit: 'bg-green-50 dark:bg-green-900/30 border-green-200 dark:border-green-800'
  }

  const categoryIcons = {
    motivation: (
      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
      </svg>
    ),
    science: (
      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
      </svg>
    ),
    benefit: (
      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z" />
      </svg>
    )
  }

  const categoryLabels = {
    motivation: 'Motivation',
    science: 'Science',
    benefit: 'Health Benefit'
  }

  return (
    <div className={`rounded-lg border p-4 ${categoryColors[quote.category]} transition-all duration-300 ${isRefreshing ? 'opacity-50' : 'opacity-100'}`}>
      <div className="flex items-start justify-between gap-2 mb-2">
        <div className="flex items-center gap-1.5 text-xs text-gray-500">
          {categoryIcons[quote.category]}
          <span>{categoryLabels[quote.category]}</span>
        </div>
        <button
          onClick={handleRefresh}
          disabled={isRefreshing}
          className="p-1.5 rounded-full hover:bg-white/50 transition-colors"
          aria-label="Get new quote"
        >
          <svg
            className={`w-4 h-4 text-gray-400 ${isRefreshing ? 'animate-spin' : ''}`}
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"
            />
          </svg>
        </button>
      </div>

      <blockquote className="text-sm text-gray-700 dark:text-gray-300 italic leading-relaxed">
        "{quote.text}"
      </blockquote>

      <p className="text-xs text-gray-500 dark:text-gray-400 mt-2 text-right">
        — {quote.author}
      </p>
    </div>
  )
}
