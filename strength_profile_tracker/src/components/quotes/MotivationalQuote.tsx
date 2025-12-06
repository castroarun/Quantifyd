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

  return (
    <div className={`text-center py-4 transition-all duration-300 ${isRefreshing ? 'opacity-50' : 'opacity-100'}`}>
      <blockquote className="text-sm text-gray-500 dark:text-gray-400 italic leading-relaxed">
        "{quote.text}"
      </blockquote>

      <div className="flex items-center justify-center gap-3 mt-2">
        <p className="text-xs text-gray-400 dark:text-gray-500">
          — {quote.author}
        </p>
        <button
          onClick={handleRefresh}
          disabled={isRefreshing}
          className="p-1 rounded-full hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
          aria-label="Get new quote"
        >
          <svg
            className={`w-3.5 h-3.5 text-gray-400 dark:text-gray-500 ${isRefreshing ? 'animate-spin' : ''}`}
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
    </div>
  )
}
