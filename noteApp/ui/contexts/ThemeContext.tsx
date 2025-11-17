'use client'

/**
 * Theme Context Provider for Dark Mode
 *
 * Input: children components
 * Output: Theme state and toggle function
 *
 * Manages dark mode preference and persists to Supabase
 */

import { createContext, useContext, useEffect, useState } from 'react'
import { createClient } from '@/lib/supabase-client'

type Theme = 'light' | 'dark'

interface ThemeContextType {
  theme: Theme
  toggleTheme: () => void
  isLoading: boolean
}

const ThemeContext = createContext<ThemeContextType>({
  theme: 'light',
  toggleTheme: () => {},
  isLoading: true,
})

export function useTheme() {
  return useContext(ThemeContext)
}

export function ThemeProvider({ children }: { children: React.ReactNode }) {
  const [theme, setTheme] = useState<Theme>('light')
  const [isLoading, setIsLoading] = useState(true)
  const supabase = createClient()

  // Load theme preference from Supabase or localStorage
  useEffect(() => {
    const loadTheme = async () => {
      try {
        // Try to get from localStorage first (faster)
        const localTheme = localStorage.getItem('theme') as Theme
        if (localTheme) {
          setTheme(localTheme)
          applyTheme(localTheme)
        }

        // Then sync with Supabase
        const { data: { user } } = await supabase.auth.getUser()
        if (user) {
          const { data, error } = await supabase
            .from('profiles')
            .select('theme_preference')
            .eq('user_id', user.id)
            .single()

          if (!error && data?.theme_preference) {
            const dbTheme = data.theme_preference as Theme
            setTheme(dbTheme)
            applyTheme(dbTheme)
            localStorage.setItem('theme', dbTheme)
          }
        }
      } catch (error) {
        console.error('Error loading theme:', error)
      } finally {
        setIsLoading(false)
      }
    }

    loadTheme()
  }, [])

  const applyTheme = (newTheme: Theme) => {
    const root = document.documentElement
    if (newTheme === 'dark') {
      root.classList.add('dark')
    } else {
      root.classList.remove('dark')
    }
  }

  const toggleTheme = async () => {
    const newTheme = theme === 'light' ? 'dark' : 'light'
    setTheme(newTheme)
    applyTheme(newTheme)
    localStorage.setItem('theme', newTheme)

    // Save to Supabase
    try {
      const { data: { user } } = await supabase.auth.getUser()
      if (user) {
        await supabase
          .from('profiles')
          .upsert({
            user_id: user.id,
            theme_preference: newTheme,
            updated_at: new Date().toISOString(),
          })
      }
    } catch (error) {
      console.error('Error saving theme preference:', error)
    }
  }

  return (
    <ThemeContext.Provider value={{ theme, toggleTheme, isLoading }}>
      {children}
    </ThemeContext.Provider>
  )
}
