import type { Metadata } from 'next'
import '@/ui/styles/globals.css'
import { ThemeProvider } from '@/ui/contexts/ThemeContext'

export const metadata: Metadata = {
  title: 'NoteApp - Clean & Simple Note Taking',
  description: 'Fast, clean note-taking app with Jira and Gmail integration',
  icons: {
    icon: '/icon.svg',
  },
}

/**
 * Root layout component for the entire application
 *
 * Input: children (React nodes to render)
 * Output: HTML structure with global styles and metadata
 *
 * Called by: Next.js app router
 * Calls: ThemeProvider
 */
export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" className="h-full">
      <body className="h-full m-0 p-0 overflow-hidden" suppressHydrationWarning>
        <ThemeProvider>
          {children}
        </ThemeProvider>
      </body>
    </html>
  )
}
