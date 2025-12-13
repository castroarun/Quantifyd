import type { Metadata, Viewport } from 'next'
import { ThemeProvider, UnitProvider } from '@/contexts'
import './globals.css'

export const metadata: Metadata = {
  title: 'Strength Profile Tracker',
  description: 'Track your strength standards and progress across key compound lifts'
}

export const viewport: Viewport = {
  width: 'device-width',
  initialScale: 1,
  maximumScale: 1
}

export default function RootLayout({
  children
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className="bg-gray-50 dark:bg-gray-900 min-h-screen antialiased transition-colors" suppressHydrationWarning>
        <ThemeProvider>
          <UnitProvider>
            {children}
          </UnitProvider>
        </ThemeProvider>
      </body>
    </html>
  )
}
