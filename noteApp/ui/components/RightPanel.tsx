'use client'

/**
 * Right Panel Utility component
 *
 * Input: None
 * Output: Collapsible utility panel with web browser, calculator, and shortcuts
 *
 * Called by: MainApp component
 * Calls: WebBrowserPanel, CalculatorPanel, ShortcutsPanel
 */

import { useState } from 'react'

type PanelType = 'browser' | 'calculator' | 'shortcuts' | null

export function RightPanel() {
  const [activePanel, setActivePanel] = useState<PanelType>(null)
  const [isExpanded, setIsExpanded] = useState(false)

  const handlePanelToggle = (panel: PanelType) => {
    if (activePanel === panel) {
      // Close if same panel clicked
      setActivePanel(null)
      setIsExpanded(false)
    } else {
      // Open new panel
      setActivePanel(panel)
      setIsExpanded(true)
    }
  }

  return (
    <div className="flex h-full">
      {/* Expanded Panel Content */}
      {isExpanded && (
        <div className="w-96 border-l border-border bg-background flex flex-col">
          {/* Panel Header */}
          <div className="flex items-center justify-between p-4 border-b border-border">
            <h3 className="font-medium text-text-primary">
              {activePanel === 'browser' && 'Web Browser'}
              {activePanel === 'calculator' && 'Calculator'}
              {activePanel === 'shortcuts' && 'Shortcuts'}
            </h3>
            <button
              onClick={() => {
                setActivePanel(null)
                setIsExpanded(false)
              }}
              className="p-1 hover:bg-border rounded transition-colors"
              title="Close panel"
            >
              <svg className="w-4 h-4 text-text-secondary" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>

          {/* Panel Content */}
          <div className="flex-1 overflow-y-auto">
            {activePanel === 'browser' && <WebBrowserPanel />}
            {activePanel === 'calculator' && <CalculatorPanel />}
            {activePanel === 'shortcuts' && <ShortcutsPanel />}
          </div>
        </div>
      )}

      {/* Icon Bar (Always Visible) */}
      <div className="w-12 border-l border-border bg-background-secondary flex flex-col items-center py-4 gap-3">
        {/* Web Browser Icon */}
        <button
          onClick={() => handlePanelToggle('browser')}
          className={`p-2.5 rounded transition-colors ${
            activePanel === 'browser'
              ? 'bg-accent-blue text-white'
              : 'hover:bg-border text-text-secondary'
          }`}
          title="Web Browser"
        >
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 01-9 9m9-9a9 9 0 00-9-9m9 9H3m9 9a9 9 0 01-9-9m9 9c1.657 0 3-4.03 3-9s-1.343-9-3-9m0 18c-1.657 0-3-4.03-3-9s1.343-9 3-9m-9 9a9 9 0 019-9" />
          </svg>
        </button>

        {/* Calculator Icon */}
        <button
          onClick={() => handlePanelToggle('calculator')}
          className={`p-2.5 rounded transition-colors ${
            activePanel === 'calculator'
              ? 'bg-accent-blue text-white'
              : 'hover:bg-border text-text-secondary'
          }`}
          title="Calculator"
        >
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 7h6m0 10v-3m-3 3h.01M9 17h.01M9 14h.01M12 14h.01M15 11h.01M12 11h.01M9 11h.01M7 21h10a2 2 0 002-2V5a2 2 0 00-2-2H7a2 2 0 00-2 2v14a2 2 0 002 2z" />
          </svg>
        </button>

        {/* Shortcuts Reference Icon */}
        <button
          onClick={() => handlePanelToggle('shortcuts')}
          className={`p-2.5 rounded transition-colors ${
            activePanel === 'shortcuts'
              ? 'bg-accent-blue text-white'
              : 'hover:bg-border text-text-secondary'
          }`}
          title="Keyboard Shortcuts"
        >
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6V4m0 2a2 2 0 100 4m0-4a2 2 0 110 4m-6 8a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4m6 6v10m6-2a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4" />
          </svg>
        </button>
      </div>
    </div>
  )
}

/**
 * Web Browser Panel Component
 */
function WebBrowserPanel() {
  const [url, setUrl] = useState('https://www.google.com')
  const [inputUrl, setInputUrl] = useState('https://www.google.com')

  const handleNavigate = () => {
    let finalUrl = inputUrl
    if (!inputUrl.startsWith('http://') && !inputUrl.startsWith('https://')) {
      finalUrl = 'https://' + inputUrl
    }
    setUrl(finalUrl)
  }

  return (
    <div className="flex flex-col h-full">
      {/* URL Bar */}
      <div className="p-3 border-b border-border">
        <div className="flex gap-2">
          <input
            type="text"
            value={inputUrl}
            onChange={(e) => setInputUrl(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleNavigate()}
            placeholder="Enter URL..."
            className="flex-1 px-3 py-2 text-sm border border-border rounded focus:outline-none focus:ring-2 focus:ring-accent-blue"
          />
          <button
            onClick={handleNavigate}
            className="px-4 py-2 bg-accent-blue text-white text-sm rounded hover:bg-blue-600 transition-colors"
          >
            Go
          </button>
        </div>
      </div>

      {/* Browser Frame */}
      <div className="flex-1 bg-white">
        <iframe
          src={url}
          className="w-full h-full border-0"
          sandbox="allow-same-origin allow-scripts allow-forms"
          title="Web Browser"
        />
      </div>
    </div>
  )
}

/**
 * Calculator Panel Component
 */
function CalculatorPanel() {
  const [display, setDisplay] = useState('0')
  const [previousValue, setPreviousValue] = useState<number | null>(null)
  const [operation, setOperation] = useState<string | null>(null)
  const [newNumber, setNewNumber] = useState(true)

  const handleNumber = (num: string) => {
    if (newNumber) {
      setDisplay(num)
      setNewNumber(false)
    } else {
      setDisplay(display === '0' ? num : display + num)
    }
  }

  const handleDecimal = () => {
    if (newNumber) {
      setDisplay('0.')
      setNewNumber(false)
    } else if (!display.includes('.')) {
      setDisplay(display + '.')
    }
  }

  const handleOperation = (op: string) => {
    const currentValue = parseFloat(display)

    if (previousValue !== null && operation && !newNumber) {
      const result = calculate(previousValue, currentValue, operation)
      setDisplay(String(result))
      setPreviousValue(result)
    } else {
      setPreviousValue(currentValue)
    }

    setOperation(op)
    setNewNumber(true)
  }

  const handleEquals = () => {
    if (previousValue !== null && operation) {
      const currentValue = parseFloat(display)
      const result = calculate(previousValue, currentValue, operation)
      setDisplay(String(result))
      setPreviousValue(null)
      setOperation(null)
      setNewNumber(true)
    }
  }

  const calculate = (a: number, b: number, op: string): number => {
    switch (op) {
      case '+': return a + b
      case '-': return a - b
      case '*': return a * b
      case '/': return b !== 0 ? a / b : 0
      default: return b
    }
  }

  const handleClear = () => {
    setDisplay('0')
    setPreviousValue(null)
    setOperation(null)
    setNewNumber(true)
  }

  const CalcButton = ({ value, onClick, className = '' }: { value: string; onClick: () => void; className?: string }) => (
    <button
      onClick={onClick}
      className={`p-4 text-lg font-medium rounded hover:bg-border-light transition-colors border border-border ${className}`}
    >
      {value}
    </button>
  )

  return (
    <div className="p-4">
      {/* Display */}
      <div className="mb-4 p-4 bg-background-secondary border border-border rounded text-right">
        <div className="text-2xl font-mono text-text-primary break-all">{display}</div>
      </div>

      {/* Buttons */}
      <div className="grid grid-cols-4 gap-2">
        <CalcButton value="7" onClick={() => handleNumber('7')} />
        <CalcButton value="8" onClick={() => handleNumber('8')} />
        <CalcButton value="9" onClick={() => handleNumber('9')} />
        <CalcButton value="/" onClick={() => handleOperation('/')} className="bg-accent-blue text-white hover:bg-blue-600" />

        <CalcButton value="4" onClick={() => handleNumber('4')} />
        <CalcButton value="5" onClick={() => handleNumber('5')} />
        <CalcButton value="6" onClick={() => handleNumber('6')} />
        <CalcButton value="*" onClick={() => handleOperation('*')} className="bg-accent-blue text-white hover:bg-blue-600" />

        <CalcButton value="1" onClick={() => handleNumber('1')} />
        <CalcButton value="2" onClick={() => handleNumber('2')} />
        <CalcButton value="3" onClick={() => handleNumber('3')} />
        <CalcButton value="-" onClick={() => handleOperation('-')} className="bg-accent-blue text-white hover:bg-blue-600" />

        <CalcButton value="0" onClick={() => handleNumber('0')} />
        <CalcButton value="." onClick={handleDecimal} />
        <CalcButton value="=" onClick={handleEquals} className="bg-green-600 text-white hover:bg-green-700" />
        <CalcButton value="+" onClick={() => handleOperation('+')} className="bg-accent-blue text-white hover:bg-blue-600" />

        <CalcButton value="C" onClick={handleClear} className="col-span-4 bg-red-600 text-white hover:bg-red-700" />
      </div>
    </div>
  )
}

/**
 * Shortcuts Reference Panel Component
 */
function ShortcutsPanel() {
  const shortcuts = [
    { category: 'Text Formatting', items: [
      { key: 'Ctrl + B', description: 'Make selected text Bold' },
      { key: 'Ctrl + I', description: 'Make selected text Italic' },
      { key: 'Ctrl + U', description: 'Make selected text Underlined' },
      { key: 'Ctrl + H', description: 'Make selected text a Header' },
    ]},
    { category: 'Lists', items: [
      { key: 'Tab', description: 'Create or indent bullet list' },
      { key: 'Shift + Tab', description: 'Un-indent or exit bullet list' },
    ]},
    { category: 'Editor', items: [
      { key: 'Ctrl + Z', description: 'Undo changes' },
      { key: 'Ctrl + Shift + Z', description: 'Redo changes' },
    ]},
  ]

  return (
    <div className="p-4">
      <div className="space-y-6">
        {shortcuts.map((section) => (
          <div key={section.category}>
            <h4 className="font-semibold text-text-primary mb-3 text-sm uppercase tracking-wide">
              {section.category}
            </h4>
            <div className="space-y-2">
              {section.items.map((shortcut) => (
                <div key={shortcut.key} className="flex items-start gap-3">
                  <kbd className="px-2 py-1 bg-background-secondary border border-border rounded text-xs font-mono text-text-primary whitespace-nowrap">
                    {shortcut.key}
                  </kbd>
                  <span className="text-sm text-text-secondary flex-1">
                    {shortcut.description}
                  </span>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>

      {/* Additional Info */}
      <div className="mt-6 p-3 bg-blue-50 border border-blue-200 rounded">
        <p className="text-xs text-blue-800">
          <strong>Tip:</strong> Keyboard shortcuts only work when text is selected. Without selection, default Windows behavior applies.
        </p>
      </div>
    </div>
  )
}
