'use client'

/**
 * Right Panel Templates component
 *
 * Input: onCreateFromTemplate callback
 * Output: Template icons that create new notes with pre-filled content
 *
 * Called by: MainApp component
 * Calls: onCreateFromTemplate when a template is clicked
 */

import React, { useState } from 'react'

interface Template {
  id: string
  name: string
  description: string
  icon: React.ReactNode
  title: string
  content: string
  plainText: string
}

interface RightPanelProps {
  onCreateFromTemplate: (title: string, content: string, plainText: string) => void
}

const templates: Template[] = [
  {
    id: 'weekend-planner',
    name: 'Weekend Planner',
    description: 'Plan your Saturday and Sunday activities',
    icon: (
      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" />
      </svg>
    ),
    title: 'Weekend Planner',
    content: `<h2>Weekend Planner</h2>
<p></p>
<h3>Saturday</h3>
<ul style="list-style-type: disc; padding-left: 1.5em; margin-left: 0.5em;">
<li style="display: list-item;"><em>Morning:</em> </li>
<li style="display: list-item;"><em>Afternoon:</em> </li>
<li style="display: list-item;"><em>Evening:</em> </li>
</ul>
<p></p>
<h3>Sunday</h3>
<ul style="list-style-type: disc; padding-left: 1.5em; margin-left: 0.5em;">
<li style="display: list-item;"><em>Morning:</em> </li>
<li style="display: list-item;"><em>Afternoon:</em> </li>
<li style="display: list-item;"><em>Evening:</em> </li>
</ul>
<p></p>
<p><strong>Notes:</strong></p>
<p></p>`,
    plainText: `Weekend Planner

Saturday
Morning:
Afternoon:
Evening:

Sunday
Morning:
Afternoon:
Evening:

Notes:
`,
  },
  {
    id: 'goal-tracker',
    name: 'Goal Tracker',
    description: 'Track your goals and progress',
    icon: (
      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
      </svg>
    ),
    title: 'Goal Tracker',
    content: `<h2>Goal Tracker</h2>
<p></p>
<h3>Main Goal</h3>
<p><strong>What I want to achieve:</strong> </p>
<p><strong>Deadline:</strong> </p>
<p></p>
<h3>Action Steps</h3>
<ol style="list-style-type: decimal; padding-left: 1.5em; margin-left: 0.5em;">
<li style="display: list-item;"></li>
<li style="display: list-item;"></li>
<li style="display: list-item;"></li>
</ol>
<p></p>
<h3>Progress</h3>
<ul style="list-style-type: disc; padding-left: 1.5em; margin-left: 0.5em;">
<li style="display: list-item;">Week 1: </li>
<li style="display: list-item;">Week 2: </li>
<li style="display: list-item;">Week 3: </li>
</ul>
<p></p>
<h3>Obstacles & Solutions</h3>
<p></p>`,
    plainText: `Goal Tracker

Main Goal
What I want to achieve:
Deadline:

Action Steps
1.
2.
3.

Progress
Week 1:
Week 2:
Week 3:

Obstacles & Solutions
`,
  },
  {
    id: 'lecture-notes',
    name: 'Lecture Notes',
    description: 'Take structured notes during lectures',
    icon: (
      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
      </svg>
    ),
    title: 'Lecture Notes',
    content: `<h2>Lecture Notes</h2>
<p></p>
<p><strong>Subject:</strong> </p>
<p><strong>Date:</strong> </p>
<p><strong>Topic:</strong> </p>
<p></p>
<h3>Key Concepts</h3>
<ul style="list-style-type: disc; padding-left: 1.5em; margin-left: 0.5em;">
<li style="display: list-item;"></li>
<li style="display: list-item;"></li>
<li style="display: list-item;"></li>
</ul>
<p></p>
<h3>Important Points</h3>
<ol style="list-style-type: decimal; padding-left: 1.5em; margin-left: 0.5em;">
<li style="display: list-item;"></li>
<li style="display: list-item;"></li>
<li style="display: list-item;"></li>
</ol>
<p></p>
<h3>Questions to Review</h3>
<ul style="list-style-type: disc; padding-left: 1.5em; margin-left: 0.5em;">
<li style="display: list-item;"></li>
</ul>
<p></p>
<h3>Summary</h3>
<p></p>`,
    plainText: `Lecture Notes

Subject:
Date:
Topic:

Key Concepts



Important Points
1.
2.
3.

Questions to Review


Summary
`,
  },
  {
    id: 'meeting-notes',
    name: 'Meeting Notes',
    description: 'Capture meeting agenda and action items',
    icon: (
      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z" />
      </svg>
    ),
    title: 'Meeting Notes',
    content: `<h2>Meeting Notes</h2>
<p></p>
<p><strong>Date:</strong> </p>
<p><strong>Attendees:</strong> </p>
<p><strong>Purpose:</strong> </p>
<p></p>
<h3>Agenda</h3>
<ol style="list-style-type: decimal; padding-left: 1.5em; margin-left: 0.5em;">
<li style="display: list-item;"></li>
<li style="display: list-item;"></li>
<li style="display: list-item;"></li>
</ol>
<p></p>
<h3>Discussion Points</h3>
<ul style="list-style-type: disc; padding-left: 1.5em; margin-left: 0.5em;">
<li style="display: list-item;"></li>
<li style="display: list-item;"></li>
</ul>
<p></p>
<h3>Action Items</h3>
<ul style="list-style-type: disc; padding-left: 1.5em; margin-left: 0.5em;">
<li style="display: list-item;"><strong>Owner:</strong>  - Task: </li>
<li style="display: list-item;"><strong>Owner:</strong>  - Task: </li>
</ul>
<p></p>
<h3>Next Meeting</h3>
<p></p>`,
    plainText: `Meeting Notes

Date:
Attendees:
Purpose:

Agenda
1.
2.
3.

Discussion Points



Action Items
Owner:  - Task:
Owner:  - Task:

Next Meeting
`,
  },
  {
    id: 'daily-journal',
    name: 'Daily Journal',
    description: 'Reflect on your day with prompts',
    icon: (
      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15.232 5.232l3.536 3.536m-2.036-5.036a2.5 2.5 0 113.536 3.536L6.5 21.036H3v-3.572L16.732 3.732z" />
      </svg>
    ),
    title: 'Daily Journal',
    content: `<h2>Daily Journal</h2>
<p></p>
<p><strong>Date:</strong> </p>
<p></p>
<h3>How am I feeling today?</h3>
<p></p>
<p></p>
<h3>3 Things I'm Grateful For</h3>
<ol style="list-style-type: decimal; padding-left: 1.5em; margin-left: 0.5em;">
<li style="display: list-item;"></li>
<li style="display: list-item;"></li>
<li style="display: list-item;"></li>
</ol>
<p></p>
<h3>Today's Highlights</h3>
<ul style="list-style-type: disc; padding-left: 1.5em; margin-left: 0.5em;">
<li style="display: list-item;"></li>
<li style="display: list-item;"></li>
</ul>
<p></p>
<h3>What I Learned</h3>
<p></p>
<p></p>
<h3>Tomorrow's Focus</h3>
<p></p>`,
    plainText: `Daily Journal

Date:

How am I feeling today?


3 Things I'm Grateful For
1.
2.
3.

Today's Highlights



What I Learned


Tomorrow's Focus
`,
  },
  {
    id: 'project-ideas',
    name: 'Project Ideas',
    description: 'Brainstorm and organize project ideas',
    icon: (
      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
      </svg>
    ),
    title: 'Project Ideas',
    content: `<h2>Project Ideas</h2>
<p></p>
<h3>Idea Name</h3>
<p><strong>Problem it solves:</strong> </p>
<p><strong>Target audience:</strong> </p>
<p></p>
<h3>Key Features</h3>
<ul style="list-style-type: disc; padding-left: 1.5em; margin-left: 0.5em;">
<li style="display: list-item;"></li>
<li style="display: list-item;"></li>
<li style="display: list-item;"></li>
</ul>
<p></p>
<h3>Resources Needed</h3>
<ul style="list-style-type: disc; padding-left: 1.5em; margin-left: 0.5em;">
<li style="display: list-item;"></li>
<li style="display: list-item;"></li>
</ul>
<p></p>
<h3>First Steps</h3>
<ol style="list-style-type: decimal; padding-left: 1.5em; margin-left: 0.5em;">
<li style="display: list-item;"></li>
<li style="display: list-item;"></li>
<li style="display: list-item;"></li>
</ol>
<p></p>
<h3>Notes</h3>
<p></p>`,
    plainText: `Project Ideas

Idea Name
Problem it solves:
Target audience:

Key Features




Resources Needed



First Steps
1.
2.
3.

Notes
`,
  },
]

export function RightPanel({ onCreateFromTemplate }: RightPanelProps) {
  const [hoveredTemplate, setHoveredTemplate] = useState<string | null>(null)

  const handleTemplateClick = (template: Template) => {
    onCreateFromTemplate(template.title, template.content, template.plainText)
  }

  return (
    <div className="flex h-full">
      {/* Icon Bar with Templates */}
      <div className="w-12 border-l border-border bg-background-secondary flex flex-col items-center py-4 gap-2">
        {/* Templates Header */}
        <div className="text-[10px] text-text-muted uppercase tracking-wide mb-2 -rotate-90 origin-center whitespace-nowrap mt-8">
          Templates
        </div>

        {/* Template Icons */}
        {templates.map((template) => (
          <div key={template.id} className="relative">
            <button
              onClick={() => handleTemplateClick(template)}
              onMouseEnter={() => setHoveredTemplate(template.id)}
              onMouseLeave={() => setHoveredTemplate(null)}
              className="p-2.5 rounded transition-colors hover:bg-border text-text-secondary hover:text-accent-blue"
              title={template.name}
            >
              {template.icon}
            </button>

            {/* Hover Tooltip */}
            {hoveredTemplate === template.id && (
              <div className="absolute right-full mr-2 top-1/2 -translate-y-1/2 z-50">
                <div className="bg-text-primary text-background px-3 py-2 rounded-lg shadow-lg whitespace-nowrap">
                  <div className="font-medium text-sm">{template.name}</div>
                  <div className="text-xs opacity-80">{template.description}</div>
                </div>
                {/* Arrow */}
                <div className="absolute left-full top-1/2 -translate-y-1/2 border-8 border-transparent border-l-text-primary" />
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  )
}
