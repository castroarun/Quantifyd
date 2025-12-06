# Strength Profile Tracker - Project Instructions

**Status:** Ready for Development
**Last Updated:** 2025-12-02

---

## Global References

> **Development Workflow:** Follow the 9-step AI-assisted development process.
> See: [dev-workflow.md](../../_claude-shared/dev-workflow.md)

> **Coding Standards:** All TypeScript/React/Next.js coding patterns.
> See: [coding-standards.md](../../_claude-shared/coding-standards.md)

---

## Project Overview

A mobile-first strength profile tracking application that calculates personalized strength standards based on user body metrics and tracks progress across compound lifts.

**Primary Input:** [strength_profile_tracker_PRD.md](../docs/strength_profile_tracker_PRD.md)

---

## Current Phase

```
[x] Step 1: DEV-CLOCK Setup     ← docs/DEV-CLOCK.md created
[x] Step 2: DESIGN.md           ← Embedded in PRD (approved)
[x] Step 3: TEST-PLAN.csv       ← Created (58 test cases) - IN REVIEW
[ ] Step 4: Build               ← After test plan approval
[ ] Step 5-9: ...               ← After build
```

**Status:** PRD approved, design approved. TEST-PLAN.csv awaiting review (Jira: SPT-3 → In Review).

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| Framework | Next.js 14+ (App Router) |
| Language | TypeScript (strict mode) |
| Styling | Tailwind CSS v3 |
| Storage | localStorage (MVP) → Supabase (Production) |

---

## Core Features (from PRD)

### Exercises
1. Bench Press
2. Squat
3. Deadlift
4. Shoulder Press

### Progression Levels

| Level | Multiplier Formula | Color |
|-------|-------------------|-------|
| Beginner | See PRD table | `#2ECC71` (Green) |
| Novice | See PRD table | `#3498DB` (Blue) |
| Intermediate | See PRD table | `#F39C12` (Orange) |
| Advanced | See PRD table | `#E74C3C` (Red) |

### Body Weight Multipliers

| Exercise | BEG | NOV | INT | ADV |
|----------|-----|-----|-----|-----|
| Bench Press | 0.59 | 0.82 | 1.06 | 1.29 |
| Squat | 0.76 | 1.12 | 1.47 | 1.88 |
| Deadlift | 0.94 | 1.29 | 1.71 | 2.18 |
| Shoulder Press | 0.41 | 0.59 | 0.76 | 1.00 |

---

## Data Model

```typescript
interface Profile {
  id: string
  name: string                  // max 50 chars
  age: number                   // 13-100
  height: number                // 100-250 cm
  weight: number                // 30-300 kg
  currentLevels: {
    benchPress: Level
    squat: Level
    deadlift: Level
    shoulderPress: Level
  }
  createdAt: Date
  updatedAt: Date
}

type Level = 'beginner' | 'novice' | 'intermediate' | 'advanced'
```

---

## Validation Rules

| Field | Min | Max | Type |
|-------|-----|-----|------|
| Name | 1 char | 50 chars | string |
| Age | 13 | 100 | number |
| Height | 100 cm | 250 cm | number |
| Weight | 30 kg | 300 kg | number |
| Profiles | - | 5 max | count |

---

## Folder Structure

```
strength_profile_tracker/
├── src/
│   ├── app/                    # Next.js App Router
│   │   ├── page.tsx           # Profile list
│   │   ├── layout.tsx
│   │   └── profile/
│   │       ├── new/page.tsx   # New profile
│   │       └── [id]/page.tsx  # Profile detail
│   ├── components/
│   │   ├── ui/               # Reusable UI
│   │   ├── profile/          # Profile components
│   │   └── strength/         # Strength components
│   ├── lib/
│   │   ├── storage/          # localStorage logic
│   │   ├── calculations/     # Strength formulas
│   │   └── utils/
│   ├── hooks/
│   └── types/
├── docs/
│   ├── strength_profile_tracker_PRD.md
│   ├── DEV-CLOCK.md
│   ├── DESIGN.md             # To create
│   └── TEST-PLAN.csv         # To create
└── .claude/
    └── instructions.md       # This file
```

---

## UI Design (from PRD)

### Color Palette

**Primary:**
- Header: `#2C3E50` (Dark blue-grey)
- Accent: `#3498DB` (Bright blue)
- Success: `#27AE60` (Green)

**Level Colors:**
- Beginner: `#2ECC71`
- Novice: `#3498DB`
- Intermediate: `#F39C12`
- Advanced: `#E74C3C`

### Typography
- Font: System Default (San Francisco / Roboto)
- Screen Title: 13pt, weight 600
- Section Header: 10pt, weight 600
- Body: 9pt, normal
- Buttons: 11pt, weight 600

### Components
- Buttons: 44-48pt height, 8pt radius
- Cards: White/light grey, 1pt border, 8pt radius

---

## Key Constraints

1. **Max 5 profiles** - System must prevent 6th profile
2. **Local storage only** (Phase 1) - No internet required
3. **Mobile-first** - Design for mobile viewport
4. **Touch-friendly** - Min 44pt touch targets

---

## Project Documents

| Document | Status | Purpose |
|----------|--------|---------|
| [PRD](../docs/strength_profile_tracker_PRD.md) | Complete | Requirements |
| [DEV-CLOCK](../docs/DEV-CLOCK.md) | Started | Time tracking |
| DESIGN.md | Pending | Design spec |
| TEST-PLAN.csv | Pending | Test cases |
| WALKTHROUGH.md | After build | Code docs |

---

## Next Actions

1. ~~Create `docs/DESIGN.md` based on PRD~~ (in PRD)
2. ~~Create `docs/TEST-PLAN.csv` with test cases~~ (done - 58 cases)
3. ~~Get approval before proceeding to build~~ (PRD approved)
4. **START BUILD PHASE** - Project scaffolding (Next.js, Tailwind)

---

**Project Repository:** strength_profile_tracker
**Global Shared Files:** `_claude-shared/`
