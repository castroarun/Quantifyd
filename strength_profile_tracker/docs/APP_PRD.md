# APP PRD: Strength Profile Tracker

**Version:** 3.0
**Date:** 2025-12-06
**Status:** In Development

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-12-01 | Initial PRD with profile management and 4 core exercises |
| 2.0 | 2025-12-02 | Expanded to 25 exercises, body part filtering, motivational quotes |
| 3.0 | 2025-12-06 | Added dark mode, Workout Logger, Achievements, Strength Score, AI Coach Tips |

---

## 1. Overview

### 1.1 Problem
Users lack a simple way to track strength standards for multiple people and understand where they stand relative to their body weight.

### 1.2 Solution
A mobile-first web app that stores up to 5 user profiles, calculates personalized strength standards, and tracks performance across difficulty levels.

### 1.3 Target Users
- Fitness enthusiasts
- Personal trainers managing clients
- Gym-goers tracking progression

---

## 2. Features

### 2.1 Profile Management

#### Requirements
- Create up to 5 profiles
- Each profile: name, age, height, weight
- Edit and delete profiles
- Profiles persist locally

#### Design

```typescript
interface Profile {
  id: string                    // crypto.randomUUID()
  name: string                  // max 50 characters
  age: number                   // 13-100
  height: number                // 100-250 cm
  weight: number                // 30-300 kg
  currentLevels: ExerciseLevels
  createdAt: string
  updatedAt: string
}

// Storage
const STORAGE_KEY = 'spt_profiles'
localStorage.setItem(STORAGE_KEY, JSON.stringify(profiles))
```

#### Components
```
<ProfileList />
  └── <ProfileCard profile={} onEdit={} onDelete={} />
  └── <AddProfileButton disabled={profiles.length >= 5} />

<ProfileForm mode="create|edit" />
  └── <Input name="name" maxLength={50} required />
  └── <Input name="age" type="number" min={13} max={100} />
  └── <Input name="height" type="number" min={100} max={250} />
  └── <Input name="weight" type="number" min={30} max={300} />
  └── <Button type="submit">Save Profile</Button>
```

#### Test Cases
- [ ] Can create profile with valid data
- [ ] Cannot create 6th profile (button disabled)
- [ ] Name field rejects >50 characters
- [ ] Age field rejects <13 or >100
- [ ] Profile persists after page refresh
- [ ] Can edit existing profile
- [ ] Can delete profile with confirmation

---

### 2.2 Exercise System

#### Requirements
- 20+ exercises across 6 body parts
- Filter exercises by body part
- Each exercise has 4 difficulty levels
- Display calculated weight based on user's body weight

#### Design

```typescript
type BodyPart = 'chest' | 'back' | 'shoulders' | 'legs' | 'arms' | 'core'
type Level = 'beginner' | 'novice' | 'intermediate' | 'advanced'

interface Exercise {
  id: string
  name: string
  bodyPart: BodyPart
  multipliers: {
    beginner: number
    novice: number
    intermediate: number
    advanced: number
  }
  unit: 'barbell' | 'dumbbell' | 'machine' | 'bodyweight'
}

interface ExerciseLevels {
  [exerciseId: string]: Level | null
}
```

#### Exercise Data

**Chest**
| Exercise | BEG | NOV | INT | ADV |
|----------|-----|-----|-----|-----|
| Bench Press | 0.59 | 0.82 | 1.06 | 1.29 |
| Incline Bench | 0.50 | 0.70 | 0.90 | 1.10 |
| Dumbbell Press | 0.25 | 0.35 | 0.45 | 0.55 |
| Cable Fly | 0.15 | 0.22 | 0.30 | 0.40 |

**Back**
| Exercise | BEG | NOV | INT | ADV |
|----------|-----|-----|-----|-----|
| Deadlift | 0.94 | 1.29 | 1.71 | 2.18 |
| Barbell Row | 0.50 | 0.70 | 0.90 | 1.15 |
| Lat Pulldown | 0.50 | 0.70 | 0.90 | 1.10 |
| Pull-ups | 0.5x | 0.8x | 1.0x | 1.3x |
| Cable Row | 0.50 | 0.70 | 0.90 | 1.10 |

**Shoulders**
| Exercise | BEG | NOV | INT | ADV |
|----------|-----|-----|-----|-----|
| Shoulder Press (Barbell) | 0.41 | 0.59 | 0.76 | 1.00 |
| Shoulder Press (Machine) | 0.35 | 0.50 | 0.65 | 0.85 |
| Shoulder Press (Dumbbell) | 0.18 | 0.26 | 0.35 | 0.45 |
| Side Lateral (Dumbbell) | 0.08 | 0.12 | 0.16 | 0.22 |
| Side Lateral (Cable) | 0.06 | 0.10 | 0.14 | 0.20 |
| Front Raise | 0.10 | 0.15 | 0.20 | 0.28 |

**Legs**
| Exercise | BEG | NOV | INT | ADV |
|----------|-----|-----|-----|-----|
| Squat | 0.76 | 1.12 | 1.47 | 1.88 |
| Leg Press | 1.50 | 2.20 | 3.00 | 3.80 |
| Romanian Deadlift | 0.60 | 0.85 | 1.10 | 1.40 |
| Leg Curl | 0.30 | 0.45 | 0.60 | 0.80 |
| Leg Extension | 0.40 | 0.55 | 0.75 | 0.95 |

**Arms**
| Exercise | BEG | NOV | INT | ADV |
|----------|-----|-----|-----|-----|
| Bicep Curl (Barbell) | 0.25 | 0.38 | 0.50 | 0.65 |
| Bicep Curl (Dumbbell) | 0.12 | 0.18 | 0.25 | 0.32 |
| Tricep Pushdown | 0.25 | 0.38 | 0.50 | 0.65 |

> **Note:** Dumbbell exercises show per-hand multiplier. Pull-ups use bodyweight multiplier.

#### Components
```
<ExerciseList>
  └── <BodyPartFilter selected={} onChange={} />
  └── <ExerciseCard exercise={} userWeight={} selectedLevel={} onLevelSelect={} />

<ExerciseCard>
  └── <ExerciseName />
  └── <LevelSelector>
        └── <LevelBadge level="beginner" weight={calculated} selected={} />
        └── <LevelBadge level="novice" weight={calculated} selected={} />
        └── <LevelBadge level="intermediate" weight={calculated} selected={} />
        └── <LevelBadge level="advanced" weight={calculated} selected={} />
      </LevelSelector>
```

#### Test Cases
- [ ] All 20+ exercises render correctly
- [ ] Body part filter works (shows only filtered exercises)
- [ ] Weight calculation: userWeight × multiplier = displayed weight
- [ ] Level selection saves to profile
- [ ] Level selection persists after refresh
- [ ] "All" filter shows all exercises

---

### 2.3 Motivational Quotes

#### Requirements
- Display fitness quote at bottom of app
- New quote on each app open
- Works offline with cached quotes

#### Design

```typescript
interface Quote {
  id: string
  text: string
  author?: string
  category: 'motivation' | 'science' | 'benefit'
  source?: string
}

// Local quotes file: /lib/quotes.json (100+ quotes)
// Fallback: ZenQuotes API or API Ninjas
```

#### Components
```
<QuoteDisplay>
  └── <QuoteText />
  └── <QuoteAuthor />
  └── <RefreshButton onClick={getNewQuote} />
</QuoteDisplay>
```

#### Test Cases
- [ ] Quote displays on app load
- [ ] Different quote on refresh
- [ ] Works offline (uses cached quote)
- [ ] Author/source displays when available

---

### 2.4 Workout Logger

#### Requirements
- Log workout sets for any exercise
- Display last 3 sessions for reference
- Pre-filled rep suggestions (12, 10, 8)
- Smart tips based on logged data
- Number of exercises doesn't affect scoring - quality over quantity

#### Design

```typescript
interface WorkoutSet {
  weight: number      // kg
  reps: number
}

interface WorkoutSession {
  id: string
  date: string        // ISO date
  exerciseId: string
  sets: WorkoutSet[]  // Always 3 sets
}

interface WorkoutLog {
  [profileId: string]: WorkoutSession[]
}

// Storage
const WORKOUT_KEY = 'spt_workouts'
localStorage.setItem(WORKOUT_KEY, JSON.stringify(workoutLog))
```

#### UI Mockup - Expanded Exercise Card

```
┌─────────────────────────────────────────────────────────┐
│  Bench Press                                    [Chest] │
├─────────────────────────────────────────────────────────┤
│  Level:  [BEG]  [NOV]  [INT✓]  [ADV]                    │
│           59kg   82kg   106kg   129kg                   │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  WORKOUT LOG                                            │
│  ─────────────────────────────────────────────────────  │
│           Nov 28    Dec 2     Dec 4      TODAY          │
│  ┌────────┬─────────┬─────────┬─────────┬─────────────┐ │
│  │ Set 1  │ 70×12   │ 75×12   │ 80×10   │ [    ] ×12  │ │
│  ├────────┼─────────┼─────────┼─────────┼─────────────┤ │
│  │ Set 2  │ 75×10   │ 80×10   │ 85×8    │ [    ] ×10  │ │
│  ├────────┼─────────┼─────────┼─────────┼─────────────┤ │
│  │ Set 3  │ 80×8    │ 85×8    │ 87.5×6  │ [    ] ×8   │ │
│  └────────┴─────────┴─────────┴─────────┴─────────────┘ │
│                                                         │
│  ─────────────────────────────────────────────────────  │
│  💡 You lifted 87.5kg last time - try 90kg today!      │
│  🏆 PR: 87.5kg × 6 (Dec 4)                              │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

#### UI Mockup - Empty State (No History)

```
┌─────────────────────────────────────────────────────────┐
│  Squat                                          [Legs]  │
├─────────────────────────────────────────────────────────┤
│  Level:  [BEG]  [NOV]  [INT]  [ADV]    (not rated)      │
│           89kg  123kg  160kg  194kg                     │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  WORKOUT LOG                                            │
│  ─────────────────────────────────────────────────────  │
│                                            TODAY        │
│  ┌────────┬─────────┬─────────┬─────────┬─────────────┐ │
│  │ Set 1  │         │         │         │ [    ] ×12  │ │
│  ├────────┼─────────┼─────────┼─────────┼─────────────┤ │
│  │ Set 2  │         │         │         │ [    ] ×10  │ │
│  ├────────┼─────────┼─────────┼─────────┼─────────────┤ │
│  │ Set 3  │         │         │         │ [    ] ×8   │ │
│  └────────┴─────────┴─────────┴─────────┴─────────────┘ │
│                                                         │
│  ─────────────────────────────────────────────────────  │
│  💡 Log your first workout to start tracking progress!  │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

#### UI Mockup - Mobile Collapsed vs Expanded

**Collapsed (default):**
```
┌─────────────────────────────────────┐
│  Bench Press              [Chest] ▼ │
│  [BEG] [NOV] [INT✓] [ADV]           │
└─────────────────────────────────────┘
```

**Expanded (tap to open):**
```
┌─────────────────────────────────────┐
│  Bench Press              [Chest] ▲ │
│  [BEG] [NOV] [INT✓] [ADV]           │
│─────────────────────────────────────│
│  Nov28  Dec2  Dec4   TODAY          │
│  70×12  75×12 80×10  [  ]×12        │
│  75×10  80×10 85×8   [  ]×10        │
│  80×8   85×8  87×6   [  ]×8         │
│─────────────────────────────────────│
│  💡 Try 90kg today!  🏆 PR: 87.5kg  │
└─────────────────────────────────────┘
```

#### Interaction Flow
| Action | Result |
|--------|--------|
| Tap exercise card | Expands to show workout log |
| Enter weight in TODAY column | Auto-saves, reps pre-filled (12/10/8) |
| Tap reps number | Can edit reps if different |
| After logging | Smart tip updates based on new data |

#### Test Cases
- [ ] Exercise card expands on tap
- [ ] Last 3 sessions display correctly
- [ ] Empty state shows placeholder reps (12, 10, 8)
- [ ] Weight entry auto-saves
- [ ] Reps are editable
- [ ] PR detection works
- [ ] Smart tips update after logging

---

### 2.5 Achievements/Badges

#### Requirements
- Reward quality over quantity
- Number of exercises rated doesn't matter
- Based on level achieved, not exercise count

#### Badge Definitions

| Badge | Name | Condition |
|-------|------|-----------|
| 🏋️ | First Steps | Rate your first exercise |
| 🔥 | On Fire | Reach Intermediate on any exercise |
| 👑 | Elite Lifter | Reach Advanced on any exercise |
| 📈 | Level Up | Improve any exercise by one level |
| 💪 | Double Advanced | Have 2 exercises at Advanced |
| ⚖️ | Balanced | All your rated exercises at same level |
| 🎯 | Focused | All rated exercises in same body part |
| 🌟 | Peak Performance | Average level is Advanced |

#### Design

```typescript
interface Badge {
  id: string
  name: string
  icon: string
  description: string
  unlockedAt?: string  // ISO date when earned
}

interface ProfileBadges {
  [profileId: string]: Badge[]
}
```

#### Test Cases
- [ ] Badge unlocks when condition met
- [ ] Badge shows unlock date
- [ ] Locked badges appear grayed out
- [ ] No badge requires minimum exercise count

---

### 2.6 Strength Score

#### Requirements
- Pure average of rated exercises
- 4 exercises or 25 - same formula
- Score reflects quality, not quantity
- Scale: 0-100

#### Formula

```
Score = Average Level × 25

Level values:
- Beginner = 1 → Score 25
- Novice = 2 → Score 50
- Intermediate = 3 → Score 75
- Advanced = 4 → Score 100

Example:
• 4 exercises, all Advanced → avg 4.0 → Score: 100
• 5 exercises, all Intermediate → avg 3.0 → Score: 75
• 3 exercises (1 Beg, 1 Nov, 1 Int) → avg 2.0 → Score: 50
```

#### Design

```typescript
function calculateStrengthScore(ratings: ExerciseRatings): number {
  const levels = Object.values(ratings).filter(Boolean)
  if (levels.length === 0) return 0

  const levelValues = { beginner: 1, novice: 2, intermediate: 3, advanced: 4 }
  const sum = levels.reduce((acc, level) => acc + levelValues[level], 0)
  const average = sum / levels.length

  return Math.round(average * 25)
}
```

#### UI Display
- Large circular gauge showing score (0-100)
- Color gradient: red (0-25) → yellow (26-50) → green (51-75) → gold (76-100)
- Label showing level name (e.g., "Intermediate Level")
- No mention of exercise count

#### Test Cases
- [ ] Score calculates correctly with any number of exercises
- [ ] Empty profile shows 0
- [ ] All Advanced = 100
- [ ] All Beginner = 25
- [ ] Mixed levels average correctly

---

### 2.7 AI Coach Tips

#### Requirements
- Contextual tips based on profile data
- Focus on quality, not quantity
- Never suggest "add more exercises"
- Encouraging and actionable

#### Tip Scenarios

| Scenario | Tip |
|----------|-----|
| All Beginner | "Everyone starts somewhere! Focus on form before adding weight." |
| Mix of levels | "Your Bench Press is stronger than your Squat - that's common! Work on what feels right." |
| All same level | "Solid consistency across your lifts. You're ready to push for the next level!" |
| One Advanced, rest lower | "Your Deadlift is elite! The other lifts will catch up with time." |
| All Advanced | "You're in the top tier. Time to set new PRs or try new variations!" |
| No exercises rated | "Tap any exercise and select your level to get started!" |
| Close to PR | "You lifted 87.5kg last time - try 90kg today!" |

#### Design

```typescript
interface CoachTip {
  id: string
  message: string
  type: 'encouragement' | 'suggestion' | 'achievement'
  priority: number
}

function generateCoachTips(profile: Profile, workoutLog: WorkoutSession[]): CoachTip[]
```

#### Test Cases
- [ ] Tip displays based on current profile state
- [ ] Tips update after changes
- [ ] No tips about exercise count
- [ ] Workout-specific tips show when relevant

---

## 3. UI Specifications

### 3.1 Color Palette

**Primary Colors**
| Name | Hex | Usage |
|------|-----|-------|
| Primary | `#2C3E50` | Headers, primary text |
| Secondary | `#3498DB` | Buttons, accents |
| Success | `#27AE60` | Save actions |

**Level Colors**
| Level | Hex | Color |
|-------|-----|-------|
| Beginner | `#2ECC71` | Green |
| Novice | `#3498DB` | Blue |
| Intermediate | `#F39C12` | Orange |
| Advanced | `#E74C3C` | Red |

### 3.2 Typography

| Element | Size | Weight |
|---------|------|--------|
| Screen Title | 13pt | 600 |
| Section Header | 10pt | 600 |
| Body Text | 9pt | 400 |
| Button Text | 11pt | 600 |

**Font:** System default (San Francisco / Roboto)

### 3.3 Components

**Buttons**
- Height: 44-48pt (touch-friendly)
- Border radius: 8pt
- Primary: `#3498DB` fill, white text

**Cards**
- Background: white
- Border: 1pt solid `#E0E0E0`
- Border radius: 8pt
- No shadow (flat design)

**Inputs**
- Height: 44pt
- Border: 1pt solid `#E0E0E0`
- Border radius: 4pt
- Focus: `#3498DB` border

---

## 4. Technical Stack

| Layer | Technology |
|-------|------------|
| Framework | Next.js 14+ (App Router) |
| Language | TypeScript (strict) |
| Styling | Tailwind CSS v3 |
| Storage | localStorage (Phase 1) |
| Hosting | Vercel |

### 4.1 Folder Structure

```
src/
├── app/
│   ├── page.tsx              # Profile list
│   ├── layout.tsx
│   └── profile/
│       ├── new/page.tsx      # Create profile
│       └── [id]/page.tsx     # Profile detail + exercises
├── components/
│   ├── ui/                   # Button, Input, Card
│   ├── profile/              # ProfileCard, ProfileForm
│   └── exercise/             # ExerciseCard, LevelBadge
├── lib/
│   ├── storage.ts            # localStorage helpers
│   ├── calculations.ts       # Weight calculations
│   ├── exercises.ts          # Exercise data
│   └── quotes.ts             # Quotes data + logic
├── hooks/
│   └── useProfiles.ts
└── types/
    └── index.ts
```

---

## 5. Development Phases

### Phase 1: MVP ✅
- [x] Profile CRUD (create, read, update, delete)
- [x] 4 core exercises (Bench, Squat, Deadlift, Shoulder Press)
- [x] Level selection and persistence
- [x] Basic UI

### Phase 2: Enhancement ✅
- [x] Expand to 25 exercises
- [x] Body part filtering
- [x] Rated/Unrated exercise sections
- [x] Overall profile level calculation
- [x] Motivational quotes (110 quotes)
- [x] Dark mode toggle
- [ ] Units toggle (kg/lbs)

### Phase 3: Fun Features (Current)
- [ ] Workout Logger (expandable exercise cards)
- [ ] Achievements/Badges system
- [ ] Strength Score (0-100)
- [ ] AI Coach Tips

### Phase 4: Polish
- [ ] Animations & transitions
- [ ] PWA support (offline, installable)
- [ ] Performance optimization
- [ ] Share profile feature

---

## 6. Success Criteria

| Metric | Target |
|--------|--------|
| Profile creation | <2 minutes |
| Page load | <2 seconds |
| Works offline | Yes (localStorage) |
| Mobile responsive | Yes |

---

## Sources

- [Strength Level](https://strengthlevel.com/) - Strength standards data
- [Legion Athletics](https://legionathletics.com/strength-standards/) - Body weight multipliers

---

**Document Status:** Active Development
**Current Phase:** Phase 3 - Fun Features
**Next:** Implement Workout Logger, Achievements, Strength Score, AI Coach Tips
