# APP PRD: Strength Profile Tracker

**Version:** 3.2
**Date:** 2025-12-09
**Status:** In Development

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-12-01 | Initial PRD with profile management and 4 core exercises |
| 2.0 | 2025-12-02 | Expanded to 25 exercises, body part filtering, motivational quotes |
| 3.0 | 2025-12-06 | Added dark mode, Workout Logger, Achievements, Strength Score, AI Coach Tips |
| 3.1 | 2025-12-06 | Added Progress Visualizations (5 chart types) |
| 3.2 | 2025-12-09 | Added Phase 5 enhancements: Auto-level updates, PR celebrations, scrollable history |
| 4.0 | 2025-12-20 | Added Rest Timer (2.9) and Transformation Tracker (2.10) features |

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

### 2.8 Progress Visualizations

#### Requirements
- Dedicated progress page accessible from profile
- Visual representation of strength journey
- Uses data from workout logger and exercise ratings
- Works with any number of exercises (quality over quantity)

#### Chart Types

**1. Strength Score History (Line Chart)**
- Tracks score changes over time
- Shows start → current comparison
- Point-based progression

**2. Personal Records**
- Top 3 PRs with medal styling (🥇🥈🥉)
- Shows exercise, weight, date, improvement amount
- Celebrates achievements

**3. Body Part Balance (Radar Chart)**
- Spider/pentagon chart
- Shows strength across Chest, Back, Shoulders, Legs, Arms
- Color-coded by percentage

**4. Workout Frequency (Calendar Heatmap)**
- GitHub-style activity grid
- Shows workout intensity by day
- Tracks streaks and totals

**5. Exercise Progression (Line Chart)**
- Per-exercise weight progression
- Gradient fill under line
- Date labels on x-axis, weight on y-axis

#### UI Mockup - Strength Score History

```
┌─────────────────────────────────────────────┐
│  Strength Score History                      │
├─────────────────────────────────────────────┤
│  100 ─┬──────────────────────────────────   │
│       │                              ●63    │
│   75 ─┤                        ●55          │
│       │                  ●42                │
│   50 ─┤            ●35                      │
│       │                                     │
│   25 ─┤                                     │
│       │                                     │
│    0 ─┴────────────────────────────────────  │
│       Nov 1   Nov 15   Dec 1    Dec 6       │
├─────────────────────────────────────────────┤
│   Started: 35  →  Current: 63    [+28 pts]  │
└─────────────────────────────────────────────┘
```

#### UI Mockup - Personal Records

```
┌─────────────────────────────────────────────┐
│  🏆 Personal Records                         │
├─────────────────────────────────────────────┤
│  🥇 Bench Press       80kg    Dec 5   +5kg  │
│  🥈 Squat            100kg    Dec 3  +10kg  │
│  🥉 Deadlift         120kg   Nov 28  +7.5kg │
└─────────────────────────────────────────────┘
```

#### UI Mockup - Body Part Balance (Radar)

```
┌─────────────────────────────────────────────┐
│  Body Part Balance                           │
├─────────────────────────────────────────────┤
│              Chest                           │
│                ▲                             │
│          Arms / \ Back                       │
│              /   \                           │
│             /     \                          │
│   Shoulders ─── ─── Legs                     │
│                                              │
│  ● Chest: 75%  ● Back: 60%  ● Arms: 85%     │
│  ● Shoulders: 50%  ● Legs: 40%              │
└─────────────────────────────────────────────┘
```

#### UI Mockup - Workout Frequency (Heatmap)

```
┌─────────────────────────────────────────────┐
│  Workout Frequency                           │
├─────────────────────────────────────────────┤
│       S  M  T  W  T  F  S                    │
│  W1  ░░ ▓▓ ░░ ██ ░░ ██ ░░                   │
│  W2  ▓▓ ░░ ██ ░░ ▓▓ ░░ ░░                   │
│  W3  ░░ ██ ░░ ██ ░░ ██ ▓▓                   │
│  W4  ██ ░░ ▓▓ ░░ ██ ░░ ░░                   │
│                                              │
│  Less ░░▓▓██ More                           │
├─────────────────────────────────────────────┤
│  Total: 12    This Week: 3    Streak: 5     │
└─────────────────────────────────────────────┘
```

#### UI Mockup - Exercise Progression (Line)

```
┌─────────────────────────────────────────────┐
│  Exercise Progression          [Bench Press] │
├─────────────────────────────────────────────┤
│  75kg ─┬──────────────────────────────●     │
│        │                         ●          │
│  70kg ─┤                    ●               │
│        │               ●                    │
│  65kg ─┤          ●                         │
│        │                                    │
│  60kg ─┴────────────────────────────────    │
│        Nov20  Nov25  Nov30  Dec3   Dec6     │
├─────────────────────────────────────────────┤
│              ↑ +10kg in last 3 weeks        │
└─────────────────────────────────────────────┘
```

#### Design

```typescript
// Route: /profile/[id]/progress

interface ProgressPageProps {
  profileId: string
}

// Data sources
- Strength Score: calculateStrengthScore(ratings) over time
- Personal Records: Max weight from WorkoutSession[]
- Body Part Balance: Average level per body part from ratings
- Workout Frequency: Count sessions by date
- Exercise Progression: Weight from WorkoutSession[] for exercise
```

#### Test Cases
- [ ] Score history line chart renders with data points
- [ ] PRs display correctly with medals
- [ ] Radar chart shows all 5 body parts
- [ ] Heatmap shows correct intensity colors
- [ ] Exercise progression line connects all points
- [ ] Empty states handled gracefully
- [ ] Dark mode works for all charts

---

### 2.9 Rest Timer

#### Requirements
- Countdown timer integrated with workout logging
- Configurable default rest time (30s - 5min)
- Quick adjustment buttons (+15s / -15s)
- Preset buttons for common rest times
- Optional auto-start after logging a set
- Sound and vibration alerts when timer ends
- Per-exercise timer memory (remembers last used time)

#### Research (Hevy & Strong Apps)
| App | Approach | Key Features |
|-----|----------|--------------|
| Hevy | Auto-start after set | 15s increments, per-exercise defaults, vibration/sound |
| Strong | Configurable per set type | Full-screen countdown, background continue, +30s extend |

#### Design

```typescript
interface TimerSettings {
  defaultRestTime: number        // seconds (30-300), default: 90
  autoStartAfterSet: boolean     // default: false
  soundEnabled: boolean          // default: true
  vibrationEnabled: boolean      // default: true
}

interface ExerciseTimerHistory {
  [exerciseId: string]: number   // last used time per exercise
}

// Storage
const TIMER_SETTINGS_KEY = 'spt_timer_settings'
const TIMER_HISTORY_KEY = 'spt_timer_history'
```

#### UI Mockup - Timer Below Workout Log

```
┌─────────────────────────────────────────────────────────┐
│  WORKOUT LOG                                            │
│  ─────────────────────────────────────────────────────  │
│     Dec4   TODAY   TARGET                               │
│   ┌───────┬────────┬────────┐                           │
│   │ 80×10 │ [82]×12│ 85×12  │                           │
│   │ 85×8  │ [  ]×10│ 87×10  │                           │
│   │ 87×6  │ [  ]×8 │ 90×8   │                           │
│   └───────┴────────┴────────┘                           │
│  ─────────────────────────────────────────────────────  │
│                                                         │
│   ⏱️ REST TIMER                                         │
│   ┌─────────────────────────────────────────────────┐   │
│   │              01:30                              │   │
│   │         [-15s]  [▶ START]  [+15s]               │   │
│   │                                                 │   │
│   │  Quick: [30s] [1:00] [1:30] [2:00] [3:00]      │   │
│   └─────────────────────────────────────────────────┘   │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

#### UI Mockup - Timer Running

```
┌─────────────────────────────────────────────────────────┐
│   ⏱️ REST TIMER                                         │
│   ┌─────────────────────────────────────────────────┐   │
│   │              00:47                              │   │
│   │           ████████████░░░░░░░░                  │   │
│   │                                                 │   │
│   │         [+15s]   [⏸ PAUSE]   [RESET]           │   │
│   └─────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

#### Components

| Component | Location | Purpose |
|-----------|----------|---------|
| `RestTimer.tsx` | `src/components/timer/` | Main timer UI with controls |
| `TimerSettings.tsx` | `src/components/settings/` | Timer configuration panel |
| `useTimer.ts` | `src/hooks/` | Timer state, countdown logic |
| `useTimerSettings.ts` | `src/hooks/` | Settings persistence |

#### Timer Settings UI

```
┌─────────────────────────────────────────────────────────┐
│  ⚙️ Timer Settings                                      │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Default Rest Time                                      │
│  ┌─────────────────────────────────────────────────┐   │
│  │  [-]  ████████████░░░░░░  1:30  [+]             │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
│  Auto-start after logging set    [  OFF  |  ON  ]      │
│                                                         │
│  Sound alert                     [  OFF  |  ON  ]      │
│                                                         │
│  Vibration                       [  OFF  |  ON  ]      │
│                                                         │
│  Reset All Exercise Timers       [ RESET ]             │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

#### Test Cases
- [ ] Timer displays in expanded exercise card
- [ ] Countdown works correctly (1s intervals)
- [ ] +15s/-15s buttons adjust time
- [ ] Quick preset buttons set correct time
- [ ] Start/Pause/Reset controls work
- [ ] Sound plays when timer reaches 0
- [ ] Vibration triggers on complete
- [ ] Timer persists when scrolling (doesn't reset)
- [ ] Per-exercise time memory saves/loads
- [ ] Settings persist across sessions
- [ ] Auto-start triggers after set logged (when enabled)

---

### 2.10 Transformation Tracker

#### Requirements
- Capture progress photos via phone camera (PWA)
- Store photos locally using IndexedDB (supports large blobs)
- Gallery view with date-organized photos
- Three pose types: Front, Side, Back
- Ghost overlay of previous photo for consistent positioning
- Before/after comparison with slider
- Photo tips for consistency (lighting, timing, pose)
- Delete photos with confirmation

#### Research (Body Journey, Shapez, SnapTrack)
| App | Key Feature | User Benefit |
|-----|-------------|--------------|
| Body Journey | Ghost overlay of previous photo | Consistent pose alignment |
| Shapez | In-app camera with countdown timer | Hands-free capture |
| SnapTrack | Side-by-side comparison slider | Easy before/after view |

#### Design

```typescript
interface ProgressPhoto {
  id: string
  profileId: string
  date: string              // YYYY-MM-DD
  pose: 'front' | 'side' | 'back'
  imageBlob: Blob           // Full size (compressed ~1MB)
  thumbnailBlob: Blob       // Small for gallery (~50KB)
  createdAt: string
}

// IndexedDB schema (not localStorage - blobs too large)
const DB_NAME = 'spt_photos'
const DB_VERSION = 1
const STORE_NAME = 'progress_photos'

// IndexedDB operations
async function savePhoto(photo: ProgressPhoto): Promise<void>
async function getPhotosByProfile(profileId: string): Promise<ProgressPhoto[]>
async function getPhotosByDate(profileId: string, date: string): Promise<ProgressPhoto[]>
async function deletePhoto(photoId: string): Promise<void>
async function getStorageUsage(): Promise<{ used: number; available: number }>
```

#### PWA Camera Access

```typescript
// Request camera permission
const stream = await navigator.mediaDevices.getUserMedia({
  video: {
    facingMode: 'user',  // or 'environment' for back camera
    width: { ideal: 1920 },
    height: { ideal: 1080 }
  }
})

// Capture frame from video to canvas
const canvas = document.createElement('canvas')
const ctx = canvas.getContext('2d')
ctx.drawImage(videoElement, 0, 0)

// Compress and convert to blob
canvas.toBlob(blob => savePhoto(blob), 'image/jpeg', 0.8)
```

#### UI Mockup - Photo Gallery Page

```
┌─────────────────────────────────────────────────────────┐
│  ← Transformation Tracker              📷 [Take Photo]  │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  📸 Your Progress Journey                               │
│                                                         │
│  December 2024                                          │
│  ┌─────────────────────────────────────────────────┐   │
│  │  Dec 1      Dec 7      Dec 14     Dec 20        │   │
│  │  ┌─────┐   ┌─────┐   ┌─────┐   ┌─────┐         │   │
│  │  │ 📷  │   │ 📷  │   │ 📷  │   │  +  │         │   │
│  │  │Front│   │Front│   │Front│   │     │         │   │
│  │  └─────┘   └─────┘   └─────┘   └─────┘         │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
│  November 2024                                          │
│  ┌─────────────────────────────────────────────────┐   │
│  │  Nov 15     Nov 22     Nov 29                   │   │
│  │  ┌─────┐   ┌─────┐   ┌─────┐                    │   │
│  │  │ 📷  │   │ 📷  │   │ 📷  │                    │   │
│  │  │Front│   │Front│   │Front│   │               │   │
│  │  └─────┘   └─────┘   └─────┘                    │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
│  ─────────────────────────────────────────────────────  │
│  [🔄 Compare Mode]                                      │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

#### UI Mockup - Camera Capture

```
┌─────────────────────────────────────────────────────────┐
│  ← Take Photo                         [🔄 Switch Cam]   │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌─────────────────────────────────────────────────┐   │
│  │                                                 │   │
│  │            [CAMERA VIEWFINDER]                  │   │
│  │                                                 │   │
│  │     ╭─────────────────────────────╮             │   │
│  │     │                             │   Ghost    │   │
│  │     │    👤 Pose Silhouette       │   overlay  │   │
│  │     │       (alignment guide)     │   (40%)    │   │
│  │     │                             │             │   │
│  │     ╰─────────────────────────────╯             │   │
│  │                                                 │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
│  Pose: [● Front] [ Side ] [ Back ]                     │
│                                                         │
│  Ghost Overlay: [OFF] [Previous] [First]               │
│                                                         │
│        [⏱️ 3s Timer]        [📷 Capture]               │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

#### UI Mockup - Before/After Comparison

```
┌─────────────────────────────────────────────────────────┐
│  ← Compare                                    [Share]   │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌─────────────────────────────────────────────────┐   │
│  │                    │                            │   │
│  │      BEFORE        │         AFTER              │   │
│  │      Dec 1         │         Dec 20             │   │
│  │                    │                            │   │
│  │     ┌──────┐       │       ┌──────┐            │   │
│  │     │      │       │       │      │            │   │
│  │     │  📷  │◄──────┼──────►│  📷  │            │   │
│  │     │      │    ◄──┼───    │      │            │   │
│  │     │      │   DRAG│SLIDER │      │            │   │
│  │     └──────┘       │       └──────┘            │   │
│  │                    │                            │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
│  Select dates:                                          │
│  Before: [Dec 1 ▼]     After: [Dec 20 ▼]               │
│                                                         │
│  Journey: 19 days  |  Photos: 4                        │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

#### Components

| Component | Location | Purpose |
|-----------|----------|---------|
| `PhotoGallery.tsx` | `src/components/photos/` | Grid view of all photos |
| `CameraCapture.tsx` | `src/components/photos/` | Camera interface with controls |
| `PhotoCompare.tsx` | `src/components/photos/` | Side-by-side slider comparison |
| `PoseGuide.tsx` | `src/components/photos/` | Silhouette overlay for alignment |
| `PhotoTips.tsx` | `src/components/photos/` | First-use tips modal |
| `useCamera.ts` | `src/hooks/` | Camera access and capture |
| `useIndexedDB.ts` | `src/hooks/` | IndexedDB CRUD operations |

#### Photo Tips (shown on first use)

```
┌─────────────────────────────────────────────────────────┐
│  📸 Tips for Great Progress Photos                      │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ☀️  Same lighting each time (natural light best)       │
│                                                         │
│  🕐  Same time of day (morning recommended)             │
│                                                         │
│  📏  Same distance from camera                          │
│                                                         │
│  👕  Same or minimal clothing                           │
│                                                         │
│  🧍  Relaxed pose, arms at sides                        │
│                                                         │
│  📅  Weekly photos recommended                          │
│                                                         │
│                    [Got it!]                            │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

#### Route Structure

```
/profile/[id]/photos          - Gallery page
/profile/[id]/photos/capture  - Camera capture
/profile/[id]/photos/compare  - Before/after comparison
```

#### Test Cases
- [ ] Camera permission requested on first use
- [ ] Front/back camera switch works
- [ ] Photo captures and saves to IndexedDB
- [ ] Thumbnail generated for gallery
- [ ] Gallery displays photos by month
- [ ] Pose filter works (Front/Side/Back)
- [ ] Ghost overlay shows previous photo
- [ ] 3-second timer countdown works
- [ ] Compare mode slider drags correctly
- [ ] Date selectors in compare mode work
- [ ] Delete photo with confirmation
- [ ] Storage usage displayed
- [ ] Works offline after first load
- [ ] Photos persist after app restart

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

### Phase 3: Fun Features ✅
- [x] Workout Logger (expandable exercise cards)
- [x] Achievements/Badges system (8 badges)
- [x] Strength Score (0-100 with circular gauge)
- [x] AI Coach Tips (contextual tips)
- [x] Progress Visualizations (mockup with 5 chart types)

### Phase 4: Progress Visualizations (Current)
- [ ] Connect Score History to real data
- [ ] Connect Personal Records to workout data
- [ ] Connect Body Part Balance to ratings
- [ ] Connect Workout Frequency to session data
- [ ] Connect Exercise Progression to workout data
- [ ] Add link to progress page from profile

### Phase 5: Enhanced UX ✅
- [x] Auto-update exercise levels based on workout PRs
- [x] PR celebration messages (10 variations)
- [x] Horizontally scrollable workout log history

### Phase 6: Rest Timer
- [ ] Timer component with countdown display
- [ ] Start/Pause/Reset controls
- [ ] +15s/-15s adjustment buttons
- [ ] Quick preset buttons (30s, 1:00, 1:30, 2:00, 3:00)
- [ ] Sound and vibration alerts
- [ ] Per-exercise timer memory
- [ ] Timer settings page
- [ ] Auto-start after set option

### Phase 7: Transformation Tracker (Future)
- [ ] IndexedDB storage layer for photos
- [ ] Camera capture with PWA getUserMedia
- [ ] Photo gallery by month
- [ ] Pose selection (Front/Side/Back)
- [ ] Ghost overlay for alignment
- [ ] Before/after comparison slider
- [ ] Photo tips modal
- [ ] Storage usage display

### Phase 8: Polish
- [ ] Animations & transitions
- [ ] Performance optimization
- [ ] Share profile feature

---

## 7. Upcoming Enhancements

### 7.1 Auto-Update Exercise Levels

#### Requirements
- When user logs a new PR, automatically update their exercise level if it exceeds the threshold
- Levels should only increase, never decrease (progress protection)
- Show notification when level increases

#### Logic

```typescript
// On workout save, check if PR triggers level upgrade
function checkLevelUpgrade(
  exercise: Exercise,
  userWeight: number,
  newPR: number,
  currentLevel: Level | null
): Level | null {
  // Calculate thresholds for each level
  const thresholds = {
    beginner: userWeight * exercise.multipliers.beginner,
    novice: userWeight * exercise.multipliers.novice,
    intermediate: userWeight * exercise.multipliers.intermediate,
    advanced: userWeight * exercise.multipliers.advanced
  }

  // Find highest level achieved
  let achievedLevel: Level = 'beginner'
  if (newPR >= thresholds.advanced) achievedLevel = 'advanced'
  else if (newPR >= thresholds.intermediate) achievedLevel = 'intermediate'
  else if (newPR >= thresholds.novice) achievedLevel = 'novice'

  // Only upgrade, never downgrade
  const levelOrder = ['beginner', 'novice', 'intermediate', 'advanced']
  const currentIndex = currentLevel ? levelOrder.indexOf(currentLevel) : -1
  const achievedIndex = levelOrder.indexOf(achievedLevel)

  return achievedIndex > currentIndex ? achievedLevel : null
}
```

#### Test Cases
- [ ] Level upgrades when PR exceeds threshold
- [ ] Level never downgrades
- [ ] Notification displays on level up
- [ ] Works for all exercises

---

### 7.2 PR Celebration Messages

#### Requirements
- Display congratulatory message when user achieves new PR
- 8-10 different message variations
- Exciting tone with emojis
- Shows exercise name in message

#### Celebration Messages

```typescript
const PR_CELEBRATIONS = [
  "🎉 NEW PR! {exercise}! You're crushing it!",
  "🔥 BOOM! New {exercise} record! Way to go!",
  "💪 YES! That's a new {exercise} PR! Beast mode!",
  "🏆 NEW PERSONAL BEST! {exercise}! You're unstoppable!",
  "⚡ INCREDIBLE! New {exercise} PR! Keep that energy!",
  "🚀 LIFT OFF! You just hit a new {exercise} record!",
  "👑 KING/QUEEN of {exercise}! New PR achieved!",
  "🌟 STELLAR! New {exercise} PR! The gains are real!",
  "💥 SMASHED IT! New {exercise} record! Legendary!",
  "🎊 CELEBRATION TIME! New {exercise} PR! You're on fire!"
]

function getCelebrationMessage(exercise: string): string {
  const index = Math.floor(Math.random() * PR_CELEBRATIONS.length)
  return PR_CELEBRATIONS[index].replace('{exercise}', exercise)
}
```

#### UI Display
- Modal/toast notification
- Large emoji animation
- Fade out after 3 seconds
- Optional confetti effect

#### Test Cases
- [ ] Message displays on new PR
- [ ] Different messages each time (random)
- [ ] Exercise name appears in message
- [ ] Auto-dismisses after delay

---

### 7.3 Scrollable Workout History

#### Requirements
- Workout log shows all previous sessions, not just last 3
- Horizontal scroll (swipe left/right) on mobile
- Most recent session on right
- Visual scroll indicators

#### UI Mockup

```
┌─────────────────────────────────────────────────────────────────┐
│  WORKOUT LOG                              ← scroll →            │
│  ─────────────────────────────────────────────────────────────  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Nov20  Nov22  Nov25  Nov28  Dec2   Dec4   Dec6   TODAY  │→│ │
│  │──────────────────────────────────────────────────────────│  │
│  │ 60×12  65×12  65×10  70×12  75×12  80×10  82×8   [  ]×12│  │
│  │ 65×10  65×10  70×10  75×10  80×10  85×8   85×6   [  ]×10│  │
│  │ 70×8   70×8   70×8   80×8   85×8   87×6   88×5   [  ]×8 │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                           ────●●●○○───         │
│                                          (scroll indicator)    │
└─────────────────────────────────────────────────────────────────┘
```

#### Implementation

```typescript
// Horizontal scroll container
<div className="overflow-x-auto scrollbar-hide">
  <div className="flex gap-2 min-w-max">
    {sessions.map(session => (
      <SessionColumn key={session.id} data={session} />
    ))}
    <TodayColumn onSave={handleSave} />
  </div>
</div>

// CSS
.scrollbar-hide {
  -ms-overflow-style: none;
  scrollbar-width: none;
}
.scrollbar-hide::-webkit-scrollbar {
  display: none;
}
```

#### Test Cases
- [ ] All sessions visible via scroll
- [ ] Smooth horizontal scroll on mobile
- [ ] Most recent on right side
- [ ] Auto-scroll to today on load
- [ ] Scroll indicators show position

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
**Current Phase:** Phase 6 - Rest Timer
**Next:** Implement rest timer with settings, then Transformation Tracker (Phase 7)
