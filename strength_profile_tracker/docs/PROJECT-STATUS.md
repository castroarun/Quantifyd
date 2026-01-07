# REPPIT - Project Status

> **App Name:** REPPIT (formerly Strength Profile Tracker)
> **Tagline:** Track your reps. Build your strength.

**Last Updated:** 2026-01-07
**Jira Board:** https://castroarun.atlassian.net/jira/software/projects/SPT/board

---

## Jira Backlog - Created Today (Jan 7, 2026)

### Bugs (P1 - High Priority)
| Key | Summary | Status |
|-----|---------|--------|
| SPT-56 | Reppit icon notification not appearing in mobile status bar when app minimized | ✅ Done |
| SPT-57 | Timer countdown/countup flip resets to zero instead of preserving elapsed time | ✅ Done |
| SPT-60 | Full screen timer goes out of window in landscape mode on mobile | ✅ Done |
| SPT-62 | Cannot scroll left to see beyond 1-2 last workout logs | ✅ Done |

### Bugs (P2 - Medium Priority)
| Key | Summary | Status |
|-----|---------|--------|
| SPT-59 | 30-second timer remaining should have double blip sound in sync | To Do |
| SPT-61 | Set suggestions should match user actual workout history pattern | To Do |

### Enhancements (P2)
| Key | Summary | Status |
|-----|---------|--------|
| SPT-58 | Add floating timer window when switching to other apps | To Do |
| SPT-63 | Add machine brand logging feature for workout equipment | To Do |

---

## Quick Summary

**Completed Features:**
- Profile management (create/edit/delete up to 5 profiles)
- 46 exercises across 6 body parts (including Core) with strength standards
- 7 predefined workout routines (PPL, Upper/Lower, Bro Split, Full Body, etc.)
- Per-routine exercise selection with persistence
- Workout logging with history (last 2 sessions + today)
- Rest timer with per-exercise memory and auto-start
- PR detection (triggers celebration when you beat records)
- Auto level-up when PR exceeds threshold
- Smart suggestions - TARGET column with recommended weights/reps
- Copy buttons to quickly fill today's entry from suggestions
- Progress page with muscle heatmap and calendar visualization
- Unit conversion (kg/lbs)
- Dark mode toggle
- Motivational quotes on app load
- Contextual tips system (onboarding)
- Wake lock during workouts (keep screen on)
- Full-screen timer mode

**Recently Added (Dec 28, 2025):**
- 22 new exercises: Core (Ab Crunch Machine/Cable, Hanging Leg Raise, Cable Woodchop, Russian Twist, Leg Raise), Chest (Chest Press Machine, Pec Deck, Decline Bench), Back (T-Bar Row, Dumbbell Row, Seated Row), Shoulders (Rear Delt Fly, Face Pull), Legs (Hip Thrust, Bulgarian Split Squat, Goblet Squat, Hack Squat), Arms (Hammer Curl, Tricep Dip, Overhead Tricep Extension, Preacher Curl, Cable Curl)
- Expanded default exercise selections (18 exercises per routine vs. 6)
- Updated all predefined routines with 7-10 exercises per workout day
- Per-routine exercise persistence (selections saved per routine)
- Separate "Activate" toggle inside routines (viewing doesn't auto-activate)
- Training Program status shows "Active: [name]" in Settings
- Fixed Done button to only show when there are completable sets
- Added contextual tips for training program setup and timer settings

**Pending Features:**
- [ ] Reorder by Workout Log toggle - auto-reorder exercises based on user's logging order (default ON first time, then user-controlled)

---

## Detailed Status

| Phase | Item | Actual Status | Jira Status |
|-------|------|---------------|-------------|
| Design | PRD Document (APP_PRD.md) | Done | In Review |
| Design | UI Prototype (ui-prototype.html) | Done | In Review |
| Design | - 46 Exercises (6 body parts incl. Core) | Done | - |
| Design | - Dark Mode Toggle | Done | - |
| Design | - 50 Motivational Quotes | Done | - |
| Design | - 4 User Difficulty Levels | Done | - |
| Design | - Exercise Filtering by Body Part | Done | - |
| Design | - Rated Exercises Sort to Top | Done | - |
| Design | - Muscle Heatmap Visualization | Done | - |
| Design | - Anatomy Visualization Mockups | Done | - |
| Design | - Target BMI Indicator | Done | - |
| Design | - kg/lbs Unit Toggle | Done | - |
| Design | Test Cases | Pending | To Do |
| Setup | Jira Backlog | Done | - |
| Setup | GitHub Repository | Done | - |
| Build | Profile Management | Done | To Do |
| Build | Exercise Tracking | Done | To Do |
| Build | Standards Calculator | Done | To Do |
| Build | Data Persistence | Done | To Do |
| Build | Theme Toggle (Dark Mode) | Done | - |
| Build | Unit Toggle (kg/lbs) | Done | - |
| Build | Muscle Heatmap | Done | - |
| Build | Motivational Quotes | Done | - |
| Build | Workout Logging (3 sets/exercise) | Done | - |
| Build | Workout History (last 2 sessions) | Done | - |
| Build | PR Detection & Celebration | Done | - |
| Build | Auto Level-Up on PR | Done | - |
| Build | Smart Suggestions (TARGET column) | Done | - |
| Build | Copy-to-Today Buttons | Done | - |
| Build | Progress Page with Heatmap | Done | - |
| Build | Rest Timer (per-exercise, auto-start) | Done | - |
| Build | 7 Predefined Workout Routines | Done | - |
| Build | Per-routine Exercise Selection | Done | - |
| Build | Contextual Tips/Onboarding | Done | - |
| Build | Wake Lock (keep screen on) | Done | - |
| Build | Full-Screen Timer Mode | Done | - |
| Build | Reorder by Workout Log | Pending | - |

---

## Current Project Status (9-Step Workflow)

| Step | Name | Actual Status | Jira Status | Jira Task |
|------|------|---------------|-------------|-----------|
| 1 | DEV-CLOCK | In Progress | In Progress | SPT-2 |
| 2 | PRD & Design | Done | In Review | SPT-1 |
| 3 | Test Cases | Pending | To Do | SPT-3 |
| 4 | Build | Done | To Do | SPT-4 |
| 5 | Manual Testing | In Progress | To Do | SPT-5 |
| 6 | Debug & Feedback | In Progress | To Do | SPT-6 |
| 7 | Code Walkthrough | Not Started | To Do | SPT-7 |
| 8 | Ship | Not Started | To Do | SPT-8 |
| 9 | Time Retrospective | Not Started | To Do | SPT-9 |

---

## Next Actions

- [ ] Implement "Reorder by Workout Log" toggle feature
  - Toggle in Settings (default ON first time, then OFF)
  - Auto-reorder exercises based on logging order
  - Persist order for next session
- [ ] Complete manual testing of all features
- [ ] Test PR detection and level-up flow
- [ ] Test smart suggestions accuracy
- [ ] Verify mobile responsiveness
- [ ] Code walkthrough before ship

---

## Key Files

| File | Description |
|------|-------------|
| [docs/APP_PRD.md](APP_PRD.md) | Product Requirements Document |
| [docs/ui-prototype.html](ui-prototype.html) | Interactive HTML mockup |
| [docs/DEV-CLOCK.md](DEV-CLOCK.md) | Time tracking |
| [src/components/strength/WorkoutLogger.tsx](../src/components/strength/WorkoutLogger.tsx) | Workout logging with suggestions |
| [src/lib/storage/workouts.ts](../src/lib/storage/workouts.ts) | PR detection & level-up logic |

---

## Tech Stack

- **Framework:** Next.js 16.0.7
- **Language:** TypeScript
- **Styling:** Tailwind CSS
- **Storage:** localStorage (PWA-ready)
- **Deployment:** Vercel (planned)
