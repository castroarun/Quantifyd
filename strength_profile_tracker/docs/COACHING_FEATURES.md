# REPPIT Coaching Features Documentation

This document describes all the coaching mechanisms built into the REPPIT app, how they work, and suggestions for additional features.

---

## 1. Automatic Level System (Performance-Based)

### How It Works
Levels are automatically determined by workout performance, not manual selection. The system compares the weight you lift against body-weight-based thresholds.

**Level Thresholds:**
Each exercise has multipliers that calculate thresholds based on your body weight:
- **Beginner**: `bodyWeight × beginner_multiplier`
- **Novice**: `bodyWeight × novice_multiplier`
- **Intermediate**: `bodyWeight × intermediate_multiplier`
- **Advanced**: `bodyWeight × advanced_multiplier`

**Example (Bench Press for 80kg user):**
- Beginner: 80 × 0.5 = 40kg
- Novice: 80 × 0.75 = 60kg
- Intermediate: 80 × 1.0 = 80kg
- Advanced: 80 × 1.25 = 100kg

### Level Upgrade
When you log a workout weight that meets or exceeds a threshold, you automatically level up:
- Instant notification with animated banner
- Level indicator updates immediately
- Celebration message displayed

### Level Downgrade (Detraining Detection)
If you don't hit your current level threshold in **4 consecutive workouts**:
- System checks your recent performance
- Downgrades you to the highest level you actually achieved
- Shows supportive notification encouraging comeback

**Downgrade Messages:**
```
"Level adjusted to {level}. Keep pushing to get back up!"
"Dropped to {level}. You've got this - time to rebuild!"
"Now at {level}. Every champion has setbacks. Come back stronger!"
"Reset to {level}. Focus on form and the gains will follow!"
```

---

## 2. Smart Weight/Rep Progression System

### Logic Overview
The system analyzes your most recent workout session and calculates smart targets for today.

**Decision Tree:**
```
IF Set 1 reps >= 10:
   → PROGRESS MODE: Add 5kg to all sets
ELSE:
   → MAINTAIN MODE: Same weights, try to beat reps
```

### PROGRESS Mode (S1 reps >= 10)
You're ready to increase weight:
| Set | Weight | Target Reps |
|-----|--------|-------------|
| S1 | Previous + 5kg | 12 |
| S2 | Previous + 5kg | 10 |
| S3 | Previous + 5kg | max(5, S3_reps - 1) |

### MAINTAIN Mode (S1 reps < 10)
Focus on improving reps before adding weight:
| Set | Weight | Target Reps |
|-----|--------|-------------|
| S1 | Same | min(12, S1_reps + 1) |
| S2 | Same | min(10, S2_reps + 1) |
| S3 | Same | min(8, S3_reps + 1) |

### Visual Display
- **TARGET column** shows recommended weights/reps
- **Copy buttons** let you quickly transfer targets to TODAY
- Targets are clickable to auto-fill

---

## 3. PR (Personal Record) Detection & Celebration

### How PRs Are Detected
The system tracks your maximum weight for each exercise and compares:
1. Against historical data (all past workouts excluding today)
2. Against other sets logged today

### PR Celebrations (10 variations)
When you hit a new PR, a colorful animated banner appears:
```
"NEW PR! {exercise}! You're crushing it!"
"BOOM! New {exercise} record! Way to go!"
"YES! That's a new {exercise} PR! Beast mode!"
"NEW PERSONAL BEST! {exercise}! You're unstoppable!"
"INCREDIBLE! New {exercise} PR! Keep that energy!"
"LIFT OFF! You just hit a new {exercise} record!"
"KING/QUEEN of {exercise}! New PR achieved!"
"STELLAR! New {exercise} PR! The gains are real!"
"SMASHED IT! New {exercise} record! Legendary!"
"CELEBRATION TIME! New {exercise} PR! You're on fire!"
```

### PR Encouragement
When your TARGET weight would be a new PR, a motivational message appears:
```
"Go for {weight}kg! That's a new PR!"
"Time to progress! Aim for {weight}kg!"
```

---

## 4. AI Coach Tips System

### Tip Categories
The coach generates contextual tips based on your profile:

| Type | Icon | Purpose |
|------|------|---------|
| Encouragement | 👋 | Welcome new users, motivate |
| Progress | 📈💪🔥 | Level-specific advice |
| Balance | ⚖️ | Body part imbalance warnings |
| Achievement | 🎯🏆 | Celebrate milestones |

### Priority Logic
1. **Body part imbalances** (highest priority)
2. **Progress advice** based on majority level
3. **Generic encouragement**

### Example Tips
- **New User:** "Welcome! Start by rating a few exercises you regularly perform."
- **Weak Body Part:** "Weak legs: Focus on leg exercises to balance your strength profile."
- **Beginner Level:** "Focus on progressive overload - add small weight each week."
- **Novice Level:** "Consider adding compound movements to accelerate strength gains."
- **Intermediate:** "To reach Advanced, focus on periodization and adequate recovery."
- **Advanced:** "Elite performance! Prioritize injury prevention."

---

## 5. Motivational Quotes System

### Collection Stats
- **110 total quotes**
- **3 categories:** Motivation (45), Science (30), Benefits (35)

### Display Modes
1. **Daily Quote:** Same quote shown throughout the day (consistent experience)
2. **Random Quote:** Random from entire collection
3. **By Category:** Random from specific category

### Quote Examples
**Motivation:**
> "The only bad workout is the one that didn't happen."

**Science:**
> "Muscle protein synthesis remains elevated for 24-48 hours after resistance training."

**Benefits:**
> "Strength training reduces symptoms of anxiety and depression by up to 50%."

---

## 6. Rest Timer with Memory

### Features
- **Countdown/Countup modes**
- **Presets:** 2:30, 3:00, 3:30, 4:00, 4:30, 5:00 (optimized for strength training)
- **Per-exercise memory:** Each exercise remembers your preferred rest time
- **Auto-start:** Opens full-screen timer when you mark a set as "Done"
- **Sound + Vibration alerts** when timer completes
- **+15s/-15s adjustment buttons**
- **Full-screen mode:** Shows contextual info during rest:
  - Set number just completed
  - Weight lifted
  - Reps performed
  - Large countdown display
- **Double-tap to minimize:** Returns to compact timer view
- **Wake Lock integration:** Keeps screen on during rest (when enabled)

### Timer Settings
- Default duration (adjustable from presets)
- Sound on/off
- Vibration on/off
- Auto-start on/off
- **Keep Screen Awake** on/off (prevents screen sleep during workout)

---

## 7. Body Part Balance Analysis

### Detection Logic
The system calculates average level per body part:
- Compares each body part against overall average
- Identifies "weak" body parts (significantly below average)
- Generates coach tips to address imbalances

### Visualization
- Progress page shows muscle heatmap
- Color coding: Green (Advanced) → Yellow (Intermediate) → Red (Beginner)

---

## 8. Achievement Badges System

### Available Badges (8 total)
| Badge | Requirement |
|-------|-------------|
| First Step | Rate your first exercise |
| Getting Started | Rate 5 exercises |
| Dedicated Trainer | Rate 10 exercises |
| Full Assessment | Rate all 23 exercises |
| Novice Milestone | Reach Novice in any exercise |
| Breaking Intermediate | Reach Intermediate in any exercise |
| Advanced Lifter | Reach Advanced in any exercise |
| Elite Status | Reach Advanced in 5+ exercises |

---

---

## 9. Form Tips (IMPLEMENTED)

### How It Works
Level-aware form tips that rotate daily so users see different advice each session.

**Implementation:**
- Small 💡 lightbulb icon next to "Workout Log" title
- Tap to reveal tip - collapsed by default (not cluttering UI)
- Tips are specific to user's current level (Beginner → Advanced)
- Different tips shown each day (rotates based on day of year)

**Coverage:**
- All major compound exercises have 3 tips per level (12 tips per exercise)
- Tips progress from basic form cues to advanced techniques

**Example (Bench Press - by level):**
- Beginner: "Grip slightly wider than shoulders. Keep wrists straight."
- Novice: "Retract scapula before unracking. Create slight arch."
- Intermediate: "Leg drive: push feet into floor as you press up."
- Advanced: "Optimize arch for your leverages. Work on weak points."

---

## 10. Dynamic Warm-Up Calculator (IMPLEMENTED)

### How It Works
Calculates personalized warm-up sets based on your target working weight.

**Display:**
- Small "🔥 Warm-up (N sets)" link - collapsed by default
- Only shows when target weight > 40kg (meaningful warm-up)
- Expands to show calculated warm-up plan

**Calculation Logic:**
```
Target: 100kg
├── Bar (20kg) × 10 → Mobility & blood flow
├── 50kg × 8       → Groove pattern
├── 70kg × 5       → Neural activation
└── 85kg × 2       → Prime for work sets
```

**Smart Features:**
- Rounds to nearest 2.5kg (plate-friendly)
- Skips redundant sets for lighter weights
- Shows purpose of each warm-up set

---

## 11. Unit System (kg/lbs) (IMPLEMENTED)

### How It Works
Users can toggle between kilograms and pounds throughout the app. All weights are stored internally in kg and converted for display.

**Features:**
- Toggle accessible from profile header (kg ↔ lbs button)
- Conversion happens throughout all displays:
  - Workout logger (Today's entries)
  - Historical sessions
  - Target suggestions
  - Warm-up recommendations
  - Full-screen timer context
  - Level thresholds

**Technical:**
- Internal storage: Always kg
- Display conversion: `kg × 2.20462 = lbs`
- Input conversion: `lbs × 0.453592 = kg` (rounded to nearest kg)

---

## 12. Keep Screen Awake (IMPLEMENTED)

### How It Works
Uses the Screen Wake Lock API to prevent the phone from sleeping during an active workout.

**Behavior:**
- Activates when you expand an exercise card to log a workout
- Releases when you collapse the card or navigate away
- Respects user preference (can be disabled in Timer Settings)

**Settings Location:**
Timer Settings modal → "Keep Screen Awake" toggle

**Browser Support:**
Works on modern browsers (Chrome, Edge, Safari iOS 16.4+). Falls back gracefully on unsupported browsers.

---

## Suggested Additional Coaching Features

### 1. Streak Tracking
**What:** Track consecutive workout days/weeks
**Implementation:** Simple counter in localStorage
**UI:** Badge showing current streak, celebration at milestones (7 days, 30 days)

### 2. Volume Tracking
**What:** Track total volume (sets × reps × weight) per session
**Coaching:** Alert when volume drops significantly (potential overtraining indicator)
**Display:** Weekly volume graph on progress page

### 3. Rest Day Reminder
**What:** Suggest rest day after 3-4 consecutive training days
**Implementation:** Check last 4 workout dates
**Display:** Gentle notification: "Consider a rest day for optimal recovery"

### 4. Plateau Detection
**What:** Detect when user hasn't progressed in 3+ weeks
**Coaching:** Suggest deload week, form check, or program change
**Example:** "You've been at 80kg for 3 weeks. Try a deload week?"

### 7. Nutrition Reminders
**What:** Post-workout reminder about protein intake
**Timing:** Show 30 mins after workout completion
**Message:** "Great workout! Don't forget your protein within 2 hours."

### 8. Weekly Summary
**What:** End-of-week summary notification
**Content:**
- Total workouts
- New PRs hit
- Level changes
- Exercises trained
**Display:** Sunday evening notification

### 9. Fatigue Indicator
**What:** Track if performance is declining (3+ sessions with lower weights)
**Coaching:** "Your performance has dipped. Consider recovery focus."
**Visual:** Small fatigue meter on profile

### 10. Goal Setting
**What:** Let user set target weights for specific exercises
**Tracking:** Show progress toward goal
**Celebration:** Special animation when goal achieved

---

## Implementation Priority

### Quick Wins (1-2 hours each)
1. Streak tracking
2. Form tips (static data)
3. Rest day reminder
4. Warm-up recommendations

### Medium Effort (4-8 hours each)
5. Weekly summary
6. Volume tracking
7. Plateau detection
8. Fatigue indicator

### Larger Features (1-2 days)
9. Goal setting
10. Nutrition reminders

---

## Current Coaching Flow Summary

```
User opens exercise card →
  ↓
Wake Lock activated (if Keep Screen Awake enabled) →
  ↓
Check for level downgrade (4-workout lookback) →
  ↓ (if downgrade needed)
  Show notification, update level
  ↓
Display past sessions + TODAY + TARGET columns (in current unit) →
  ↓
Show warm-up suggestion (if target > 40kg) →
  ↓
User logs weight (input converted from display unit to kg) →
  ↓
Check for PR → Show celebration if new record →
  ↓
Check for level upgrade → Show notification if level up →
  ↓
User taps ✓ "Done" button →
  ↓
Open full-screen timer (shows set, weight, reps context) →
  ↓
Timer completes → Sound + vibration alert →
  ↓
Save session to localStorage →
  ↓
User collapses card → Wake Lock released
```

This comprehensive coaching system provides automated guidance without overwhelming the user, focusing on progressive overload and balanced training.

---

## Feature Implementation Status

| # | Feature | Status |
|---|---------|--------|
| 1 | Automatic Level System | ✅ Implemented |
| 2 | Smart Weight/Rep Progression | ✅ Implemented |
| 3 | PR Detection & Celebration | ✅ Implemented |
| 4 | AI Coach Tips | ✅ Implemented |
| 5 | Motivational Quotes | ✅ Implemented |
| 6 | Rest Timer with Memory | ✅ Implemented |
| 7 | Body Part Balance Analysis | ✅ Implemented |
| 8 | Achievement Badges | ✅ Implemented |
| 9 | Form Tips | ✅ Implemented |
| 10 | Dynamic Warm-Up Calculator | ✅ Implemented |
| 11 | Unit System (kg/lbs) | ✅ Implemented |
| 12 | Keep Screen Awake | ✅ Implemented |
