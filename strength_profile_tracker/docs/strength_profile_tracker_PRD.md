# Product Requirements Document: Strength Profile Tracker

**Document Version:** 1.0
**Date:** November 4, 2025
**Status:** Draft

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Product Goals & Objectives](#2-product-goals--objectives)
3. [Functional Requirements](#3-functional-requirements)
4. [Design Specifications](#4-design-specifications)
5. [Technical Architecture](#5-technical-architecture)
6. [Development Phases](#6-development-phases)
7. [Success Criteria](#7-success-criteria)
8. [Appendices](#8-appendices)

---

## 1. Executive Summary

### 1.1 Product Overview

Strength Profile Tracker is a mobile application that helps users track their weightlifting progress by storing multiple user profiles and displaying personalized strength standards for key compound exercises. The app calculates ideal lifting weights based on user body metrics and allows users to track their current performance level.

### 1.2 Target Audience

- Fitness enthusiasts who engage in strength training
- Personal trainers managing multiple clients
- Gym-goers tracking progression across different training phases
- Families sharing fitness goals

### 1.3 Problem Statement

Users lack a simple, personalized way to:
- Track strength standards for multiple people (self, clients, family)
- Understand where they stand relative to their body weight
- Monitor progression from beginner to advanced levels
- Set realistic lifting goals based on scientifically-backed standards

### 1.4 Solution

A mobile application that stores up to 5 user profiles, calculates personalized strength standards based on body metrics, and allows users to track their current lifting performance across four difficulty levels.

---

## 2. Product Goals & Objectives

### 2.1 Primary Goals

1. Enable users to create and store up to 5 distinct lifting profiles
2. Calculate and display personalized strength standards for major compound lifts
3. Allow users to track and update their current performance levels
4. Provide clear progression pathways from beginner to advanced

### 2.2 Success Metrics

| Metric | Target |
|--------|--------|
| User engagement | 70% of users create at least one profile within first session |
| Retention | 60% of users return to update their levels within 30 days |
| Usability | 90% of users can complete profile creation in under 2 minutes |
| Accuracy | Strength calculations align with established standards (StrengthLevel.com reference) |

### 2.3 Out of Scope (Phase 1)

- Workout planning or logging features
- Social sharing capabilities
- Progress charts and historical tracking
- Exercise form videos or tutorials
- Integration with fitness wearables
- Cloud backup/sync across devices

---

## 3. Functional Requirements

### 3.1 Profile Management

#### 3.1.1 Create Profile

**Priority:** P0 (Must Have)

**Requirements:**
- System shall allow users to create up to 5 profiles
- Each profile requires:
  - Profile Name (text, max 50 characters)
  - Age (numeric, 13-100 years)
  - Height (numeric, cm, 100-250 cm)
  - Weight (numeric, kg, 30-300 kg)
- System shall validate all inputs before saving
- System shall prevent creation of 6th profile
- Profiles shall be stored locally on device

#### 3.1.2 View Profiles

**Priority:** P0 (Must Have)

**Requirements:**
- System shall display all profiles on home screen
- Each profile card shows: Name, Age, Height, Weight
- Empty profile slots shall be visually indicated
- Profile counter shall display "X of 5 profiles"

### 3.2 Strength Standards Calculation

#### 3.2.1 Exercise Coverage

**Priority:** P0 (Must Have)

**Exercises to Include:**
1. Bench Press
2. Squat
3. Deadlift
4. Shoulder Press (Overhead Press)

#### 3.2.2 Difficulty Levels

**Priority:** P0 (Must Have)

**Four Standard Levels:**

| Level | Description |
|-------|-------------|
| Beginner | Just starting strength training |
| Novice | 6-12 months of training |
| Intermediate | 1-2 years of consistent training |
| Advanced | 2+ years of serious training |

#### 3.2.3 Calculation Method - Body Weight Multipliers

| Exercise | Beginner | Novice | Intermediate | Advanced |
|----------|----------|--------|--------------|----------|
| Bench Press | 0.59 × BW | 0.82 × BW | 1.06 × BW | 1.29 × BW |
| Squat | 0.76 × BW | 1.12 × BW | 1.47 × BW | 1.88 × BW |
| Deadlift | 0.94 × BW | 1.29 × BW | 1.71 × BW | 2.18 × BW |
| Shoulder Press | 0.41 × BW | 0.59 × BW | 0.76 × BW | 1.00 × BW |

> **Note:** BW = Body Weight in kg

### 3.3 User Level Tracking

#### 3.3.1 Select Current Level

**Priority:** P0 (Must Have)

**Requirements:**
- Users can select one level per exercise
- Selected level shall be marked with checkmark
- Selected level shall have highlighted border
- System shall save selections immediately
- Each exercise can have different selected level

#### 3.3.2 Persistent Storage

**Priority:** P0 (Must Have)

**Requirements:**
- Selected levels shall persist across app sessions
- Selections stored locally per profile
- System shall display saved selections by default on profile view
- No internet required for data persistence

---

## 4. Design Specifications

### 4.1 Color Palette

**Primary Colors:**

| Name | Hex | Usage |
|------|-----|-------|
| Primary | `#2C3E50` | Dark blue-grey - Headers, primary text |
| Secondary | `#3498DB` | Bright blue - Buttons, accents |
| Success | `#27AE60` | Green - Save actions |

**Level Colors:**

| Level | Hex | Color |
|-------|-----|-------|
| Beginner | `#2ECC71` | Green |
| Novice | `#3498DB` | Blue |
| Intermediate | `#F39C12` | Orange |
| Advanced | `#E74C3C` | Coral red |

### 4.2 Typography

**Font Family:** System Default
- iOS: San Francisco
- Android: Roboto

**Type Scale:**

| Element | Size | Weight |
|---------|------|--------|
| Screen Title | 13pt | 600 |
| Section Header | 10pt | 600 |
| Body Text | 9pt | normal |
| Button Text | 11pt | 600 |

### 4.3 UI Components

**Buttons:**
- Primary: Solid fill, white text, 8pt corner radius
- Height: 44-48pt (touch-friendly)

**Cards:**
- Background: White or light grey
- Border: 1pt solid `#E0E0E0`
- Corner radius: 8pt
- Shadow: None (flat design)

---

## 5. Technical Architecture

### 5.1 Technology Stack

| Layer | Recommended Technology |
|-------|------------------------|
| Framework | React Native (cross-platform iOS + Android) |
| Data Storage | SQLite or AsyncStorage (local persistence) |
| State Management | Redux Toolkit or Context API |

### 5.2 Data Models

**Profile Schema:**

```typescript
interface Profile {
  id: string                    // UUID
  name: string                  // max 50 characters
  age: number                   // 13-100
  height: number                // 100-250 cm
  weight: number                // 30-300 kg
  sex?: 'male' | 'female'       // For gender-specific standards & BMR (Phase 4)
  dailySteps?: number           // Average daily steps, 0-50000 (Phase 4)
  activityLevel?: ActivityLevel // General activity level (Phase 4)
  goal?: Goal                   // Fitness/nutrition goal (Phase 4)
  exerciseRatings: ExerciseRatings  // User's level per exercise
  createdAt: Date
  updatedAt: Date
}

type Level = 'beginner' | 'novice' | 'intermediate' | 'advanced'
type Sex = 'male' | 'female'
type ActivityLevel = 'sedentary' | 'light' | 'moderate' | 'active' | 'very_active'
type Goal = 'lose' | 'maintain' | 'gain'
```

**Exercise Schema (with Gender-Specific Multipliers - Phase 4):**

```typescript
interface ExerciseInfo {
  id: Exercise
  name: string
  bodyPart: BodyPart
  multipliers: {              // Male multipliers (default)
    beginner: number
    novice: number
    intermediate: number
    advanced: number
  }
  femaleMultipliers?: {       // Female multipliers (Phase 4)
    beginner: number
    novice: number
    intermediate: number
    advanced: number
  }
  isDumbbell?: boolean        // Per-hand weight indicator
}
```

---

## 6. Development Phases

### Phase 1: MVP (4-6 weeks)

**Scope:**
- [x] Create, view, delete profiles (max 5)
- [x] Calculate strength standards for 4 exercises
- [x] Display 4 difficulty levels per exercise
- [x] Select and save current performance levels
- [x] Local data persistence
- [x] Basic UI implementation

### Phase 2: Enhancement (2-3 weeks)

**Scope:**
- [x] Edit existing profiles
- [x] Add 2-3 more exercises
- [x] Improved animations and transitions
- [x] Gender-specific calculations
- [ ] Units toggle (kg/lbs)
- [x] Dark mode support

### Phase 3: Polish & Launch (2 weeks)

**Scope:**
- [x] Comprehensive testing
- [x] Performance optimization
- [x] Accessibility improvements
- [x] Dark mode support
- [x] Progress visualizations (charts)

### Phase 4: Nutrition & Gender Features (Completed)

**Scope:**

#### 4.1 Profile Schema Updates
- [x] Add `sex` field to profile (male/female)
- [x] Add `dailySteps` field (average daily steps)
- [x] Add `activityLevel` field (sedentary/light/moderate/active/very active)
- [x] Add `goal` field (maintain/lose/gain)

#### 4.2 Profile Editing
- [x] Create Edit Profile page
- [x] Allow updating all profile fields (name, age, height, weight, sex, dailySteps)
- [x] Validate all fields before saving
- [x] Navigation from profile detail page to edit

#### 4.3 Gender-Specific Strength Standards
- [x] Add female multipliers to exercise definitions
- [x] Update `calculateStrength()` to accept sex parameter
- [x] Recalculate all strength standards based on profile sex
- [x] Female multipliers typically ~65-70% of male standards

**Female Strength Multipliers (Implemented):**

| Exercise | BEG (F) | NOV (F) | INT (F) | ADV (F) |
|----------|---------|---------|---------|---------|
| Bench Press | 0.35 | 0.50 | 0.65 | 0.85 |
| Squat | 0.53 | 0.78 | 1.03 | 1.32 |
| Deadlift | 0.66 | 0.90 | 1.20 | 1.53 |
| Shoulder Press | 0.29 | 0.41 | 0.53 | 0.70 |

> **Note:** Female multipliers are approximately 60-70% of male multipliers, based on physiological differences and strength standards from StrengthLevel.com

#### 4.4 Calorie Calculations (Detailed)

**Step 1: Calculate BMR (Basal Metabolic Rate)**

Using the Mifflin-St Jeor Equation (most accurate):

```
Male:   BMR = (10 × weight in kg) + (6.25 × height in cm) - (5 × age) + 5
Female: BMR = (10 × weight in kg) + (6.25 × height in cm) - (5 × age) - 161
```

**Step 2: Calculate Activity Factor**

| Activity Level | Description | Multiplier |
|----------------|-------------|------------|
| Sedentary | Little to no exercise, desk job | 1.2 |
| Light | Light exercise 1-3 days/week | 1.375 |
| Moderate | Moderate exercise 3-5 days/week | 1.55 |
| Active | Hard exercise 6-7 days/week | 1.725 |
| Very Active | Very hard exercise, physical job | 1.9 |

**Step 3: Daily Steps Adjustment**

Additional calories burned from walking:
```
Steps Calories = (steps × 0.04)  // Approx 40 calories per 1000 steps
```

**Step 4: Calculate TDEE (Total Daily Energy Expenditure)**

```
TDEE = (BMR × Activity Multiplier) + Steps Calories
```

**Step 5: Goal-Based Recommendations**

| Goal | Calorie Adjustment |
|------|-------------------|
| Lose Weight | TDEE - 500 (moderate deficit) |
| Maintain | TDEE |
| Gain Weight | TDEE + 300 (lean bulk) |

**Step 6: Body Composition Analysis**

```typescript
// BMI Categories
const BMI = weight / (height/100)²

Underweight: BMI < 18.5  → Recommend: Gain weight, calorie surplus
Normal:      BMI 18.5-24.9 → Recommend: Based on goal
Overweight:  BMI 25-29.9 → Suggest: Moderate deficit possible
Obese:       BMI ≥ 30    → Suggest: Calorie deficit recommended
```

**Example Calculation:**
```
Profile: Male, 28 years, 178cm, 82kg, Moderate activity, 8000 steps/day

BMR = (10 × 82) + (6.25 × 178) - (5 × 28) + 5
    = 820 + 1112.5 - 140 + 5
    = 1797.5 kcal

TDEE = (1797.5 × 1.55) + (8000 × 0.04)
     = 2786 + 320
     = 3106 kcal/day (maintenance)

If goal = Lose Weight:
    Target = 3106 - 500 = 2606 kcal/day
```

#### 4.5 Nutrition Display UI
- [x] Show maintenance calories on profile page
- [x] Display recommended intake based on goal
- [x] Show BMI category with health indicator
- [x] Provide simple nutrition guidance

**UI Mockup:**
```
┌─────────────────────────────────────────┐
│  📊 Nutrition Summary                    │
├─────────────────────────────────────────┤
│  Maintenance: 3,106 kcal/day            │
│  Your Goal: Lose Weight                  │
│  Target: 2,606 kcal/day                  │
│                                          │
│  BMI: 25.9 (Slightly overweight)        │
│  💡 A 500 calorie deficit = ~0.5kg/week │
└─────────────────────────────────────────┘
```

---

## 7. Success Criteria

### 7.1 Launch Criteria

Before launching, the app must:
- [ ] Pass all functional tests
- [ ] Support primary devices (latest 2 iOS/Android versions)
- [ ] Meet performance benchmarks
- [ ] Have no critical bugs
- [ ] Meet accessibility standards (WCAG 2.1 AA)

### 7.2 Post-Launch Metrics

**Week 1:**
- 100+ downloads
- < 5% crash rate
- Average session length > 3 minutes

**Month 1:**
- 500+ downloads
- 70% profile creation rate
- 4+ star rating
- 60% user retention

**Month 3:**
- 2000+ downloads
- Active feature usage metrics
- Positive user feedback
- Clear product-market fit signals

---

## 8. Appendices

### Appendix A: Reference Sources

- [StrengthLevel.com](https://strengthlevel.com) - Strength standards methodology
- [ExRx.net](https://exrx.net) - Exercise performance standards
- NSCA guidelines - Strength training benchmarks

### Appendix B: Competitive Analysis

**Similar Apps:**

| App | Description | Our Differentiation |
|-----|-------------|---------------------|
| Strong | Full workout tracker | We're simpler, focused on standards only |
| JEFIT | Exercise library + tracking | We have multi-profile support |
| FitNotes | Workout journal | No workout logging complexity |

**Our Key Differentiators:**
- Simpler, focused on standards only
- Multi-profile support (up to 5)
- No workout logging complexity
- Quick reference tool

### Appendix C: Future Enhancements

Post-Phase 3 considerations:
- Progress charts and historical tracking
- Cloud sync across devices
- Export/print functionality
- Additional exercises (pull-ups, rows, etc.)
- One-rep-max calculator
- Training program suggestions
- Social features
- Apple Health / Google Fit integration

---

## Document Approval

| Role | Name | Date |
|------|------|------|
| Product Manager | _________________ | _______ |
| Engineering Lead | _________________ | _______ |
| Design Lead | _________________ | _______ |

---

*— End of Document —*
