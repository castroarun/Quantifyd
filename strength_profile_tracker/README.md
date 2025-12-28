# REPPIT - Track Your Reps. Build Your Strength.

A mobile-first strength tracking app that automatically measures your fitness level and guides your progression. Built with Next.js 16, TypeScript, Tailwind CSS, and Capacitor for Android.

## Features

- **Automatic Level Detection**: Calculates your strength level (Beginner, Novice, Intermediate, Advanced) for each exercise based on performance relative to body weight
- **Smart Progression System**: Personalized weight and rep targets for every session with PROGRESS/MAINTAIN logic
- **PR Tracking & Celebrations**: Automatic personal record detection with celebratory animations
- **Intelligent Rest Timer**: Full-screen countdown with customizable presets (2:30-5:00), per-exercise memory, sound/vibration alerts
- **Muscle Heatmap**: Visual progress tracking showing strength balance across body parts
- **Multi-Profile Support**: Create up to 5 profiles for family members or training partners
- **Dark Mode**: Easy on the eyes during early morning or late night sessions
- **Unit Flexibility**: Switch between kg and lbs anytime with automatic conversions

## Tech Stack

- **Framework**: Next.js 16.0.7 (App Router with Turbopack)
- **Language**: TypeScript 5.x
- **Styling**: Tailwind CSS 4.x
- **Mobile**: Capacitor 8.x for Android
- **Storage**: localStorage (PWA-ready, privacy-first)
- **Database**: Supabase (PostgreSQL) - optional cloud sync
- **Visualization**: react-body-highlighter for muscle heatmap

## Project Structure

```
strength_profile_tracker/
├── src/
│   ├── app/                    # Next.js app router
│   │   ├── page.tsx           # Home page
│   │   ├── layout.tsx         # Root layout
│   │   └── timer/             # Rest timer page
│   ├── components/            # React components
│   │   ├── strength/          # Workout logger, exercise list
│   │   ├── profile/           # Profile management
│   │   └── ui/                # Shared UI components
│   ├── contexts/              # React context providers
│   ├── data/                  # Exercise data, quotes
│   ├── hooks/                 # Custom React hooks
│   ├── lib/                   # Utilities, storage logic
│   └── types/                 # TypeScript type definitions
├── android/                   # Capacitor Android project
├── docs/                      # Documentation
│   ├── APP_PRD.md            # Product requirements
│   ├── PROJECT-STATUS.md     # Development status
│   ├── DEV-CLOCK.md          # Time tracking
│   └── PLAY_STORE_LISTING.md # Store listing content
└── public/                    # Static assets
```

## Getting Started

### Prerequisites

- Node.js 18+ and npm
- Android Studio (for mobile builds)

### 1. Clone and Install

```bash
git clone <your-repo-url>
cd strength_profile_tracker
npm install
```

### 2. Configure Environment Variables

Create `.env.local` with your credentials (optional for cloud sync):

```env
# Supabase (optional)
NEXT_PUBLIC_SUPABASE_URL=your_supabase_project_url
NEXT_PUBLIC_SUPABASE_ANON_KEY=your_supabase_anon_key

# Application
NEXT_PUBLIC_APP_URL=http://localhost:3000
```

### 3. Run Development Server

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

### 4. Build for Android

```bash
npm run build
npx cap sync android
npx cap open android
```

Then build APK/AAB from Android Studio.

## Exercise Coverage

23 exercises across 5 body parts:

| Body Part | Exercises |
|-----------|-----------|
| **Chest** | Bench Press, Incline Press, Dumbbell Press, Cable Fly |
| **Back** | Deadlift, Barbell Row, Lat Pulldown, Pull-ups, Cable Row |
| **Shoulders** | Overhead Press, Lateral Raises, Front Raise |
| **Legs** | Squat, Leg Press, Romanian Deadlift, Leg Curl |
| **Arms** | Bicep Curl, Tricep Pushdown, Hammer Curl |

## Core Features

### Workout Logger

- Log weight x reps x sets for each exercise
- View last 2 sessions + today's entry
- Smart TARGET column with progression suggestions
- One-tap copy from target to today's entry

### Level System

| Level | Description |
|-------|-------------|
| Beginner | Just starting out |
| Novice | Some gym experience |
| Intermediate | Consistent training |
| Advanced | Elite performance |

Levels auto-update when you hit PRs that exceed the next threshold.

### Rest Timer

- Full-screen countdown display
- Customizable presets per exercise
- Sound and vibration alerts
- Keep screen awake option
- Set context display

## Development

### Build for Production

```bash
npm run build
npm run start
```

### Type Checking

```bash
npx tsc --noEmit
```

### Linting

```bash
npm run lint
```

## Privacy

All data is stored locally on your device. No account required, no data collection. Your workout data never leaves your phone.

## Future Enhancements (Backlog)

- Cloud sync with Supabase
- Workout templates
- Social sharing of PRs
- Apple Watch companion app
- Barcode scanner for gym equipment

## Acknowledgments

- Built with [Next.js](https://nextjs.org/)
- UI with [Tailwind CSS](https://tailwindcss.com/)
- Mobile with [Capacitor](https://capacitorjs.com/)
- Muscle visualization by [react-body-highlighter](https://www.npmjs.com/package/react-body-highlighter)

---

**Version**: 1.0.2
**Last Updated**: 2025-12-28
**Play Store**: Coming Soon (Closed Testing)
