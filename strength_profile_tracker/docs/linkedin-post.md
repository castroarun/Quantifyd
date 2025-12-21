# LinkedIn Posts - REPPIT Launch

**Reference:** `_claude-shared/linkedin-instructions.md`
**Project:** REPPIT - Strength Profile Tracker
**Platform:** Web (Vercel) + Android (Play Store)
**Last Updated:** 2025-12-20

---

## Project Differentiation

| Aspect | Common Workout Apps | REPPIT |
|--------|---------------------|--------|
| Tracking | Log sets/reps/weight | Log + see WHERE you stand (Beginner → Advanced) |
| Progress | Weight lifted over time | Strength SCORE + level progression |
| Targets | Generic goals | Exercise-specific standards based on body weight |
| Nutrition | None or separate app | BMI + calorie advisor (cut/maintain/bulk) |
| Multi-user | One account per person | Track family or gym partners (up to 5 profiles) |
| Rest | Basic stopwatch | Smart rest timer with per-exercise memory |
| Visualization | Charts | Muscle heatmap showing strong/weak areas |
| Coaching | None | Warmup tips, form cues, today's targets |
| Guidance | None or paid coaches | AI coach tips based on your ratings |

**Core differentiator:** Most workout apps tell you WHAT you lifted. REPPIT tells you WHERE you stand and WHAT to go for.

---

## Stack

- **Frontend:** Next.js 14, TypeScript, Tailwind CSS
- **Mobile:** Capacitor for Android native (WebView)
- **Storage:** localStorage (offline-first)
- **Deployment:** Vercel (web), Google Play (Android)
- **AI Assist:** Claude Code (Opus 4.5)

---

## Post 1/4 - The Ship

```
𝗦𝗵𝗶𝗽𝗽𝗲𝗱 𝗥𝗘𝗣𝗣𝗜𝗧 - A 𝘀𝘁𝗿𝗲𝗻𝗴𝘁𝗵 𝘁𝗿𝗮𝗰𝗸𝗲𝗿 app 𝗯𝘂𝗶𝗹𝘁 𝗶𝗻 𝟯 𝘄𝗲𝗲𝗸𝘀 (1/4)

𝗧𝗵𝗲 𝗽𝗿𝗼𝗯𝗹𝗲𝗺:
Lifters track sets, reps, and weight - but struggle to answer: "Am I intermediate or advanced at bench press? What should I be lifting at my body weight or setting my tarege at?"

REPPIT addressed the above. Most workout apps tell you WHAT you did. REPPIT tells you WHERE you stand and WHAT to go for.

Built from my own training experience — features I wanted but couldn't find in one app.

𝗗𝗲𝘀𝗶𝗴𝗻 & 𝗲𝘅𝗲𝗰𝘂𝘁𝗶𝗼𝗻 𝗴𝘂𝗶𝗱𝗲𝗱 𝗯𝘆:
• "Hours of planning saves weeks of coding"
• Fix it early at 1x. Fix it late at 100x. (Boehm)
→ Invested upfront in design. Execution was butter smooth.

𝗪𝗵𝗮𝘁 𝗺𝗮𝗸𝗲𝘀 𝗥𝗘𝗣𝗣𝗜𝗧 𝗱𝗶𝗳𝗳𝗲𝗿𝗲𝗻𝘁:
• Auto Strength levels for 23 core exercises gender wise (Beginner → Novice → Intermediate → Advanced)
• Target weights based on YOUR body weight
• Today's targets based on past workouts
• BMI calculator with healthy weight targets
• Calorie advisor: cut, maintain, or bulk - with exact daily numbers
• Auto-level updates when you hit new PRs
• Track family or gym partners (up to 5 profiles)
• Warmup tips and form cues per exercise
• AI coach tips based on your ratings

→ Feels like a personal trainer in your pocket - contextual guidance without the cost.

𝗢𝘁𝗵𝗲𝗿 𝗰𝗿𝗶𝘁𝗶𝗰𝗮𝗹 𝗳𝗲𝗮𝘁𝘂𝗿𝗲𝘀:
• PR tracker with celebration messages
• Smart rest timer with per-exercise memory
• Strength score history chart
• Personal records board (top 3 PRs with medals)
• Body part balance radar chart
• Workout frequency heatmap
• Individual exercise progression charts

𝗨𝗫 𝗱𝗲𝘁𝗮𝗶𝗹𝘀 𝘁𝗵𝗮𝘁 𝗺𝗮𝘁𝘁𝗲𝗿:
• Keep screen awake during workouts
• Full screen timer with double-tap to minimize
• Unit toggle (kg/lbs)
• Dark mode
• Offline-first (works without internet)

𝗦𝘁𝗮𝗰𝗸:
Next.js 14 • TypeScript • Capacitor • Tailwind

Built with Claude Code as pair programmer.

Try it → https://reppit-fitness.vercel.app
Play Store → [link when ready]
```

**Screenshot:** Profile detail page showing:
- User avatar with level badge (e.g., "Intermediate")
- Strength score circle (e.g., 64/100)
- Stats grid (age, height, weight, BMI)

---

## Post 2/4 - The Workflow

```
𝗔𝗜-𝗮𝘀𝘀𝗶𝘀𝘁𝗲𝗱 𝗱𝗲𝘃 𝘄𝗼𝗿𝗸𝗳𝗹𝗼𝘄 𝘁𝗵𝗮𝘁 𝗮𝗰𝘁𝘂𝗮𝗹𝗹𝘆 𝘄𝗼𝗿𝗸𝘀 (2/4)

𝗕𝘂𝗶𝗹𝘁 𝗺𝗼𝘀𝘁𝗹𝘆 𝗼𝗻 𝘁𝗵𝗲 𝗴𝗼:
80% of this project was built remotely - via Claude Code on web, execution on cloud. Pulled to local only for the last 20%: final packaging and Play Store prep.

𝗟𝗲𝗮𝗿𝗻𝗶𝗻𝗴𝘀 𝗳𝗿𝗼𝗺 𝗹𝗮𝘀𝘁 𝗽𝗿𝗼𝗷𝗲𝗰𝘁:
Previous project = firefighting in end phases. Unplanned debugging, scope creep, time overruns. The retro was clear: invest more upfront in design.

𝗖𝗼𝗿𝗲 𝗽𝗿𝗶𝗻𝗰𝗶𝗽𝗹𝗲𝘀 𝘄𝗲 𝗳𝗼𝗹𝗹𝗼𝘄𝗲𝗱:
• Slow down to speed up — "Hours of planning saves weeks of coding"
• Fix it early at 1x. Fix it late at 100x. (Boehm)
• "Voluminous documentation is part of the problem, not part of the solution."
→ Good design doesn't just prevent bugs—it makes execution inevitable.

𝗪𝗵𝗮𝘁 𝘄𝗲 𝗱𝗶𝗱 𝗱𝗶𝗳𝗳𝗲𝗿𝗲𝗻𝘁𝗹𝘆:

𝗦𝗽𝗲𝗰𝗶𝗮𝗹𝗶𝘇𝗲𝗱 𝗮𝗴𝗲𝗻𝘁𝘀 𝘄𝗶𝘁𝗵 𝗵𝗮𝗻𝗱𝘀𝗵𝗮𝗸𝗲𝘀:
• @designer → Deep research, HTML mockups, design principles:
  ◦ Mobile-first (start small, scale up)
  ◦ Rule of Three (abstract on 3rd occurrence, not before)
  ◦ Separation of concerns (UI vs logic vs data)
  ◦ Idempotent ops (safe to retry offline syncs)
• @architect → PRD with data models, ASCII UI layouts, delivery gates
• @qa → Test cases before code, validation at cheapest phase

𝗣𝗿𝗼𝗷𝗲𝗰𝘁 𝗰𝗼𝗺𝗺𝗮𝗻𝗱𝘀:
• /newproject → Scaffold with workflow templates
• /checkprd → Jira PRD review loop
• /deploy → One-command production push

𝗝𝗶𝗿𝗮 𝗶𝗻𝘁𝗲𝗴𝗿𝗮𝘁𝗶𝗼𝗻:
• Uncluttered workspace - todos in Jira, not in code
• Reviews in independent space - clear handoffs
• Human touchpoints enforced at gates

𝗧𝗵𝗲 𝗿𝗲𝘀𝘂𝗹𝘁:
Execution was butter smooth. Zero firefighting. Design upfront → build fast → ship clean.

𝗥𝗲𝘁𝗿𝗼 → 𝗟𝗲𝗮𝗿𝗻 → 𝗔𝗱𝗮𝗽𝘁. That's the cycle.
```

**Screenshot:** None (text-focused) OR ecosystem diagram

---

## Post 3/4 - The Features

```
𝗙𝗲𝗮𝘁𝘂𝗿𝗲𝘀 𝘁𝗵𝗮𝘁 𝗺𝗮𝗸𝗲 𝗥𝗘𝗣𝗣𝗜𝗧 𝗱𝗶𝗳𝗳𝗲𝗿𝗲𝗻𝘁 (3/4)

𝗦𝘁𝗿𝗲𝗻𝗴𝘁𝗵 𝗦𝘁𝗮𝗻𝗱𝗮𝗿𝗱𝘀
"80kg bench at 70kg body weight = Intermediate level"
→ Based on established strength standards, not arbitrary numbers

𝗦𝗺𝗮𝗿𝘁 𝗥𝗲𝘀𝘁 𝗧𝗶𝗺𝗲𝗿
• Per-exercise memory (remembers your preferred rest for each lift)
• Quick presets: 30s, 1:00, 1:30, 2:00, 3:00
• +15s/-15s adjustments
• Sound + vibration alerts
• Keep screen awake option
• Double-tap to minimize

𝗣𝗥 𝗧𝗿𝗮𝗰𝗸𝗶𝗻𝗴
• Log sets, app tracks PRs automatically
• Celebration messages when you hit new PRs
• Auto-level updates based on performance
• Today's targets based on past workouts

𝗪𝗼𝗿𝗸𝗼𝘂𝘁 𝗖𝗼𝗮𝗰𝗵𝗶𝗻𝗴
• Warmup suggestions per exercise
• Form tips and cues while you lift
• Target weights shown for next level
→ Like having a trainer guide you through each set

𝗕𝗠𝗜 & 𝗖𝗮𝗹𝗼𝗿𝗶𝗲 𝗔𝗱𝘃𝗶𝘀𝗼𝗿
• BMI calculation with healthy range targets
• Personalized calorie recommendations
• Goal-based: cut (-500), maintain, or bulk (+300)
• Shows exactly how much to eat daily

𝗠𝘂𝘀𝗰𝗹𝗲 𝗛𝗲𝗮𝘁𝗺𝗮𝗽
Visual body showing strong (green) vs weak (red) areas
→ See imbalances at a glance

𝗔𝗜 𝗖𝗼𝗮𝗰𝗵 𝗧𝗶𝗽𝘀
"Your bench is stronger than squat - that's common!"
→ Contextual tips based on YOUR ratings
```

**Screenshot:** Workout logger showing:
- Exercise card with level badges
- Rest timer with quick presets
- Today's sets input

---

## Post 4/4 - The Reflection

```
𝗪𝗵𝗮𝘁 𝘄𝗼𝗿𝗸𝗲𝗱 / 𝗪𝗵𝗮𝘁 𝗯𝗿𝗼𝗸𝗲 (4/4)

𝗪𝗵𝗮𝘁 𝘄𝗼𝗿𝗸𝗲𝗱:
• PRD with ASCII UI mockups → AI built exactly what I envisioned
• HTML mockups before React → faster iteration, no wasted code
• Per-exercise timer memory → small detail, big UX win
• Capacitor web-to-Android → 30 minutes, updates bypass store review

𝗪𝗵𝗮𝘁 𝗯𝗿𝗼𝗸𝗲:
• Double-click issues on mobile → had to add specific handling
• Unit conversion bugs → weights stored in kg, displayed in user's unit
• Timer auto-start → needed careful state management
• Dark mode + SVG heatmaps → 2 hours debugging fill colors

𝗪𝗵𝗮𝘁 𝘀𝘁𝗶𝗹𝗹 𝗻𝗲𝗲𝗱𝗲𝗱 𝗺𝗲:
• Design decisions (what features, what UX)
• Testing on actual device
• Understanding the code (not just accepting it)
• Knowing when to revert (simpler is often better)

𝗧𝗵𝗲 𝗿𝗲𝗮𝗹 𝗹𝗲𝘀𝘀𝗼𝗻:
AI accelerates implementation. But design taste, testing discipline, and understanding your code - that's still on you.

𝗟𝗶𝗻𝗸𝘀:
• Web app → https://reppit-fitness.vercel.app
• Play Store → [link when ready]
• GitHub → https://github.com/castroarun/strength_profile_tracker

What's your process for knowing if you're actually getting stronger?
```

**Screenshot:** Timer running with:
- Countdown display
- +15s/-15s buttons
- Quick preset buttons
- Minimize option

---

## Screenshots to Capture

| Post | Screen | Key Elements |
|------|--------|--------------|
| 1/4 | Profile detail | Avatar, level badge, strength score, stats |
| 2/4 | Timer mockup | docs/mockups/timer-options.html |
| 3/4 | Workout logger | Exercise card with timer and set inputs |
| 4/4 | Timer running | Full-screen countdown with controls |

---

## Hashtags

```
#buildinpublic #nextjs #typescript #capacitor #mobiledev #fitness #strengthtraining #claudecode #ai
```

---

## Posting Schedule

| Day | Post | Focus |
|-----|------|-------|
| Day 1 | 1/4 - The Ship | What REPPIT is, differentiators |
| Day 3 | 2/4 - The Workflow | 9-step process, Claude Code setup |
| Day 5 | 3/4 - The Features | Rest timer, PR tracking, heatmap |
| Day 7 | 4/4 - The Reflection | What worked, what broke, links |
