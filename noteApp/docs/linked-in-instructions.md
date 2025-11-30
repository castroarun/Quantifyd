# LinkedIn Showcase - Agentic Template

**Purpose:** Reusable framework for showcasing AI-assisted builds on LinkedIn
**Format:** Posts (not articles)
**Last Updated:** 2025-11-30

---

## Core Principles

| Principle | Do | Don't |
|-----------|-----|-------|
| **No philosophy** | Share what you built, what broke, what worked | Pontificate about "the future of AI" or deep thoughts |
| **Workflow-first** | Show the process so others can learn | Just show the final product |
| **Work > Me** | Let the project speak | Make it about personal brand |

---

## Post Structure

```
1. HOOK (1 line)      → Stop the scroll, state what you did
2. CONTEXT (2-3 lines)→ What you built, what stack, why
3. WORKFLOW (bullet)  → The steps you followed (learnable)
4. INSIGHT (1-2 lines)→ One thing that surprised you or broke
5. VISUAL             → Screenshot, GIF, or diagram
6. SOFT CTA           → Invite reflection, not engagement bait
```

---

## Language Guidelines

| Use | Avoid |
|-----|-------|
| "Built X using Y" | "Excited to announce..." |
| "Here's how" | "I believe..." |
| "This broke: [detail]" | "It was a journey" |
| "31 hours total" | "After weeks of hard work" |
| Active voice | Passive voice |
| Specific numbers | Vague timeframes |

**Tone:** Direct, technical, helpful. Like explaining to a colleague.

---

## Visual Requirements

- **Primary:** App screenshot or GIF showing the feature
- **Secondary:** Terminal output, time breakdown chart, architecture diagram
- **Avoid:** Stock images, motivational quotes, selfies

**Available for NoteApp:**
- `C:\Users\Castro\Pictures\claude\NoteApp\01.png`
- `C:\Users\Castro\Pictures\claude\NoteApp\noteApp2.gif`
- `C:\Users\Castro\Pictures\claude\NoteApp\sample01.jfif`
- `C:\Users\Castro\Pictures\claude\NoteApp\Terminal01.jpg`

---

## AI-Assisted Build Workflow (Template)

This is the workflow to reference when showcasing AI collaboration:

```
1. PRD Document       → Design decisions for human approval
2. Test Case Document → Validates AI understood requirements
3. Build              → AI writes, human reviews
4. Manual Testing     → Human verifies functionality
5. Debug & Feedback   → Iterative fixes
6. Code Walkthrough   → Documentation for code familiarity
7. Ship               → Deploy to production
8. Retro: Time Spent  → Analyze where time went
9. Retro: Workflow    → Process improvements (optional)
```

---

## Project-Specific Data: NoteApp

### Stack
- **Frontend:** Next.js 16.0.3 with Turbopack
- **Language:** TypeScript 5.x
- **Styling:** Tailwind CSS 4.x
- **Editor:** TipTap 2.x (ProseMirror-based)
- **Database:** Supabase (PostgreSQL)
- **Auth:** Supabase Auth

### Time Breakdown
| Phase | Hours | % |
|-------|-------|---|
| Design & Planning | 4 | 13% |
| Documentation | 3 | 10% |
| Building | 15 | 48% |
| Debugging | 6 | 19% |
| Testing | 3 | 10% |
| **Total** | **31** | 100% |

### Why This Project
- Step outside Python comfort zone
- Learn modern web stack (Next.js, TypeScript, Tailwind)
- Use AI to accelerate learning, not replace understanding

### App Purpose
Clean, simple note-taking. Start writing immediately without worrying about titles (auto-generated from first line).

---

## NoteApp LinkedIn Post (Thread Format)

Thread-style posts work on LinkedIn - reply to your own post with (1/4), (2/4) etc. Less common than Twitter but effective for workflow content.

---

### (1 of 4)

```
𝗦𝗵𝗶𝗽𝗽𝗲𝗱 𝗺𝘆 𝗳𝗶𝗿𝘀𝘁 𝗔𝗜-𝗲𝗿𝗮 𝘀𝗶𝗱𝗲 𝗽𝗿𝗼𝗷𝗲𝗰𝘁. (1 of 4)

A note-taking app in 32 hours using Claude AI as my pair programmer.

• Tracked every hour in a DEV-CLOCK
• Followed a 9-step workflow I set for Claude AI
• More on that in the replies

Built it to step outside my Python comfort zone and learn modern stacks.

𝗦𝘁𝗮𝗰𝗸:
Next.js 16 • TypeScript • TipTap editor • Supabase • Tailwind CSS

𝗙𝗲𝗮𝘁𝘂𝗿𝗲𝘀 𝗜 𝘄𝗮𝗻𝘁𝗲𝗱:
• Auto-generated titles (from first line)
• Pinning notes
• Templates
• Simple formatting
• Google authenticated login

The result: A clean slate where you start typing immediately.

Try it → https://buildfolio-woad.vercel.app/
```

**Visual:** noteApp2.gif

---

### (2 of 4)

```
𝗠𝘆 𝟵-𝘀𝘁𝗲𝗽 𝘄𝗼𝗿𝗸𝗳𝗹𝗼𝘄 𝘄𝗶𝘁𝗵 𝗖𝗹𝗮𝘂𝗱𝗲 𝗔𝗜 (2 of 4)

Instructions I gave before we started:

𝟭. Start a 𝘁𝗶𝗺𝗲 𝘁𝗿𝗮𝗰𝗸𝗲𝗿 - log hours by phase
   → Knew I may not get it right the first time, so baked in measurement from the start

𝟮. Present 𝗣𝗥𝗗 with design details for my approval
   → My structural inputs: tech stack, minimal file structure, capabilities summary, UI layout
   → AI drafts the formal document with my instructions, I approve or modify

𝟯. Prepare 𝘁𝗲𝘀𝘁 𝗰𝗮𝘀𝗲𝘀
   → Quick hack: my shortcut to validate AI got it right

𝟰. 𝗕𝘂𝗶𝗹𝗱
   → Let AI do its thing, I stay in review mode

𝟱. 𝗠𝗮𝗻𝘂𝗮𝗹 𝘁𝗲𝘀𝘁𝗶𝗻𝗴
   → Catch what AI misses

𝟲. Debug loop with my feedback
   → I describe the issue, AI proposes the fix

𝟳. Code walkthrough documentation
   → So I'm code-familiar, not code-naive

𝟴. 𝗦𝗵𝗶𝗽
   → Done is better than perfect

𝟵. 𝗧𝗶𝗺𝗲 𝗮𝗻𝗮𝗹𝘆𝘀𝗶𝘀 𝗿𝗲𝘁𝗿𝗼
   → Learn where time actually goes
```

**Visual:** None (text-focused)

---

### (3 of 4)

```
𝗧𝗶𝗺𝗲 𝗥𝗲𝘁𝗿𝗼𝘀𝗽𝗲𝗰𝘁𝗶𝘃𝗲 (3 of 4)

Where the 32 hours went:

• 39% building
• 19% debugging (CSS frameworks fighting each other)
• 13% design
• 11% shipping (Vercel setup, deployment, domain config)
• 9% docs
• 9% testing

Building is less than half the effort.

More time on debugging and shipping than expected - probably the learning curve of a first build. The rest - design, testing, docs - that's where intent meets execution.

𝗪𝗵𝗮𝘁 𝘄𝗼𝗿𝗸𝗲𝗱:
AI accelerates boilerplate and debugging.

𝗪𝗵𝗮𝘁 𝘀𝘁𝗶𝗹𝗹 𝗻𝗲𝗲𝗱𝗲𝗱 𝗺𝗲:
Design decisions, testing, understanding why.
```

**Visual:** Time breakdown chart

---

### (4 of 4)

```
𝗥𝗲𝗳𝗹𝗲𝗰𝘁𝗶𝗼𝗻𝘀 (4 of 4)

𝗪𝗵𝗮𝘁 𝘄𝗲𝗻𝘁 𝗿𝗶𝗴𝗵𝘁:
• AI accelerated boilerplate and debugging
• PRD → Test cases caught misalignments early, before code was written
• Time tracking gave real data, not guesses

𝗪𝗵𝗮𝘁 𝘁𝗼 𝗶𝗺𝗽𝗿𝗼𝘃𝗲:
• Reduce debugging time - learn the stack better upfront
• Stick with manual testing for now - fear of knowledge gaps creeping in with automation
• Template the shipping setup for next project

𝗟𝗶𝗻𝗸𝘀:
• Code walkthrough → https://github.com/castroarun/buildfolio/blob/main/noteApp/docs/CODE-WALKTHROUGH.md
• GitHub → https://github.com/castroarun/buildfolio/tree/main/noteApp#readme

What's your workflow when building with AI?
```

**Visual:** Screenshot of code walkthrough

---

## How to Use This Document

**For current project:**
1. Copy the relevant post draft
2. Customize with your screenshots
3. Post

**For future projects:**
1. Update "Project-Specific Data" section with new stack/times
2. Follow the Post Structure template
3. Reference the AI Workflow steps

**Command:** `/linkedin` loads this document

---

## Posting Schedule

| Post | Content | Visual |
|------|---------|--------|
| 1 | The Ship (overview) | noteApp2.gif |
| 2 | The Workflow (9 steps) | Terminal01.jpg |
| 3 | The Numbers (time breakdown) | time chart |

Space posts 3-4 days apart for engagement without fatigue.
