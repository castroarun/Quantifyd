# Castronix Portfolio - Project Status

> **Project:** Castronix Portfolio Website
> **Purpose:** Showcase projects, workflow, and engineering practices

**Last Updated:** 2025-12-28

---

## Quick Summary

**Goal:** Create a best-in-class portfolio website to showcase:
- REPPIT, NoteApp, PRIMMO and future projects
- 9-Step development workflow
- Engineering practices and principles

**Current Phase:** Design & Mockup

---

## Mockups

| Version | File | Description |
|---------|------|-------------|
| V1 | [v1-basic.html](mockups/v1-basic.html) | Initial Augen-style design |
| V2 | [v2-premium.html](mockups/v2-premium.html) | Best-in-class Linear-style design |
| V5 | [v5-combined.html](mockups/v5-combined.html) | Combined features, project cards |
| V6 | [v6-combined.html](mockups/v6-combined.html) | Carousel showcase, spotlight effect, mobile menu |
| V7 | [v7-principles.html](mockups/v7-principles.html) | Principles showcase experiments (5 approaches) |
| V8 | [v8-brittany.html](mockups/v8-brittany.html) | **LATEST** - Brittany Chiang style, alternating projects, vertical email |

### V8 Features (Latest - Brittany + V6/V7 Combined)
**Layout (Brittany Chiang style):**
- Alternating featured project layout (image left/right)
- Fixed vertical email on right side (rotated 90°)
- Fixed social links on left side
- Numbered section headings with extending line (01., 02., etc.)
- Mobile slide-in menu with backdrop

**Theme & Effects (V6 style):**
- V6 dark foundation (#0A0A0F) with cyan accent (#22D3EE)
- Cursor glow spotlight effect (follows mouse)
- Grid background pattern
- Gradient orbs (top-right, bottom-left)
- Inter Tight font family

**Sections (V7 style):**
- Engineering Principles section with contrast cards
- "Instead of / We do" pattern from V7
- Success/Error color scheme for visual contrast
- Link to V7 for full principles showcase

**Other Features:**
- Grayscale-to-color image hover effect
- Project cards with folder icon
- Fade-up scroll animations
- Accessibility-focused focus states (dashed cyan outline)

### V6 Features
- Project carousel with horizontal scroll
- Card popup effect (15% scale) with spotlight dimming
- Scroll indicator with pulse animation
- Mobile hamburger menu with animated X transition
- Split-view project showcase
- Waitlist modal with domain chips
- Section labels: "Featured Work", "All Projects"

### V7 Principles Showcase (Experimental)
Five creative approaches to present engineering principles:
1. **Contrast Cards** - "Instead of / We do" side-by-side comparison
2. **Visual Equations** - Icon math: Problem + Principle = Solution
3. **Real Examples** - Actual decisions from REPPIT, NoteApp, PRIMMO
4. **Flip + Contrast Cards** - Combined: flip to reveal "Instead of / We do" (recommended)
5. **One Metric** - Each principle gets one impactful number

### V2 Features (Premium Version)
- Cursor glow spotlight effect
- Glassmorphism cards with backdrop blur
- Scroll-triggered fade animations
- Animated stat counters
- Interactive project mockups
- Gradient orbs and grid background
- Easter egg interaction
- Linear-style thin borders and accents

---

## Design Inspiration Sources

Based on research from:
- [Awwwards Portfolio Winners](https://www.awwwards.com/websites/portfolio/)
- [Muzli Top 100 Portfolios 2025](https://muz.li/blog/top-100-most-creative-and-unique-portfolio-websites-of-2025/)
- [Brittany Chiang](https://brittanychiang.com) - Cursor glow, accessibility focus
- [Linear.app](https://linear.app) - Dark theme, thin borders, Inter font
- [Frontend Horse - Linear Look](https://frontend.horse/articles/the-linear-look/)

---

## Current Project Status (9-Step Workflow)

| Step | Name | Status |
|------|------|--------|
| 1 | DEV-CLOCK | Not Started |
| 2 | PRD & Design | In Progress |
| 3 | Test Cases | Not Started |
| 4 | Build | Not Started |
| 5 | Manual Testing | Not Started |
| 6 | Debug & Feedback | Not Started |
| 7 | Code Walkthrough | Not Started |
| 8 | Ship | Not Started |
| 9 | Retrospective | Not Started |

---

## Next Actions

- [ ] Review V1 vs V2 mockups and choose direction
- [ ] Add actual project screenshots/GIFs
- [ ] Convert to Next.js project
- [ ] Deploy to Vercel
- [ ] Connect custom domain (castronix.dev)

---

## Future Ideas (Backlog)

### Interactive TOGAF Architecture Diagram
Display the 9-step workflow as an interactive architecture diagram (inspired by TOGAF ADM cycle):
- Visual circular/flowchart representation of the workflow
- Each stage is clickable/hoverable
- Shows deliverables at each stage
- Links to actual artifacts (PRD, test cases, code)
- Git commit links for each phase
- Progress indicator showing current project stage

**Implementation notes:**
- Could be SVG-based for crisp rendering at any size
- Consider using D3.js or Framer Motion for animations
- Mobile: could transform to vertical timeline
- Each node could expand to show sub-deliverables

### Other Ideas
- Project timeline showing development history
- Live GitHub contribution graph
- Tech stack visualization with competency levels
- Before/after code comparisons for refactoring stories

---

## Tech Stack (Planned)

- **Framework:** Next.js 14+ with App Router
- **Language:** TypeScript
- **Styling:** Tailwind CSS
- **Animations:** Framer Motion
- **Deployment:** Vercel