# Quantifyd unified design system

## Scope

This design system applies to **all three apps** in the Quantifyd product family:

1. **Quantifyd** — options selling platform (scanner, strategies, positions, journal)
2. **Trading Cockpit** — 5-tab live monitoring dashboard (Kite Connect, React, SQLite)
3. **Desktop Command Center** — personal productivity hub (Tauri 2.0 + React)

They must feel like one product. Same tokens, same components, same rules.

---

## Philosophy

The aesthetic is inspired by Zoom AI Companion — clean, light, professional, generous whitespace. The design relies on **restraint, not decoration**. No gradients, no shadows, no glow, no glassmorphism. Typography and whitespace do the heavy lifting.

### The eight rules

1. **One canvas color, one surface color.** `#FAFAF9` for the page/canvas, `#FFFFFF` for cards and surfaces. No tertiary backgrounds. Card lift comes from a hairline border, not a fill change.
2. **0.5px hairlines at 10% black.** Never thicker, never darker. `border: 0.5px solid rgba(0,0,0,0.10)`. For softer dividers inside sidebars: `rgba(0,0,0,0.06)`.
3. **Two type weights — 400 and 500.** No 600, no 700, no italics. Hierarchy comes from size and color, not weight.
4. **Sentence case everywhere.** Including buttons, tabs, card titles, nav items. "Range squeeze scan" not "Range Squeeze Scan".
5. **Color is reserved.** Black ink for primary text, muted gray for secondary. Color only earns a place when it carries meaning — green for positive P&L, red for negative, one accent for active nav state. No decorative color.
6. **No shadows, no gradients, no glow.** Ever. If something needs emphasis, it gets a hairline border or a subtle background fill — not a drop shadow.
7. **Sidebar is 200px, icons are 15px, items use 13px text.** Active state is a 4% black fill with no border, no pill, no accent bar. Whisper, don't shout.
8. **Font: Inter (system-ui fallback).** One font family. Two weights. That's it.

---

## Design tokens (CSS custom properties)

```css
:root {
  /* Canvas & surfaces */
  --canvas: #FAFAF9;
  --surface: #FFFFFF;

  /* Ink (text) */
  --ink: #1B1B1A;
  --ink-muted: #888780;
  --ink-faint: #B4B2A9;
  --ink-secondary: #5F5E5A;
  --ink-subtle: #444441;

  /* Borders */
  --hairline: rgba(0, 0, 0, 0.10);
  --hairline-soft: rgba(0, 0, 0, 0.06);
  --hairline-softest: rgba(0, 0, 0, 0.05);

  /* Semantic accent — P&L only */
  --accent-pos: #0F6E56;
  --accent-neg: #A32D2D;

  /* Active nav state */
  --nav-active-bg: rgba(0, 0, 0, 0.04);
  --nav-active-ink: var(--ink);

  /* Status dot (connectivity) */
  --status-connected: #1D9E75;

  /* Pill/chip backgrounds */
  --chip-bg: #F1EFE8;
  --chip-ink: var(--ink-muted);

  /* Avatar (user initial circle) */
  --avatar-bg: #E1F5EE;
  --avatar-ink: #0F6E56;

  /* Radii */
  --radius-control: 6px;
  --radius-card: 10px;
  --radius-input: 12px;

  /* Typography */
  --font-family: 'Inter', system-ui, -apple-system, sans-serif;
  --weight-regular: 400;
  --weight-medium: 500;

  /* Type scale */
  --text-xl: 22px;     /* Page title / hero heading */
  --text-lg: 20px;     /* Metric card numbers */
  --text-md: 14px;     /* Input placeholder, body text */
  --text-sm: 13px;     /* Nav items, card titles, table data */
  --text-xs: 12px;     /* Card descriptions, secondary labels */
  --text-xxs: 11px;    /* Metric labels, chips, status text */

  /* Letter spacing */
  --tracking-tight: -0.02em;  /* Headings, large numbers */
  --tracking-normal: -0.01em; /* Logo, brand text */
}
```

---

## Typography rules

| Element | Size | Weight | Color | Tracking |
|---|---|---|---|---|
| Page title | `--text-xl` (22px) | 500 | `--ink` | `--tracking-tight` |
| Metric value | `--text-lg` (20px) | 500 | `--ink` or `--accent-pos/neg` | `--tracking-tight` |
| Input placeholder | `--text-md` (14px) | 400 | `--ink-faint` | normal |
| Nav item (active) | `--text-sm` (13px) | 500 | `--ink` | normal |
| Nav item (inactive) | `--text-sm` (13px) | 400 | `--ink-subtle` | normal |
| Card title | `--text-sm` (13px) | 500 | `--ink` | normal |
| Card description | `--text-xs` (12px) | 400 | `--ink-muted` | normal |
| Table header | `--text-xxs` (11px) | 400 | `--ink-muted` | normal |
| Table data | `--text-sm` (13px) | 400 | `--ink` | normal |
| Metric label | `--text-xxs` (11px) | 400 | `--ink-muted` | normal |
| Chip/pill text | `--text-xxs` (11px) | 400 | `--ink-muted` | normal |
| Status text | `--text-xs` (12px) | 400 | `--ink-secondary` | normal |

**Never use:**
- Font weight 600 or 700
- Italic text
- UPPERCASE or Title Case (sentence case only)
- Font sizes below 11px

---

## Component specifications

### Sidebar (Quantifyd, Command Center)

```
Width: 200px
Background: var(--canvas)
Right border: 0.5px solid var(--hairline-soft)
Padding: 20px 12px
```

**Logo block** (top):
```
Padding: 6px 10px
Margin-bottom: 18px
Icon: 22×22px rounded square (radius 6px), background --ink
Brand text: 14px / weight 500 / --ink / tracking --tracking-normal
```

**Nav item**:
```
Padding: 7px 10px
Border-radius: var(--radius-control)
Icon: 15×15px SVG, stroke-width 1.6
Text: 13px / weight 400 / --ink-subtle
Icon stroke: --ink-secondary (#5F5E5A)
Gap between icon and text: 10px
```

**Nav item (active)**:
```
Background: var(--nav-active-bg)
Text: 13px / weight 500 / --ink
Icon stroke: --ink (#1B1B1A)
No border. No accent bar. No pill.
```

**User avatar** (bottom, margin-top auto):
```
Circle: 24×24px
Background: var(--avatar-bg)
Text: 11px / weight 500 / --avatar-ink
Name text: 12px / --ink-secondary
```

### Top tab bar (Trading Cockpit)

Use when the app is single-purpose and doesn't need a sidebar.

```
Container: flex, gap 28px, padding 0 24px
Bottom border: 0.5px solid var(--hairline-soft)
```

**Tab (inactive)**:
```
Padding: 12px 0
Text: 13px / weight 400 / --ink-muted
```

**Tab (active)**:
```
Padding: 12px 0
Text: 13px / weight 500 / --ink
Border-bottom: 1.5px solid --ink
Margin-bottom: -0.5px (to overlap container border)
```

### Top header bar (Trading Cockpit)

```
Padding: 14px 24px
Border-bottom: 0.5px solid var(--hairline-soft)
Flex: space-between, center
```

Left side: logo + timestamp (12px / --ink-faint, with margin-left 6px)
Right side: status dot (7×7px circle, --status-connected) + status text + user avatar

### Suggestion / module cards

```
Background: var(--surface)
Border: 0.5px solid rgba(0,0,0,0.08)
Border-radius: var(--radius-card)
Padding: 14px 16px (or 16px for larger cards)
Cursor: pointer
```

**Card title**: 13px / weight 500 / --ink / margin-bottom 4px
**Card description**: 12px / weight 400 / --ink-muted / line-height 1.45

**Grid layout**: `grid-template-columns: repeat(3, minmax(0, 1fr))` with `gap: 12px`. Max-width 760px, centered.

### Metric cards (Trading Cockpit, Command Center)

```
Background: var(--surface)
Border: 0.5px solid rgba(0,0,0,0.08)
Border-radius: var(--radius-card)
Padding: 14px 16px
```

**Label**: 11px / weight 400 / --ink-muted / margin-bottom 6px
**Value**: 20px / weight 500 / --ink (or --accent-pos/neg for P&L) / tracking --tracking-tight

**Grid**: `repeat(4, minmax(0, 1fr))` with `gap: 12px`

### Data table (positions, scanner results)

**Container**:
```
Background: var(--surface)
Border: 0.5px solid rgba(0,0,0,0.08)
Border-radius: var(--radius-card)
Overflow: hidden
```

**Header row**:
```
Background: var(--canvas)
Padding: 10px 16px
Font: 11px / weight 400 / --ink-muted
Border-bottom: 0.5px solid var(--hairline-soft)
```

**Data row**:
```
Padding: 11px 16px
Font: 13px / weight 400 / --ink
Border-bottom: 0.5px solid var(--hairline-softest)
Last row: no bottom border
```

**P&L column**: weight 500, color --accent-pos or --accent-neg
**Secondary data** (qty, avg): color --ink-secondary
**System/tag column**: 11px / --ink-muted

Use CSS grid for table columns, not `<table>` elements. Example:
```css
.table-row {
  display: grid;
  grid-template-columns: 1.6fr 1fr 1fr 1fr 1fr 0.8fr;
  padding: 11px 16px;
}
```

### Chips / pills

```
Font: 11px / weight 400 / --chip-ink
Background: var(--chip-bg)
Border-radius: var(--radius-control)
Padding: 4px 10px
```

Use for keyboard shortcuts (⌘ K), slash commands (/scan, /pos), or tags.

### Status indicator

```
Dot: 7×7px circle, border-radius 50%
Color: var(--status-connected) for connected
Gap: 6px to status text
Text: 12px / --ink-secondary
```

### Tab underline (horizontal, e.g. "Suggested / Recent / Saved")

```
Container: flex, gap 24px, centered, border-bottom 0.5px solid var(--hairline-soft)
```

**Tab (active)**: 13px / weight 500 / --ink / border-bottom 1.5px solid --ink
**Tab (inactive)**: 13px / weight 400 / --ink-muted

---

## Layout patterns

### Sidebar + content (Quantifyd, Command Center)

```
Container: flex, min-height 100vh
Sidebar: width 200px, fixed
Content area: flex 1, padding 56px 48px (top/sides) for landing pages
  Inner content: max-width 620px for input areas, 760px for card grids, 920px for tables
  Centering: margin 0 auto
```

### Top bar + tabs + content (Trading Cockpit)

```
Header: full width, fixed height
Tab bar: full width, directly below header
Content area: padding 36px 48px
  Inner content: max-width 720px for hero areas, 920px for data tables
```

### Content hierarchy on landing pages

1. Page title or brand mark (centered)
2. Subtitle / context line (centered, --ink-muted)
3. Suggestion / module card grid
4. Status summary cards (optional, at bottom)

---

## Color usage rules

### When to use color

| Scenario | Color |
|---|---|
| Positive P&L, net theta gain | `--accent-pos` (#0F6E56) |
| Negative P&L, loss | `--accent-neg` (#A32D2D) |
| Active nav item icon/text | `--ink` (#1B1B1A) |
| Connected status dot | `--status-connected` (#1D9E75) |
| User avatar circle | `--avatar-bg` / `--avatar-ink` |

### When NOT to use color

- Card backgrounds (always white)
- Card borders (always --hairline)
- Decorative accents, hover states
- Badges, tags (use chip style with --chip-bg instead)
- Icons in inactive nav (use --ink-secondary gray)

### P&L formatting

```
Positive: color --accent-pos, prefix with "+"
  Example: +₹18,420
Negative: color --accent-neg, prefix with "−" (minus sign, not hyphen)
  Example: −₹2,062
Neutral/zero: color --ink
```

Always use weight 500 for P&L values in tables. Use `Intl.NumberFormat('en-IN')` for Indian number formatting.

---

## Spacing system

| Token | Value | Usage |
|---|---|---|
| Page padding (top) | 56px | Landing page content start |
| Page padding (sides) | 48px | Left/right on content area |
| Section gap | 32px | Between major content sections |
| Card grid gap | 12px | Between cards in a grid |
| Card internal padding | 14–16px | Inside cards |
| Nav item padding | 7px 10px | Sidebar nav items |
| Nav item gap | 4px | Between nav items vertically |
| Table row padding | 11px 16px | Data rows |
| Table header padding | 10px 16px | Column headers |
| Component internal gap | 6–10px | Between icon and text, label and value |

---

## Interaction states

### Hover (cards)

```css
.card:hover {
  border-color: rgba(0, 0, 0, 0.15);
  transition: border-color 150ms ease;
}
```

No background change, no shadow, no scale. Just a slightly darker border.

### Hover (nav items)

```css
.nav-item:hover {
  background: rgba(0, 0, 0, 0.03);
  transition: background 100ms ease;
}
```

### Focus (inputs)

```css
input:focus {
  outline: none;
  border-color: rgba(0, 0, 0, 0.25);
}
```

No colored focus rings. Just a darker border.

### Active/pressed

```css
.card:active, .nav-item:active {
  transform: scale(0.99);
  transition: transform 50ms ease;
}
```

Subtle. Not bouncy.

---

## Iconography

- Style: outline/stroke only, no fills
- Stroke width: 1.6px for sidebar (15×15), 2px for 12×12 icons
- Source: Lucide Icons (https://lucide.dev) or hand-drawn SVG paths
- Color: `--ink` for active, `--ink-secondary` for inactive
- Never use emoji as icons
- Never use filled/solid icon variants

---

## What this system does NOT include

- **AI command input / "ask the cockpit" prompt box** — not applicable, no AI integration
- **Dark mode** — the system is light-only. If a dark mode is needed in the future, define a separate token set under `@media (prefers-color-scheme: dark)` or a `.dark` class
- **Mobile responsive breakpoints** — all three apps are desktop-first (Tauri desktop app, or desktop web). Add responsive rules only if needed later
- **Animation/motion** — keep transitions under 150ms, ease timing. No spring physics, no bouncing, no loading spinners that spin for show

---

## Implementation notes for Claude Code

### React component structure

```
src/
  tokens/
    design-tokens.css          ← CSS custom properties above
  components/
    Sidebar/
      Sidebar.tsx
      SidebarNavItem.tsx
    TopBar/
      TopBar.tsx
      TabBar.tsx
    Cards/
      SuggestionCard.tsx
      MetricCard.tsx
    DataTable/
      DataTable.tsx
      DataTableRow.tsx
    Chip/
      Chip.tsx
    StatusDot/
      StatusDot.tsx
    Avatar/
      Avatar.tsx
```

### Key implementation rules

1. **Use CSS modules or plain CSS files** — not styled-components, not Tailwind, not CSS-in-JS. The token system is CSS variables; keep it native.
2. **Use CSS Grid for tables** — not HTML `<table>`. Grid gives precise column control with `grid-template-columns`.
3. **Use `minmax(0, 1fr)`** — not plain `1fr`. Prevents grid blowout from long content.
4. **Format all numbers** — use `Intl.NumberFormat('en-IN')` for currency, `toFixed()` for decimals. Never show raw float math.
5. **No component libraries** — no Material UI, no Ant Design, no Chakra. Build from primitives with these tokens.
6. **Icons: inline SVG** — import as React components or paste inline. Do not use icon font libraries.
7. **Transitions: CSS only** — no Framer Motion, no React Spring. Use `transition: property 150ms ease` on the element.

### Tauri-specific (Command Center)

- Window chrome: use Tauri's native decorations on macOS, custom title bar on Windows matching `--canvas` background
- Window default size: 1200×800px minimum
- Sidebar should be non-resizable, fixed 200px

### Tech stack reference

| App | Framework | State | Backend |
|---|---|---|---|
| Quantifyd | React + TypeScript | Local state + SQLite | Kite Connect API, Contabo VPS |
| Trading Cockpit | React + TypeScript | Local state + SQLite | Kite Connect API, Contabo VPS |
| Command Center | Tauri 2.0 + React | Tauri store / SQLite | Local filesystem |

---

## Quick checklist before shipping any screen

- [ ] Canvas is `#FAFAF9`, cards are `#FFFFFF`
- [ ] All borders are `0.5px solid rgba(0,0,0,0.10)` or softer
- [ ] No font weight above 500
- [ ] All text is sentence case
- [ ] Color appears ONLY for P&L values and status indicators
- [ ] No shadows, gradients, or glow anywhere
- [ ] Numbers are formatted (no floating point artifacts)
- [ ] Icons are 15px outline SVGs with stroke-width 1.6
- [ ] Card border-radius is 10px, controls are 6px, input shells are 12px
- [ ] Active nav state uses 4% black fill, nothing more
