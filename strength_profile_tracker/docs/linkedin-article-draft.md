# A 9-Step Development Workflow Using AI Agents

**Most solo developers are still doing everything manually.**

Requirements gathering. Architecture design. Test planning. Documentation. All bottlenecks that slow down what should be fast: shipping working software.

This article breaks down an AI-powered development system that handles 80% of the typical workflow. Here's exactly how it works.

[SCREENSHOT: ecosystem-diagram.png - The complete Claude Code agent ecosystem showing inputs, agents, commands, and outputs]

---

## The Problem: The Default Way Doesn't Scale

The patterns that slow developers down are well-known:

- **Vague requirements** lead to rework
- **No design docs** mean inconsistent architecture
- **Test cases written after bugs** appear
- **Zero documentation** for future maintenance

The goal: a system that prevents these problems before they start.

---

## The Solution: Specialized AI Agents

The approach: a team of specialized AI agents - each with a defined role, specific deliverables, and clear handover points between them. Think TOGAF principles applied to AI: structured phases, defined outputs, and explicit "Definition of Done" before moving to the next stage.

### Agent 1: @designer (Requirements Analyst)

Before any design work begins, this agent conducts deep research:
- Searches for similar apps and analyzes competitors
- Researches best practices for the project type
- Asks methodical questions to capture complete requirements

The output? A structured requirements document covering:
- Project type, purpose, and target users
- Design system preferences
- Page structure and navigation
- Components needed
- Technical stack decisions

**Key insight:** The designer agent uses the Opus model for complex reasoning. It doesn't assume - it asks until requirements are complete.

[SCREENSHOT: designer-questions.png - Terminal showing @designer asking methodical questions about project requirements]

### Agent 2: @architect (System Designer)

With requirements captured, the architect takes over with a two-phase approach:

**Phase 1: Discovery**
- Reviews requirements
- Proposes 2-3 implementation approaches
- Presents trade-offs for each option
- Waits for human approval before proceeding

**Phase 2: PRD Creation**
- Creates detailed Product Requirements Document
- Generates visual workflow diagrams (Draw.io format)
- Creates HTML mockups with Tailwind CSS
- Submits everything for human review via Jira

The PRD includes database schemas, API structures, component hierarchies, state management strategy, and a task breakdown with complexity estimates.

[SCREENSHOT: architect-prd-snippet.png - Section of APP_PRD.md showing database schema and component breakdown]

[SCREENSHOT: html-mockup-browser.png - HTML mockup rendered in browser showing the actual UI design]

### Agent 3: @qa (Test Plan Creator)

Here's the counterintuitive part: test cases come BEFORE development.

The QA agent reads the approved PRD and generates comprehensive test cases:
- Positive tests (happy path)
- Negative tests (error handling)
- Boundary tests (edge values)

Each test case includes specific steps, expected behavior, test data, and priority level. The output is a CSV file that tracks execution during manual testing.

**Why test before code?** It validates understanding. Forces clarity on requirements. Catches gaps before any code is written.

[SCREENSHOT: test-plan-csv.png - TEST-PLAN.csv showing actual test cases with IDs, steps, and expected results]

---

## The 9-Step Workflow

These agents fit into a structured development process:

| Step | Activity | Agent | Output |
|------|----------|-------|--------|
| 1 | DEV-CLOCK Setup | - | Time tracking document |
| 2 | PRD & Design | @architect | APP_PRD.md, workflow.drawio |
| 3 | Test Cases | @qa | TEST-PLAN.csv |
| 4 | Build | Claude Code | Working code |
| 5 | Manual Testing | Human | Test execution results |
| 6 | Debug & Feedback | Claude Code | Bug fixes |
| 7 | Code Walkthrough | @walkthrough | WALKTHROUGH.md |
| 8 | Ship | Deploy scripts | Live application |
| 9 | Time Retrospective | - | Lessons learned |

**The critical rule:** No coding without an approved design. No coding without a test plan.

Human approval gates at Steps 2, 3, and 8 ensure AI proposals get validated before investment.

---

## The Review Loop: Jira Integration

The workflow integrates with Jira for structured reviews.

When @architect completes a PRD:
1. Creates a Jira task: "Review PRD: [Feature]"
2. Attaches PRD document, workflow diagram, and mockups
3. Assigns to the reviewer

The review process:
- Add comments describing changes needed
- Keep status as "In Review"
- Run `/checkprd` command

The architect reads feedback, revises the PRD, and resubmits. This loop continues until approval.

When approved:
- Move Jira task to "Done"
- Architect automatically creates Epic, Stories, and Tasks
- Development begins with full traceability

[SCREENSHOT: jira-prd-review.png - Jira task "Review PRD: [Feature]" with attached documents and review comments]

---

## Real Example: Strength Profile Tracker

A fitness tracking PWA built using this workflow.

**Step 2 output:** PRD covering profile management, exercise tracking, strength calculations, and progress visualization. Architect created HTML mockups for each screen.

**Step 3 output:** 50+ test cases covering CRUD operations, validation rules, data persistence, and UI components. Priority P0 tests defined the "smoke test" suite.

**Step 4:** Built with Next.js, TypeScript, Tailwind CSS. The approved design made implementation straightforward - no architectural debates mid-sprint.

**Step 5:** Executed test plan. Found 3 edge cases. Logged as Jira bugs linked to specific test case IDs.

**Result:** A fully-tested PWA with comprehensive documentation. Every requirement traced from initial capture through deployment.

[SCREENSHOT: spt-app-running.png - Strength Profile Tracker PWA running on mobile showing the actual interface]

[SCREENSHOT: docs-folder-structure.png - docs/ folder showing APP_PRD.md, TEST-PLAN.csv, WALKTHROUGH.md, and mockups/]

---

## Key Takeaways

### 1. Specialization beats generalization

One agent doing everything produces mediocre outputs. Specialized agents with defined responsibilities produce expert-level work in their domain.

### 2. Human gates matter

AI proposes, human approves. The review loops aren't overhead - they're quality control. They catch wrong assumptions before they become wrong code.

### 3. Documentation is a product

The PRD, test plan, and walkthrough aren't bureaucracy. They're deliverables that make the codebase maintainable.

### 4. Front-loaded effort saves time

Spending time on requirements and design feels slow. But it's faster than refactoring half-built features because something obvious was missed.

---

## Implementation Details

This system runs on Claude Code with custom agents and slash commands. The core components:

**Agents** (stored in `~/.claude/agents/`):
- `designer.md` - Requirements gathering
- `architect.md` - PRD and design
- `qa.md` - Test case generation
- `walkthrough.md` - Documentation

**Commands** (stored in `~/.claude/commands/`):
- `/newproject` - Initialize project templates
- `/checkprd` - Check Jira review status
- `/jirastatus` - View sprint progress

**Integration:**
- MCP server for Jira (mcp-atlassian)
- Git for version control (including agent configs)

Start simple: Create one specialized agent for the biggest bottleneck. Document its responsibilities, inputs, and expected outputs. Iterate from there.

---

**What's your biggest bottleneck in solo development?**

Drop a comment with your approach - or the problem you're still trying to solve.

---

*Building in public at github.com/castroarun*
