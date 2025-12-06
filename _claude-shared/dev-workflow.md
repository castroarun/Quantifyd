# AI-Assisted Development Workflow

**Version:** 1.0
**Purpose:** Standardized 9-step development process for all projects in buildfolio.

---

## 9-Step Development Process

```
Step 1: DEV-CLOCK Setup        → Time tracking from day 1
Step 2: PRD & Design Document  → Design decisions for approval
Step 3: Test Cases             → Validate understanding before code
Step 4: Build                  → AI writes, human reviews
Step 5: Manual Testing         → Human verifies functionality
Step 6: Debug & Feedback       → Iterative fixes
Step 7: Code Walkthrough       → Documentation for code familiarity
Step 8: Ship                   → Deploy to production
Step 9: Time Retrospective     → Analyze where time went
```

---

## Step Details

### Step 1: DEV-CLOCK Setup

Create `docs/DEV-CLOCK.md` to track time spent on each phase.

| Phase | Purpose |
|-------|---------|
| Design & Planning | PRD review, tech decisions |
| Documentation | Creating project docs |
| Building | Coding implementation |
| Debugging | Fixing issues |
| Testing | Manual verification |
| Shipping | Deployment |

### Step 2: PRD & Design Document

Create design documentation covering:
- Objective and capabilities
- Database schema / data models
- UI component breakdown
- File structure
- Color palette and typography

**Note:** Design can be embedded in PRD if comprehensive enough.

### Step 3: Test Cases

Create `docs/TEST-PLAN.csv` with test cases **before** coding.

**Why test plan before development?**
1. **Validates understanding** - Forces clarity on requirements before code
2. **Defines "done"** - Clear acceptance criteria for each feature
3. **Catches gaps early** - Missing requirements surface before coding
4. **Enables TDD mindset** - Build with testability in mind
5. **Review checkpoint** - Human approval before investment in code

**CSV Format:**
```csv
Test ID,Category,Test Case,Steps,Expected Result,Priority,Status
TC-001,Category,Test name,1. Step 2. Step,Expected outcome,P0,Not Tested
```

### Step 4: Build

- Follow approved design
- Create reusable components
- Add clear function comments
- Human reviews code regularly

### Step 5: Manual Testing

- Test on target viewport (mobile-first)
- Cross-browser testing
- Edge case verification
- Run through TEST-PLAN.csv

### Step 6: Debug & Feedback

- Log issues found
- Fix and verify
- Update test plan status

### Step 7: Code Walkthrough

Create `docs/WALKTHROUGH.md` documenting:
- Key code paths
- Architecture decisions
- How to extend features

### Step 8: Ship

- Deploy to production (Vercel, etc.)
- Environment setup
- Domain configuration
- Final verification

### Step 9: Time Retrospective

- Update DEV-CLOCK with actual hours
- Analyze time distribution
- Document lessons learned

---

## Review Checklist Format

When presenting documents for review, use this table format for clarity:

**Example for Test Plan Review:**

```markdown
| Category | Cases | Key Coverage |
|----------|-------|--------------|
| Profile Management | 9 | CRUD, max limits |
| Validation | 11 | All boundary values |
| Core Logic | 18 | Business calculations |
| Data Persistence | 4 | Offline, app restart |
| UI Components | 5 | Colors, touch targets |
| Edge Cases | 7 | Special chars, decimals |
```

**Benefits:**
- Quick visual scan of coverage
- Shows category distribution
- Highlights key areas tested
- Easy to spot gaps

---

## Jira Integration Workflow

### Status Flow

```
To Do → In Review → In Development → Testing → Done
              ↓
        In Review (Redo)  ← Changes requested
```

### Task Mapping

| Step | Jira Task | Status on Completion |
|------|-----------|---------------------|
| Step 2 | PRD Review | In Review → Done |
| Step 2 | Design Review | In Review → Done |
| Step 3 | Test Plan Review | In Review → Done |
| Step 4 | Build Tasks | In Development → Testing |
| Step 5-6 | Testing | Testing → Done |
| Step 8 | Ship | Done |

---

## Document Checklist

| Document | Location | Created |
|----------|----------|---------|
| DEV-CLOCK.md | `docs/` | Step 1 |
| PRD.md or DESIGN.md | `docs/` | Step 2 |
| TEST-PLAN.csv | `docs/` | Step 3 |
| WALKTHROUGH.md | `docs/` | Step 7 |

---

## Key Principles

1. **No coding without approved design** - Step 2 must be approved
2. **No coding without test plan** - Step 3 defines acceptance criteria
3. **Human reviews at each step** - AI proposes, human approves
4. **Track time from day 1** - DEV-CLOCK enables retrospectives
5. **Document for future self** - WALKTHROUGH ensures maintainability

---

*This workflow is referenced by all projects in buildfolio.*
