# Jira PRD Review Workflow

**Purpose:** Documents the approval flow for PRD review via Jira
**Managed by:** `@architect` agent
**Triggered by:** `/checkprd` command

---

## Workflow Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         PRD REVIEW WORKFLOW                              │
└─────────────────────────────────────────────────────────────────────────┘

 @architect [feature]
       │
       ├── 1. Complete docs/APP_PRD.md
       ├── 2. Generate docs/[feature]-workflow.drawio
       ├── 3. Generate docs/mockups/[feature].html (UI features)
       ├── 4. Create Jira: "📋 Review PRD: [Feature]"
       │      - Type: Task
       │      - Status: To Do → In Review
       │      - Assignee: User
       │      - Attachments: APP_PRD.md, workflow.drawio, mockup.html
       └── 5. STOP - Wait for review

═══════════════════════════════════════════════════════════════════════════
                              USER REVIEWS
═══════════════════════════════════════════════════════════════════════════

                         ┌─────────────────┐
                         │  User Reviews   │
                         │   APP_PRD.md    │
                         └────────┬────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    │                           │
                    ▼                           ▼
           ┌───────────────┐           ┌───────────────┐
           │   APPROVED    │           │ NEEDS CHANGES │
           └───────┬───────┘           └───────┬───────┘
                   │                           │
                   ▼                           ▼
           Add Jira comment:           Add Jira comment:
           "approved" / "lgtm"         "Need X, Y, Z..."
           "ok" / "good to go"         Keep status: In Review
           Move to: Done                       │
                   │                           │
═══════════════════╪═══════════════════════════╪═══════════════════════════
                   │         /checkprd         │
═══════════════════╪═══════════════════════════╪═══════════════════════════
                   │                           │
                   ▼                           ▼
           ┌───────────────┐           ┌───────────────┐
           │ @architect    │           │ @architect    │
           │ APPROVED MODE │           │ REVISION MODE │
           └───────┬───────┘           └───────┬───────┘
                   │                           │
                   ▼                           ▼
           Create Jira items:          1. Read feedback comments
           • Epic (feature)            2. Revise APP_PRD.md
           • Stories (components)      3. Add Jira comment:
           • Tasks (subtasks)             "Changes made: [list]"
                   │                   4. Status stays: In Review
                   │                           │
                   ▼                           ▼
              ✅ DONE                   🔄 LOOP BACK
           Implementation                (User reviews again)
           can begin
```

---

## Jira Status Mapping

| Jira Status | User Action | /checkprd → @architect Response |
|-------------|-------------|----------------------------------|
| `To Do` | Not reviewed yet | "PRD awaiting review. Please review docs/APP_PRD.md" |
| `In Review` | Added feedback | **REVISION MODE** - Revise PRD based on comments |
| `In Review` | No new comments | "Waiting for your feedback in Jira" |
| `Done` | Approved | **APPROVED MODE** - Create Epic/Stories/Tasks |

---

## Approval Keywords

Comments containing these words trigger approval:
- `approved`
- `lgtm` (looks good to me)
- `ok` / `okay`
- `good to go`
- `reviewed`
- `ship it`
- `👍` (thumbs up emoji)

---

## Revision Mode Details

When status = "In Review" with feedback comments:

### 1. Parse Feedback
Extract action items from user comments:
```
User comment: "Need more detail on error handling.
              Also add offline sync section."

Parsed items:
- Add detail on error handling
- Add offline sync section
```

### 2. Revise APP_PRD.md
@architect updates the PRD addressing each item

### 3. Add Jira Comment
```
Revisions complete:
✅ Added error handling section (lines 45-62)
✅ Added offline sync section (lines 78-95)

Resubmitting for review.
```

### 4. Status Remains "In Review"
User will review again and either:
- Add more feedback → Another revision cycle
- Approve → Move to Done

---

## Approved Mode Details

When status = "Done" with approval comment:

### 1. Create Epic
```
Type: Epic
Summary: [Feature Name]
Description: Link to APP_PRD.md
```

### 2. Create Stories
From implementation plan major sections:
```
Type: Story
Summary: [Component/Section Name]
Parent: Epic
Story Points: Based on complexity
```

### 3. Create Tasks
From implementation plan task table:
```
Type: Task (or Sub-task)
Summary: [Task from plan]
Parent: Story
Labels: complexity-low/medium/high
```

---

## File Locations

| File | Purpose |
|------|---------|
| `docs/APP_PRD.md` | PRD document (revised during workflow) |
| `docs/[feature]-workflow.drawio` | Implementation diagram |
| `inits_n_info/jira-prd-workflow.md` | This workflow documentation |

---

## Commands

| Command | Purpose |
|---------|---------|
| `@architect [feature]` | Initial PRD creation + Review Jira task |
| `/checkprd` | Check review status, trigger appropriate mode |

---

## Example Session

```bash
# Initial PRD creation
> @architect PDF export feature

✅ Created docs/APP_PRD.md
✅ Created Jira SPT-45: "📋 Review PRD: PDF Export"
⏸️ Waiting for your review in Jira

# After user adds feedback and keeps "In Review"
> /checkprd

📋 SPT-45 Status: In Review
📝 Feedback found:
   - "Add error handling for large files"
   - "Clarify supported chart types"

🔄 Revising PRD...
✅ Updated APP_PRD.md
✅ Added Jira comment with changes
⏸️ Resubmitted for review

# After user approves and moves to "Done"
> /checkprd

📋 SPT-45 Status: Done ✅
📝 Approval: "lgtm, approved"

Creating implementation items...
✅ Epic: SPT-50 "PDF Export Feature"
✅ Story: SPT-51 "PDF Components" (5 tasks)
✅ Story: SPT-52 "Export Integration" (3 tasks)
✅ Story: SPT-53 "Mobile Support" (2 tasks)

🚀 Ready for development!
```
