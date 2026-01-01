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

---

## Commands

| Command | Purpose |
|---------|---------|
| `@architect [feature]` | Initial PRD creation + Review Jira task |
| `/checkprd` | Check review status, trigger appropriate mode |

---

**Document Version:** 1.0
