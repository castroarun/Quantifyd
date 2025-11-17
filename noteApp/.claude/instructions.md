# Note App with Jira & Gmail Integration - Project Instructions

> **Coding Standards:** All general coding standards, TypeScript patterns, and best practices are documented in [coding-standards.md](./coding-standards.md).

---

## Project Overview

A web-based note-taking application combining simple note management with powerful Jira task creation and Gmail communication, featuring real-time markdown file synchronization with local filesystem.  Objective is to make this note app clean, simple, quick to load and use. Must be integrated with jira and gmail for quick task creations and emailing.

**Primary Goal:** Learn full-stack development (steeping away from python frameworks) while building a production-ready note app with AI-assisted development.

**Plan:** Do not get into development until we design the product end-end, review and finalize the PRD and immplementation plan


## Tech Stack Requirements

Use the following technologies:
- **Next.js** for application
- **TypeScript** for programming
- **Tailwind CSS v3** for styling

First, let us research more on this code stack and fine tune/enhance the same.
- Front end: I want to arrive at the **color palette** and **themes** that will be used in the application as the objective is to make this note app clean, simple, quick to load and use.
- Database: Must use **supabase**
---

## Folder Structure Requirements

Organize the project following these rules:

1. **Only the main app file** stays in the project folder
2. **UI related files** under ui folder
3. **Jira integrations, Gmail integrations, Google integrations** under api directory etc.
4. **All .md files** must be within docs folder

---

## Code Writing Requirements

### Function Development

Write functions keeping **reusability in mind**.

### Function Comments

Write clear and simple comments on:
- What the function does
- What is the input and output
- What is the calling function/code etc.

**Example format:**
```typescript
/**
 * [What the function does]
 *
 * Input: [describe input parameters]
 * Output: [describe return value]
 *
 * Called by: [calling function/component name]
 * Calls: [functions this calls]
 */
```
## Prototyping Requirements
- I want to be able to vew the prototypes of the UI and navigations before proceeding to the full fledged development

---

## Editor Requirements

### Text Formatting
The editor should have **simple keyboard shortcuts only** - no toolbar icons for formatting:

- **Ctrl+B** - Make selected text **Bold**
- **Ctrl+H** - Make selected text a **Header**
- **Ctrl+I** - Make selected text *Italic*
- **Ctrl+U** - Make selected text <u>Underlined</u>
- **Ctrl+Z** - Undo changes
- **Tab** - Make selected text bulleted
- **Tab (again)** - Make sub-bullet under the bulleted line

**Important:** If these shortcuts are pressed without any text selected, perform default Windows operations (don't interfere).

### Auto-Generated Note Headlines
- When user finishes typing (switches to another note or creates new note), automatically derive a suitable headline from the note content
- The auto-generated headline should be **editable** by the user
- Headline should be generated intelligently from the first line or key content
- **Important:** Auto-titling should NOT remove or modify the note content - title is separate from content

---

## Right Panel (Utility Panel) Requirements

The right side of the app should have a **collapsible vertical panel** with utility icons:

### Panel Icons & Features

1. **Web Browser Icon**
   - Opens an embedded web browser panel
   - Allows users to browse websites, search, and copy content
   - Easy copy-paste into notes without leaving the app

2. **Calculator Icon**
   - Opens a simple calculator panel
   - Quick calculations while taking notes

3. **Shortcuts Reference Icon**
   - Shows all keyboard shortcuts including:
     - Text formatting shortcuts (Ctrl+B, Ctrl+I, etc.)
     - New note creation
     - Panel toggle shortcuts
     - Navigation shortcuts

### Panel Behavior
- Panel should be collapsible/expandable
- Icons should be vertically aligned on the right edge
- Clicking an icon expands the panel with that tool
- Panel should not interfere with the main editor area

---

## Additional Core Capabilities

### Pin Important Notes
- Users should be able to **pin** frequently accessed notes
- Pinned notes stay at the top of the notes list
- Visual indicator (pin icon) to show pinned status
- Easy toggle on/off for pinning

### Dark Mode
- **Essential** for long writing sessions
- Toggle between light and dark themes
- Smooth transition between themes
- Save user's theme preference
- Dark mode should use eye-friendly colors

---

## Documentation Requirements

### 1. Product Design Document (`docs/DESIGN.md`)

**Create BEFORE building code**

Must include:

#### A. Objective in Short
- Brief description of what the app does

#### B. Capabilities List
- List all major capabilities

#### C. Capabilities Explanation
For each capability, explain:
- How we are going to achieve the same
- API know-hows for API integrations (if applicable)

#### D. File Structures
- Simple tree structure format
- File names with purpose in a single line

**Important:** The product design document should **not be too heavy**.

---

### 2. Test Plan (`docs/TEST-PLAN.csv`)

**Create BEFORE building code**

Prepare a simple test plan in CSV format.

**Purpose:**
- Review the PRD and test plan
- Make corrections if required
- Only proceed to coding after approval

---

### 3. Technical Walkthrough Document (`docs/WALKTHROUGH.md`)

**Create AFTER project completion**

A document to technically understand the codebase with respect to the app's capabilities.

**Requirements:**
- Should **not be exhaustive**
- Code snippets must be mentioned with file name and directory with the explanations
- Explain the capabilities and code flow in order

**Purpose:** Help understand how each capability is implemented in the codebase.

---

## Development Workflow

### Step 1: Design & Planning
1. Create `docs/DESIGN.md` (Product Design Document)
2. Create `docs/TEST-PLAN.csv` (Test Plan)
3. **Wait for review and approval**
4. Make corrections if requested
5. Get explicit approval to proceed

### Step 2: Implementation
- Follow the folder structure requirements
- Write reusable functions with proper comments
- Implement features as per approved design

### Step 3: Post-Completion Documentation
1. Create `docs/WALKTHROUGH.md` (Technical Walkthrough)
2. Document how capabilities are implemented
3. Include code snippets with file paths

---

## Key Principles

1. **Reusability First** - Write functions that can be reused
2. **Clear Documentation** - Every function must have clear comments
3. **Organized Structure** - Follow folder structure strictly
4. **Approval Required** - Get design and test plan approved before coding
5. **Not Too Heavy** - Keep documentation concise and focused

---

## Important References

- **[coding-standards.md](./coding-standards.md)** - General coding standards and best practices
- **[docs/NOTE-APP-PRD.md](../docs/NOTE-APP-PRD.md)** - Product requirements document
- **[docs/NOTE-APP-PROJECT.md](../docs/NOTE-APP-PROJECT.md)** - Detailed project guidelines

---

**Last Updated:** 2025-11-16
**Document Version:** 2.0
**Status:** Planning Phase
