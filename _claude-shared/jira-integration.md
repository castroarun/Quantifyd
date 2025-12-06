# Jira API Integration

**Purpose:** Automate Jira task updates during AI-assisted development workflow.

---

## Setup

### 1. Get Jira API Token

1. Go to: https://id.atlassian.com/manage-profile/security/api-tokens
2. Click "Create API token"
3. Name it: `buildfolio-claude`
4. Copy the token (shown only once)

### 2. Create Config File

Copy `jira-config.example.json` to `jira-config.json` and fill in:

```json
{
  "domain": "your-domain.atlassian.net",
  "email": "your-email@example.com",
  "projectKey": "SPT",
  "apiToken": "your-api-token-here"
}
```

**Important:** `jira-config.json` is gitignored - never commit credentials.

---

## API Operations

### Attach File to Issue

```bash
# Attach TEST-PLAN.csv to SPT-3
curl -X POST \
  -H "Authorization: Basic $(echo -n 'email:token' | base64)" \
  -H "X-Atlassian-Token: no-check" \
  -F "file=@docs/TEST-PLAN.csv" \
  "https://domain.atlassian.net/rest/api/3/issue/SPT-3/attachments"
```

### Update Issue Status

```bash
# Get available transitions
curl -X GET \
  -H "Authorization: Basic $(echo -n 'email:token' | base64)" \
  "https://domain.atlassian.net/rest/api/3/issue/SPT-3/transitions"

# Move to "In Review" (transition ID varies by workflow)
curl -X POST \
  -H "Authorization: Basic $(echo -n 'email:token' | base64)" \
  -H "Content-Type: application/json" \
  -d '{"transition": {"id": "21"}}' \
  "https://domain.atlassian.net/rest/api/3/issue/SPT-3/transitions"
```

### Add Comment

```bash
curl -X POST \
  -H "Authorization: Basic $(echo -n 'email:token' | base64)" \
  -H "Content-Type: application/json" \
  -d '{
    "body": {
      "type": "doc",
      "version": 1,
      "content": [{
        "type": "paragraph",
        "content": [{"type": "text", "text": "TEST-PLAN.csv attached with 58 test cases."}]
      }]
    }
  }' \
  "https://domain.atlassian.net/rest/api/3/issue/SPT-3/comment"
```

---

## Workflow Integration

| Development Step | Jira Action |
|-----------------|-------------|
| Test Plan Created | Attach CSV, move to "In Review" |
| Test Plan Approved | Move to "Done", create Build tasks |
| Build Complete | Move Build tasks to "Testing" |
| Testing Passed | Move to "Done" |

---

## Status IDs (Configure per project)

| Status | Typical ID |
|--------|-----------|
| To Do | 11 |
| In Review | 21 |
| In Development | 31 |
| Testing | 41 |
| Done | 51 |

*Note: Run transitions GET to find your actual IDs.*

---

## Security Notes

1. Never commit `jira-config.json`
2. Use environment variables in CI/CD
3. API token has same permissions as your account
4. Rotate tokens periodically
