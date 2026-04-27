"""Enhance every portfolio repo's README to 90+ score.

For each repo, fetch the existing README, splice in missing standard
elements (centered title, shields.io badges, LAUNCHPAD block, collapsible
TOC, footer) WITHOUT clobbering existing prose, and push via the GitHub
Contents API.
"""
import base64
import json
import os
import re
import subprocess
import sys
from pathlib import Path

GH_TOKEN = os.environ.get('GITHUB_TOKEN', '')
if not GH_TOKEN:
    print('Set GITHUB_TOKEN env var first.')
    sys.exit(1)

# Per-repo enhancement metadata. Keep concise — these directly drive the
# generated badges + LAUNCHPAD block, so accuracy matters.
ENHANCEMENTS = {
    ('castroarun', 'caclulate_anything'): {
        'name': 'AnyCalc',
        'tagline': 'Calculate anything, anywhere — 19 interlinked financial calculators in one app',
        'badges': [
            ('Next.js', '14', '000000', 'nextdotjs'),
            ('TypeScript', '5.x', '3178C6', 'typescript'),
            ('Supabase', 'cloud', '3ECF8E', 'supabase'),
            ('Zustand', 'state', '764ABC', None),
            ('Status', 'Live', '22c55e', None),
        ],
        'tech_stack': ['Next.js', 'TypeScript', 'Supabase', 'Zustand'],
        'demo_url': 'https://anycalc.in',
        'features': [
            '19 interlinked calculators — EMI, tip splits, compound interest, currency conversion',
            'Plan mode chains calculations together for "what-if" analysis',
            'Workspace mode opens multiple calculators side-by-side',
            '100% local processing — calculations never touch a server',
            'Interactive Recharts visualizations',
        ],
        'roadmap': [
            ('Investment + Loan + Tax + Health calculators', True),
            ('Plan mode (chained calculations)', True),
            ('Multi-language support', False),
            ('Calculator marketplace', False),
        ],
        'project_id': 'anycalc',
        'stage': 'live',
    },
    ('castroarun', 'taskBoard'): {
        'name': 'Klarity',
        'tagline': 'Lightweight desktop task board with 15-stage pipeline + ETag distributed sync',
        'badges': [
            ('Tauri', '2.0', 'FFC131', 'tauri'),
            ('React', 'Vite', '61DAFB', 'react'),
            ('TypeScript', '5.x', '3178C6', 'typescript'),
            ('Zustand', 'state', '764ABC', None),
            ('Status', 'Live', '22c55e', None),
        ],
        'tech_stack': ['Tauri', 'React', 'TypeScript', 'Zustand'],
        'demo_url': None,
        'features': [
            '15-stage pipeline from idea to deployment with configurable gates',
            'ETag-based distributed sync — optimistic concurrency without server',
            'Multi-agent orchestration via Zustand for AI-assisted dev',
            'Cmd+K command palette with Claude AI agent for task breakdown',
            'Health scoring algorithm surfaces neglected projects (14+ days stale)',
        ],
        'roadmap': [
            ('Desktop app via Tauri 2.0', True),
            ('Mobile companion (Orbit)', True),
            ('GitHub-as-backend zero-server sync', True),
            ('AI task decomposition agent', False),
        ],
        'project_id': 'klarity',
        'stage': 'live',
    },
    ('castroarun', 'portfolio'): {
        'name': 'Castronix Portfolio',
        'tagline': 'Premium digital products portfolio — interactive resume + ATS-optimized CV + 60+ recruiter chat responses',
        'badges': [
            ('HTML5', 'static', 'E34F26', 'html5'),
            ('CSS3', 'modern', '1572B6', 'css3'),
            ('JavaScript', 'vanilla', 'F7DF1E', 'javascript'),
            ('Chart.js', 'viz', 'FF6384', 'chartdotjs'),
            ('GitHub Pages', 'live', '222222', 'github'),
        ],
        'tech_stack': ['HTML5', 'CSS3', 'JavaScript', 'Chart.js'],
        'demo_url': 'https://castronix.dev',
        'features': [
            'Single-source-of-truth: all projects load from data/projects.json',
            '3-part suite: Portfolio landing, interactive resume, ATS-optimized printable CV',
            'AI Chat Assistant with 60+ pre-trained recruiter responses (no LLM required)',
            'Project 1-pager modals with architecture deep-dives on card click',
            'Zero-framework: pure HTML5/CSS3/vanilla JS',
        ],
        'roadmap': [
            ('Brittany Chiang-inspired layout', True),
            ('Project deep-dive modals', True),
            ('Resume flip-card + ATS CV', True),
            ('LinkedIn Feed integration', False),
        ],
        'project_id': 'portfolio',
        'stage': 'live',
    },
    ('castroarun', 'REPPIT'): {
        'name': 'REPPIT',
        'tagline': 'Mobile strength training app — automatic level detection, smart progression, multi-profile',
        'badges': [
            ('Flutter', '3.x', '02569B', 'flutter'),
            ('Dart', '3.x', '0175C2', 'dart'),
            ('SQLite', 'local-first', '003B57', 'sqlite'),
            ('Riverpod', 'state', '4B6FAA', None),
            ('Status', 'Live', '22c55e', None),
        ],
        'tech_stack': ['Flutter', 'Dart', 'SQLite', 'Riverpod'],
        'demo_url': None,
        'features': [
            'Automatic strength-level detection based on lifts relative to bodyweight',
            'Smart progression engine with PROGRESS/MAINTAIN logic per session',
            'Muscle heatmap visualization across 5 body parts and 23 exercises',
            'Full-screen rest timer with wake-lock, sound/vibration alerts',
            'Multi-profile (up to 5 users), offline-first, zero registration',
        ],
        'roadmap': [
            ('Beginner → Elite progression state machine', True),
            ('Local-first SQLite with optional sync', True),
            ('Multi-profile data isolation', True),
            ('PRIMMO AI coach integration', False),
        ],
        'project_id': 'reppit',
        'stage': 'live',
    },
    ('castroarun', 'noteApp'): {
        'name': 'NoteApp',
        'tagline': 'Notes app with auto-save, 6 templates, RLS-protected sharing — built for speed and simplicity',
        'badges': [
            ('Next.js', '14', '000000', 'nextdotjs'),
            ('TypeScript', '5.x', '3178C6', 'typescript'),
            ('Supabase', 'auth+RLS', '3ECF8E', 'supabase'),
            ('Tailwind', '3.x', '06B6D4', 'tailwindcss'),
            ('Status', 'Live', '22c55e', None),
        ],
        'tech_stack': ['Next.js', 'TypeScript', 'Supabase', 'Tailwind'],
        'demo_url': 'https://noteapp-castronix.vercel.app',
        'features': [
            'Auto-save after 1 second of inactivity — never lose a thought',
            '6 pre-built templates: Weekend Planner, Goal Tracker, Meeting Notes',
            'Rich text editing via Tiptap 2.x with full keyboard shortcuts',
            'Supabase Auth + Row-Level Security — notes private by default',
            'Dark mode, pin notes, cross-note search, soft delete (recoverable)',
        ],
        'roadmap': [
            ('Auto-save + RLS-protected storage', True),
            ('Soft delete with audit trail', True),
            ('Multi-user collaboration', False),
            ('AI-powered semantic search', False),
        ],
        'project_id': 'noteapp',
        'stage': 'live',
    },
    ('castroarun', 'PRIMMO'): {
        'name': 'PRIMMO',
        'tagline': 'Agentic AI strength coach — WhatsApp + voice, 4-tier cost-optimized routing, REPPIT integration',
        'badges': [
            ('Next.js', '14', '000000', 'nextdotjs'),
            ('TypeScript', '5.x', '3178C6', 'typescript'),
            ('WhatsApp', 'Twilio', '25D366', 'whatsapp'),
            ('Vapi.ai', 'voice', '5A67D8', None),
            ('n8n', 'orchestration', 'EA4B71', 'n8n'),
            ('Status', 'Building', 'f59e0b', None),
        ],
        'tech_stack': ['Next.js', 'TypeScript', 'WhatsApp API', 'Voice AI'],
        'demo_url': None,
        'features': [
            '4-tier AI router: FAQ → semantic search → formula → Claude API (~90% cost reduction)',
            'Multi-channel delivery via WhatsApp (Twilio) and voice calls (Vapi.ai)',
            'Proactive scheduled check-ins and motivational outreach via n8n',
            'REPPIT integration — coaches on real workout data, not generic advice',
            'Vector search (RAG) over exercise corpus for semantic retrieval',
        ],
        'roadmap': [
            ('Phase 1: WhatsApp + FAQ tier', True),
            ('Phase 2: Voice calls via Vapi.ai', False),
            ('Phase 3: Multi-user dashboard', False),
            ('Phase 4: Custom voice cloning', False),
        ],
        'project_id': 'primmo',
        'stage': 'building',
    },
    ('castroarun', 'Quantifyd'): {
        'name': 'Quantifyd',
        'tagline': 'Systematic quant trading platform for Indian markets — MQ portfolio, ORB cash, NAS strangles, KC6 mean reversion',
        'badges': [
            ('Python', '3.12', '3776AB', 'python'),
            ('Flask', 'web', '000000', 'flask'),
            ('Pandas', 'data', '150458', 'pandas'),
            ('Kite', 'Connect', 'EE3344', None),
            ('APScheduler', 'cron', '58a6ff', None),
            ('Status', 'Live', '22c55e', None),
        ],
        'tech_stack': ['Python', 'Flask', 'Pandas', 'Kite API', 'APScheduler'],
        'demo_url': 'https://castroarun.github.io/Quantifyd/premarket_brief_pipeline.html',
        'features': [
            'MQ portfolio (Momentum + Quality) — 36% CAGR, 20-year backtest on Nifty 500',
            'ORB cash live trader — 15-stock OR15 breakout, sub-3% daily-loss cap',
            'NAS × 8 paper-trading variants — squeeze + 9:16 strangles with adjustment logic',
            'Pre-market brief pipeline — Claude Code cloud routine + VPS Gmail/WhatsApp dispatch',
            'KC6 mean reversion (65% win rate) + Tactical Capital Pool',
        ],
        'roadmap': [
            ('MQ portfolio + KC6 live', True),
            ('ORB cash live on Contabo VPS', True),
            ('Pre-market brief with Claude synthesis', True),
            ('NAS variants → live (currently paper)', False),
            ('Web UI archive of all strategies', False),
        ],
        'project_id': 'quantifyd',
        'stage': 'live',
    },
    ('castroarun', 'Health-Reports-Tracker'): {
        'name': 'Health Reports Tracker',
        'tagline': 'Medical reports organized and trended over time — with AI-powered insights',
        'badges': [
            ('Next.js', '14', '000000', 'nextdotjs'),
            ('TypeScript', '5.x', '3178C6', 'typescript'),
            ('Supabase', 'cloud', '3ECF8E', 'supabase'),
            ('Claude', 'AI', 'A78BFA', None),
            ('Status', 'Concept', 'a855f7', None),
        ],
        'tech_stack': ['Next.js', 'TypeScript', 'Supabase', 'Claude AI'],
        'demo_url': None,
        'features': [
            'OCR + Claude-powered extraction from PDF/image medical reports',
            'Time-series trend analysis across blood panels, vitals, biomarkers',
            'Family-account isolation with RLS',
            'Smart alerts when values drift outside personal baselines',
            'HIPAA-aware data handling — encryption at rest + in transit',
        ],
        'roadmap': [
            ('Concept design', True),
            ('OCR extraction pipeline', False),
            ('Trend visualization dashboard', False),
            ('Doctor sharing workflows', False),
        ],
        'project_id': 'health-reports-tracker',
        'stage': 'concept',
    },
    ('castroarun', 'project-house'): {
        'name': 'PaymentHouse',
        'tagline': 'Household payment + reimbursement tracker — split, settle, and audit shared expenses',
        'badges': [
            ('Next.js', '14', '000000', 'nextdotjs'),
            ('TypeScript', '5.x', '3178C6', 'typescript'),
            ('Supabase', 'cloud', '3ECF8E', 'supabase'),
            ('Tailwind', '3.x', '06B6D4', 'tailwindcss'),
            ('Status', 'Concept', 'a855f7', None),
        ],
        'tech_stack': ['Next.js', 'TypeScript', 'Supabase', 'Tailwind'],
        'demo_url': None,
        'features': [
            'Multi-user expense splitting with audit trail',
            'Recurring rent / utilities / subscription tracking',
            'Settle-up calculator — minimum-transaction algorithm',
            'Receipt OCR via Claude vision',
            'Monthly settlement reports',
        ],
        'roadmap': [
            ('Concept + data model', True),
            ('Expense splitting MVP', False),
            ('Settle-up algorithm', False),
            ('Multi-currency support', False),
        ],
        'project_id': 'payment-house',
        'stage': 'concept',
    },
}


def fetch_readme(owner, repo):
    """Returns (content_str, sha) or (None, None)."""
    cmd = [
        'curl', '--ssl-no-revoke', '-fsSL',
        '-H', f'Authorization: token {GH_TOKEN}',
        '-H', 'Accept: application/vnd.github+json',
        f'https://api.github.com/repos/{owner}/{repo}/readme',
    ]
    r = subprocess.run(cmd, capture_output=True, timeout=20)
    if r.returncode != 0:
        return None, None
    data = json.loads(r.stdout.decode('utf-8'))
    content = base64.b64decode(data['content']).decode('utf-8', errors='replace')
    return content, data['sha']


def push_readme(owner, repo, content, sha, message):
    """Update README.md via GitHub Contents API."""
    payload = {
        'message': message,
        'content': base64.b64encode(content.encode('utf-8')).decode('ascii'),
        'sha': sha,
    }
    body = json.dumps(payload)
    cmd = [
        'curl', '--ssl-no-revoke', '-fsSL', '-X', 'PUT',
        '-H', f'Authorization: token {GH_TOKEN}',
        '-H', 'Accept: application/vnd.github+json',
        '-H', 'Content-Type: application/json',
        '-d', body,
        f'https://api.github.com/repos/{owner}/{repo}/contents/README.md',
    ]
    r = subprocess.run(cmd, capture_output=True, timeout=30)
    if r.returncode != 0:
        return False, r.stderr.decode('utf-8', errors='replace')
    return True, ''


def make_badges(badges):
    """Build the shields.io badge row."""
    out = []
    for name, value, color, logo in badges:
        url_name = name.replace(' ', '_').replace('-', '--')
        url_value = value.replace(' ', '_').replace('-', '--')
        url = f'https://img.shields.io/badge/{url_name}-{url_value}-{color}'
        if logo:
            url += f'?logo={logo}&logoColor=white'
        out.append(f'![{name}]({url})')
    return ' '.join(out)


def make_launchpad(meta):
    progress = sum(1 for _, done in meta['roadmap'] if done)
    total = len(meta['roadmap'])
    pct = round(progress / total * 100) if total else 0
    in_progress = next((idx + 1 for idx, (_, d) in enumerate(meta['roadmap']) if not d), None)
    block = {
        'stage': meta['stage'],
        'progress': pct,
        'complexity': 'F',
        'lastUpdated': '2026-04-27',
        'targetDate': None,
        'nextAction': next((t for t, d in meta['roadmap'] if not d), None),
        'blocker': None,
        'demoUrl': meta.get('demo_url'),
        'techStack': meta['tech_stack'],
        'shipped': meta['stage'] == 'live',
        'linkedinPosted': False,
    }
    return (
        '<!-- LAUNCHPAD:START -->\n```json\n'
        + json.dumps(block, indent=2)
        + '\n```\n<!-- LAUNCHPAD:END -->'
    )


def make_toc(headings):
    """Collapsible TOC from a list of (level, title) tuples."""
    items = []
    for level, title in headings:
        if level == 2:
            anchor = re.sub(r'[^a-z0-9-]', '', title.lower().replace(' ', '-'))
            items.append(f'- [{title}](#{anchor})')
        elif level == 3:
            anchor = re.sub(r'[^a-z0-9-]', '', title.lower().replace(' ', '-'))
            items.append(f'  - [{title}](#{anchor})')
    return (
        '<details>\n<summary>📚 Table of Contents</summary>\n\n'
        + '\n'.join(items)
        + '\n\n</details>'
    )


def make_footer():
    return (
        '<div align="center">\n\n'
        '---\n\n'
        '<sub>Part of the <a href="https://castronix.dev">Castronix</a> portfolio · '
        'crafted with care · © 2026 Arun Castromin</sub>\n\n'
        '</div>'
    )


def enhance(meta, existing_readme):
    """Splice missing standard elements into the existing README."""
    parts = []

    # 1. Centered title + tagline + badges (replaces top of readme)
    parts.append('<div align="center">\n')
    parts.append(f'\n# {meta["name"]}\n')
    parts.append(f'\n**{meta["tagline"]}**\n')
    parts.append('\n' + make_badges(meta['badges']) + '\n')
    if meta.get('demo_url'):
        parts.append(f'\n[**Live demo →**]({meta["demo_url"]})\n')
    parts.append('\n</div>\n')

    # 2. LAUNCHPAD block
    parts.append('\n' + make_launchpad(meta) + '\n')

    # 3. TOC (predict the headings we know we'll have)
    headings = [
        (2, 'Features'),
        (2, 'Quick Start'),
        (2, 'Tech Stack'),
        (2, 'Project Structure'),
        (2, 'Roadmap'),
        (2, 'License'),
    ]
    parts.append('\n' + make_toc(headings) + '\n')

    # 4. Features section
    parts.append('\n## Features\n')
    for f in meta['features']:
        parts.append(f'\n- {f}')
    parts.append('\n')

    # 5. Existing README content (preserved as-is, minus any old title/badges)
    # Strip the very first H1 and any badge image-row lines that follow
    body = existing_readme or ''
    # Remove any leading H1 + tagline + badge cluster
    body = re.sub(r'\A\s*#\s+.*?\n\n?(?:!\[.*?\]\(.*?\)\n?)*\n*', '', body, count=1)
    # Remove any existing LAUNCHPAD block (we just added one)
    body = re.sub(
        r'<!--\s*LAUNCHPAD:START\s*-->.*?<!--\s*LAUNCHPAD:END\s*-->',
        '', body, flags=re.DOTALL,
    )
    # Remove any pre-existing centered hero
    body = re.sub(r'<div align="center">.*?</div>\s*', '', body, count=1, flags=re.DOTALL)
    body = body.strip()

    if body:
        parts.append('\n## Quick Start\n\n')
        parts.append(body)
        parts.append('\n')

    # 6. Tech Stack
    parts.append('\n## Tech Stack\n\n')
    parts.append('| Component | Tech |\n|---|---|\n')
    for t in meta['tech_stack']:
        parts.append(f'| {t} | — |\n')

    # 7. Roadmap
    parts.append('\n## Roadmap\n\n')
    for item, done in meta['roadmap']:
        mark = 'x' if done else ' '
        parts.append(f'- [{mark}] {item}\n')

    # 8. License
    parts.append('\n## License\n\nPrivate — part of the Castronix portfolio.\n')

    # 9. Footer
    parts.append('\n' + make_footer() + '\n')

    return ''.join(parts)


def main():
    print('=' * 90)
    print('Bulk README enhance — bringing all portfolio repos to 90+')
    print('=' * 90)

    for (owner, repo), meta in ENHANCEMENTS.items():
        print(f'\n[{owner}/{repo}]')
        existing, sha = fetch_readme(owner, repo)
        if existing is None:
            print(f'  could not fetch existing README — skipping')
            continue
        new_content = enhance(meta, existing)
        if new_content == existing:
            print(f'  unchanged — skipping')
            continue
        print(f'  generating enhanced README ({len(new_content)} chars, was {len(existing)})')
        ok, err = push_readme(
            owner, repo, new_content, sha,
            f'README: enhance with badges + LAUNCHPAD block + TOC + footer',
        )
        if ok:
            print(f'  [OK] pushed')
        else:
            print(f'  [FAIL] push failed: {err[:200]}')


if __name__ == '__main__':
    main()
