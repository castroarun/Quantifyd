"""Audit README quality across all repos referenced in the portfolio's
projects.json. Scoring matches /readme command's 100-point scale."""
import json
import re
import subprocess
from pathlib import Path

import os
GH_TOKEN = os.environ.get('GITHUB_TOKEN', '')
if not GH_TOKEN:
    print('Set GITHUB_TOKEN env var (export GITHUB_TOKEN=ghp_...) then re-run.')
    raise SystemExit(1)

PROJECTS_JSON = Path(r'C:\Users\Castro\Documents\Projects\castronix-portfolio\data\projects.json')
projects = json.loads(PROJECTS_JSON.read_text(encoding='utf-8'))['projects']

# Build the unique repo list
repos = []
seen = set()
for p in projects:
    g = (p.get('links') or {}).get('github') or ''
    m = re.match(r'https?://github\.com/([^/]+)/([^/]+)', g)
    if not m:
        continue
    owner, repo = m.group(1), m.group(2)
    repo = repo.split('/')[0]  # strip /blob/...
    key = (owner, repo)
    if key in seen:
        continue
    seen.add(key)
    repos.append({'project_id': p['id'], 'owner': owner, 'repo': repo, 'gh': g})


def fetch_readme(owner, repo):
    """Fetch raw README.md content from GitHub API."""
    cmd = [
        'curl', '--ssl-no-revoke', '-fsSL',
        '-H', f'Authorization: token {GH_TOKEN}',
        '-H', 'Accept: application/vnd.github.raw',
        f'https://api.github.com/repos/{owner}/{repo}/readme',
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=20)
        if r.returncode != 0:
            return None
        return r.stdout.decode('utf-8', errors='replace')
    except Exception:
        return None


def score(readme):
    """Return (score:int, breakdown:dict) for a README string."""
    if not readme:
        return 0, {'README exists': (0, 10, 'no README.md')}

    breakdown = {}

    # README exists
    breakdown['README exists'] = (10, 10, 'OK')

    # Title/Hero (looking for # Title near top + image OR <div align="center">)
    has_title = bool(re.search(r'^#\s+\S+', readme, re.MULTILINE))
    has_centered = '<div align="center">' in readme[:1500] or '<p align="center">' in readme[:1500]
    has_logo = bool(re.search(r'!\[.*\]\(.*(logo|hero|banner)', readme, re.IGNORECASE))
    if has_title and (has_centered or has_logo):
        breakdown['Title/Hero'] = (10, 10, 'centered + title')
    elif has_title:
        breakdown['Title/Hero'] = (5, 10, 'title only, no centered/logo')
    else:
        breakdown['Title/Hero'] = (0, 10, 'no title')

    # Description (one-sentence between title and next ##)
    desc_match = re.search(r'^#\s+\S.*\n+([^#].*?)(?:\n\n|\n#)', readme, re.MULTILINE | re.DOTALL)
    if desc_match and len(desc_match.group(1).strip()) > 20:
        breakdown['Description'] = (10, 10, 'present')
    else:
        breakdown['Description'] = (0, 10, 'missing or very short')

    # Badges (shields.io image links count)
    badges = re.findall(r'!\[.*?\]\(https?://(?:img\.)?shields\.io/[^)]+\)', readme)
    if 4 <= len(badges) <= 7:
        breakdown['Badges'] = (10, 10, f'{len(badges)} badges')
    elif 1 <= len(badges) < 4:
        breakdown['Badges'] = (5, 10, f'only {len(badges)} (target 4-7)')
    elif len(badges) > 7:
        breakdown['Badges'] = (7, 10, f'{len(badges)} (slightly over target)')
    else:
        breakdown['Badges'] = (0, 10, 'no badges')

    # LAUNCHPAD block
    if '<!-- LAUNCHPAD:START -->' in readme and '<!-- LAUNCHPAD:END -->' in readme:
        breakdown['LAUNCHPAD block'] = (15, 15, 'present')
    elif 'LAUNCHPAD' in readme.upper():
        breakdown['LAUNCHPAD block'] = (8, 15, 'mention only, no block')
    else:
        breakdown['LAUNCHPAD block'] = (0, 15, 'MISSING')

    # Collapsible TOC
    has_toc = '<details>' in readme and '<summary>' in readme and 'table of contents' in readme.lower()
    has_toc = has_toc or ('## Table of Contents' in readme or '## TOC' in readme)
    if has_toc:
        breakdown['Collapsible TOC'] = (10, 10, 'present')
    else:
        breakdown['Collapsible TOC'] = (0, 10, 'missing')

    # Features section
    has_features = bool(re.search(r'##\s+(?:🌟\s*)?Features?\b', readme, re.IGNORECASE))
    if has_features:
        breakdown['Features section'] = (10, 10, 'present')
    else:
        breakdown['Features section'] = (0, 10, 'missing')

    # Quick Start
    has_qs = bool(re.search(
        r'##\s+(?:🚀\s*)?(?:Quick\s*Start|Getting\s*Started|Installation|Setup|Usage)\b',
        readme, re.IGNORECASE,
    ))
    if has_qs:
        breakdown['Quick Start'] = (10, 10, 'present')
    else:
        breakdown['Quick Start'] = (0, 10, 'missing')

    # Project Structure (folder tree with ├── or └──)
    has_struct = '├──' in readme or '└──' in readme or bool(
        re.search(r'##\s+(?:📁\s*)?(?:Project\s+Structure|Folder\s+Structure|Structure)\b', readme, re.IGNORECASE),
    )
    if has_struct:
        breakdown['Project Structure'] = (5, 5, 'present')
    else:
        breakdown['Project Structure'] = (0, 5, 'missing')

    # Roadmap (## Roadmap with - [ ] checkboxes)
    has_roadmap = bool(re.search(r'##\s+(?:🗺?\s*)?Roadmap\b', readme, re.IGNORECASE))
    has_checkboxes = bool(re.search(r'-\s*\[\s*[xX ]?\s*\]', readme))
    if has_roadmap and has_checkboxes:
        breakdown['Roadmap'] = (5, 5, 'with checkboxes')
    elif has_roadmap:
        breakdown['Roadmap'] = (3, 5, 'header only, no checkboxes')
    else:
        breakdown['Roadmap'] = (0, 5, 'missing')

    # Footer (last paragraph mentions author or castronix or </div align="center">)
    tail = readme[-1200:]
    has_footer = (
        '<div align="center">' in tail
        or 'castronix' in tail.lower()
        or 'made with' in tail.lower()
        or 'designed' in tail.lower() and 'shipped' in tail.lower()
    )
    if has_footer:
        breakdown['Footer'] = (5, 5, 'present')
    else:
        breakdown['Footer'] = (0, 5, 'missing')

    total = sum(v[0] for v in breakdown.values())
    return total, breakdown


def status_for(score):
    if score >= 90:
        return '[OK] Healthy'
    if score >= 70:
        return '[~]  Okay'
    if score >= 50:
        return '[!]  Needs Work'
    return '[X]  Critical'


print(f'Auditing {len(repos)} repos...\n')

results = []
for r in repos:
    rd = fetch_readme(r['owner'], r['repo'])
    s, breakdown = score(rd or '')
    r['score'] = s
    r['breakdown'] = breakdown
    r['readme_size'] = len(rd or '')
    results.append(r)

results.sort(key=lambda x: x['score'])

# Summary
print('=' * 92)
print(f'{"Project":24s} {"Repo":30s} {"Score":>7s}  {"Status"}')
print('=' * 92)
for r in results:
    print(f'{r["project_id"]:24s} {r["repo"]:30s} {r["score"]:>5}/100  {status_for(r["score"])}')

# Detail per repo
print('\n\nDETAIL per repo (failing checks only)')
print('=' * 92)
for r in results:
    fails = [(name, got, max_, note) for name, (got, max_, note) in r['breakdown'].items() if got < max_]
    if not fails:
        continue
    print(f'\n>> {r["project_id"]:24s} ({r["repo"]})  - {r["score"]}/100')
    for name, got, max_, note in fails:
        print(f'    [X] {name:22s} {got:>3}/{max_:<3}  {note}')
