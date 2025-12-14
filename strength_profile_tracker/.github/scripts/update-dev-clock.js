/**
 * Dev Clock - Automatic Time Tracker
 *
 * Calculates development time from git commit history.
 * Sessions are grouped by 2-hour gaps between commits.
 * Phases are detected from commit message prefixes.
 *
 * Reusable across any project - just copy .github folder.
 */

const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');

// Configuration
const CONFIG = {
  sessionGapMinutes: 120, // New session after 2hr gap
  maxSessionMinutes: 480, // Cap single session at 8hrs
  minSessionMinutes: 5,   // Ignore sessions < 5min
  firstCommitBufferMinutes: 30, // Assume work started 30min before first commit
  devClockPath: 'strength_profile_tracker/docs/DEV-CLOCK.md',

  // Project path filter - only count commits touching files in this path
  // Set to null or '' to count all commits in repo
  projectPath: 'strength_profile_tracker',

  // Phase detection from commit prefixes
  phases: {
    'Design & Planning': ['design', 'plan', 'rfc', 'spec', 'architecture'],
    'Documentation': ['docs', 'readme', 'doc:', 'documentation'],
    'Building': ['feat', 'feature', 'add', 'implement', 'create', 'build', 'ui', 'component'],
    'Debugging': ['fix', 'bug', 'hotfix', 'patch', 'debug', 'resolve'],
    'Testing': ['test', 'spec', 'e2e', 'unit', 'coverage'],
    'Shipping': ['deploy', 'release', 'version', 'publish', 'ci', 'cd', 'build:', 'chore']
  }
};

function getCommitHistory() {
  try {
    // If projectPath is set, only get commits that touched files in that path
    const pathFilter = CONFIG.projectPath ? ` -- "${CONFIG.projectPath}"` : '';
    const log = execSync(
      `git log --format="%H|%aI|%s" --reverse${pathFilter}`,
      { encoding: 'utf-8', maxBuffer: 10 * 1024 * 1024 }
    );

    return log.trim().split('\n').filter(Boolean).map(line => {
      const [hash, date, ...messageParts] = line.split('|');
      return {
        hash,
        date: new Date(date),
        message: messageParts.join('|')
      };
    });
  } catch (error) {
    console.error('Error getting commit history:', error.message);
    return [];
  }
}

function detectPhase(message) {
  const lowerMessage = message.toLowerCase();

  for (const [phase, keywords] of Object.entries(CONFIG.phases)) {
    if (keywords.some(kw => lowerMessage.includes(kw))) {
      return phase;
    }
  }

  return 'Building'; // Default phase
}

function parseStartedTime(message, commitDate) {
  // Parse [started:8am] or [started:14:30] from commit message
  const match = message.match(/\[started:(\d{1,2}):?(\d{2})?\s*(am|pm)?\]/i);
  if (!match) return null;

  let hours = parseInt(match[1]);
  const minutes = parseInt(match[2] || '0');
  const ampm = match[3]?.toLowerCase();

  if (ampm === 'pm' && hours < 12) hours += 12;
  if (ampm === 'am' && hours === 12) hours = 0;

  const startTime = new Date(commitDate);
  startTime.setHours(hours, minutes, 0, 0);

  // If parsed time is after commit time, assume previous day
  if (startTime > commitDate) {
    startTime.setDate(startTime.getDate() - 1);
  }

  return startTime;
}

function calculateSessions(commits) {
  if (commits.length === 0) return [];

  const sessions = [];
  let currentSession = {
    start: commits[0].date,
    end: commits[0].date,
    commits: [commits[0]],
    phases: {},
    explicitStart: parseStartedTime(commits[0].message, commits[0].date)
  };

  for (let i = 1; i < commits.length; i++) {
    const commit = commits[i];
    const prevCommit = commits[i - 1];
    const gapMinutes = (commit.date - prevCommit.date) / (1000 * 60);

    if (gapMinutes > CONFIG.sessionGapMinutes) {
      // End current session, start new one
      sessions.push(currentSession);
      currentSession = {
        start: commit.date,
        end: commit.date,
        commits: [commit],
        phases: {},
        explicitStart: parseStartedTime(commit.message, commit.date)
      };
    } else {
      currentSession.end = commit.date;
      currentSession.commits.push(commit);
      // Check if this commit has explicit start time
      const explicitStart = parseStartedTime(commit.message, commit.date);
      if (explicitStart && (!currentSession.explicitStart || explicitStart < currentSession.explicitStart)) {
        currentSession.explicitStart = explicitStart;
      }
    }
  }

  sessions.push(currentSession); // Don't forget last session

  // Calculate duration and phases for each session
  return sessions.map(session => {
    // Use explicit start time if provided, otherwise apply buffer
    let effectiveStart = session.explicitStart ||
      new Date(session.start.getTime() - CONFIG.firstCommitBufferMinutes * 60 * 1000);

    let durationMinutes = (session.end - effectiveStart) / (1000 * 60);

    // Add 15 min buffer for single-commit sessions (minimum work time)
    if (session.commits.length === 1 && !session.explicitStart) {
      durationMinutes = Math.max(durationMinutes, 15);
    }

    // Add 10 min buffer for context switching at end
    durationMinutes += 10;

    // Apply caps
    durationMinutes = Math.min(durationMinutes, CONFIG.maxSessionMinutes);

    // Skip tiny sessions
    if (durationMinutes < CONFIG.minSessionMinutes) {
      return null;
    }

    // Count phases
    session.commits.forEach(commit => {
      const phase = detectPhase(commit.message);
      session.phases[phase] = (session.phases[phase] || 0) + 1;
    });

    // Determine primary phase
    const primaryPhase = Object.entries(session.phases)
      .sort((a, b) => b[1] - a[1])[0]?.[0] || 'Building';

    return {
      date: session.start.toISOString().split('T')[0],
      start: session.start,
      end: session.end,
      durationMinutes,
      durationHours: Math.round(durationMinutes / 60 * 100) / 100,
      primaryPhase,
      commitCount: session.commits.length
    };
  }).filter(Boolean);
}

function aggregateByPhase(sessions) {
  const phaseStats = {};

  for (const phase of Object.keys(CONFIG.phases)) {
    phaseStats[phase] = { hours: 0, sessions: 0, commits: 0 };
  }

  sessions.forEach(session => {
    const phase = session.primaryPhase;
    if (phaseStats[phase]) {
      phaseStats[phase].hours += session.durationHours;
      phaseStats[phase].sessions += 1;
      phaseStats[phase].commits += session.commitCount;
    }
  });

  // Round hours
  for (const phase of Object.keys(phaseStats)) {
    phaseStats[phase].hours = Math.round(phaseStats[phase].hours * 100) / 100;
  }

  return phaseStats;
}

function aggregateByDay(sessions) {
  const dailyStats = {};

  sessions.forEach(session => {
    if (!dailyStats[session.date]) {
      dailyStats[session.date] = { hours: 0, commits: 0, phases: {} };
    }
    dailyStats[session.date].hours += session.durationHours;
    dailyStats[session.date].commits += session.commitCount;
    dailyStats[session.date].phases[session.primaryPhase] =
      (dailyStats[session.date].phases[session.primaryPhase] || 0) + session.durationHours;
  });

  return dailyStats;
}

function generateMarkdown(sessions, phaseStats, dailyStats) {
  const totalHours = sessions.reduce((sum, s) => sum + s.durationHours, 0);
  const totalCommits = sessions.reduce((sum, s) => sum + s.commitCount, 0);
  const startDate = sessions[0]?.date || 'N/A';
  const lastDate = sessions[sessions.length - 1]?.date || 'N/A';

  const phaseColors = {
    'Design & Planning': '#3B82F6',
    'Documentation': '#10B981',
    'Building': '#F59E0B',
    'Debugging': '#EF4444',
    'Testing': '#8B5CF6',
    'Shipping': '#EC4899'
  };

  const estimatedHours = {
    'Design & Planning': 2.5,
    'Documentation': 1.5,
    'Building': 10,
    'Debugging': 4,
    'Testing': 2.5,
    'Shipping': 1.5
  };

  const projectName = CONFIG.projectPath
    ? CONFIG.projectPath.split('/').pop().replace(/-|_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())
    : 'Project';

  let md = `# ${projectName} - DEV-CLOCK

Time tracker for development phases. **Auto-updated from git commits.**
${CONFIG.projectPath ? `\n> Project: \`${CONFIG.projectPath}\`` : ''}

> Last updated: ${new Date().toISOString().split('T')[0]} | Total: **${Math.round(totalHours * 10) / 10} hours** | ${totalCommits} commits

---

## Summary Statistics

| Phase | Estimated | Actual | Progress |
|-------|-----------|--------|----------|
`;

  for (const [phase, stats] of Object.entries(phaseStats)) {
    const estimated = estimatedHours[phase] || 0;
    const percentage = estimated > 0 ? Math.round((stats.hours / estimated) * 100) : 0;
    const bar = percentage > 0 ? `${'█'.repeat(Math.min(Math.round(percentage / 10), 10))}${'░'.repeat(Math.max(10 - Math.round(percentage / 10), 0))}` : '░░░░░░░░░░';
    md += `| ${phase} | ${estimated}h | ${stats.hours}h | ${bar} ${percentage}% |\n`;
  }

  const totalEstimated = Object.values(estimatedHours).reduce((a, b) => a + b, 0);
  const totalPercentage = Math.round((totalHours / totalEstimated) * 100);
  md += `| **Total** | **${totalEstimated}h** | **${Math.round(totalHours * 10) / 10}h** | **${totalPercentage}%** |\n`;

  md += `
---

## Daily Log

| Date | Hours | Commits | Primary Phase |
|------|-------|---------|---------------|
`;

  const sortedDays = Object.entries(dailyStats).sort((a, b) => b[0].localeCompare(a[0]));
  for (const [date, stats] of sortedDays.slice(0, 30)) { // Last 30 days
    const primaryPhase = Object.entries(stats.phases).sort((a, b) => b[1] - a[1])[0]?.[0] || '-';
    md += `| ${date} | ${Math.round(stats.hours * 10) / 10}h | ${stats.commits} | ${primaryPhase} |\n`;
  }

  md += `
---

## Visualization Data (JSON)

\`\`\`json
${JSON.stringify({
  project: projectName,
  totalHours: Math.round(totalHours * 100) / 100,
  totalCommits,
  startDate,
  lastUpdate: new Date().toISOString(),
  phases: Object.entries(phaseStats).map(([name, stats]) => ({
    name,
    estimatedHours: estimatedHours[name] || 0,
    actualHours: stats.hours,
    color: phaseColors[name] || '#6B7280',
    commits: stats.commits,
    sessions: stats.sessions
  }))
}, null, 2)}
\`\`\`

---

## How It Works

- **Automatic tracking** from git commit timestamps
- **Session detection**: commits within 2hr gaps = same session
- **Phase detection**: parsed from commit message prefixes (feat:, fix:, docs:, etc.)
- **Updates on every push** via GitHub Actions

**Started:** ${startDate}
**Status:** ${totalHours > 0 ? 'In Development' : 'Pre-Development'}
`;

  return md;
}

function main() {
  console.log('📊 Dev Clock - Calculating development time...\n');

  if (CONFIG.projectPath) {
    console.log(`Project filter: ${CONFIG.projectPath}`);
  }

  const commits = getCommitHistory();
  console.log(`Found ${commits.length} commits${CONFIG.projectPath ? ` touching ${CONFIG.projectPath}` : ''}`);

  if (commits.length === 0) {
    console.log('No commits found. Skipping update.');
    return;
  }

  const sessions = calculateSessions(commits);
  console.log(`Detected ${sessions.length} work sessions`);

  const phaseStats = aggregateByPhase(sessions);
  const dailyStats = aggregateByDay(sessions);

  const totalHours = sessions.reduce((sum, s) => sum + s.durationHours, 0);
  console.log(`Total development time: ${Math.round(totalHours * 10) / 10} hours\n`);

  console.log('Phase breakdown:');
  for (const [phase, stats] of Object.entries(phaseStats)) {
    if (stats.hours > 0) {
      console.log(`  ${phase}: ${stats.hours}h (${stats.commits} commits)`);
    }
  }

  const markdown = generateMarkdown(sessions, phaseStats, dailyStats);

  // Ensure directory exists
  const dir = path.dirname(CONFIG.devClockPath);
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true });
  }

  fs.writeFileSync(CONFIG.devClockPath, markdown);
  console.log(`\n✅ Updated ${CONFIG.devClockPath}`);
}

main();
