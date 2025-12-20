import { Exercise, Level } from '@/types'

/**
 * Form tips organized by exercise and level
 * Each exercise has tips for different skill levels
 * Tips rotate - we pick based on session count or day
 */

interface FormTipSet {
  beginner: string[]
  novice: string[]
  intermediate: string[]
  advanced: string[]
}

export const FORM_TIPS: Partial<Record<Exercise, FormTipSet>> = {
  // Chest
  benchPress: {
    beginner: [
      'Grip slightly wider than shoulders. Keep wrists straight.',
      'Plant feet flat on floor. Squeeze shoulder blades together.',
      'Lower bar to mid-chest. Keep elbows at 45° angle.'
    ],
    novice: [
      'Retract scapula before unracking. Create slight arch.',
      'Touch chest lightly - no bouncing. Control the descent.',
      'Drive through feet as you press. Stay tight throughout.'
    ],
    intermediate: [
      'Leg drive: push feet into floor as you press up.',
      'Bar path: slight diagonal from chest to over shoulders.',
      'Pause at chest for 1 second to build strength off the bottom.'
    ],
    advanced: [
      'Optimize arch for your leverages. Work on weak points.',
      'Vary grip width periodically to target different fibers.',
      'Practice commands if competing: start, press, rack.'
    ]
  },
  inclineBench: {
    beginner: [
      'Set bench to 30-45°. Grip just outside shoulders.',
      'Lower to upper chest/collarbone area.',
      'Keep lower back in contact with bench.'
    ],
    novice: [
      'Slightly narrower grip than flat bench works well.',
      'Focus on squeezing chest at the top.',
      'Don\'t let elbows flare past 60°.'
    ],
    intermediate: [
      'Try different angles (15°, 30°, 45°) to find your sweet spot.',
      'Control eccentric for 2-3 seconds for more tension.',
      'Keep chest up - don\'t let shoulders roll forward.'
    ],
    advanced: [
      'Pre-exhaust with flyes before pressing for more chest activation.',
      'Partial reps at top for continuous tension.',
      'Vary between barbell and dumbbell for complete development.'
    ]
  },
  dumbbellPress: {
    beginner: [
      'Start with dumbbells on thighs, kick up one at a time.',
      'Lower until upper arms are parallel to floor.',
      'Press up and slightly inward - dumbbells almost touch at top.'
    ],
    novice: [
      'Rotate palms slightly inward (neutral grip) to protect shoulders.',
      'Keep elbows under wrists throughout the movement.',
      'Control the weight - don\'t let dumbbells drift outward.'
    ],
    intermediate: [
      'Squeeze chest hard at top. Hold for 1 second.',
      'Try slight incline for better upper chest activation.',
      'Go deeper than barbell allows for extra stretch.'
    ],
    advanced: [
      'Alternate arms for core stability challenge.',
      'Use floor press variation to work lockout strength.',
      'Try 1.5 reps: full rep + half rep = 1 rep.'
    ]
  },

  // Back
  deadlift: {
    beginner: [
      'Bar over mid-foot. Shins 1 inch from bar before lifting.',
      'Grip just outside knees. Keep arms straight like ropes.',
      'Push floor away rather than pulling bar up.'
    ],
    novice: [
      'Brace core hard - like bracing for a punch.',
      'Keep bar close to body - drag it up your shins/thighs.',
      'Lock out by squeezing glutes, not hyperextending back.'
    ],
    intermediate: [
      'Pull slack out of bar before lift. Hear the click.',
      'Lead with chest, not hips. Don\'t let hips shoot up first.',
      'Pause at knee height occasionally to build positional strength.'
    ],
    advanced: [
      'Optimize stance width and grip for your proportions.',
      'Work weak points: deficit, block pulls, pause reps.',
      'Breathing: big breath, brace, lift, exhale at top.'
    ]
  },
  barbellRow: {
    beginner: [
      'Hinge at hips to ~45°. Keep back flat, not rounded.',
      'Pull to lower chest/upper abs. Squeeze at top.',
      'Let arms fully extend at bottom for full stretch.'
    ],
    novice: [
      'Initiate with elbows, not hands. Think "elbow to hip pocket".',
      'Keep core tight - don\'t use momentum to swing weight.',
      'Grip hard - this activates more back muscles.'
    ],
    intermediate: [
      'Try different angles: more upright = upper back, more bent = lats.',
      'Use straps on heavy sets - back strength shouldn\'t be limited by grip.',
      'Pause at top for 1-2 seconds to eliminate momentum.'
    ],
    advanced: [
      'Experiment with underhand grip for different muscle emphasis.',
      'Meadows row variation for unique lat stretch.',
      'Control the negative - 3 seconds down builds size.'
    ]
  },
  latPulldown: {
    beginner: [
      'Grip 1.5x shoulder width. Lean back slightly.',
      'Pull to upper chest, not behind neck.',
      'Lead with elbows down and back.'
    ],
    novice: [
      'Squeeze shoulder blades together at bottom.',
      'Control the return - don\'t let weight yank you up.',
      'Keep chest up throughout the movement.'
    ],
    intermediate: [
      'Try close grip for lower lat emphasis.',
      'Pause at bottom, stretch at top for full range.',
      'Single-arm variation for better mind-muscle connection.'
    ],
    advanced: [
      'Behind-neck pulls with light weight for rear delt work.',
      'Kneeling pulldowns remove leg drive cheating.',
      'Straight-arm pulldown as pre-exhaust before rows.'
    ]
  },
  pullUps: {
    beginner: [
      'Dead hang start - full arm extension.',
      'Pull until chin clears bar.',
      'Use assisted machine or bands if needed.'
    ],
    novice: [
      'Initiate by depressing shoulder blades first.',
      'Drive elbows down and back, not just pulling with arms.',
      'Control the descent - don\'t just drop.'
    ],
    intermediate: [
      'Vary grip: wide for lats, narrow/neutral for biceps.',
      'Add pause at top for 2 seconds.',
      'Try L-sit pull-ups for core challenge.'
    ],
    advanced: [
      'Weighted pull-ups: belt or vest for progression.',
      'Archer pull-ups for unilateral strength.',
      'Muscle-ups for explosive power.'
    ]
  },

  // Shoulders
  shoulderPressBarbell: {
    beginner: [
      'Grip just outside shoulders. Bar starts at collarbone.',
      'Press straight up, move head back slightly then forward.',
      'Keep core tight - don\'t arch lower back.'
    ],
    novice: [
      'Squeeze glutes to protect lower back.',
      'Lock out fully at top - ears visible in front of arms.',
      'Control the descent - touch collarbone softly.'
    ],
    intermediate: [
      'Use leg drive for push press on last 1-2 reps.',
      'Try behind-neck press (light) for rear delt work.',
      'Pause at bottom to eliminate stretch reflex.'
    ],
    advanced: [
      'Z-press (seated on floor) for pure shoulder strength.',
      'Single-arm landmine press for shoulder health.',
      'Tempo work: 4 seconds down, 1 second up.'
    ]
  },
  shoulderPressDumbbell: {
    beginner: [
      'Start with dumbbells at shoulder height, palms forward.',
      'Press up and slightly inward.',
      'Keep elbows slightly forward of torso.'
    ],
    novice: [
      'Don\'t lock out aggressively - slight bend at top is fine.',
      'Control descent - 2 seconds down minimum.',
      'Try neutral grip (palms facing) for shoulder comfort.'
    ],
    intermediate: [
      'Arnold press for complete delt activation.',
      'Single-arm for core stability.',
      'Seated removes momentum - harder but purer.'
    ],
    advanced: [
      'Standing for more core and stability work.',
      'Alternate arms for time under tension.',
      '21s method: 7 bottom half, 7 top half, 7 full reps.'
    ]
  },
  sideLateralDumbbell: {
    beginner: [
      'Lead with elbows, not hands. Pinkies up at top.',
      'Stop at shoulder height. Don\'t swing.',
      'Light weight, high reps - this is an isolation move.'
    ],
    novice: [
      'Slight bend in elbows throughout.',
      'Control the negative - don\'t just drop the weights.',
      'Think "pouring water" at the top of the movement.'
    ],
    intermediate: [
      'Lean forward slightly for middle delt focus.',
      'Seated removes momentum cheating.',
      'Drop sets work great for lateral raises.'
    ],
    advanced: [
      'Cable laterals for constant tension.',
      'Behind-back cable laterals for peak contraction.',
      'Partial reps in top range after full ROM failure.'
    ]
  },

  // Legs
  squat: {
    beginner: [
      'Feet shoulder-width, toes slightly out (15-30°).',
      'Sit back and down - weight stays mid-foot.',
      'Knees track over toes - it\'s OK for them to pass toes.'
    ],
    novice: [
      'Brace core before descending. Big breath, hold it.',
      'Break at hips and knees simultaneously.',
      'Drive through whole foot, not just heels or toes.'
    ],
    intermediate: [
      'Pause at bottom for 1-2 seconds to build strength.',
      'Work on ankle mobility if heels lift.',
      'Try high bar vs low bar to find your best position.'
    ],
    advanced: [
      'Tempo squats: 3-4 seconds down builds muscle.',
      'Box squats to work out of the hole.',
      'Front squats for quad emphasis and core work.'
    ]
  },
  legPress: {
    beginner: [
      'Feet shoulder-width on platform.',
      'Lower until thighs are ~90° to torso.',
      'Don\'t lock knees at top - keep slight bend.'
    ],
    novice: [
      'Higher foot placement = more glutes/hams.',
      'Lower foot placement = more quads.',
      'Control the descent - don\'t bounce at bottom.'
    ],
    intermediate: [
      'Narrow stance for outer quad emphasis.',
      'Single leg for unilateral strength.',
      'Pause at bottom to eliminate stretch reflex.'
    ],
    advanced: [
      '1.5 reps: full rep + half rep from bottom.',
      'Drop sets: strip weight 3-4 times to failure.',
      'Slow negatives (5 seconds) for hypertrophy.'
    ]
  },
  romanianDeadlift: {
    beginner: [
      'Slight knee bend - maintain same bend throughout.',
      'Hinge at hips, push butt back.',
      'Stop when you feel hamstring stretch - don\'t round back.'
    ],
    novice: [
      'Keep bar close to legs - drag it down your thighs.',
      'Squeeze glutes at top to finish the rep.',
      'Look forward, not down - helps keep back flat.'
    ],
    intermediate: [
      'Single-leg RDL for balance and unilateral strength.',
      'Pause at bottom stretch for 2 seconds.',
      'Deficit RDL (stand on plate) for extra range.'
    ],
    advanced: [
      'Snatch-grip RDL for upper back and longer ROM.',
      'Banded RDL for accommodating resistance.',
      'Tempo: 4 seconds down, explosive up.'
    ]
  },
  legCurl: {
    beginner: [
      'Adjust pad to sit just above ankles.',
      'Curl all the way up, squeeze at top.',
      'Control the descent - don\'t let weight slam.'
    ],
    novice: [
      'Point toes toward shins (dorsiflexion) for better contraction.',
      'Don\'t lift hips off pad as you curl.',
      'Try different foot positions: neutral, toes in, toes out.'
    ],
    intermediate: [
      'Pause at peak contraction for 1-2 seconds.',
      'Negatives: 4-5 seconds on the way down.',
      'Single-leg to address imbalances.'
    ],
    advanced: [
      'Nordic curls for eccentric strength.',
      'Glute-ham raise for complete posterior chain.',
      'Partials in contracted position after failure.'
    ]
  },

  // Arms
  bicepCurlBarbell: {
    beginner: [
      'Elbows pinned to sides. Only forearms move.',
      'Full range: arms straight at bottom, squeeze at top.',
      'Control the descent - no swinging.'
    ],
    novice: [
      'Slight wrist curl at top for peak contraction.',
      'EZ-bar is easier on wrists if straight bar hurts.',
      'Lean slightly forward at start, back at top.'
    ],
    intermediate: [
      'Cheat curls on last 1-2 reps with controlled negative.',
      'Try 21s: 7 bottom half, 7 top half, 7 full.',
      'Pause at top for 2 seconds.'
    ],
    advanced: [
      'Drag curls keep tension on bicep throughout.',
      'Heavy negatives with partner assist on concentric.',
      'Preacher curls to eliminate momentum completely.'
    ]
  },
  bicepCurlDumbbell: {
    beginner: [
      'Start with palms facing thighs, rotate as you curl.',
      'Keep elbows stationary at your sides.',
      'Full extension at bottom, full contraction at top.'
    ],
    novice: [
      'Alternate arms for better focus on each rep.',
      'Incline curls for better stretch.',
      'Hammer curls for brachialis and forearms.'
    ],
    intermediate: [
      'Concentration curls for peak.',
      'Spider curls for constant tension.',
      'Slow negatives: 4 seconds down.'
    ],
    advanced: [
      'Cross-body hammer for brachialis emphasis.',
      'Mechanical drop set: incline → seated → standing.',
      'Isometric holds at 90° between sets.'
    ]
  },
  tricepPushdown: {
    beginner: [
      'Elbows pinned to sides throughout.',
      'Push all the way down until arms are straight.',
      'Control the return - don\'t let cable pull you up.'
    ],
    novice: [
      'Try different attachments: rope, V-bar, straight bar.',
      'Lean forward slightly for better contraction.',
      'Rope: spread handles apart at bottom for extra squeeze.'
    ],
    intermediate: [
      'Overhead extension for long head emphasis.',
      'Single-arm for focus and balance.',
      'Pause at full extension for 1 second.'
    ],
    advanced: [
      'Reverse grip for lateral head focus.',
      'Behind-back cable pushdown for peak contraction.',
      'Drop sets with 3-4 weight reductions.'
    ]
  },
  skullCrushers: {
    beginner: [
      'Lower bar to forehead or just behind head.',
      'Keep elbows pointing at ceiling, not flaring.',
      'Arms vertical at the start position.'
    ],
    novice: [
      'EZ-bar is more comfortable than straight bar.',
      'Try lowering behind head for more stretch.',
      'Keep upper arms stationary throughout.'
    ],
    intermediate: [
      'Decline bench for constant tension.',
      'Pause at bottom stretch for 1-2 seconds.',
      'Floor skulls eliminate bounce and build strength.'
    ],
    advanced: [
      'Rolling skulls: extend toward face, roll back, press.',
      'Dumbbell variation for individual arm work.',
      'Close-grip bench as compound finisher.'
    ]
  }
}

/**
 * Get a form tip for an exercise based on user's level
 * Rotates tips based on day of year
 */
export function getFormTip(exerciseId: Exercise, level: Level): string | null {
  const tips = FORM_TIPS[exerciseId]
  if (!tips) return null

  const levelTips = tips[level]
  if (!levelTips || levelTips.length === 0) return null

  // Rotate based on day of year so user sees different tips
  const dayOfYear = Math.floor(
    (Date.now() - new Date(new Date().getFullYear(), 0, 0).getTime()) / (1000 * 60 * 60 * 24)
  )
  const tipIndex = dayOfYear % levelTips.length

  return levelTips[tipIndex]
}

/**
 * Get all tips for an exercise at a given level
 */
export function getAllFormTips(exerciseId: Exercise, level: Level): string[] {
  const tips = FORM_TIPS[exerciseId]
  if (!tips) return []
  return tips[level] || []
}
