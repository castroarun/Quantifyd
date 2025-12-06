import { Quote } from '@/types'

export const QUOTES: Quote[] = [
  // Motivation quotes
  { id: 1, text: "The only bad workout is the one that didn't happen.", author: "Unknown", category: 'motivation' },
  { id: 2, text: "Strength does not come from physical capacity. It comes from an indomitable will.", author: "Mahatma Gandhi", category: 'motivation' },
  { id: 3, text: "The pain you feel today will be the strength you feel tomorrow.", author: "Arnold Schwarzenegger", category: 'motivation' },
  { id: 4, text: "Success usually comes to those who are too busy to be looking for it.", author: "Henry David Thoreau", category: 'motivation' },
  { id: 5, text: "The body achieves what the mind believes.", author: "Napoleon Hill", category: 'motivation' },
  { id: 6, text: "Don't limit your challenges. Challenge your limits.", author: "Unknown", category: 'motivation' },
  { id: 7, text: "The clock is ticking. Are you becoming the person you want to be?", author: "Greg Plitt", category: 'motivation' },
  { id: 8, text: "Your body can stand almost anything. It's your mind that you have to convince.", author: "Unknown", category: 'motivation' },
  { id: 9, text: "The harder you work, the luckier you get.", author: "Gary Player", category: 'motivation' },
  { id: 10, text: "Wake up with determination. Go to bed with satisfaction.", author: "Unknown", category: 'motivation' },
  { id: 11, text: "No pain, no gain. Shut up and train.", author: "Unknown", category: 'motivation' },
  { id: 12, text: "Champions aren't made in gyms. Champions are made from something deep inside them.", author: "Muhammad Ali", category: 'motivation' },
  { id: 13, text: "The resistance that you fight physically in the gym and the resistance that you fight in life can only build a strong character.", author: "Arnold Schwarzenegger", category: 'motivation' },
  { id: 14, text: "Motivation is what gets you started. Habit is what keeps you going.", author: "Jim Ryun", category: 'motivation' },
  { id: 15, text: "The successful warrior is the average man, with laser-like focus.", author: "Bruce Lee", category: 'motivation' },
  { id: 16, text: "You don't have to be great to start, but you have to start to be great.", author: "Zig Ziglar", category: 'motivation' },
  { id: 17, text: "The only way to define your limits is by going beyond them.", author: "Arthur C. Clarke", category: 'motivation' },
  { id: 18, text: "Fall seven times, stand up eight.", author: "Japanese Proverb", category: 'motivation' },
  { id: 19, text: "It's not about having time. It's about making time.", author: "Unknown", category: 'motivation' },
  { id: 20, text: "The difference between try and triumph is just a little umph!", author: "Marvin Phillips", category: 'motivation' },
  { id: 21, text: "What seems impossible today will one day become your warm-up.", author: "Unknown", category: 'motivation' },
  { id: 22, text: "Rome wasn't built in a day, but they worked on it every single day.", author: "Unknown", category: 'motivation' },
  { id: 23, text: "Today I will do what others won't, so tomorrow I can accomplish what others can't.", author: "Jerry Rice", category: 'motivation' },
  { id: 24, text: "Once you learn to quit, it becomes a habit.", author: "Vince Lombardi", category: 'motivation' },
  { id: 25, text: "I don't count my sit-ups. I only start counting when it starts hurting because they're the only ones that count.", author: "Muhammad Ali", category: 'motivation' },
  { id: 26, text: "The iron never lies to you. You can walk outside and listen to all kinds of talk. But the iron will always kick you the real deal.", author: "Henry Rollins", category: 'motivation' },
  { id: 27, text: "Discipline is doing what you hate to do, but nonetheless doing it like you love it.", author: "Mike Tyson", category: 'motivation' },
  { id: 28, text: "Whether you think you can, or you think you can't – you're right.", author: "Henry Ford", category: 'motivation' },
  { id: 29, text: "If you want something you've never had, you must be willing to do something you've never done.", author: "Thomas Jefferson", category: 'motivation' },
  { id: 30, text: "The last three or four reps is what makes the muscle grow.", author: "Arnold Schwarzenegger", category: 'motivation' },
  { id: 31, text: "Pain is temporary. Quitting lasts forever.", author: "Lance Armstrong", category: 'motivation' },
  { id: 32, text: "Strength does not come from winning. Your struggles develop your strengths.", author: "Arnold Schwarzenegger", category: 'motivation' },
  { id: 33, text: "Training gives us an outlet for suppressed energies created by stress and thus tones the spirit just as exercise conditions the body.", author: "Arnold Schwarzenegger", category: 'motivation' },
  { id: 34, text: "A year from now you will wish you had started today.", author: "Karen Lamb", category: 'motivation' },
  { id: 35, text: "Sweat is just fat crying.", author: "Unknown", category: 'motivation' },

  // Science-based quotes
  { id: 36, text: "Muscle tissue burns 7-10 calories per pound per day, versus 2-3 calories for fat.", author: "Exercise Physiology Research", category: 'science' },
  { id: 37, text: "Resistance training increases bone density by stimulating osteoblast activity.", author: "Journal of Bone & Mineral Research", category: 'science' },
  { id: 38, text: "Progressive overload is the gradual increase of stress placed upon the body during training.", author: "Thomas Delorme", category: 'science' },
  { id: 39, text: "Strength training can increase your metabolic rate by up to 15% for hours after exercise.", author: "Sports Medicine Research", category: 'science' },
  { id: 40, text: "The principle of specificity states that training should be specific to the desired outcome.", author: "Exercise Science", category: 'science' },
  { id: 41, text: "Muscle protein synthesis remains elevated for 24-48 hours after resistance training.", author: "Journal of Physiology", category: 'science' },
  { id: 42, text: "Compound movements recruit multiple muscle groups and promote greater hormonal response.", author: "Strength & Conditioning Research", category: 'science' },
  { id: 43, text: "Rest between sets allows for ATP regeneration, crucial for subsequent performance.", author: "Exercise Biochemistry", category: 'science' },
  { id: 44, text: "The eccentric phase of a lift causes more muscle damage and subsequent growth.", author: "Journal of Applied Physiology", category: 'science' },
  { id: 45, text: "Neural adaptations account for most strength gains in the first 8 weeks of training.", author: "Neuroscience Research", category: 'science' },
  { id: 46, text: "Muscle hypertrophy requires a combination of mechanical tension, metabolic stress, and muscle damage.", author: "Brad Schoenfeld", category: 'science' },
  { id: 47, text: "The mind-muscle connection can increase muscle activation by up to 22%.", author: "European Journal of Sport Science", category: 'science' },
  { id: 48, text: "Proper breathing technique helps maintain intra-abdominal pressure for spinal stability.", author: "Spine Biomechanics Research", category: 'science' },
  { id: 49, text: "Sleep is when most muscle repair and growth hormone release occurs.", author: "Sleep Medicine Research", category: 'science' },
  { id: 50, text: "Protein intake of 1.6-2.2g per kg bodyweight optimizes muscle protein synthesis.", author: "International Society of Sports Nutrition", category: 'science' },
  { id: 51, text: "Training to failure is not necessary for muscle growth in experienced lifters.", author: "Journal of Strength & Conditioning", category: 'science' },
  { id: 52, text: "Deload weeks help manage accumulated fatigue and prevent overtraining.", author: "Periodization Research", category: 'science' },
  { id: 53, text: "Time under tension between 40-70 seconds per set optimizes hypertrophy.", author: "Muscle & Fitness Research", category: 'science' },
  { id: 54, text: "The repeated bout effect reduces muscle damage with consistent training.", author: "Exercise Immunology Review", category: 'science' },
  { id: 55, text: "Full range of motion training produces greater strength gains than partial reps.", author: "Journal of Strength Research", category: 'science' },
  { id: 56, text: "Resistance training improves insulin sensitivity and glucose metabolism.", author: "Diabetes Care Journal", category: 'science' },
  { id: 57, text: "Exercise increases brain-derived neurotrophic factor (BDNF) for cognitive benefits.", author: "Neurobiology of Aging", category: 'science' },
  { id: 58, text: "The rate of force development is trainable and crucial for athletic performance.", author: "Sports Biomechanics", category: 'science' },
  { id: 59, text: "Proper warm-up increases muscle temperature and reduces injury risk by up to 50%.", author: "British Journal of Sports Medicine", category: 'science' },
  { id: 60, text: "Tracking your workouts leads to 30% better results on average.", author: "Behavioral Psychology Research", category: 'science' },
  { id: 61, text: "Bilateral deficit allows single-limb exercises to produce more total force.", author: "Biomechanics Research", category: 'science' },
  { id: 62, text: "Strength training twice per week maintains muscle mass in most individuals.", author: "American College of Sports Medicine", category: 'science' },
  { id: 63, text: "The stretch-shortening cycle enhances power output in plyometric movements.", author: "Journal of Applied Biomechanics", category: 'science' },
  { id: 64, text: "Consistent training can increase muscle fiber cross-sectional area by 20-35%.", author: "Muscle Physiology Research", category: 'science' },
  { id: 65, text: "Post-workout nutrition within 2 hours optimizes muscle recovery.", author: "Sports Nutrition Guidelines", category: 'science' },

  // Benefit quotes
  { id: 66, text: "Strength training reduces symptoms of anxiety and depression by up to 50%.", author: "Mental Health Research", category: 'benefit' },
  { id: 67, text: "Regular exercise adds an average of 7 years to your life expectancy.", author: "Longevity Studies", category: 'benefit' },
  { id: 68, text: "Weight training improves sleep quality and reduces time to fall asleep.", author: "Sleep Research", category: 'benefit' },
  { id: 69, text: "Stronger muscles protect joints and reduce injury risk in daily activities.", author: "Physical Therapy Research", category: 'benefit' },
  { id: 70, text: "Exercise is the most effective way to improve energy levels naturally.", author: "Fatigue Research", category: 'benefit' },
  { id: 71, text: "Strength training improves posture and reduces chronic back pain.", author: "Spine Health Journal", category: 'benefit' },
  { id: 72, text: "Regular lifters have 40% lower risk of heart disease.", author: "Cardiovascular Research", category: 'benefit' },
  { id: 73, text: "Building muscle increases confidence and self-esteem significantly.", author: "Psychology of Exercise", category: 'benefit' },
  { id: 74, text: "Weight training preserves cognitive function as you age.", author: "Gerontology Research", category: 'benefit' },
  { id: 75, text: "Strong muscles improve balance and reduce fall risk by 40% in older adults.", author: "Aging Research", category: 'benefit' },
  { id: 76, text: "Exercise releases endorphins, your body's natural mood elevators.", author: "Neurochemistry Research", category: 'benefit' },
  { id: 77, text: "Resistance training helps maintain healthy blood pressure levels.", author: "Hypertension Journal", category: 'benefit' },
  { id: 78, text: "Building muscle increases your metabolic rate even at rest.", author: "Metabolism Research", category: 'benefit' },
  { id: 79, text: "Strong core muscles reduce lower back pain and improve daily function.", author: "Rehabilitation Science", category: 'benefit' },
  { id: 80, text: "Exercise improves immune function and reduces sick days.", author: "Immunology Research", category: 'benefit' },
  { id: 81, text: "Weight training increases bone density and prevents osteoporosis.", author: "Bone Health Research", category: 'benefit' },
  { id: 82, text: "Regular exercise improves skin health through increased blood flow.", author: "Dermatology Research", category: 'benefit' },
  { id: 83, text: "Strength training enhances athletic performance in all sports.", author: "Sports Science", category: 'benefit' },
  { id: 84, text: "Building muscle helps regulate blood sugar and prevent diabetes.", author: "Endocrinology Research", category: 'benefit' },
  { id: 85, text: "Exercise increases productivity and mental clarity at work.", author: "Occupational Health Research", category: 'benefit' },
  { id: 86, text: "Strong muscles make everyday tasks easier and less tiring.", author: "Functional Fitness Research", category: 'benefit' },
  { id: 87, text: "Weight training improves flexibility when combined with proper stretching.", author: "Flexibility Research", category: 'benefit' },
  { id: 88, text: "Regular exercise reduces chronic inflammation throughout the body.", author: "Inflammation Research", category: 'benefit' },
  { id: 89, text: "Strength training helps maintain independence as you age.", author: "Geriatric Medicine", category: 'benefit' },
  { id: 90, text: "Building muscle improves your body's ability to burn fat.", author: "Body Composition Research", category: 'benefit' },
  { id: 91, text: "Exercise improves digestion and gut health.", author: "Gastroenterology Research", category: 'benefit' },
  { id: 92, text: "Weight training reduces symptoms of chronic fatigue syndrome.", author: "Chronic Illness Research", category: 'benefit' },
  { id: 93, text: "Strong muscles support joint health and reduce arthritis symptoms.", author: "Rheumatology Research", category: 'benefit' },
  { id: 94, text: "Exercise increases creativity and problem-solving abilities.", author: "Cognitive Psychology", category: 'benefit' },
  { id: 95, text: "Regular lifters report higher life satisfaction scores.", author: "Quality of Life Research", category: 'benefit' },
  { id: 96, text: "Building strength improves respiratory function and lung capacity.", author: "Pulmonology Research", category: 'benefit' },
  { id: 97, text: "Weight training helps manage and prevent metabolic syndrome.", author: "Metabolic Health Research", category: 'benefit' },
  { id: 98, text: "Strong muscles improve reaction time and coordination.", author: "Motor Control Research", category: 'benefit' },
  { id: 99, text: "Exercise provides natural stress relief better than most medications.", author: "Stress Research", category: 'benefit' },
  { id: 100, text: "Building muscle increases your functional lifespan significantly.", author: "Longevity Research", category: 'benefit' },

  // Additional motivation quotes
  { id: 101, text: "Every rep brings you closer to your goals.", author: "Unknown", category: 'motivation' },
  { id: 102, text: "Your only limit is you.", author: "Unknown", category: 'motivation' },
  { id: 103, text: "Be stronger than your excuses.", author: "Unknown", category: 'motivation' },
  { id: 104, text: "The gym is my therapy.", author: "Unknown", category: 'motivation' },
  { id: 105, text: "Excuses don't burn calories.", author: "Unknown", category: 'motivation' },
  { id: 106, text: "Sore today, strong tomorrow.", author: "Unknown", category: 'motivation' },
  { id: 107, text: "Good things come to those who sweat.", author: "Unknown", category: 'motivation' },
  { id: 108, text: "Train insane or remain the same.", author: "Unknown", category: 'motivation' },
  { id: 109, text: "The only impossible journey is the one you never begin.", author: "Tony Robbins", category: 'motivation' },
  { id: 110, text: "Make yourself proud.", author: "Unknown", category: 'motivation' }
]

/**
 * Get a random quote from the collection
 */
export function getRandomQuote(): Quote {
  const randomIndex = Math.floor(Math.random() * QUOTES.length)
  return QUOTES[randomIndex]
}

/**
 * Get a random quote from a specific category
 */
export function getRandomQuoteByCategory(category: Quote['category']): Quote {
  const categoryQuotes = QUOTES.filter(q => q.category === category)
  const randomIndex = Math.floor(Math.random() * categoryQuotes.length)
  return categoryQuotes[randomIndex]
}

/**
 * Get the daily quote based on the current date
 * This ensures the same quote is shown throughout the day
 */
export function getDailyQuote(): Quote {
  const today = new Date()
  const dayOfYear = Math.floor(
    (today.getTime() - new Date(today.getFullYear(), 0, 0).getTime()) / (1000 * 60 * 60 * 24)
  )
  const quoteIndex = dayOfYear % QUOTES.length
  return QUOTES[quoteIndex]
}
