import { Profile, ExerciseLevels, VALIDATION } from '@/types'

const STORAGE_KEY = 'strength_profiles'

/**
 * Generate a unique ID for profiles
 */
function generateId(): string {
  return `profile_${Date.now()}_${Math.random().toString(36).substring(2, 9)}`
}

/**
 * Get all profiles from localStorage
 */
export function getProfiles(): Profile[] {
  if (typeof window === 'undefined') return []

  try {
    const data = localStorage.getItem(STORAGE_KEY)
    return data ? JSON.parse(data) : []
  } catch {
    console.error('Error reading profiles from localStorage')
    return []
  }
}

/**
 * Get a single profile by ID
 */
export function getProfileById(id: string): Profile | null {
  const profiles = getProfiles()
  return profiles.find(p => p.id === id) || null
}

/**
 * Save profiles to localStorage
 */
function saveProfiles(profiles: Profile[]): void {
  if (typeof window === 'undefined') return

  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(profiles))
  } catch {
    console.error('Error saving profiles to localStorage')
  }
}

/**
 * Create a new profile
 * @throws Error if max profiles reached or validation fails
 */
export function createProfile(
  data: Omit<Profile, 'id' | 'currentLevels' | 'createdAt' | 'updatedAt'>
): Profile {
  const profiles = getProfiles()

  // Check max profiles limit
  if (profiles.length >= VALIDATION.maxProfiles) {
    throw new Error(`Maximum of ${VALIDATION.maxProfiles} profiles allowed`)
  }

  // Validate data
  validateProfileData(data)

  const now = new Date().toISOString()
  const defaultLevels: ExerciseLevels = {
    benchPress: 'beginner',
    squat: 'beginner',
    deadlift: 'beginner',
    shoulderPress: 'beginner'
  }

  const newProfile: Profile = {
    id: generateId(),
    name: data.name.trim(),
    age: data.age,
    height: data.height,
    weight: data.weight,
    currentLevels: defaultLevels,
    createdAt: now,
    updatedAt: now
  }

  profiles.push(newProfile)
  saveProfiles(profiles)

  return newProfile
}

/**
 * Update an existing profile
 * @throws Error if profile not found or validation fails
 */
export function updateProfile(
  id: string,
  data: Partial<Omit<Profile, 'id' | 'createdAt' | 'updatedAt'>>
): Profile {
  const profiles = getProfiles()
  const index = profiles.findIndex(p => p.id === id)

  if (index === -1) {
    throw new Error('Profile not found')
  }

  // Validate updated data
  const updatedData = { ...profiles[index], ...data }
  validateProfileData(updatedData)

  profiles[index] = {
    ...updatedData,
    updatedAt: new Date().toISOString()
  }

  saveProfiles(profiles)

  return profiles[index]
}

/**
 * Update exercise level for a profile
 */
export function updateProfileLevel(
  profileId: string,
  exercise: keyof ExerciseLevels,
  level: ExerciseLevels[keyof ExerciseLevels]
): Profile {
  const profiles = getProfiles()
  const index = profiles.findIndex(p => p.id === profileId)

  if (index === -1) {
    throw new Error('Profile not found')
  }

  profiles[index].currentLevels[exercise] = level
  profiles[index].updatedAt = new Date().toISOString()

  saveProfiles(profiles)

  return profiles[index]
}

/**
 * Delete a profile
 * @throws Error if profile not found
 */
export function deleteProfile(id: string): void {
  const profiles = getProfiles()
  const index = profiles.findIndex(p => p.id === id)

  if (index === -1) {
    throw new Error('Profile not found')
  }

  profiles.splice(index, 1)
  saveProfiles(profiles)
}

/**
 * Get the count of profiles
 */
export function getProfileCount(): number {
  return getProfiles().length
}

/**
 * Check if more profiles can be created
 */
export function canCreateProfile(): boolean {
  return getProfileCount() < VALIDATION.maxProfiles
}

/**
 * Validate profile data against constraints
 * @throws Error if validation fails
 */
function validateProfileData(
  data: Partial<Pick<Profile, 'name' | 'age' | 'height' | 'weight'>>
): void {
  if (data.name !== undefined) {
    const name = data.name.trim()
    if (name.length < VALIDATION.name.min) {
      throw new Error('Name is required')
    }
    if (name.length > VALIDATION.name.max) {
      throw new Error(`Name must be ${VALIDATION.name.max} characters or less`)
    }
  }

  if (data.age !== undefined) {
    if (data.age < VALIDATION.age.min || data.age > VALIDATION.age.max) {
      throw new Error(`Age must be between ${VALIDATION.age.min} and ${VALIDATION.age.max}`)
    }
  }

  if (data.height !== undefined) {
    if (data.height < VALIDATION.height.min || data.height > VALIDATION.height.max) {
      throw new Error(`Height must be between ${VALIDATION.height.min} and ${VALIDATION.height.max} cm`)
    }
  }

  if (data.weight !== undefined) {
    if (data.weight < VALIDATION.weight.min || data.weight > VALIDATION.weight.max) {
      throw new Error(`Weight must be between ${VALIDATION.weight.min} and ${VALIDATION.weight.max} kg`)
    }
  }
}
