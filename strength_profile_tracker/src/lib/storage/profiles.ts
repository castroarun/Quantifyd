import { Profile, Exercise, Level, VALIDATION } from '@/types'
import { addToSyncQueue, pullFromCloud } from './sync'

const STORAGE_KEY = 'strength_profiles_v2'
const LAST_SYNC_KEY = 'profiles_last_sync'

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
  data: Omit<Profile, 'id' | 'exerciseRatings' | 'createdAt' | 'updatedAt'>
): Profile {
  const profiles = getProfiles()

  // Check max profiles limit
  if (profiles.length >= VALIDATION.maxProfiles) {
    throw new Error(`Maximum of ${VALIDATION.maxProfiles} profiles allowed`)
  }

  // Validate data
  validateProfileData(data)

  const now = new Date().toISOString()

  const newProfile: Profile = {
    id: generateId(),
    name: data.name.trim(),
    age: data.age,
    height: data.height,
    weight: data.weight,
    sex: data.sex,
    dailySteps: data.dailySteps,
    activityLevel: data.activityLevel,
    goal: data.goal,
    exerciseRatings: {},  // Start with no ratings
    createdAt: now,
    updatedAt: now
  }

  profiles.push(newProfile)
  saveProfiles(profiles)

  // Queue for cloud sync
  addToSyncQueue('profile', 'create', newProfile)

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

  // Queue for cloud sync
  addToSyncQueue('profile', 'update', profiles[index])

  return profiles[index]
}

/**
 * Update exercise rating for a profile
 */
export function updateExerciseRating(
  profileId: string,
  exercise: Exercise,
  level: Level
): Profile {
  const profiles = getProfiles()
  const index = profiles.findIndex(p => p.id === profileId)

  if (index === -1) {
    throw new Error('Profile not found')
  }

  profiles[index].exerciseRatings[exercise] = level
  profiles[index].updatedAt = new Date().toISOString()

  saveProfiles(profiles)

  // Queue for cloud sync
  addToSyncQueue('profile', 'update', profiles[index])

  return profiles[index]
}

/**
 * Remove exercise rating from a profile
 */
export function removeExerciseRating(
  profileId: string,
  exercise: Exercise
): Profile {
  const profiles = getProfiles()
  const index = profiles.findIndex(p => p.id === profileId)

  if (index === -1) {
    throw new Error('Profile not found')
  }

  delete profiles[index].exerciseRatings[exercise]
  profiles[index].updatedAt = new Date().toISOString()

  saveProfiles(profiles)

  // Queue for cloud sync
  addToSyncQueue('profile', 'update', profiles[index])

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

  // Queue for cloud sync
  addToSyncQueue('profile', 'delete', { id })
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
 * Sync profiles from cloud (pull)
 * Merges cloud data with local data using last-write-wins
 */
export async function syncProfilesFromCloud(): Promise<boolean> {
  try {
    const cloudData = await pullFromCloud()
    if (!cloudData) return false

    const localProfiles = getProfiles()
    const mergedProfiles = mergeProfiles(localProfiles, cloudData.profiles)

    saveProfiles(mergedProfiles)
    localStorage.setItem(LAST_SYNC_KEY, new Date().toISOString())

    return true
  } catch (error) {
    console.error('Failed to sync profiles from cloud:', error)
    return false
  }
}

/**
 * Merge local and cloud profiles using last-write-wins strategy
 */
function mergeProfiles(local: Profile[], cloud: Profile[]): Profile[] {
  const merged = new Map<string, Profile>()

  // Add all local profiles
  for (const profile of local) {
    merged.set(profile.id, profile)
  }

  // Merge cloud profiles
  for (const cloudProfile of cloud) {
    const existing = merged.get(cloudProfile.id)

    if (!existing) {
      // New profile from cloud
      merged.set(cloudProfile.id, cloudProfile)
    } else {
      // Compare timestamps - cloud wins if newer
      const localTime = new Date(existing.updatedAt).getTime()
      const cloudTime = new Date(cloudProfile.updatedAt).getTime()

      if (cloudTime > localTime) {
        merged.set(cloudProfile.id, cloudProfile)
      }
    }
  }

  return Array.from(merged.values())
}

/**
 * Get last sync timestamp
 */
export function getLastSyncTime(): string | null {
  if (typeof window === 'undefined') return null
  return localStorage.getItem(LAST_SYNC_KEY)
}

/**
 * Validate profile data against constraints
 * @throws Error if validation fails
 */
function validateProfileData(
  data: Partial<Pick<Profile, 'name' | 'age' | 'height' | 'weight' | 'dailySteps'>>
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

  if (data.dailySteps !== undefined) {
    if (data.dailySteps < VALIDATION.dailySteps.min || data.dailySteps > VALIDATION.dailySteps.max) {
      throw new Error(`Daily steps must be between ${VALIDATION.dailySteps.min} and ${VALIDATION.dailySteps.max}`)
    }
  }
}
