'use client'

import { useState, useRef } from 'react'
import { useRouter } from 'next/navigation'
import {
  Profile,
  VALIDATION,
  Sex,
  ActivityLevel,
  Goal,
  ACTIVITY_LEVEL_INFO,
  GOAL_INFO,
  ALL_ACTIVITY_LEVELS,
  ALL_GOALS
} from '@/types'
import { Button, Input } from '@/components/ui'
import { createProfile, updateProfile } from '@/lib/storage/profiles'

interface ProfileFormProps {
  profile?: Profile
  onCancel?: () => void
}

interface FormData {
  name: string
  age: string
  height: string
  weight: string
  sex: Sex | ''
  avatarUrl: string
  dailySteps: string
  activityLevel: ActivityLevel | ''
  goal: Goal | ''
}

interface FormErrors {
  name?: string
  age?: string
  height?: string
  weight?: string
  dailySteps?: string
  general?: string
}

export default function ProfileForm({ profile, onCancel }: ProfileFormProps) {
  const router = useRouter()
  const isEditing = !!profile

  const [formData, setFormData] = useState<FormData>({
    name: profile?.name || '',
    age: profile?.age?.toString() || '',
    height: profile?.height?.toString() || '',
    weight: profile?.weight?.toString() || '',
    sex: profile?.sex || '',
    avatarUrl: profile?.avatarUrl || '',
    dailySteps: profile?.dailySteps?.toString() || '',
    activityLevel: profile?.activityLevel || '',
    goal: profile?.goal || ''
  })

  const [errors, setErrors] = useState<FormErrors>({})
  const [isSubmitting, setIsSubmitting] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)

  // Handle file upload and convert to base64
  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return

    // Validate file type
    if (!file.type.startsWith('image/')) {
      setErrors({ ...errors, general: 'Please select an image file' })
      return
    }

    // Validate file size (max 2MB)
    if (file.size > 2 * 1024 * 1024) {
      setErrors({ ...errors, general: 'Image must be less than 2MB' })
      return
    }

    const reader = new FileReader()
    reader.onload = (event) => {
      const base64 = event.target?.result as string
      // Use functional update to avoid stale closure
      setFormData(prev => ({ ...prev, avatarUrl: base64 }))
    }
    reader.readAsDataURL(file)
  }

  const handleRemoveAvatar = () => {
    setFormData({ ...formData, avatarUrl: '' })
    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
  }

  const validateForm = (): boolean => {
    const newErrors: FormErrors = {}

    // Validate name
    const name = formData.name.trim()
    if (!name) {
      newErrors.name = 'Name is required'
    } else if (name.length > VALIDATION.name.max) {
      newErrors.name = `Name must be ${VALIDATION.name.max} characters or less`
    }

    // Validate age
    const age = parseInt(formData.age, 10)
    if (!formData.age || isNaN(age)) {
      newErrors.age = 'Age is required'
    } else if (age < VALIDATION.age.min || age > VALIDATION.age.max) {
      newErrors.age = `Age must be between ${VALIDATION.age.min} and ${VALIDATION.age.max}`
    }

    // Validate height
    const height = parseInt(formData.height, 10)
    if (!formData.height || isNaN(height)) {
      newErrors.height = 'Height is required'
    } else if (height < VALIDATION.height.min || height > VALIDATION.height.max) {
      newErrors.height = `Height must be between ${VALIDATION.height.min} and ${VALIDATION.height.max} cm`
    }

    // Validate weight
    const weight = parseInt(formData.weight, 10)
    if (!formData.weight || isNaN(weight)) {
      newErrors.weight = 'Weight is required'
    } else if (weight < VALIDATION.weight.min || weight > VALIDATION.weight.max) {
      newErrors.weight = `Weight must be between ${VALIDATION.weight.min} and ${VALIDATION.weight.max} kg`
    }

    // Validate daily steps (optional but must be valid if provided)
    if (formData.dailySteps) {
      const steps = parseInt(formData.dailySteps, 10)
      if (isNaN(steps) || steps < VALIDATION.dailySteps.min || steps > VALIDATION.dailySteps.max) {
        newErrors.dailySteps = `Steps must be between ${VALIDATION.dailySteps.min} and ${VALIDATION.dailySteps.max}`
      }
    }

    setErrors(newErrors)
    return Object.keys(newErrors).length === 0
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()

    if (!validateForm()) return

    setIsSubmitting(true)

    try {
      const data: {
        name: string
        age: number
        height: number
        weight: number
        sex?: Sex
        avatarUrl?: string
        dailySteps?: number
        activityLevel?: ActivityLevel
        goal?: Goal
      } = {
        name: formData.name.trim(),
        age: parseInt(formData.age, 10),
        height: parseInt(formData.height, 10),
        weight: parseInt(formData.weight, 10)
      }

      // Add optional fields if provided
      if (formData.sex) {
        data.sex = formData.sex as Sex
      }
      if (formData.avatarUrl.trim()) {
        data.avatarUrl = formData.avatarUrl.trim()
      }
      if (formData.dailySteps) {
        data.dailySteps = parseInt(formData.dailySteps, 10)
      }
      if (formData.activityLevel) {
        data.activityLevel = formData.activityLevel as ActivityLevel
      }
      if (formData.goal) {
        data.goal = formData.goal as Goal
      }

      if (isEditing && profile) {
        updateProfile(profile.id, data)
        router.push(`/profile/${profile.id}`)
      } else {
        const newProfile = createProfile(data)
        router.push(`/profile/${newProfile.id}`)
      }
    } catch (error) {
      setErrors({
        general: error instanceof Error ? error.message : 'An error occurred'
      })
    } finally {
      setIsSubmitting(false)
    }
  }

  const handleCancel = () => {
    if (onCancel) {
      onCancel()
    } else {
      router.back()
    }
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      {errors.general && (
        <div className="p-3 bg-red-50 dark:bg-red-900/30 border border-red-200 dark:border-red-800 rounded-lg text-red-600 dark:text-red-400 text-sm">
          {errors.general}
        </div>
      )}

      <Input
        label="Name"
        placeholder="Enter profile name"
        value={formData.name}
        onChange={e => setFormData({ ...formData, name: e.target.value })}
        error={errors.name}
        maxLength={VALIDATION.name.max}
      />

      <Input
        label="Age"
        type="number"
        placeholder="Enter age"
        value={formData.age}
        onChange={e => setFormData({ ...formData, age: e.target.value })}
        error={errors.age}
        min={VALIDATION.age.min}
        max={VALIDATION.age.max}
        hint={`${VALIDATION.age.min}-${VALIDATION.age.max} years`}
      />

      <Input
        label="Height (cm)"
        type="number"
        placeholder="Enter height in cm"
        value={formData.height}
        onChange={e => setFormData({ ...formData, height: e.target.value })}
        error={errors.height}
        min={VALIDATION.height.min}
        max={VALIDATION.height.max}
        hint={`${VALIDATION.height.min}-${VALIDATION.height.max} cm`}
      />

      <Input
        label="Weight (kg)"
        type="number"
        placeholder="Enter weight in kg"
        value={formData.weight}
        onChange={e => setFormData({ ...formData, weight: e.target.value })}
        error={errors.weight}
        min={VALIDATION.weight.min}
        max={VALIDATION.weight.max}
        hint={`${VALIDATION.weight.min}-${VALIDATION.weight.max} kg`}
      />

      {/* Profile Picture */}
      <div className="space-y-2">
        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
          Profile Picture
        </label>

        {/* Preview */}
        {formData.avatarUrl && (
          <div className="flex items-center gap-3">
            <img
              src={formData.avatarUrl}
              alt="Preview"
              className="w-16 h-16 rounded-full object-cover border-2 border-gray-200 dark:border-gray-700"
            />
            <button
              type="button"
              onClick={handleRemoveAvatar}
              className="text-sm text-red-500 hover:text-red-700"
            >
              Remove
            </button>
          </div>
        )}

        {/* Upload & URL options */}
        <div className="flex gap-2">
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            onChange={handleFileUpload}
            className="hidden"
            id="avatar-upload"
          />
          <label
            htmlFor="avatar-upload"
            className="flex-1 py-2 px-4 border-2 border-dashed border-gray-300 dark:border-gray-600 rounded-lg text-center cursor-pointer hover:border-blue-400 dark:hover:border-blue-500 transition-colors"
          >
            <span className="text-sm text-gray-600 dark:text-gray-400">
              📷 Upload from device
            </span>
          </label>
        </div>

        {/* URL input */}
        <Input
          placeholder="Or paste image URL..."
          type="url"
          value={formData.avatarUrl.startsWith('data:') ? '' : formData.avatarUrl}
          onChange={e => setFormData({ ...formData, avatarUrl: e.target.value })}
        />
        <p className="text-xs text-gray-500 dark:text-gray-400">
          Max 2MB for uploaded images
        </p>
      </div>

      {/* Sex Selection */}
      <div className="space-y-1">
        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
          Sex <span className="text-gray-400 text-xs">(for accurate calculations)</span>
        </label>
        <div className="flex gap-3">
          <button
            type="button"
            onClick={() => setFormData({ ...formData, sex: 'male' })}
            className={`flex-1 py-3 px-4 rounded-lg border-2 transition-all ${
              formData.sex === 'male'
                ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300'
                : 'border-gray-200 dark:border-gray-700 hover:border-gray-300 dark:hover:border-gray-600'
            }`}
          >
            <span className="text-lg">♂</span>
            <span className="ml-2 font-medium">Male</span>
          </button>
          <button
            type="button"
            onClick={() => setFormData({ ...formData, sex: 'female' })}
            className={`flex-1 py-3 px-4 rounded-lg border-2 transition-all ${
              formData.sex === 'female'
                ? 'border-pink-500 bg-pink-50 dark:bg-pink-900/30 text-pink-700 dark:text-pink-300'
                : 'border-gray-200 dark:border-gray-700 hover:border-gray-300 dark:hover:border-gray-600'
            }`}
          >
            <span className="text-lg">♀</span>
            <span className="ml-2 font-medium">Female</span>
          </button>
        </div>
      </div>

      {/* Daily Steps */}
      <Input
        label="Daily Steps (average)"
        type="number"
        placeholder="e.g., 8000"
        value={formData.dailySteps}
        onChange={e => setFormData({ ...formData, dailySteps: e.target.value })}
        error={errors.dailySteps}
        min={VALIDATION.dailySteps.min}
        max={VALIDATION.dailySteps.max}
        hint="Check your phone's health app"
      />

      {/* Activity Level */}
      <div className="space-y-1">
        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
          Activity Level
        </label>
        <div className="space-y-2">
          {ALL_ACTIVITY_LEVELS.map(level => (
            <button
              key={level}
              type="button"
              onClick={() => setFormData({ ...formData, activityLevel: level })}
              className={`w-full text-left p-3 rounded-lg border-2 transition-all ${
                formData.activityLevel === level
                  ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/30'
                  : 'border-gray-200 dark:border-gray-700 hover:border-gray-300 dark:hover:border-gray-600'
              }`}
            >
              <div className="font-medium text-gray-900 dark:text-gray-100">
                {ACTIVITY_LEVEL_INFO[level].name}
              </div>
              <div className="text-xs text-gray-500 dark:text-gray-400">
                {ACTIVITY_LEVEL_INFO[level].description}
              </div>
            </button>
          ))}
        </div>
      </div>

      {/* Goal */}
      <div className="space-y-1">
        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
          Goal
        </label>
        <div className="grid grid-cols-3 gap-2">
          {ALL_GOALS.map(goal => (
            <button
              key={goal}
              type="button"
              onClick={() => setFormData({ ...formData, goal })}
              className={`py-3 px-2 rounded-lg border-2 transition-all text-center ${
                formData.goal === goal
                  ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/30'
                  : 'border-gray-200 dark:border-gray-700 hover:border-gray-300 dark:hover:border-gray-600'
              }`}
            >
              <div className="text-sm font-medium text-gray-900 dark:text-gray-100">
                {GOAL_INFO[goal].name}
              </div>
            </button>
          ))}
        </div>
      </div>

      <div className="flex gap-3 pt-4">
        <Button
          type="button"
          variant="secondary"
          onClick={handleCancel}
          fullWidth
          disabled={isSubmitting}
        >
          Cancel
        </Button>
        <Button
          type="submit"
          variant="primary"
          fullWidth
          disabled={isSubmitting}
        >
          {isSubmitting ? 'Saving...' : isEditing ? 'Save Changes' : 'Create Profile'}
        </Button>
      </div>
    </form>
  )
}
