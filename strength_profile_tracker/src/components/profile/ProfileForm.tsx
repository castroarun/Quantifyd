'use client'

import { useState } from 'react'
import { useRouter } from 'next/navigation'
import { Profile, VALIDATION } from '@/types'
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
}

interface FormErrors {
  name?: string
  age?: string
  height?: string
  weight?: string
  general?: string
}

export default function ProfileForm({ profile, onCancel }: ProfileFormProps) {
  const router = useRouter()
  const isEditing = !!profile

  const [formData, setFormData] = useState<FormData>({
    name: profile?.name || '',
    age: profile?.age?.toString() || '',
    height: profile?.height?.toString() || '',
    weight: profile?.weight?.toString() || ''
  })

  const [errors, setErrors] = useState<FormErrors>({})
  const [isSubmitting, setIsSubmitting] = useState(false)

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

    setErrors(newErrors)
    return Object.keys(newErrors).length === 0
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()

    if (!validateForm()) return

    setIsSubmitting(true)

    try {
      const data = {
        name: formData.name.trim(),
        age: parseInt(formData.age, 10),
        height: parseInt(formData.height, 10),
        weight: parseInt(formData.weight, 10)
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
        <div className="p-3 bg-red-50 border border-red-200 rounded-lg text-red-600 text-sm">
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
