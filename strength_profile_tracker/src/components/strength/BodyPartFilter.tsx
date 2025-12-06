'use client'

import { BodyPart, ALL_BODY_PARTS, BODY_PART_NAMES } from '@/types'

interface BodyPartFilterProps {
  selected: BodyPart | 'all'
  onChange: (bodyPart: BodyPart | 'all') => void
}

export default function BodyPartFilter({ selected, onChange }: BodyPartFilterProps) {
  return (
    <div className="overflow-x-auto -mx-4 px-4 pb-2">
      <div className="flex gap-2 min-w-max">
        <FilterButton
          label="All"
          isSelected={selected === 'all'}
          onClick={() => onChange('all')}
        />
        {ALL_BODY_PARTS.map(bodyPart => (
          <FilterButton
            key={bodyPart}
            label={BODY_PART_NAMES[bodyPart]}
            isSelected={selected === bodyPart}
            onClick={() => onChange(bodyPart)}
          />
        ))}
      </div>
    </div>
  )
}

interface FilterButtonProps {
  label: string
  isSelected: boolean
  onClick: () => void
}

function FilterButton({ label, isSelected, onClick }: FilterButtonProps) {
  return (
    <button
      onClick={onClick}
      className={`
        px-4 py-2 rounded-full text-sm font-medium transition-colors
        whitespace-nowrap min-h-[36px]
        ${isSelected
          ? 'bg-[#3498DB] text-white'
          : 'bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600'
        }
      `}
    >
      {label}
    </button>
  )
}
