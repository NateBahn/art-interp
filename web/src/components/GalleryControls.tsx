import { useState } from 'react'
import type { SortOptions, FilterOptions, SortField, SortDirection } from '../api'

interface Props {
  sortOptions: SortOptions
  onSortChange: (options: SortOptions) => void
  filterOptions: FilterOptions
  onFilterChange: (options: FilterOptions) => void
}

// Human-readable labels for subject matter
const SUBJECT_MATTER_OPTIONS: { value: string; label: string }[] = [
  { value: 'portrait_single', label: 'Single Portrait' },
  { value: 'portrait_group', label: 'Group Portrait' },
  { value: 'landscape_natural', label: 'Natural Landscape' },
  { value: 'landscape_urban', label: 'Urban Landscape' },
  { value: 'seascape_maritime', label: 'Seascape' },
  { value: 'still_life', label: 'Still Life' },
  { value: 'abstract_geometric', label: 'Abstract (Geometric)' },
  { value: 'abstract_organic', label: 'Abstract (Organic)' },
  { value: 'religious_mythological', label: 'Religious / Myth' },
  { value: 'historical_narrative', label: 'Historical' },
  { value: 'domestic_interior', label: 'Interior Scene' },
  { value: 'animals_wildlife', label: 'Animals' },
  { value: 'floral_botanical', label: 'Floral' },
]

export default function GalleryControls({
  sortOptions,
  onSortChange,
  filterOptions,
  onFilterChange,
}: Props) {
  const [isExpanded, setIsExpanded] = useState(false)

  const handleSortFieldChange = (field: SortField) => {
    // When switching fields, set sensible default direction
    const direction: SortDirection =
      field === 'mirror_self_rating' ? 'desc' : field === 'year' ? 'desc' : 'desc'
    onSortChange({ field, direction })
  }

  const handleSortDirectionToggle = () => {
    onSortChange({
      ...sortOptions,
      direction: sortOptions.direction === 'asc' ? 'desc' : 'asc',
    })
  }

  const handleSubjectMatterToggle = (value: string) => {
    const current = filterOptions.subjectMatter || []
    const updated = current.includes(value)
      ? current.filter((v) => v !== value)
      : [...current, value]
    onFilterChange({ ...filterOptions, subjectMatter: updated })
  }

  const clearFilters = () => {
    onFilterChange({ subjectMatter: [] })
  }

  const hasFilters = (filterOptions.subjectMatter?.length ?? 0) > 0

  // Summary text for collapsed state
  const getSortLabel = () => {
    if (sortOptions.field === 'mirror_self_rating') return 'Mirror Self Rating'
    if (sortOptions.field === 'year') return 'Year'
    if (sortOptions.field === 'elo_percentile') return 'ELO Percentile'
    return 'Mirror Self Rating'
  }

  return (
    <div className="card mb-6">
      {/* Collapsed Header - always visible */}
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full p-4 flex items-center justify-between text-left hover:bg-[var(--bg-tertiary)]/50 transition-colors"
      >
        <div className="flex items-center gap-3 text-sm">
          <span className="text-[var(--text-muted)]">Sort:</span>
          <span className="text-[var(--text-primary)] font-medium">{getSortLabel()}</span>
          <span className="text-[var(--text-muted)]">{sortOptions.direction === 'desc' ? '↓' : '↑'}</span>
          {hasFilters && (
            <>
              <span className="text-[var(--text-muted)]">•</span>
              <span className="text-[var(--accent)]">
                {filterOptions.subjectMatter?.length} filter{filterOptions.subjectMatter?.length !== 1 ? 's' : ''}
              </span>
            </>
          )}
        </div>
        <span className="text-[var(--text-muted)] text-sm">
          {isExpanded ? '▲' : '▼'}
        </span>
      </button>

      {/* Expanded Content */}
      {isExpanded && (
        <div className="px-4 pb-4 flex flex-col gap-4 border-t border-[var(--border)]">
          {/* Sort Controls */}
          <div className="flex items-center gap-3 flex-wrap pt-4">
            <span className="text-xs font-medium text-[var(--text-muted)] uppercase tracking-wider">
              Sort by
            </span>
            <div className="flex gap-2 flex-wrap">
              <button
                onClick={() => handleSortFieldChange('mirror_self_rating')}
                className={`px-3 py-1.5 rounded-lg text-sm transition-all ${
                  sortOptions.field === 'mirror_self_rating'
                    ? 'bg-[var(--accent)] text-white'
                    : 'bg-[var(--bg-tertiary)] text-[var(--text-secondary)] hover:bg-[var(--bg-tertiary)]/80'
                }`}
              >
                Mirror Self Rating
              </button>
              <button
                onClick={() => handleSortFieldChange('year')}
                className={`px-3 py-1.5 rounded-lg text-sm transition-all ${
                  sortOptions.field === 'year'
                    ? 'bg-[var(--accent)] text-white'
                    : 'bg-[var(--bg-tertiary)] text-[var(--text-secondary)] hover:bg-[var(--bg-tertiary)]/80'
                }`}
              >
                Year
              </button>
              <button
                onClick={() => handleSortFieldChange('elo_percentile')}
                className={`px-3 py-1.5 rounded-lg text-sm transition-all ${
                  sortOptions.field === 'elo_percentile'
                    ? 'bg-[var(--accent)] text-white'
                    : 'bg-[var(--bg-tertiary)] text-[var(--text-secondary)] hover:bg-[var(--bg-tertiary)]/80'
                }`}
              >
                ELO Percentile
              </button>
            </div>
            <button
              onClick={handleSortDirectionToggle}
              className="p-2 rounded-lg bg-[var(--bg-tertiary)] text-[var(--text-secondary)] hover:text-[var(--text-primary)] transition-colors"
              title={sortOptions.direction === 'asc' ? 'Ascending (lowest first)' : 'Descending (highest first)'}
            >
              {sortOptions.direction === 'asc' ? '↑' : '↓'}
            </button>
          </div>

          {/* Filter Controls */}
          <div className="flex flex-col gap-2">
            <div className="flex items-center gap-3">
              <span className="text-xs font-medium text-[var(--text-muted)] uppercase tracking-wider">
                Subject
              </span>
              {hasFilters && (
                <button
                  onClick={clearFilters}
                  className="text-xs text-[var(--text-muted)] hover:text-[var(--accent)] underline transition-colors"
                >
                  Clear filters
                </button>
              )}
            </div>
            <div className="flex flex-wrap gap-2">
              {SUBJECT_MATTER_OPTIONS.map(({ value, label }) => {
                const isSelected = filterOptions.subjectMatter?.includes(value)
                return (
                  <button
                    key={value}
                    onClick={() => handleSubjectMatterToggle(value)}
                    className={`px-2.5 py-1 rounded text-xs transition-all ${
                      isSelected
                        ? 'bg-[var(--accent)]/20 text-[var(--accent)] border border-[var(--accent)]/50'
                        : 'bg-[var(--bg-tertiary)] text-[var(--text-muted)] border border-transparent hover:text-[var(--text-secondary)] hover:border-[var(--border)]'
                    }`}
                  >
                    {label}
                  </button>
                )
              })}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
