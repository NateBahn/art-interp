import { useState } from 'react'
import {
  AVAILABLE_LAYERS,
  LAYER_COLORS,
  type FeatureSortOptions,
  type FeatureFilterOptions,
  type FeatureSortField,
  type SortDirection,
  type Layer,
} from '../api'

interface Props {
  sortOptions: FeatureSortOptions
  onSortChange: (options: FeatureSortOptions) => void
  filterOptions: FeatureFilterOptions
  onFilterChange: (options: FeatureFilterOptions) => void
}

const SORT_FIELDS: { field: FeatureSortField; label: string }[] = [
  { field: 'strongest_correlation', label: 'Rating Correlation' },
  { field: 'elo_correlation', label: 'ELO Correlation' },
  { field: 'monosemanticity_score', label: 'Interpretability' },
]

export default function FeatureBrowserControls({
  sortOptions,
  onSortChange,
  filterOptions,
  onFilterChange,
}: Props) {
  const [expanded, setExpanded] = useState(false)

  const handleSortFieldChange = (field: FeatureSortField) => {
    onSortChange({ ...sortOptions, field })
  }

  const handleSortDirectionToggle = () => {
    const newDirection: SortDirection = sortOptions.direction === 'asc' ? 'desc' : 'asc'
    onSortChange({ ...sortOptions, direction: newDirection })
  }

  const handleLayerToggle = (layer: number) => {
    const currentLayers = filterOptions.layers || [7, 8, 11]
    const newLayers = currentLayers.includes(layer)
      ? currentLayers.filter(l => l !== layer)
      : [...currentLayers, layer]
    // Don't allow empty layers
    if (newLayers.length > 0) {
      onFilterChange({ ...filterOptions, layers: newLayers })
    }
  }

  const handleMonoChange = (value: number) => {
    onFilterChange({ ...filterOptions, minMonosemanticity: value })
  }

  const handleCorrelationSignChange = (sign: 'positive' | 'negative' | 'all') => {
    onFilterChange({ ...filterOptions, correlationSign: sign === 'all' ? undefined : sign })
  }

  const handleClearFilters = () => {
    onFilterChange({
      layers: [7, 8, 11],
      minMonosemanticity: 0.5,
      correlationSign: undefined,
    })
  }

  // Calculate active filter count
  const activeFilterCount =
    ((filterOptions.layers?.length || 3) < 3 ? 1 : 0) +
    ((filterOptions.minMonosemanticity ?? 0.5) !== 0.5 ? 1 : 0) +
    (filterOptions.correlationSign ? 1 : 0)

  const sortLabel = SORT_FIELDS.find(s => s.field === sortOptions.field)?.label || sortOptions.field

  return (
    <div className="card">
      {/* Collapsible Header */}
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full px-4 py-3 flex items-center justify-between hover:bg-[var(--bg-tertiary)] transition-colors rounded-xl"
      >
        <div className="flex items-center gap-3 text-sm">
          <span className="text-[var(--text-muted)]">Sort:</span>
          <span className="text-[var(--text-primary)] font-medium">{sortLabel}</span>
          <span className="text-[var(--text-muted)]">{sortOptions.direction === 'asc' ? '↑' : '↓'}</span>
          {activeFilterCount > 0 && (
            <>
              <span className="text-[var(--text-muted)]">•</span>
              <span className="text-[var(--accent)]">{activeFilterCount} filter{activeFilterCount !== 1 ? 's' : ''}</span>
            </>
          )}
        </div>
        <span className="text-[var(--text-muted)]">{expanded ? '▲' : '▼'}</span>
      </button>

      {/* Expanded Controls */}
      {expanded && (
        <div className="px-4 pb-4 space-y-4">
          {/* Sort By */}
          <div>
            <h4 className="text-xs font-medium text-[var(--text-muted)] uppercase tracking-wider mb-2">
              Sort By
            </h4>
            <div className="flex flex-wrap gap-2">
              {SORT_FIELDS.map(({ field, label }) => (
                <button
                  key={field}
                  onClick={() => handleSortFieldChange(field)}
                  className={`px-3 py-1.5 rounded-lg text-sm transition-all ${
                    sortOptions.field === field
                      ? 'bg-[var(--accent)] text-white'
                      : 'bg-[var(--bg-tertiary)] text-[var(--text-secondary)] hover:brightness-110'
                  }`}
                >
                  {label}
                </button>
              ))}
              <button
                onClick={handleSortDirectionToggle}
                className="px-3 py-1.5 rounded-lg text-sm bg-[var(--bg-tertiary)] text-[var(--text-secondary)] hover:brightness-110"
              >
                {sortOptions.direction === 'asc' ? '↑ Ascending' : '↓ Descending'}
              </button>
            </div>
          </div>

          {/* Layers */}
          <div>
            <h4 className="text-xs font-medium text-[var(--text-muted)] uppercase tracking-wider mb-2">
              Layers
            </h4>
            <div className="flex flex-wrap gap-2">
              {AVAILABLE_LAYERS.map((layer) => {
                const isSelected = (filterOptions.layers || [7, 8, 11]).includes(layer)
                const color = LAYER_COLORS[layer]
                const layerLabels: Record<Layer, string> = {
                  7: 'L7 (early)',
                  8: 'L8 (middle)',
                  11: 'L11 (late)',
                }
                return (
                  <button
                    key={layer}
                    onClick={() => handleLayerToggle(layer)}
                    className={`px-3 py-1.5 rounded-lg text-sm transition-all ${
                      isSelected ? 'opacity-100' : 'opacity-40 hover:opacity-70'
                    }`}
                    style={{
                      backgroundColor: isSelected ? `${color}20` : 'var(--bg-tertiary)',
                      color: isSelected ? color : 'var(--text-secondary)',
                      border: `1px solid ${isSelected ? color : 'transparent'}`,
                    }}
                  >
                    {layerLabels[layer]}
                  </button>
                )
              })}
            </div>
          </div>

          {/* Correlation Sign */}
          <div>
            <h4 className="text-xs font-medium text-[var(--text-muted)] uppercase tracking-wider mb-2">
              Correlation Sign
            </h4>
            <div className="flex flex-wrap gap-2">
              {(['all', 'positive', 'negative'] as const).map((sign) => {
                const isSelected = sign === 'all'
                  ? !filterOptions.correlationSign
                  : filterOptions.correlationSign === sign
                const labels = { all: 'All', positive: 'Positive (+)', negative: 'Negative (-)' }
                return (
                  <button
                    key={sign}
                    onClick={() => handleCorrelationSignChange(sign)}
                    className={`px-3 py-1.5 rounded-lg text-sm transition-all ${
                      isSelected
                        ? sign === 'positive'
                          ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500'
                          : sign === 'negative'
                          ? 'bg-rose-500/20 text-rose-400 border border-rose-500'
                          : 'bg-[var(--accent)] text-white'
                        : 'bg-[var(--bg-tertiary)] text-[var(--text-secondary)] hover:brightness-110 border border-transparent'
                    }`}
                  >
                    {labels[sign]}
                  </button>
                )
              })}
            </div>
          </div>

          {/* Min Interpretability */}
          <div>
            <div className="flex items-center justify-between mb-2">
              <h4 className="text-xs font-medium text-[var(--text-muted)] uppercase tracking-wider">
                Min Interpretability
              </h4>
              <span className="text-sm text-[var(--text-secondary)] font-mono">
                {((filterOptions.minMonosemanticity ?? 0.5) * 100).toFixed(0)}%
              </span>
            </div>
            <input
              type="range"
              min="0"
              max="100"
              step="5"
              value={(filterOptions.minMonosemanticity ?? 0.5) * 100}
              onChange={(e) => handleMonoChange(parseInt(e.target.value) / 100)}
              className="w-full h-2 bg-[var(--bg-tertiary)] rounded-lg appearance-none cursor-pointer accent-[var(--accent)]"
            />
          </div>

          {/* Clear Filters */}
          {activeFilterCount > 0 && (
            <button
              onClick={handleClearFilters}
              className="text-sm text-[var(--text-muted)] hover:text-[var(--text-primary)] transition-colors"
            >
              Clear all filters
            </button>
          )}
        </div>
      )}
    </div>
  )
}
