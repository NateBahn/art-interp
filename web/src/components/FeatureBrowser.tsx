import { useState, useEffect } from 'react'
import {
  fetchFeatures,
  type FeatureSummary,
  type FeatureSortOptions,
  type FeatureFilterOptions,
} from '../api'
import FeatureGallery from './FeatureGallery'
import FeatureBrowserControls from './FeatureBrowserControls'
import FeatureDetailModal from './FeatureDetailModal'

const PAGE_SIZE = 20

export type AppView = 'landing' | 'paintings' | 'features' | 'results'

interface Props {
  onNavigate: (view: AppView) => void
  onSelectPainting?: (paintingId: string) => void
}

export default function FeatureBrowser({ onNavigate, onSelectPainting }: Props) {
  const [features, setFeatures] = useState<FeatureSummary[]>([])
  const [loading, setLoading] = useState(true)
  const [loadingMore, setLoadingMore] = useState(false)
  const [hasMore, setHasMore] = useState(true)
  const [total, setTotal] = useState(0)

  // Filter/sort state
  const [sortOptions, setSortOptions] = useState<FeatureSortOptions>({
    field: 'strongest_correlation',
    direction: 'desc',
  })
  const [filterOptions, setFilterOptions] = useState<FeatureFilterOptions>({
    layers: [7, 8, 11],
    minMonosemanticity: 0.5,
  })

  // Modal state
  const [selectedFeature, setSelectedFeature] = useState<FeatureSummary | null>(null)

  // Fetch features when filters/sort change
  useEffect(() => {
    const loadFeatures = async () => {
      setLoading(true)
      try {
        const result = await fetchFeatures({
          limit: PAGE_SIZE,
          offset: 0,
          sort: sortOptions,
          filter: filterOptions,
        })
        setFeatures(result.features)
        setTotal(result.total)
        setHasMore(result.features.length === PAGE_SIZE)
      } catch (error) {
        console.error('Failed to fetch features:', error)
        setFeatures([])
        setTotal(0)
        setHasMore(false)
      } finally {
        setLoading(false)
      }
    }
    loadFeatures()
  }, [sortOptions, filterOptions])

  // Load more features
  const handleLoadMore = async () => {
    if (loadingMore || !hasMore) return

    setLoadingMore(true)
    try {
      const result = await fetchFeatures({
        limit: PAGE_SIZE,
        offset: features.length,
        sort: sortOptions,
        filter: filterOptions,
      })
      setFeatures([...features, ...result.features])
      setHasMore(result.features.length === PAGE_SIZE)
    } catch (error) {
      console.error('Failed to load more features:', error)
    } finally {
      setLoadingMore(false)
    }
  }

  // Handle artwork click from modal
  const handleSelectPainting = (paintingId: string) => {
    if (onSelectPainting) {
      onSelectPainting(paintingId)
      onNavigate('paintings')
    }
  }

  return (
    <div className="max-w-4xl mx-auto space-y-4">
      {/* Header */}
      <div className="flex items-center gap-4">
        <button
          onClick={() => onNavigate('landing')}
          className="text-[var(--text-muted)] hover:text-[var(--text-primary)] transition-colors"
        >
          ← Back
        </button>
        <div>
          <h1 className="text-2xl font-semibold text-[var(--text-primary)]">Browse Features</h1>
          <p className="text-sm text-[var(--text-muted)]">
            Explore SAE features that correlate with aesthetic ratings
            {total > 0 && ` (${total.toLocaleString()} features)`}
          </p>
        </div>
      </div>

      {/* Controls */}
      <FeatureBrowserControls
        sortOptions={sortOptions}
        onSortChange={setSortOptions}
        filterOptions={filterOptions}
        onFilterChange={setFilterOptions}
      />

      {/* Features List */}
      {loading ? (
        <div className="flex items-center justify-center py-12">
          <div className="animate-spin rounded-full h-8 w-8 border-2 border-[var(--accent)] border-t-transparent"></div>
        </div>
      ) : (
        <>
          <FeatureGallery
            features={features}
            onSelectFeature={setSelectedFeature}
          />

          {/* Load More Button */}
          {hasMore && features.length > 0 && (
            <div className="text-center pt-4">
              <button
                onClick={handleLoadMore}
                disabled={loadingMore}
                className="px-6 py-2 rounded-lg bg-[var(--bg-tertiary)] text-[var(--text-secondary)] hover:brightness-110 transition-all disabled:opacity-50"
              >
                {loadingMore ? (
                  <span className="flex items-center gap-2">
                    <div className="animate-spin rounded-full h-4 w-4 border-2 border-[var(--accent)] border-t-transparent"></div>
                    Loading...
                  </span>
                ) : (
                  `Load More (${features.length} of ${total})`
                )}
              </button>
            </div>
          )}
        </>
      )}

      {/* Feature Detail Modal */}
      {selectedFeature && (
        <FeatureDetailModal
          featureIdx={selectedFeature.feature_idx}
          layer={selectedFeature.layer}
          onClose={() => setSelectedFeature(null)}
          onSelectPainting={handleSelectPainting}
          maxArtworks={10}
        />
      )}
    </div>
  )
}
