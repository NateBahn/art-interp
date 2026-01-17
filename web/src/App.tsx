import { useState, useEffect } from 'react'
import './index.css'
import PaintingViewer from './components/PaintingViewer'
import PaintingGallery from './components/PaintingGallery'
import GalleryControls from './components/GalleryControls'
import LandingPage from './components/LandingPage'
import FeatureBrowser from './components/FeatureBrowser'
import MethodsPage from './components/MethodsPage'
import { fetchPaintings, fetchPainting } from './api'
import type { Painting as APIPainting, SortOptions, FilterOptions } from './api'

// App view type for navigation
export type AppView = 'landing' | 'paintings' | 'features' | 'results'

// Re-export types for components
export interface Painting {
  id: string
  title: string
  artist: string | null
  year: number | null
  imageUrl: string
  thumbnailUrl: string | null
}

// Transform API painting to component painting
function transformPainting(p: APIPainting): Painting {
  return {
    id: p.id,
    title: p.title,
    artist: p.artist,
    year: p.year,
    imageUrl: p.image_url,
    thumbnailUrl: p.thumbnail_url,
  }
}

const PAGE_SIZE = 50

function App() {
  // Navigation state
  const [currentView, setCurrentView] = useState<AppView>('landing')

  // Paintings state
  const [paintings, setPaintings] = useState<Painting[]>([])
  const [selectedPainting, setSelectedPainting] = useState<Painting | null>(null)
  const [loading, setLoading] = useState(false)
  const [loadingMore, setLoadingMore] = useState(false)
  const [hasMore, setHasMore] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [sortOptions, setSortOptions] = useState<SortOptions>({ field: 'mirror_self_rating', direction: 'desc' })
  const [filterOptions, setFilterOptions] = useState<FilterOptions>({ subjectMatter: [] })

  // Navigate to a view
  const navigateTo = (view: AppView) => {
    setCurrentView(view)
    if (view !== 'paintings') {
      setSelectedPainting(null)
    }
  }

  // Fetch paintings when on paintings view (re-fetch when sort/filter changes)
  useEffect(() => {
    if (currentView !== 'paintings') return

    async function loadData() {
      try {
        setLoading(true)
        setPaintings([]) // Reset when options change
        const paintingsData = await fetchPaintings({
          limit: PAGE_SIZE,
          offset: 0,
          sort: sortOptions,
          filter: filterOptions,
        })
        setPaintings(paintingsData.map(transformPainting))
        setHasMore(paintingsData.length === PAGE_SIZE)
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load data')
      } finally {
        setLoading(false)
      }
    }
    loadData()
  }, [currentView, sortOptions, filterOptions])

  // Load more paintings
  const loadMore = async () => {
    if (loadingMore || !hasMore) return

    try {
      setLoadingMore(true)
      const paintingsData = await fetchPaintings({
        limit: PAGE_SIZE,
        offset: paintings.length,
        sort: sortOptions,
        filter: filterOptions,
      })
      const newPaintings = paintingsData.map(transformPainting)
      setPaintings(prev => [...prev, ...newPaintings])
      setHasMore(paintingsData.length === PAGE_SIZE)
    } catch (err) {
      console.error('Failed to load more paintings:', err)
    } finally {
      setLoadingMore(false)
    }
  }

  // Handle selecting a painting by ID (from feature modal top artworks)
  const handleSelectPaintingById = async (paintingId: string) => {
    // First check if it's already in our loaded paintings
    const existing = paintings.find(p => p.id === paintingId)
    if (existing) {
      setSelectedPainting(existing)
      return
    }

    // Otherwise fetch it
    try {
      const paintingData = await fetchPainting(paintingId)
      const transformed = transformPainting(paintingData)
      setSelectedPainting(transformed)
    } catch (err) {
      console.error('Failed to fetch painting:', err)
    }
  }

  // Landing page
  if (currentView === 'landing') {
    return (
      <div className="min-h-screen bg-[var(--bg-primary)]">
        <div className="max-w-7xl mx-auto px-6 py-8">
          <LandingPage onNavigate={navigateTo} />
        </div>
      </div>
    )
  }

  // Feature browser
  if (currentView === 'features') {
    return (
      <div className="min-h-screen bg-[var(--bg-primary)]">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <FeatureBrowser
            onNavigate={navigateTo}
            onSelectPainting={(paintingId) => {
              handleSelectPaintingById(paintingId)
              navigateTo('paintings')
            }}
          />
        </div>
      </div>
    )
  }

  // Methods & Results page
  if (currentView === 'results') {
    return (
      <div className="min-h-screen bg-[var(--bg-primary)]">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <MethodsPage onNavigate={navigateTo} />
        </div>
      </div>
    )
  }

  // Paintings view
  if (loading) {
    return (
      <div className="min-h-screen bg-[var(--bg-primary)] flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-8 w-8 border-2 border-[var(--accent)] border-t-transparent mx-auto"></div>
          <p className="mt-4 text-sm text-[var(--text-muted)]">Loading paintings...</p>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="min-h-screen bg-[var(--bg-primary)] flex items-center justify-center p-6">
        <div className="card p-8 text-center max-w-md">
          <div className="w-12 h-12 rounded-full bg-rose-500/10 flex items-center justify-center mx-auto mb-4">
            <span className="text-rose-400 text-xl">!</span>
          </div>
          <p className="text-[var(--text-primary)] font-medium mb-2">Unable to load data</p>
          <p className="text-sm text-[var(--text-muted)] mb-4">{error}</p>
          <p className="text-xs text-[var(--text-muted)]">
            Make sure the backend is running at localhost:8000
          </p>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-[var(--bg-primary)]">
      <div className="max-w-7xl mx-auto px-6 py-4">
        {selectedPainting ? (
          /* Main viewer */
          <div>
            <button
              onClick={() => setSelectedPainting(null)}
              className="btn-ghost mb-6 text-sm -ml-3"
            >
              <span className="mr-1">←</span> Back to gallery
            </button>
            <PaintingViewer
              painting={selectedPainting}
              onSelectPainting={handleSelectPaintingById}
            />
          </div>
        ) : (
          /* Gallery view */
          <div>
            <div className="flex items-center gap-4 mb-6">
              <button
                onClick={() => navigateTo('landing')}
                className="text-[var(--text-muted)] hover:text-[var(--text-primary)] transition-colors"
              >
                ← Back
              </button>
              <div>
                <h2 className="text-lg font-medium text-[var(--text-primary)]">
                  Browse Paintings
                </h2>
                <p className="text-sm text-[var(--text-secondary)]">
                  Choose an artwork to explore how AI reflects on it and what visual features it recognizes.
                  <span className="text-[var(--text-muted)] ml-1">({paintings.length} loaded)</span>
                </p>
              </div>
            </div>
            <GalleryControls
              sortOptions={sortOptions}
              onSortChange={setSortOptions}
              filterOptions={filterOptions}
              onFilterChange={setFilterOptions}
            />
            <PaintingGallery
              paintings={paintings}
              onSelectPainting={setSelectedPainting}
            />

            {/* Load More Button */}
            {hasMore && (
              <div className="flex justify-center mt-8 mb-4">
                <button
                  onClick={loadMore}
                  disabled={loadingMore}
                  className="px-6 py-3 rounded-lg bg-[var(--bg-secondary)] border border-[var(--border)] text-[var(--text-primary)] hover:bg-[var(--bg-tertiary)] transition-colors disabled:opacity-50"
                >
                  {loadingMore ? (
                    <span className="flex items-center gap-2">
                      <div className="animate-spin rounded-full h-4 w-4 border-2 border-[var(--accent)] border-t-transparent"></div>
                      Loading...
                    </span>
                  ) : (
                    'Load More'
                  )}
                </button>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}

export default App
