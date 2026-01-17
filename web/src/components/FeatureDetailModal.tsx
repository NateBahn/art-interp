import { useState, useEffect } from 'react'
import {
  fetchFeatureDetail,
  LAYER_COLORS,
  type FeatureDetail,
  type Layer,
} from '../api'

// Rating question ID to label mapping
const QUESTION_LABELS: Record<string, string> = {
  drawn_to: 'Drawn To',
  emotional_impact: 'Emotional Impact',
  technical_skill: 'Technical Skill',
  choose_to_look: 'Choose to Look',
  wholeness: 'Wholeness',
  inner_light: 'Inner Light',
  mirror_self: 'Mirror of Self',
  deepest_honest: 'Deepest Honest',
}

export interface FeatureDetailModalProps {
  featureIdx: number
  layer?: number
  onClose: () => void
  onSelectPainting?: (paintingId: string) => void
  maxArtworks?: number
}

export default function FeatureDetailModal({
  featureIdx,
  layer,
  onClose,
  onSelectPainting,
  maxArtworks = 10,
}: FeatureDetailModalProps) {
  const [featureDetail, setFeatureDetail] = useState<FeatureDetail | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const loadFeatureDetail = async () => {
      setLoading(true)
      setError(null)
      try {
        const data = await fetchFeatureDetail(featureIdx, layer)
        setFeatureDetail(data)
      } catch (err) {
        setError('Failed to load feature details')
        console.error(err)
      } finally {
        setLoading(false)
      }
    }
    loadFeatureDetail()
  }, [featureIdx, layer])

  // Sort correlations by absolute value
  const sortedCorrelations = featureDetail?.all_correlations
    ? Object.entries(featureDetail.all_correlations)
        .map(([id, data]) => ({ id, ...data }))
        .sort((a, b) => Math.abs(b.correlation) - Math.abs(a.correlation))
    : []

  // Get layer color
  const layerColor = layer ? LAYER_COLORS[layer as Layer] : undefined

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4" onClick={onClose}>
      <div className="absolute inset-0 bg-black/60 backdrop-blur-sm" />
      <div
        className="relative card p-6 max-w-2xl w-full max-h-[90vh] overflow-y-auto"
        onClick={(e) => e.stopPropagation()}
      >
        <button
          onClick={onClose}
          className="absolute top-4 right-4 text-[var(--text-muted)] hover:text-[var(--text-primary)] z-10"
        >
          ✕
        </button>

        {loading ? (
          <div className="flex items-center justify-center py-12">
            <div className="animate-spin rounded-full h-8 w-8 border-2 border-[var(--accent)] border-t-transparent"></div>
          </div>
        ) : error ? (
          <div className="text-center py-8">
            <p className="text-rose-400">{error}</p>
          </div>
        ) : (
          <div className="space-y-6">
            {/* Header */}
            <div>
              <div className="flex items-center gap-2 mb-1">
                {layer && layerColor && (
                  <span
                    className="px-2 py-0.5 rounded text-xs font-medium"
                    style={{ backgroundColor: `${layerColor}20`, color: layerColor }}
                  >
                    L{layer}
                  </span>
                )}
                <h3 className="text-xl font-semibold text-[var(--text-primary)]">
                  {featureDetail?.label || `Feature #${featureIdx}`}
                </h3>
              </div>
              <p className="text-sm text-[var(--text-muted)] mt-1">
                SAE Feature #{featureIdx}
              </p>
            </div>

            {/* Description */}
            {featureDetail?.description && (
              <div className="p-4 rounded-lg bg-[var(--bg-tertiary)]">
                <p className="text-sm text-[var(--text-secondary)] leading-relaxed">
                  {featureDetail.description}
                </p>
              </div>
            )}

            {/* Visual Elements */}
            {featureDetail?.visual_elements && featureDetail.visual_elements.length > 0 && (
              <div>
                <h4 className="text-xs font-medium text-[var(--text-muted)] uppercase tracking-wider mb-2">
                  Visual Elements
                </h4>
                <div className="flex flex-wrap gap-2">
                  {featureDetail.visual_elements.map((element, i) => (
                    <span key={i} className="px-2 py-1 rounded text-xs bg-[var(--bg-tertiary)] text-[var(--text-secondary)]">
                      {element}
                    </span>
                  ))}
                </div>
              </div>
            )}

            {/* Rating Explanation */}
            {featureDetail?.explains_rating && (
              <div>
                <h4 className="text-xs font-medium text-[var(--text-muted)] uppercase tracking-wider mb-2">
                  Why It Affects Ratings
                </h4>
                <p className="text-sm text-[var(--text-secondary)] leading-relaxed">
                  {featureDetail.explains_rating}
                </p>
              </div>
            )}

            {/* ELO Correlation (highlighted) */}
            {featureDetail?.eloCorrelation !== null && featureDetail?.eloCorrelation !== undefined && (
              <div className="p-3 rounded-lg bg-amber-500/10 border border-amber-500/30">
                <div className="flex items-center justify-between">
                  <div>
                    <h4 className="text-xs font-medium text-amber-400 uppercase tracking-wider">
                      ELO Correlation
                    </h4>
                    <p className="text-[10px] text-[var(--text-muted)] mt-0.5">
                      Correlation with pairwise preference rankings
                    </p>
                  </div>
                  <span className={`text-lg font-mono font-bold ${
                    featureDetail.eloCorrelation >= 0
                      ? 'text-amber-400'
                      : 'text-orange-400'
                  }`}>
                    {featureDetail.eloCorrelation >= 0 ? '+' : ''}{featureDetail.eloCorrelation.toFixed(3)}
                  </span>
                </div>
              </div>
            )}

            {/* All Correlations */}
            {sortedCorrelations.length > 0 && (
              <div>
                <h4 className="text-xs font-medium text-[var(--text-muted)] uppercase tracking-wider mb-3">
                  Rating Correlations
                </h4>
                <div className="grid grid-cols-2 gap-2">
                  {sortedCorrelations.map(({ id, correlation }) => {
                    const isPositive = correlation >= 0
                    const absCorr = Math.abs(correlation)
                    return (
                      <div
                        key={id}
                        className="flex items-center justify-between p-2 rounded bg-[var(--bg-tertiary)]"
                      >
                        <span className="text-sm text-[var(--text-secondary)] truncate">
                          {QUESTION_LABELS[id] || id}
                        </span>
                        <span className={`text-xs font-mono ml-2 ${
                          absCorr < 0.1
                            ? 'text-[var(--text-muted)]'
                            : isPositive
                              ? 'text-emerald-400'
                              : 'text-rose-400'
                        }`}>
                          {isPositive ? '+' : ''}{correlation.toFixed(3)}
                        </span>
                      </div>
                    )
                  })}
                </div>
              </div>
            )}

            {/* Interpretability Score */}
            {featureDetail?.monosemanticity !== null && featureDetail?.monosemanticity !== undefined && (
              <div>
                <div className="flex items-center justify-between text-sm mb-1">
                  <span className="text-[var(--text-muted)]">Interpretability Score</span>
                  <span className="text-[var(--text-secondary)]">
                    {(featureDetail.monosemanticity * 100).toFixed(0)}%
                  </span>
                </div>
                <div className="h-2 bg-[var(--bg-primary)] rounded-full overflow-hidden">
                  <div
                    className="h-full bg-[var(--accent)] rounded-full"
                    style={{ width: `${featureDetail.monosemanticity * 100}%` }}
                  />
                </div>
                <p className="text-xs text-[var(--text-muted)] mt-1">
                  Higher = more likely to represent a single visual concept
                </p>
              </div>
            )}

            {/* Top Activating Artworks */}
            {featureDetail?.top_artworks && featureDetail.top_artworks.length > 0 && (
              <div>
                <h4 className="text-xs font-medium text-[var(--text-muted)] uppercase tracking-wider mb-3">
                  Top Activating Artworks
                </h4>
                <div className="grid grid-cols-5 gap-2">
                  {featureDetail.top_artworks.slice(0, maxArtworks).map((artwork) => (
                    <button
                      key={artwork.id}
                      onClick={() => {
                        if (onSelectPainting) {
                          onSelectPainting(artwork.id)
                          onClose()
                        }
                      }}
                      className="group relative aspect-square rounded-lg overflow-hidden bg-[var(--bg-tertiary)] hover:ring-2 hover:ring-[var(--accent)] transition-all"
                    >
                      <img
                        src={artwork.thumbnail_url || artwork.image_url}
                        alt={artwork.title}
                        className="w-full h-full object-cover"
                      />
                      <div className="absolute inset-0 bg-black/60 opacity-0 group-hover:opacity-100 transition-opacity flex items-end p-1">
                        <p className="text-[10px] text-white line-clamp-2 leading-tight">
                          {artwork.title}
                        </p>
                      </div>
                    </button>
                  ))}
                </div>
                {onSelectPainting && (
                  <p className="text-xs text-[var(--text-muted)] mt-2">
                    Click an artwork to view it
                  </p>
                )}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
