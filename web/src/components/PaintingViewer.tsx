import { useState, useEffect } from 'react'
import type { Painting } from '../App'
import {
  fetchPaintingFeatures,
  fetchPainting,
  fetchClaudeOpinions,
  fetchEloRating,
  AVAILABLE_LAYERS,
  LAYER_COLORS,
  type FeatureCorrelation,
  type PaintingDetail,
  type ModelOpinions,
  type EloRating,
  type Layer,
} from '../api'
import FeatureDetailModal from './FeatureDetailModal'

interface Props {
  painting: Painting
  onSelectPainting?: (paintingId: string) => void
}


// Circular rating gauge component
function RatingGauge({
  rating,
  size = 80,
  label,
  sublabel,
  onClick,
  selected,
  hideCircle,
}: {
  rating: number
  size?: number
  label: string
  sublabel?: string
  onClick?: () => void
  selected?: boolean
  hideCircle?: boolean
}) {
  const strokeWidth = 6
  const radius = (size - strokeWidth) / 2
  const circumference = radius * 2 * Math.PI
  const progress = (rating / 10) * circumference
  const offset = circumference - progress
  const isClickable = !!onClick

  return (
    <div className="flex flex-col items-center">
      <button
        onClick={onClick}
        disabled={!isClickable}
        className={`relative transition-all ${
          isClickable
            ? 'cursor-pointer hover:scale-105 active:scale-95'
            : ''
        } ${
          selected
            ? 'ring-2 ring-[var(--accent)] ring-offset-2 ring-offset-[var(--bg-secondary)] rounded-full'
            : ''
        }`}
        style={{ width: size, height: size }}
      >
        {/* Background circle */}
        {!hideCircle && (
          <svg className="absolute inset-0 -rotate-90" width={size} height={size}>
            <circle
              cx={size / 2}
              cy={size / 2}
              r={radius}
              fill="none"
              stroke="var(--bg-tertiary)"
              strokeWidth={strokeWidth}
            />
            <circle
              cx={size / 2}
              cy={size / 2}
              r={radius}
              fill="none"
              stroke="var(--accent)"
              strokeWidth={strokeWidth}
              strokeLinecap="round"
              strokeDasharray={circumference}
              strokeDashoffset={offset}
              className="transition-all duration-700 ease-out"
            />
          </svg>
        )}
        {/* Center text */}
        <div className="absolute inset-0 flex items-center justify-center">
          <span className="text-2xl font-bold text-[var(--text-primary)]">{rating}</span>
        </div>
      </button>
      <div className="mt-2 text-center">
        <div className={`text-xs font-medium ${isClickable ? 'text-[var(--accent)]' : 'text-[var(--text-primary)]'}`}>
          {label}
        </div>
        {sublabel && (
          <div className="text-[10px] text-[var(--text-muted)]">{sublabel}</div>
        )}
      </div>
    </div>
  )
}

// ELO Rating Badge component (always expanded, shows percentile only)
function EloRatingBadge({
  eloRating,
}: {
  eloRating: EloRating
}) {
  return (
    <div className="flex flex-col items-center">
      <div className="px-4 py-2 rounded-lg bg-[var(--bg-tertiary)] border border-[var(--border-subtle)]">
        <div className="text-2xl font-bold text-[var(--accent)]">{eloRating.rating}</div>
      </div>
      <div className="mt-2 text-center">
        <div className="text-xs font-medium text-[var(--text-primary)]">ELO</div>
        <div className="text-[10px] text-[var(--text-muted)]">Qwen 2.5 VL 7B</div>
      </div>
      {eloRating.percentile !== null && (
        <div className="mt-1 text-xs text-[var(--text-secondary)]">
          <span className="text-[var(--accent)]">{eloRating.percentile}</span>
          <span className="font-medium"> percentile</span>
        </div>
      )}
    </div>
  )
}

export default function PaintingViewer({ painting, onSelectPainting }: Props) {
  const [paintingDetail, setPaintingDetail] = useState<PaintingDetail | null>(null)
  const [features, setFeatures] = useState<FeatureCorrelation[]>([])
  const [loading, setLoading] = useState(false)
  const [selectedFeature, setSelectedFeature] = useState<FeatureCorrelation | null>(null)
  const [modelOpinions, setModelOpinions] = useState<ModelOpinions>({ haiku35: null })
  const [eloRating, setEloRating] = useState<EloRating | null>(null)
  const [showClaudeReflection, setShowClaudeReflection] = useState(true)
    const [selectedLayers, setSelectedLayers] = useState<Set<Layer>>(new Set([11]))

  // Fetch painting detail
  useEffect(() => {
    const loadPaintingDetail = async () => {
      try {
        const data = await fetchPainting(painting.id)
        setPaintingDetail(data)
      } catch (error) {
        console.error('Failed to fetch painting detail:', error)
      }
    }
    loadPaintingDetail()
  }, [painting.id])

  // Fetch Claude's opinions on this painting (multiple models)
  useEffect(() => {
    const loadClaudeOpinions = async () => {
      try {
        const opinions = await fetchClaudeOpinions(painting.id)
        setModelOpinions(opinions)
      } catch (error) {
        console.error('Failed to fetch Claude opinions:', error)
        setModelOpinions({ haiku35: null })
      }
    }
    loadClaudeOpinions()
  }, [painting.id])

  // Fetch ELO rating
  useEffect(() => {
    const loadEloRating = async () => {
      try {
        const rating = await fetchEloRating(painting.id)
        setEloRating(rating)
      } catch (error) {
        console.error('Failed to fetch ELO rating:', error)
        setEloRating(null)
      }
    }
    loadEloRating()
  }, [painting.id])

  // Fetch features (hardcoded to mirror_self)
  useEffect(() => {
    const loadFeatures = async () => {
      setLoading(true)
      try {
        const layers = Array.from(selectedLayers)
        const data = await fetchPaintingFeatures(painting.id, 'cls', layers)
        setFeatures(data.features)
      } catch (error) {
        console.error('Failed to fetch features:', error)
        setFeatures([])
      } finally {
        setLoading(false)
      }
    }
    loadFeatures()
  }, [painting.id, selectedLayers])

  // Get the rating value for mirror_self
  const getRatingValue = (): number | null => {
    if (!paintingDetail?.labels) return null
    const labels = paintingDetail.labels as Record<string, unknown>
    const value = labels['mirror_self']
    if (typeof value === 'number') return value
    return null
  }

  const ratingValue = getRatingValue()

  return (
    <div className="space-y-4">
      {/* Question Header */}
      <div className="text-center">
        <h2 className="text-xl md:text-2xl font-light text-[var(--text-primary)] italic">
          "Does this artwork reflect something of yourself?"
        </h2>
      </div>

      {/* Main Content: Painting + Info side by side on large screens */}
      <div className="grid grid-cols-1 lg:grid-cols-5 gap-6 max-w-7xl mx-auto">
        {/* Left: Painting (3 cols) */}
        <div className="lg:col-span-3">
          <div className="card-elevated overflow-hidden rounded-xl">
            <img
              src={painting.imageUrl}
              alt={painting.title}
              className="w-full h-auto max-h-[60vh] object-contain bg-black/20"
            />
          </div>
          <div className="text-center mt-3">
            <h3 className="text-lg font-semibold text-[var(--text-primary)]">
              {painting.title}
            </h3>
            <p className="text-sm text-[var(--text-muted)]">
              {painting.artist || 'Unknown artist'}
              {painting.year && ` · ${painting.year}`}
            </p>
          </div>
        </div>

        {/* Right: AI Perspectives (2 cols) */}
        <div className="lg:col-span-2">
          <div className="card p-4 space-y-4">
            <h3 className="text-xs font-medium text-[var(--text-muted)] uppercase tracking-wider">
              AI Perspectives
            </h3>

            {/* Rating Gauges */}
            {(ratingValue !== null || modelOpinions.haiku35 || eloRating) && (
              <div className="flex justify-center gap-6 items-start">
                {/* Claude - leftmost */}
                {modelOpinions.haiku35 && (
                  <RatingGauge
                    rating={modelOpinions.haiku35.rating}
                    size={60}
                    label="Claude"
                    sublabel="Haiku 3.5"
                    onClick={() => setShowClaudeReflection(!showClaudeReflection)}
                    selected={showClaudeReflection}
                    hideCircle
                  />
                )}
                {/* Qwen Rating - center */}
                {ratingValue !== null && (
                  <RatingGauge rating={ratingValue} size={60} label="Qwen 2.5 VL" sublabel="7B" />
                )}
                {/* ELO - far right */}
                {eloRating && (
                  <EloRatingBadge eloRating={eloRating} />
                )}
              </div>
            )}

            {/* Inline Reflection Panel */}
            {showClaudeReflection && modelOpinions.haiku35 && (
              <div className="p-3 rounded-lg bg-[var(--bg-tertiary)] animate-in fade-in slide-in-from-top-2 duration-200">
                <div className="flex items-center justify-between mb-2">
                  <h4 className="text-xs font-medium text-[var(--text-muted)] uppercase tracking-wider">
                    Haiku 3.5 Reflection
                  </h4>
                  <span className="text-[10px] font-mono text-[var(--text-muted)]">
                    {modelOpinions.haiku35.model}
                  </span>
                </div>
                <p className="text-xs text-[var(--text-secondary)] leading-relaxed italic">
                  "{modelOpinions.haiku35.explanation}"
                </p>
              </div>
            )}

            {/* Hint when reflection not shown */}
            {!showClaudeReflection && modelOpinions.haiku35 && (
              <p className="text-[10px] text-[var(--text-muted)] text-center">
                Tap a Claude rating to see its reasoning
              </p>
            )}

            {!ratingValue && !modelOpinions.haiku35 && !eloRating && (
              <p className="text-sm text-[var(--text-muted)] text-center py-4">
                No AI perspectives available for this painting
              </p>
            )}
          </div>
        </div>
      </div>

      {/* Visual Features - Compact */}
      <div className="card p-3 max-w-7xl mx-auto">
        <div className="flex items-center justify-between mb-2">
          <h3 className="text-xs font-medium text-[var(--text-muted)] uppercase tracking-wider">
            Visual Features
          </h3>

          {/* Layer Filter - Horizontal */}
          <div className="flex items-center gap-2">
            <span className="text-[10px] text-[var(--text-muted)]">Layers:</span>
            <div className="flex gap-1">
              {AVAILABLE_LAYERS.map((layer) => {
                const isSelected = selectedLayers.has(layer)
                const color = LAYER_COLORS[layer]
                const layerLabels: Record<Layer, string> = {
                  7: 'L7',
                  8: 'L8',
                  11: 'L11',
                }
                return (
                  <button
                    key={layer}
                    onClick={() => {
                      const newSet = new Set(selectedLayers)
                      if (isSelected) {
                        newSet.delete(layer)
                      } else {
                        newSet.add(layer)
                      }
                      // Don't allow deselecting all layers
                      if (newSet.size > 0) {
                        setSelectedLayers(newSet)
                      }
                    }}
                    className={`px-2 py-0.5 rounded text-[10px] font-medium transition-all ${
                      isSelected
                        ? 'opacity-100'
                        : 'opacity-40 hover:opacity-70'
                    }`}
                    style={{
                      backgroundColor: isSelected ? `${color}20` : 'transparent',
                      color: color,
                      border: `1px solid ${isSelected ? color : 'transparent'}`,
                    }}
                  >
                    {layerLabels[layer]}
                  </button>
                )
              })}
            </div>
          </div>
        </div>

        {loading ? (
          <div className="text-center py-2">
            <div className="animate-spin rounded-full h-4 w-4 border-2 border-[var(--accent)] border-t-transparent mx-auto"></div>
          </div>
        ) : features.length === 0 ? (
          <p className="text-sm text-[var(--text-muted)] text-center py-2">
            No features found
          </p>
        ) : (
          <div className="max-h-32 overflow-y-auto">
            <div className="flex flex-wrap gap-1.5">
              {features.slice(0, 20).map((feature) => {
                const isPositive = feature.correlation >= 0
                const isEloPositive = (feature.eloCorrelation ?? 0) >= 0
                const layerColor = LAYER_COLORS[feature.layer as Layer]
                return (
                  <button
                    key={`${feature.layer}-${feature.feature_idx}`}
                    onClick={() => setSelectedFeature(feature)}
                    className="px-2 py-1 rounded-md bg-[var(--bg-tertiary)] hover:brightness-110 transition-all text-left flex items-center gap-1.5"
                    style={{
                      borderLeft: `2px solid ${layerColor}`,
                      borderTop: '1px solid var(--border-subtle)',
                      borderRight: '1px solid var(--border-subtle)',
                      borderBottom: '1px solid var(--border-subtle)',
                    }}
                  >
                    <span className="text-[11px] font-medium text-[var(--text-primary)]">
                      {feature.label || `#${feature.feature_idx}`}
                    </span>
                    {feature.activation !== null && feature.activation !== undefined && (
                      <span className="text-[9px] font-mono px-1 py-0.5 rounded bg-sky-500/10 text-sky-400">
                        {feature.activation.toFixed(1)}
                      </span>
                    )}
                    <span className={`text-[9px] font-mono px-1 py-0.5 rounded ${
                      isPositive
                        ? 'bg-emerald-500/10 text-emerald-400'
                        : 'bg-rose-500/10 text-rose-400'
                    }`}>
                      {isPositive ? '+' : ''}{feature.correlation.toFixed(2)}
                    </span>
                    {feature.eloCorrelation !== null && feature.eloCorrelation !== undefined && (
                      <span className={`text-[9px] font-mono px-1 py-0.5 rounded ${
                        isEloPositive
                          ? 'bg-purple-500/10 text-purple-400'
                          : 'bg-pink-500/10 text-pink-400'
                      }`}>
                        {isEloPositive ? '+' : ''}{feature.eloCorrelation.toFixed(2)}
                      </span>
                    )}
                  </button>
                )
              })}
            </div>
          </div>
        )}

        <p className="text-[9px] text-[var(--text-muted)] mt-2">
          <span className="text-sky-400">Blue</span>=activation • <span className="text-emerald-400">Green</span>/<span className="text-rose-400">Red</span>=mirror self • <span className="text-purple-400">Purple</span>/<span className="text-pink-400">Pink</span>=ELO
        </p>
      </div>

      {/* Feature detail modal */}
      {selectedFeature && (
        <FeatureDetailModal
          featureIdx={selectedFeature.feature_idx}
          layer={selectedFeature.layer}
          onClose={() => setSelectedFeature(null)}
          onSelectPainting={onSelectPainting}
          maxArtworks={5}
        />
      )}
    </div>
  )
}
