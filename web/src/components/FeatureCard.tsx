import { LAYER_COLORS, type FeatureSummary, type Layer } from '../api'

interface Props {
  feature: FeatureSummary
  onClick: () => void
}

export default function FeatureCard({ feature, onClick }: Props) {
  const layerColor = LAYER_COLORS[feature.layer as Layer]
  const hasCorrelations = feature.mirror_self_correlation !== null || feature.elo_correlation !== null

  return (
    <button
      onClick={onClick}
      className="w-full text-left card p-4 hover:brightness-110 transition-all hover:ring-1 hover:ring-[var(--accent)]"
    >
      {/* Header Row */}
      <div className="flex items-center gap-3 mb-3">
        {/* Layer Badge */}
        <span
          className="px-2 py-1 rounded text-xs font-medium shrink-0"
          style={{ backgroundColor: `${layerColor}20`, color: layerColor }}
        >
          L{feature.layer}
        </span>

        {/* Feature Label */}
        <h3 className="text-base font-semibold text-[var(--text-primary)] flex-1 truncate">
          {feature.label || `Feature #${feature.feature_idx}`}
        </h3>
      </div>

      {/* Description */}
      {feature.description && (
        <p className="text-sm text-[var(--text-secondary)] leading-relaxed mb-3 line-clamp-2">
          {feature.description}
        </p>
      )}

      {/* Correlations Row */}
      {hasCorrelations && (
        <div className="flex flex-wrap gap-3 mb-3 text-xs">
          {feature.mirror_self_correlation !== null && (
            <div className="flex items-center gap-1.5">
              <span className="text-[var(--text-muted)]">Mirror Self:</span>
              <span className={`font-mono font-medium ${
                feature.mirror_self_correlation >= 0
                  ? 'text-emerald-400'
                  : 'text-rose-400'
              }`}>
                {feature.mirror_self_correlation >= 0 ? '+' : ''}
                {feature.mirror_self_correlation.toFixed(3)}
              </span>
            </div>
          )}

          {feature.elo_correlation !== null && (
            <div className="flex items-center gap-1.5">
              <span className="text-[var(--text-muted)]">ELO:</span>
              <span className={`font-mono font-medium ${
                feature.elo_correlation >= 0
                  ? 'text-amber-400'
                  : 'text-orange-400'
              }`}>
                {feature.elo_correlation >= 0 ? '+' : ''}
                {feature.elo_correlation.toFixed(3)}
              </span>
            </div>
          )}

          {feature.monosemanticity !== null && (
            <div className="flex items-center gap-1.5">
              <span className="text-[var(--text-muted)]">Interpretability:</span>
              <span className="font-mono font-medium text-[var(--text-secondary)]">
                {(feature.monosemanticity * 100).toFixed(0)}%
              </span>
            </div>
          )}
        </div>
      )}

      {/* Top Artworks */}
      {feature.top_artworks && feature.top_artworks.length > 0 && (
        <div>
          <span className="text-[10px] text-[var(--text-muted)] uppercase tracking-wider mb-2 block">
            Top Activating Artworks
          </span>
          <div className="flex gap-2">
            {feature.top_artworks.slice(0, 5).map((artwork) => (
              <div
                key={artwork.id}
                className="w-12 h-12 rounded-lg overflow-hidden bg-[var(--bg-tertiary)] shrink-0"
              >
                <img
                  src={artwork.thumbnail_url || artwork.image_url}
                  alt={artwork.title}
                  className="w-full h-full object-cover"
                />
              </div>
            ))}
          </div>
        </div>
      )}
    </button>
  )
}
