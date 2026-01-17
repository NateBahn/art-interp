import type { FeatureSummary } from '../api'
import FeatureCard from './FeatureCard'

interface Props {
  features: FeatureSummary[]
  onSelectFeature: (feature: FeatureSummary) => void
}

export default function FeatureGallery({ features, onSelectFeature }: Props) {
  if (features.length === 0) {
    return (
      <div className="text-center py-12">
        <p className="text-[var(--text-muted)]">No features found matching your filters.</p>
      </div>
    )
  }

  return (
    <div className="space-y-3">
      {features.map((feature) => (
        <FeatureCard
          key={`${feature.layer}-${feature.feature_idx}`}
          feature={feature}
          onClick={() => onSelectFeature(feature)}
        />
      ))}
    </div>
  )
}
