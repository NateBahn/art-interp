export type AppView = 'landing' | 'paintings' | 'features' | 'results'

interface Props {
  onNavigate: (view: AppView) => void
}

interface NavCard {
  id: AppView
  title: string
  icon: string
  available: boolean
}

const NAV_CARDS: NavCard[] = [
  {
    id: 'paintings',
    title: 'Browse Paintings',
    icon: '🖼',
    available: true,
  },
  {
    id: 'features',
    title: 'Browse Features',
    icon: '🔍',
    available: true,
  },
  {
    id: 'results',
    title: 'Methods & Results',
    icon: '📊',
    available: true,
  },
]

export default function LandingPage({ onNavigate }: Props) {
  return (
    <div className="max-w-4xl mx-auto space-y-8">
      {/* Header */}
      <div className="text-center space-y-3">
        <h1 className="text-4xl md:text-5xl font-semibold text-[var(--text-primary)]">
          Self Portrait
        </h1>
        <p className="text-lg text-[var(--text-secondary)] max-w-2xl mx-auto">
          Exploring how AI perceives itself through how it identifies with visual art
        </p>
      </div>

      {/* Navigation Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {NAV_CARDS.map((card) => (
          <button
            key={card.id}
            onClick={() => card.available && onNavigate(card.id)}
            disabled={!card.available}
            className={`relative card p-6 text-left transition-all ${
              card.available
                ? 'hover:brightness-110 hover:ring-1 hover:ring-[var(--accent)] cursor-pointer'
                : 'opacity-60 cursor-not-allowed'
            }`}
          >
            {/* Coming Soon Badge */}
            {!card.available && (
              <span className="absolute top-3 right-3 px-2 py-0.5 rounded text-[10px] bg-[var(--bg-tertiary)] text-[var(--text-muted)]">
                Coming Soon
              </span>
            )}

            {/* Icon */}
            <div className="text-4xl mb-3">{card.icon}</div>

            {/* Title */}
            <h2 className="text-lg font-semibold text-[var(--text-primary)]">
              {card.title}
            </h2>
          </button>
        ))}
      </div>

    </div>
  )
}
