import type { Painting } from '../App'

interface Props {
  paintings: Painting[]
  onSelectPainting: (painting: Painting) => void
}

export default function PaintingGallery({ paintings, onSelectPainting }: Props) {
  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-5">
      {paintings.map((painting) => (
        <button
          key={painting.id}
          onClick={() => onSelectPainting(painting)}
          className="group text-left card overflow-hidden hover:shadow-lg hover:shadow-[var(--accent)]/5 hover:border-[var(--accent)]/30"
        >
          <div className="aspect-[4/3] overflow-hidden bg-[var(--bg-tertiary)]">
            <img
              src={painting.imageUrl}
              alt={painting.title}
              className="w-full h-full object-cover group-hover:scale-[1.02] transition-transform duration-500 ease-out"
            />
          </div>
          <div className="p-4">
            <h3 className="font-medium text-[var(--text-primary)] group-hover:text-[var(--accent)] transition-colors line-clamp-2">
              {painting.title}
            </h3>
            <p className="text-sm text-[var(--text-muted)] mt-1">
              {painting.artist || 'Unknown artist'}{painting.year ? ` · ${painting.year}` : ''}
            </p>
          </div>
        </button>
      ))}
    </div>
  )
}
