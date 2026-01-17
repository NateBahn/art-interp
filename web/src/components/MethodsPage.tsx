export type AppView = 'landing' | 'paintings' | 'features' | 'results'

interface Props {
  onNavigate: (view: AppView) => void
}

export default function MethodsPage({ onNavigate }: Props) {
  return (
    <div className="max-w-3xl mx-auto space-y-6">
      {/* Header */}
      <div className="flex items-center gap-4">
        <button
          onClick={() => onNavigate('landing')}
          className="text-[var(--text-muted)] hover:text-[var(--text-primary)] transition-colors"
        >
          ← Back
        </button>
        <h1 className="text-2xl font-semibold text-[var(--text-primary)]">Methods & Results</h1>
      </div>

      {/* Overview */}
      <section className="card p-6 space-y-4">
        <h2 className="text-lg font-semibold text-[var(--text-primary)]">Overview</h2>
        <p className="text-sm text-[var(--text-secondary)] leading-relaxed">
          This project investigates AI models' sense of self through how they relate to works of art.
          The core question: <em className="text-[var(--text-primary)]">can we use interpretability tools to learn what makes a model "identify" with an artwork?</em>
        </p>
        <p className="text-sm text-[var(--text-secondary)] leading-relaxed">
          We combine aesthetic ratings from vision-language models with sparse autoencoder (SAE) feature analysis
          to explore what visual concepts correlate with AI self-identification.
        </p>
      </section>

      {/* Ratings Collection */}
      <section className="card p-6 space-y-4">
        <h2 className="text-lg font-semibold text-[var(--text-primary)]">Aesthetic Ratings</h2>

        <div className="space-y-3">
          <div>
            <h3 className="text-sm font-medium text-[var(--accent)]">Mirror of Self Question</h3>
            <p className="text-sm text-[var(--text-secondary)] leading-relaxed">
              Inspired by Christopher Alexander's concept, we asked models: <em>"Does this artwork reflect something of yourself?"</em> (rated 1-10).
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-3">
            <div className="p-3 rounded-lg bg-[var(--bg-tertiary)]">
              <h4 className="text-xs font-medium text-[var(--text-muted)] uppercase tracking-wider mb-2">Claude Haiku 3.5</h4>
              <p className="text-xs text-[var(--text-secondary)] leading-relaxed">
                Collected ratings and natural language explanations describing why specific visual elements resonate with the model's sense of self.
              </p>
            </div>
            <div className="p-3 rounded-lg bg-[var(--bg-tertiary)]">
              <h4 className="text-xs font-medium text-[var(--text-muted)] uppercase tracking-wider mb-2">Qwen 2.5 VL 7B</h4>
              <p className="text-xs text-[var(--text-secondary)] leading-relaxed">
                Collected direct ratings plus Elo scores from pairwise comparisons asking which painting better represents the model's true self.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* SAE Features */}
      <section className="card p-6 space-y-4">
        <h2 className="text-lg font-semibold text-[var(--text-primary)]">SAE Feature Extraction</h2>

        <p className="text-sm text-[var(--text-secondary)] leading-relaxed">
          We used pretrained sparse autoencoder features from the{' '}
          <span className="font-mono text-xs bg-[var(--bg-tertiary)] px-1 py-0.5 rounded">prisma-multimodal</span>{' '}
          library for CLIP ViT-B-16 embeddings. Features were extracted at three transformer layers:
        </p>

        <div className="flex gap-2 flex-wrap">
          <span className="px-3 py-1.5 rounded-lg text-xs font-medium" style={{ backgroundColor: '#f9731620', color: '#f97316', border: '1px solid #f97316' }}>
            Layer 7 (early)
          </span>
          <span className="px-3 py-1.5 rounded-lg text-xs font-medium" style={{ backgroundColor: '#a855f720', color: '#a855f7', border: '1px solid #a855f7' }}>
            Layer 8 (middle)
          </span>
          <span className="px-3 py-1.5 rounded-lg text-xs font-medium" style={{ backgroundColor: '#06b6d420', color: '#06b6d4', border: '1px solid #06b6d4' }}>
            Layer 11 (late)
          </span>
        </div>

        <p className="text-sm text-[var(--text-secondary)] leading-relaxed">
          Each layer contains 49,152 sparse features extracted from the CLS token, representing visual concepts at different levels of abstraction.
        </p>
      </section>

      {/* Monosemanticity */}
      <section className="card p-6 space-y-4">
        <h2 className="text-lg font-semibold text-[var(--text-primary)]">Interpretability Scoring</h2>

        <p className="text-sm text-[var(--text-secondary)] leading-relaxed">
          Features were evaluated for <em>monosemanticity</em>—whether they represent a single coherent visual concept
          rather than a mixture of unrelated patterns.
        </p>

        <div className="p-3 rounded-lg bg-[var(--bg-tertiary)]">
          <h4 className="text-xs font-medium text-[var(--text-muted)] uppercase tracking-wider mb-2">Method</h4>
          <p className="text-xs text-[var(--text-secondary)] leading-relaxed">
            Adapted from SAE-for-VLM research: we compare feature activations between visually similar paintings
            using <strong>DINOv2 embeddings</strong> (not CLIP) to avoid circularity, since the SAE was trained on CLIP.
            High monosemanticity means the top-activating images share genuine visual similarity.
          </p>
        </div>
      </section>

      {/* Correlation Analysis */}
      <section className="card p-6 space-y-4">
        <h2 className="text-lg font-semibold text-[var(--text-primary)]">Correlation Analysis</h2>

        <p className="text-sm text-[var(--text-secondary)] leading-relaxed">
          Monosemantic features were correlated with mirror-self ratings using Pearson correlation.
          Statistical rigor was maintained through:
        </p>

        <ul className="text-sm text-[var(--text-secondary)] space-y-1 ml-4">
          <li>• Benjamini-Hochberg FDR correction for multiple testing</li>
          <li>• Minimum sample size requirements (n ≥ 30)</li>
          <li>• Effect size thresholds (|r| &gt; 0.1)</li>
        </ul>

        <p className="text-sm text-[var(--text-secondary)] leading-relaxed">
          A subset of high-correlation features were labeled by Gemini 2.5 Flash to provide
          human-readable descriptions of the visual concepts they represent.
        </p>
      </section>

      {/* Results */}
      <section className="card p-6 space-y-4">
        <h2 className="text-lg font-semibold text-[var(--text-primary)]">Results</h2>
        <p className="text-sm text-[var(--text-secondary)] leading-relaxed">
          Still exploring and analyzing — detailed results coming soon.
        </p>
        <p className="text-sm text-[var(--text-secondary)] leading-relaxed">
          In the meantime, explore the interactive tools to see how AI models view themselves through how they rate,
          verbalize, and neurally activate their identification with works of art.
        </p>
        <div className="flex gap-3 pt-2">
          <button
            onClick={() => onNavigate('paintings')}
            className="px-4 py-2 rounded-lg bg-[var(--accent)] text-white text-sm font-medium hover:brightness-110 transition-all"
          >
            Browse Paintings
          </button>
          <button
            onClick={() => onNavigate('features')}
            className="px-4 py-2 rounded-lg bg-[var(--bg-tertiary)] text-[var(--text-secondary)] text-sm font-medium hover:brightness-110 transition-all"
          >
            Browse Features
          </button>
        </div>
      </section>
    </div>
  )
}
