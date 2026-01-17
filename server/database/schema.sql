-- art-interp Database Schema
-- Run this in your Supabase SQL editor or PostgreSQL client

-- =============================================================================
-- ARTWORKS TABLE
-- Stores artwork metadata for display in the explorer
-- =============================================================================
CREATE TABLE IF NOT EXISTS artworks (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    artist_name TEXT,
    year INTEGER,
    image_url TEXT NOT NULL,
    thumbnail_url TEXT,
    labels TEXT,           -- JSON: AI-generated categorical labels
    description TEXT,      -- AI-generated description
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_artworks_labels ON artworks (labels) WHERE labels IS NOT NULL;

-- =============================================================================
-- SAE FEATURES TABLE
-- Stores SAE feature metadata and correlation data
-- =============================================================================
CREATE TABLE IF NOT EXISTS sae_features (
    id SERIAL PRIMARY KEY,
    layer INTEGER NOT NULL,                    -- Transformer layer (7, 8, 11)
    feature_idx INTEGER NOT NULL,             -- Feature index within layer
    monosemanticity_score REAL,               -- Semantic coherence [0-1]
    tier INTEGER NOT NULL DEFAULT 3,          -- Quality tier (1=elite, 2=good, 3=diverse)
    strongest_rating TEXT,                     -- Best correlated rating type
    strongest_correlation REAL,                -- Correlation coefficient
    top_artwork_ids TEXT,                      -- JSON array of artwork IDs
    all_correlations TEXT,                     -- JSON object of all rating correlations
    elo_correlation REAL,                      -- Correlation with ELO rankings
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    CONSTRAINT uq_sae_layer_feature UNIQUE (layer, feature_idx)
);

CREATE INDEX IF NOT EXISTS idx_sae_monosemanticity ON sae_features (monosemanticity_score);
CREATE INDEX IF NOT EXISTS idx_sae_layer ON sae_features (layer);
CREATE INDEX IF NOT EXISTS idx_sae_tier ON sae_features (tier);

-- =============================================================================
-- SAE FEATURE LABELS TABLE
-- Stores VLM-generated labels for interpretable features
-- =============================================================================
CREATE TABLE IF NOT EXISTS sae_feature_labels (
    id SERIAL PRIMARY KEY,
    layer INTEGER NOT NULL,
    feature_idx INTEGER NOT NULL,
    short_label TEXT NOT NULL,                 -- 2-5 word feature name
    description TEXT NOT NULL,                 -- 1-2 sentence explanation
    visual_elements TEXT,                      -- JSON array of visual properties
    explains_rating TEXT,                      -- How pattern influences rating
    non_obvious_insight TEXT,                  -- Surprising observations
    confidence REAL,                           -- VLM confidence [0-1]
    model_used TEXT,                           -- e.g., "gemini-2.0-flash-exp"
    num_images_used INTEGER,                   -- Images analyzed
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    CONSTRAINT uq_sae_label_layer_feature UNIQUE (layer, feature_idx)
);

CREATE INDEX IF NOT EXISTS idx_sae_labels_layer ON sae_feature_labels (layer);

-- =============================================================================
-- ARTWORK TOP FEATURES TABLE
-- Inverted index: which features activate strongly for each artwork
-- =============================================================================
CREATE TABLE IF NOT EXISTS artwork_top_features (
    id SERIAL PRIMARY KEY,
    artwork_id TEXT NOT NULL REFERENCES artworks(id),
    layer INTEGER NOT NULL,
    feature_idx INTEGER NOT NULL,
    activation REAL NOT NULL,
    rank INTEGER NOT NULL,                     -- Rank within this artwork (1 = highest)

    CONSTRAINT uq_artwork_layer_feature UNIQUE (artwork_id, layer, feature_idx)
);

CREATE INDEX IF NOT EXISTS idx_artwork_top_features_lookup ON artwork_top_features (artwork_id, layer, rank);
CREATE INDEX IF NOT EXISTS idx_artwork_top_features_feature ON artwork_top_features (layer, feature_idx, activation);

-- =============================================================================
-- ROW LEVEL SECURITY (optional, for Supabase)
-- =============================================================================
-- Enable RLS if using Supabase with anon key
-- ALTER TABLE artworks ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE sae_features ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE sae_feature_labels ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE artwork_top_features ENABLE ROW LEVEL SECURITY;

-- Allow public read access
-- CREATE POLICY "Public read access" ON artworks FOR SELECT USING (true);
-- CREATE POLICY "Public read access" ON sae_features FOR SELECT USING (true);
-- CREATE POLICY "Public read access" ON sae_feature_labels FOR SELECT USING (true);
-- CREATE POLICY "Public read access" ON artwork_top_features FOR SELECT USING (true);
