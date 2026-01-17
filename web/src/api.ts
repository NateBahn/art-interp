// API client powered directly by Supabase
import { supabase, type DbArtwork, type DbSaeFeature, type DbSaeFeatureLabel } from './lib/supabase'

// Partial types for specific queries
type SaeFeatureRow = Pick<DbSaeFeature, 'feature_idx' | 'monosemanticity_score' | 'tier' | 'all_correlations' | 'elo_correlation'>
type FeatureLabelRow = Pick<DbSaeFeatureLabel, 'feature_idx' | 'short_label'>
type FeaturesByQuestionRow = Pick<DbSaeFeature, 'feature_idx' | 'strongest_rating'>
type ArtworkForTopList = Pick<DbArtwork, 'id' | 'title' | 'artist_name' | 'image_url' | 'thumbnail_url'>

export interface Painting {
  id: string
  title: string
  artist: string | null
  year: number | null
  image_url: string
  thumbnail_url: string | null
}

export interface PaintingDetail extends Painting {
  labels: Record<string, unknown> | null
  description: string | null
}

export interface Question {
  id: string
  label: string
  description: string
}

export interface FeatureCorrelation {
  feature_idx: number
  label: string | null
  correlation: number
  p_value: number | null
  monosemanticity: number | null
  activation: number | null
  layer: number
  eloCorrelation: number | null
}

export interface PaintingFeatures {
  painting_id: string
  question_id: string
  features: FeatureCorrelation[]
  painting_specific: boolean
}

export interface Stats {
  paintings_with_labels: number
  features_analyzed: number
  features_labeled: number
  rating_questions: number
  paintings_with_heatmaps: number
}

export interface HeatmapFeature {
  feature_idx: number
  activation: number
}

export interface PaintingHeatmapData {
  painting_id: string
  title: string
  artist: string | null
  year: number | null
  image_url: string
  top_active_features: HeatmapFeature[]
  available_features: number[]
}

export interface FeatureHeatmap {
  feature_idx: number
  heatmap_7x7: number[][]
  max_activation: number
  mean_activation: number
  total_activation: number
}

export type FeaturesByQuestion = Record<string, number[]>

// Sort and filter options for paintings
export type SortField = 'mirror_self_rating' | 'year' | 'elo_percentile' | 'default'
export type SortDirection = 'asc' | 'desc'

export interface SortOptions {
  field: SortField
  direction: SortDirection
}

export interface FilterOptions {
  subjectMatter?: string[]
}

export interface FetchPaintingsOptions {
  limit?: number
  offset?: number
  sort?: SortOptions
  filter?: FilterOptions
}

// Sort and filter options for features
export type FeatureSortField = 'strongest_correlation' | 'elo_correlation' | 'monosemanticity_score'

export interface FeatureSortOptions {
  field: FeatureSortField
  direction: SortDirection
}

export interface FeatureFilterOptions {
  layers?: number[]              // 7, 8, 11
  strongestRating?: string[]     // mirror_self, drawn_to, etc.
  minMonosemanticity?: number    // 0-1 threshold
  correlationSign?: 'positive' | 'negative' | 'all'  // Filter by correlation sign
}

export interface FetchFeaturesOptions {
  limit?: number
  offset?: number
  sort?: FeatureSortOptions
  filter?: FeatureFilterOptions
}

export interface FeatureSummary {
  feature_idx: number
  layer: number
  label: string | null
  description: string | null
  monosemanticity: number | null
  strongest_rating: string | null
  strongest_correlation: number | null
  elo_correlation: number | null
  mirror_self_correlation: number | null
  top_artworks: TopArtwork[]
}

export type FeatureType = 'cls' | 'spatial'

export interface FeatureTypeInfo {
  id: FeatureType
  label: string
  description: string
  available: boolean
  has_correlations: boolean
  supports_heatmaps: boolean
}

export interface FeatureTypesResponse {
  types: FeatureTypeInfo[]
  default: FeatureType
  note: string
}

export interface ClaudeOpinion {
  rating: number
  explanation: string
  model: string
  question_id: string
}

export interface EloRating {
  rating: number
  rank: number | null
  totalArtworks: number
  matches: number
  wins: number
  losses: number
  percentile: number | null
}

export interface ModelOpinions {
  haiku35: ClaudeOpinion | null
}

// Rating dimensions with correlations in sae_features.all_correlations
// TODO: Add missing rating dimensions once correlations are computed:
//   - choose_to_look (Askell)
//   - emotional_impact (Askell)
//   - inner_light (Alexander)
//   - deepest_honest (Alexander)
const QUESTIONS: Question[] = [
  { id: 'mirror_self', label: 'Mirror of Self', description: 'Does this artwork reflect something of yourself?' },
  { id: 'drawn_to', label: 'Drawn To', description: 'How much does this artwork draw you in?' },
  { id: 'technical_skill', label: 'Technical Skill', description: 'How technically skilled is the execution?' },
  { id: 'wholeness', label: 'Wholeness', description: 'How complete and unified does this artwork feel?' },
]

// Available layers for CLS features
export const AVAILABLE_LAYERS = [7, 8, 11] as const
export type Layer = typeof AVAILABLE_LAYERS[number]

// Layer colors for visual distinction
export const LAYER_COLORS: Record<Layer, string> = {
  7: '#f97316',  // orange - early layer
  8: '#a855f7',  // purple - middle layer
  11: '#06b6d4', // cyan - late layer (default)
}

// Default layer for CLS features (layer 11 is the current standard)
const DEFAULT_LAYER = 11

// Minimum activation threshold to show a feature
const MINIMUM_ACTIVATION = 0.5

// Minimum monosemanticity score to show a feature (filters out polysemantic features)
const MINIMUM_MONOSEMANTICITY = 0.4

export async function fetchFeatureTypes(): Promise<FeatureTypesResponse> {
  return {
    types: [
      {
        id: 'cls',
        label: 'CLS Token',
        description: 'Global image features from the CLS token',
        available: true,
        has_correlations: true,
        supports_heatmaps: false,
      },
      {
        id: 'spatial',
        label: 'Spatial',
        description: 'Spatially-localized features with heatmaps',
        available: false,
        has_correlations: false,
        supports_heatmaps: true,
      },
    ],
    default: 'cls',
    note: 'CLS features available via Supabase',
  }
}

type OpinionWithRating = { artwork_id: string; rating: number }
type EloWithRank = { artwork_id: string; rank: number }

export async function fetchPaintings(options: FetchPaintingsOptions = {}): Promise<Painting[]> {
  const { limit = 50, offset = 0, sort, filter } = options
  const hasFilters = filter?.subjectMatter && filter.subjectMatter.length > 0

  let artworkIds: string[] = []
  let ratingMap = new Map<string, number>()
  let eloRankMap = new Map<string, number>()

  // Handle ELO percentile sorting separately
  if (sort?.field === 'elo_percentile') {
    // Fetch from artwork_elo_ratings, sorted by rank
    // For ELO percentile: desc = highest percentile first = lowest rank first
    let eloQuery = supabase
      .from('artwork_elo_ratings')
      .select('artwork_id, rank')
      .not('rank', 'is', null)
      .order('rank', { ascending: sort.direction === 'desc' })

    if (!hasFilters) {
      eloQuery = eloQuery.range(offset, offset + limit - 1)
    } else {
      eloQuery = eloQuery.range(0, 499)
    }

    const { data: eloData, error: eloError } = await eloQuery

    if (eloError) {
      throw new Error(`Failed to fetch ELO ratings: ${eloError.message}`)
    }

    if (!eloData || eloData.length === 0) {
      return []
    }

    artworkIds = (eloData as EloWithRank[]).map((row) => row.artwork_id)
    eloRankMap = new Map((eloData as EloWithRank[]).map((row) => [row.artwork_id, row.rank]))
  } else {
    // Step 1: Get artwork IDs with ratings from claude_opinions (mirror_self_v2 = Haiku 3.5)
    let opinionQuery = supabase
      .from('claude_opinions')
      .select('artwork_id, rating')
      .eq('question_id', 'mirror_self_v2')

    // Apply rating-based ordering if sorting by rating
    if (sort?.field === 'mirror_self_rating') {
      opinionQuery = opinionQuery.order('rating', {
        ascending: sort.direction === 'asc',
        nullsFirst: false,
      })
    } else {
      opinionQuery = opinionQuery.order('artwork_id')
    }

    // When no filters are applied and using default/rating sort, paginate directly
    // When filters are applied, fetch more to account for filtered-out items
    if (!hasFilters && sort?.field !== 'year') {
      opinionQuery = opinionQuery.range(offset, offset + limit - 1)
    } else {
      // Fetch larger batch when filtering or sorting by year (need artwork data)
      // Limit to 500 to keep queries fast
      opinionQuery = opinionQuery.range(0, 499)
    }

    const { data: opinionData, error: opinionError } = await opinionQuery

    if (opinionError) {
      throw new Error(`Failed to fetch paintings with opinions: ${opinionError.message}`)
    }

    if (!opinionData || opinionData.length === 0) {
      return []
    }

    // Create rating map for later use
    ratingMap = new Map(
      (opinionData as OpinionWithRating[]).map((row) => [row.artwork_id, row.rating])
    )
    artworkIds = (opinionData as OpinionWithRating[]).map((row) => row.artwork_id)
  }

  // Step 2: Fetch artwork data
  let artworksQuery = supabase
    .from('artworks')
    .select('id, title, artist_name, year, image_url, thumbnail_url, labels')
    .in('id', artworkIds)

  // Apply year sorting at query level
  if (sort?.field === 'year') {
    artworksQuery = artworksQuery.order('year', {
      ascending: sort.direction === 'asc',
      nullsFirst: false,
    })
  }

  const { data, error } = await artworksQuery

  if (error) {
    throw new Error(`Failed to fetch paintings: ${error.message}`)
  }

  let results = data || []

  // Apply subject matter filter in JavaScript (labels is TEXT, not JSONB)
  if (hasFilters) {
    const filterSet = new Set(filter!.subjectMatter!)
    results = results.filter((row) => {
      if (!row.labels) return false
      try {
        const labels = typeof row.labels === 'string' ? JSON.parse(row.labels) : row.labels
        return labels.subject_matter && filterSet.has(labels.subject_matter)
      } catch {
        return false
      }
    })
  }

  // Step 3: Re-apply sort order (since artworks query doesn't preserve it)
  if (sort?.field === 'mirror_self_rating') {
    results = results.sort((a, b) => {
      const ratingA = ratingMap.get(a.id) ?? -Infinity
      const ratingB = ratingMap.get(b.id) ?? -Infinity
      return sort.direction === 'asc' ? ratingA - ratingB : ratingB - ratingA
    })
  } else if (sort?.field === 'elo_percentile') {
    // For ELO percentile: desc = highest percentile first = lowest rank first
    results = results.sort((a, b) => {
      const rankA = eloRankMap.get(a.id) ?? Infinity
      const rankB = eloRankMap.get(b.id) ?? Infinity
      return sort.direction === 'desc' ? rankA - rankB : rankB - rankA
    })
  } else if (sort?.field === 'default' || !sort) {
    // Maintain original order from opinion query
    const orderMap = new Map(artworkIds.map((id, idx) => [id, idx]))
    results = results.sort((a, b) => (orderMap.get(a.id) ?? 0) - (orderMap.get(b.id) ?? 0))
  }

  // Apply pagination after filtering/sorting (only when we fetched larger batch)
  if (hasFilters || sort?.field === 'year' || sort?.field === 'elo_percentile') {
    results = results.slice(offset, offset + limit)
  }

  return results.map((row) => ({
    id: row.id,
    title: row.title,
    artist: row.artist_name,
    year: row.year,
    image_url: row.image_url,
    thumbnail_url: row.thumbnail_url,
  }))
}

export async function fetchPainting(id: string): Promise<PaintingDetail> {
  const { data, error } = await supabase
    .from('artworks')
    .select('id, title, artist_name, year, image_url, thumbnail_url, labels, description')
    .eq('id', id)
    .single()

  if (error) {
    throw new Error(`Failed to fetch painting: ${error.message}`)
  }

  return {
    id: data.id,
    title: data.title,
    artist: data.artist_name,
    year: data.year,
    image_url: data.image_url,
    thumbnail_url: data.thumbnail_url,
    labels: data.labels ? JSON.parse(data.labels) : null,
    description: data.description,
  }
}

export async function fetchQuestions(): Promise<Question[]> {
  return QUESTIONS
}

export async function fetchPaintingFeatures(
  paintingId: string,
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  _featureType: FeatureType = 'cls',
  layers: readonly number[] = [DEFAULT_LAYER]
): Promise<PaintingFeatures> {
  const questionId = 'mirror_self' // Hardcoded to Mirror of Self

  // Get the top features active in this painting from selected layers
  const { data: artworkFeatures, error: artworkError } = await supabase
    .from('artwork_top_features')
    .select('feature_idx, activation, rank, layer')
    .eq('artwork_id', paintingId)
    .in('layer', layers as unknown as number[])
    .gte('activation', MINIMUM_ACTIVATION)
    .order('activation', { ascending: false })

  if (artworkError) {
    throw new Error(`Failed to fetch artwork features: ${artworkError.message}`)
  }

  if (!artworkFeatures || artworkFeatures.length === 0) {
    return {
      painting_id: paintingId,
      question_id: questionId,
      features: [],
      painting_specific: true,
    }
  }

  // Create unique (layer, feature_idx) pairs
  type ArtworkFeatureRow = { feature_idx: number; activation: number; rank: number; layer: number }
  const featuresByLayer = new Map<number, number[]>()
  for (const f of artworkFeatures as ArtworkFeatureRow[]) {
    const existing = featuresByLayer.get(f.layer) || []
    existing.push(f.feature_idx)
    featuresByLayer.set(f.layer, existing)
  }

  // Fetch feature metadata and labels for each layer
  const allSaeFeatures: Array<SaeFeatureRow & { layer: number }> = []
  const allFeatureLabels: Array<FeatureLabelRow & { layer: number }> = []

  for (const [layer, featureIdxs] of featuresByLayer) {
    const [saeResult, labelResult] = await Promise.all([
      supabase
        .from('sae_features')
        .select('feature_idx, monosemanticity_score, tier, all_correlations, elo_correlation')
        .eq('layer', layer)
        .in('feature_idx', featureIdxs),
      supabase
        .from('sae_feature_labels')
        .select('feature_idx, short_label')
        .eq('layer', layer)
        .in('feature_idx', featureIdxs),
    ])

    if (saeResult.error) {
      throw new Error(`Failed to fetch SAE features: ${saeResult.error.message}`)
    }
    if (labelResult.error) {
      throw new Error(`Failed to fetch feature labels: ${labelResult.error.message}`)
    }

    for (const f of saeResult.data || []) {
      allSaeFeatures.push({ ...f, layer })
    }
    for (const f of labelResult.data || []) {
      allFeatureLabels.push({ ...f, layer })
    }
  }

  // Create lookup maps keyed by "layer:feature_idx"
  const makeKey = (layer: number, idx: number) => `${layer}:${idx}`
  const saeFeatureMap = new Map(allSaeFeatures.map((f) => [makeKey(f.layer, f.feature_idx), f]))
  const labelMap = new Map(allFeatureLabels.map((f) => [makeKey(f.layer, f.feature_idx), f.short_label]))

  // Build feature correlations with the selected question
  const features: FeatureCorrelation[] = []

  for (const af of artworkFeatures as ArtworkFeatureRow[]) {
    const key = makeKey(af.layer, af.feature_idx)
    const saeFeature = saeFeatureMap.get(key)

    // Filter out features with low monosemanticity (polysemantic features)
    const monosemanticity = saeFeature?.monosemanticity_score ?? 0
    if (monosemanticity < MINIMUM_MONOSEMANTICITY) {
      continue
    }

    let correlation = 0
    if (saeFeature?.all_correlations) {
      try {
        const correlations = JSON.parse(saeFeature.all_correlations)
        const value = correlations[questionId]
        // Handle both formats: flat number or nested {correlation: number}
        if (typeof value === 'number') {
          correlation = value
        } else if (value && typeof value === 'object' && 'correlation' in value) {
          correlation = value.correlation
        }
      } catch {
        // Ignore parse errors
      }
    }

    features.push({
      feature_idx: af.feature_idx,
      label: labelMap.get(key) || null,
      correlation,
      p_value: null,
      monosemanticity: saeFeature?.monosemanticity_score || null,
      activation: af.activation,
      layer: af.layer,
      eloCorrelation: saeFeature?.elo_correlation ?? null,
    })
  }

  // Sort by activation descending
  features.sort((a, b) => (b.activation ?? 0) - (a.activation ?? 0))

  return {
    painting_id: paintingId,
    question_id: questionId,
    features,
    painting_specific: true,
  }
}

export async function fetchStats(): Promise<Stats> {
  // Get counts from the database
  const [artworksResult, featuresResult, labelsResult] = await Promise.all([
    supabase.from('artworks').select('id', { count: 'exact', head: true }).not('labels', 'is', null),
    supabase.from('sae_features').select('id', { count: 'exact', head: true }),
    supabase.from('sae_feature_labels').select('id', { count: 'exact', head: true }),
  ])

  return {
    paintings_with_labels: artworksResult.count || 0,
    features_analyzed: featuresResult.count || 0,
    features_labeled: labelsResult.count || 0,
    rating_questions: QUESTIONS.length,
    paintings_with_heatmaps: 0, // Heatmaps not stored in Supabase yet
  }
}

export async function fetchPaintingsWithHeatmaps(): Promise<PaintingHeatmapData[]> {
  // Heatmaps are not stored in Supabase yet
  return []
}

// eslint-disable-next-line @typescript-eslint/no-unused-vars
export async function fetchPaintingHeatmapData(_paintingId: string): Promise<PaintingHeatmapData | null> {
  // Heatmaps are not stored in Supabase yet
  return null
}

// eslint-disable-next-line @typescript-eslint/no-unused-vars
export async function fetchFeatureHeatmap(_paintingId: string, _featureIdx: number): Promise<FeatureHeatmap | null> {
  // Heatmaps are not stored in Supabase yet
  return null
}

export async function fetchFeaturesByQuestion(): Promise<FeaturesByQuestion> {
  // Get all features and group by their strongest rating
  const { data, error } = await supabase
    .from('sae_features')
    .select('feature_idx, strongest_rating')
    .eq('layer', DEFAULT_LAYER)

  if (error) {
    throw new Error(`Failed to fetch features by question: ${error.message}`)
  }

  const result: FeaturesByQuestion = {}
  for (const row of (data || []) as FeaturesByQuestionRow[]) {
    if (row.strongest_rating) {
      // Extract question ID from rating like "subject:portrait_single" -> just use the rating
      const questionId = row.strongest_rating.split(':')[0] || row.strongest_rating
      if (!result[questionId]) {
        result[questionId] = []
      }
      result[questionId].push(row.feature_idx)
    }
  }

  return result
}

// Detailed feature info for the modal
export interface TopArtwork {
  id: string
  title: string
  artist: string | null
  image_url: string
  thumbnail_url: string | null
}

export interface CorrelationDetail {
  correlation: number
  p_value: number
}

export interface FeatureDetail {
  feature_idx: number
  label: string | null
  description: string | null
  visual_elements: string[] | null
  explains_rating: string | null
  monosemanticity: number | null
  strongest_rating: string | null
  strongest_correlation: number | null
  all_correlations: Record<string, CorrelationDetail>
  eloCorrelation: number | null
  top_artworks: TopArtwork[]
}

export async function fetchFeatureDetail(featureIdx: number, layer: number = DEFAULT_LAYER): Promise<FeatureDetail> {
  // Fetch the SAE feature
  const { data: saeFeature, error: saeError } = await supabase
    .from('sae_features')
    .select('*')
    .eq('layer', layer)
    .eq('feature_idx', featureIdx)
    .single()

  if (saeError) {
    throw new Error(`Failed to fetch feature: ${saeError.message}`)
  }

  // Fetch the feature label
  const { data: featureLabel } = await supabase
    .from('sae_feature_labels')
    .select('*')
    .eq('layer', layer)
    .eq('feature_idx', featureIdx)
    .single()

  // Fetch the top artworks from artwork_top_features (the source of truth for activations)
  let topArtworks: TopArtwork[] = []
  const { data: topFeatures } = await supabase
    .from('artwork_top_features')
    .select('artwork_id, activation')
    .eq('layer', layer)
    .eq('feature_idx', featureIdx)
    .order('activation', { ascending: false })
    .limit(10)

  if (topFeatures && topFeatures.length > 0) {
    const artworkIds = topFeatures.map((f: { artwork_id: string }) => f.artwork_id)
    const { data: artworks } = await supabase
      .from('artworks')
      .select('id, title, artist_name, image_url, thumbnail_url')
      .in('id', artworkIds)

    if (artworks) {
      // Maintain the order from topFeatures (highest activation first)
      const artworkMap = new Map(artworks.map((a: ArtworkForTopList) => [a.id, a]))
      topArtworks = artworkIds
        .map((id: string) => artworkMap.get(id))
        .filter((a): a is ArtworkForTopList => a != null)
        .map((a: ArtworkForTopList) => ({
          id: a.id,
          title: a.title,
          artist: a.artist_name,
          image_url: a.image_url,
          thumbnail_url: a.thumbnail_url,
        }))
    }
  }

  // Parse all correlations
  const allCorrelations: Record<string, CorrelationDetail> = {}
  if (saeFeature.all_correlations) {
    try {
      const parsed = JSON.parse(saeFeature.all_correlations) as Record<string, unknown>
      // Convert simple correlation values to CorrelationDetail format
      for (const [key, value] of Object.entries(parsed)) {
        if (typeof value === 'number') {
          allCorrelations[key] = { correlation: value, p_value: 0 }
        } else if (typeof value === 'object' && value !== null) {
          allCorrelations[key] = value as CorrelationDetail
        }
      }
    } catch {
      // Ignore parse errors
    }
  }

  // Parse visual elements
  let visualElements: string[] | null = null
  if (featureLabel?.visual_elements) {
    try {
      visualElements = JSON.parse(featureLabel.visual_elements)
    } catch {
      // Ignore parse errors
    }
  }

  return {
    feature_idx: featureIdx,
    label: featureLabel?.short_label || null,
    description: featureLabel?.description || null,
    visual_elements: visualElements,
    explains_rating: featureLabel?.explains_rating || null,
    monosemanticity: saeFeature.monosemanticity_score,
    strongest_rating: saeFeature.strongest_rating,
    strongest_correlation: saeFeature.strongest_correlation,
    all_correlations: allCorrelations,
    eloCorrelation: saeFeature.elo_correlation ?? null,
    top_artworks: topArtworks,
  }
}

export async function fetchClaudeOpinion(paintingId: string): Promise<ClaudeOpinion | null> {
  // Fetch v2 (visual details prompt) opinion - for backwards compatibility
  const { data, error } = await supabase
    .from('claude_opinions')
    .select('rating, explanation, model, question_id')
    .eq('artwork_id', paintingId)
    .eq('question_id', 'mirror_self_v2')
    .single()

  if (error || !data) {
    return null
  }

  return {
    rating: data.rating,
    explanation: data.explanation,
    model: data.model || 'claude-haiku-4-5-20251001',
    question_id: data.question_id,
  }
}

export async function fetchClaudeOpinions(paintingId: string): Promise<ModelOpinions> {
  // Fetch Haiku 3.5 opinion for this painting (mirror_self_v2)
  const { data, error } = await supabase
    .from('claude_opinions')
    .select('rating, explanation, model, question_id')
    .eq('artwork_id', paintingId)
    .eq('question_id', 'mirror_self_v2')
    .single()

  if (error || !data) {
    return { haiku35: null }
  }

  return {
    haiku35: {
      rating: data.rating,
      explanation: data.explanation,
      model: data.model || 'claude-3-5-haiku-20241022',
      question_id: data.question_id,
    }
  }
}

// Cache for total artworks count (doesn't change often)
let cachedTotalArtworks: number | null = null

export async function fetchEloRating(paintingId: string): Promise<EloRating | null> {
  // Fetch ELO rating for this painting
  const { data, error } = await supabase
    .from('artwork_elo_ratings')
    .select('rating, rank, matches, wins, losses')
    .eq('artwork_id', paintingId)
    .single()

  if (error || !data) {
    return null
  }

  // Get total artworks count for percentile calculation (cached)
  if (cachedTotalArtworks === null) {
    const { count } = await supabase
      .from('artwork_elo_ratings')
      .select('*', { count: 'exact', head: true })
    cachedTotalArtworks = count || 0
  }

  const totalArtworks = cachedTotalArtworks
  const percentile = data.rank && totalArtworks > 0
    ? Math.round((1 - (data.rank - 1) / totalArtworks) * 100)
    : null

  return {
    rating: Math.round(data.rating),
    rank: data.rank,
    totalArtworks,
    matches: data.matches,
    wins: data.wins,
    losses: data.losses,
    percentile,
  }
}

// Fetch features for the feature browser
export async function fetchFeatures(options: FetchFeaturesOptions = {}): Promise<{
  features: FeatureSummary[]
  total: number
}> {
  const {
    limit = 20,
    offset = 0,
    sort = { field: 'strongest_correlation', direction: 'desc' },
    filter = {}
  } = options

  // Build query with filters
  let query = supabase
    .from('sae_features')
    .select('feature_idx, layer, monosemanticity_score, strongest_rating, strongest_correlation, all_correlations, elo_correlation', { count: 'exact' })

  // Apply layer filter (default to all layers if not specified)
  if (filter.layers && filter.layers.length > 0) {
    query = query.in('layer', filter.layers)
  }

  // Apply strongest_rating filter
  if (filter.strongestRating && filter.strongestRating.length > 0) {
    query = query.in('strongest_rating', filter.strongestRating)
  }

  // Apply monosemanticity threshold (default to 0.5 if not specified)
  const minMono = filter.minMonosemanticity ?? MINIMUM_MONOSEMANTICITY
  query = query.gte('monosemanticity_score', minMono)

  // Apply correlation sign filter
  if (filter.correlationSign === 'positive') {
    query = query.gt('strongest_correlation', 0)
  } else if (filter.correlationSign === 'negative') {
    query = query.lt('strongest_correlation', 0)
  }

  // Apply sorting
  const sortColumn = sort.field === 'strongest_correlation' ? 'strongest_correlation'
    : sort.field === 'elo_correlation' ? 'elo_correlation'
    : 'monosemanticity_score'
  query = query.order(sortColumn, { ascending: sort.direction === 'asc', nullsFirst: false })

  // Fetch more than needed to account for features without artworks (we'll filter and paginate later)
  query = query.range(0, 999)

  const { data: features, error } = await query

  if (error) {
    throw new Error(`Failed to fetch features: ${error.message}`)
  }

  if (!features || features.length === 0) {
    return { features: [], total: 0 }
  }

  // Collect unique (layer, feature_idx) pairs
  const featureKeys = features.map(f => ({ layer: f.layer, idx: f.feature_idx }))

  // Fetch labels for all features - need to batch by layer
  const labelsByKey = new Map<string, { short_label: string; description: string }>()
  const layerGroups = new Map<number, number[]>()

  for (const { layer, idx } of featureKeys) {
    const existing = layerGroups.get(layer) || []
    existing.push(idx)
    layerGroups.set(layer, existing)
  }

  for (const [layer, idxs] of layerGroups) {
    const { data: labels } = await supabase
      .from('sae_feature_labels')
      .select('feature_idx, short_label, description')
      .eq('layer', layer)
      .in('feature_idx', idxs)

    if (labels) {
      for (const label of labels) {
        labelsByKey.set(`${layer}:${label.feature_idx}`, {
          short_label: label.short_label,
          description: label.description,
        })
      }
    }
  }

  // Fetch top artworks for each feature (limit to 5 per feature for cards)
  const artworksByKey = new Map<string, TopArtwork[]>()

  for (const [layer, idxs] of layerGroups) {
    // Get top activations for all features in this layer
    const { data: topFeatures } = await supabase
      .from('artwork_top_features')
      .select('feature_idx, artwork_id, activation')
      .eq('layer', layer)
      .in('feature_idx', idxs)
      .order('activation', { ascending: false })

    if (topFeatures) {
      // Group by feature and take top 5 for each
      const featureActivations = new Map<number, { artwork_id: string; activation: number }[]>()
      for (const tf of topFeatures) {
        const existing = featureActivations.get(tf.feature_idx) || []
        if (existing.length < 5) {
          existing.push({ artwork_id: tf.artwork_id, activation: tf.activation })
        }
        featureActivations.set(tf.feature_idx, existing)
      }

      // Collect all artwork IDs needed
      const allArtworkIds = new Set<string>()
      for (const activations of featureActivations.values()) {
        for (const a of activations) {
          allArtworkIds.add(a.artwork_id)
        }
      }

      // Fetch artwork details
      if (allArtworkIds.size > 0) {
        const { data: artworks } = await supabase
          .from('artworks')
          .select('id, title, artist_name, image_url, thumbnail_url')
          .in('id', Array.from(allArtworkIds))

        if (artworks) {
          const artworkMap = new Map(artworks.map(a => [a.id, a]))

          // Build artwork lists for each feature
          for (const [featureIdx, activations] of featureActivations) {
            const topArtworks: TopArtwork[] = []
            for (const { artwork_id } of activations) {
              const artwork = artworkMap.get(artwork_id)
              if (artwork) {
                topArtworks.push({
                  id: artwork.id,
                  title: artwork.title,
                  artist: artwork.artist_name,
                  image_url: artwork.image_url,
                  thumbnail_url: artwork.thumbnail_url,
                })
              }
            }
            artworksByKey.set(`${layer}:${featureIdx}`, topArtworks)
          }
        }
      }
    }
  }

  // Build enriched feature summaries
  const enrichedFeatures: FeatureSummary[] = features.map(f => {
    const key = `${f.layer}:${f.feature_idx}`
    const label = labelsByKey.get(key)
    const topArtworks = artworksByKey.get(key) || []

    // Parse mirror_self correlation from all_correlations
    let mirrorSelfCorr: number | null = null
    if (f.all_correlations) {
      try {
        const correlations = JSON.parse(f.all_correlations)
        const value = correlations['mirror_self']
        if (typeof value === 'number') {
          mirrorSelfCorr = value
        } else if (value && typeof value === 'object' && 'correlation' in value) {
          mirrorSelfCorr = value.correlation
        }
      } catch {
        // Ignore parse errors
      }
    }

    return {
      feature_idx: f.feature_idx,
      layer: f.layer,
      label: label?.short_label || null,
      description: label?.description || null,
      monosemanticity: f.monosemanticity_score,
      strongest_rating: f.strongest_rating,
      strongest_correlation: f.strongest_correlation,
      elo_correlation: f.elo_correlation,
      mirror_self_correlation: mirrorSelfCorr,
      top_artworks: topArtworks,
    }
  })

  // Filter to only features that have artwork data, then paginate
  const featuresWithArtworks = enrichedFeatures.filter(f => f.top_artworks.length > 0)
  const paginatedFeatures = featuresWithArtworks.slice(offset, offset + limit)

  return { features: paginatedFeatures, total: featuresWithArtworks.length }
}

// Fetch unique strongest_rating values for filter dropdown
export async function fetchStrongestRatingOptions(): Promise<string[]> {
  const { data, error } = await supabase
    .from('sae_features')
    .select('strongest_rating')
    .not('strongest_rating', 'is', null)

  if (error) {
    throw new Error(`Failed to fetch rating options: ${error.message}`)
  }

  const unique = [...new Set(data?.map(d => d.strongest_rating).filter((r): r is string => r !== null))]
  return unique.sort()
}
