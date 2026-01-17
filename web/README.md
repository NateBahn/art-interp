# art-interp Web Frontend

Interactive web UI for exploring "What Does AI See in Art?" - visualizing SAE features and their correlations with AI aesthetic ratings.

## Quick Start

```bash
# Install dependencies
npm install

# Copy .env.example to .env and configure Supabase credentials
cp .env.example .env

# Start development server
npm run dev
```

## Environment Variables

The frontend connects directly to Supabase for data access. Configure these in `.env`:

```
VITE_SUPABASE_URL=your-supabase-project-url
VITE_SUPABASE_ANON_KEY=your-supabase-anon-key
```

## Features

- **Gallery View**: Browse artworks sorted by AI ratings, ELO scores, or year
- **Feature Explorer**: Discover SAE features that correlate with aesthetic dimensions
- **Painting Detail**: View individual artworks with their active features
- **Feature Detail**: See what visual patterns each SAE feature detects

## Architecture

This is a React + TypeScript + Vite application that:
- Connects directly to Supabase for all data queries
- Uses Tailwind CSS v4 for styling
- Supports filtering by subject matter, layer, and rating dimension

## Database Schema (Supabase)

The frontend expects these tables:

- `artworks` - Artwork metadata, images, labels
- `sae_features` - SAE feature metadata, correlations
- `sae_feature_labels` - Gemini-generated feature labels
- `artwork_top_features` - Top feature activations per artwork
- `claude_opinions` - AI model ratings
- `artwork_elo_ratings` - ELO tournament rankings

## Development

```bash
npm run dev      # Start dev server
npm run build    # Production build
npm run preview  # Preview production build
npm run lint     # Run ESLint
```

## Tech Stack

- React 19
- TypeScript
- Vite 7
- Tailwind CSS 4
- Supabase (database + auth)
