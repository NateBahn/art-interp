"""
Feature Selection for Interpretation

Selects which SAE features to prioritize for labeling/interpretation.
Uses a tiered approach based on monosemanticity and correlation strength.

Tiers:
- Tier 1 (Elite): High monosemanticity AND high correlation with ratings
- Tier 2 (Good): Moderate monosemanticity OR high correlation
- Tier 3 (Diverse): Exploratory - top by monosemanticity alone

This stratified selection ensures we interpret:
1. The most clearly interpretable features (Tier 1)
2. Features that predict human ratings (all tiers)
3. Diverse representation across feature space (Tier 3)

Example:
    >>> selector = FeatureSelector(
    ...     monosemanticity_scores=mono_scores,
    ...     correlation_results=corr_results,
    ... )
    >>> selected = selector.select_features(n_per_tier=50)
    >>> for tier, features in selected.items():
    ...     print(f"{tier}: {len(features)} features")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from interpretability.analysis.correlations import CorrelationResults
    from interpretability.analysis.monosemanticity import MonosemanticityResult

logger = logging.getLogger(__name__)

# Default thresholds (can be overridden)
DEFAULT_MONO_THRESHOLD_ELITE = 0.50  # Monosemantic (+2σ)
DEFAULT_MONO_THRESHOLD_GOOD = 0.35  # Moderate (+1σ)
DEFAULT_CORR_THRESHOLD_ELITE = 0.25  # Strong correlation
DEFAULT_CORR_THRESHOLD_GOOD = 0.15  # Moderate correlation


@dataclass
class SelectedFeature:
    """A feature selected for interpretation."""

    feature_idx: int
    tier: int  # 1, 2, or 3
    monosemanticity_score: float
    strongest_correlation: float
    strongest_rating: str
    top_items: list[str]
    selection_reason: str


class FeatureSelector:
    """
    Select features for interpretation based on monosemanticity and correlations.

    Implements a tiered selection strategy to balance:
    - Interpretability (high monosemanticity)
    - Predictive power (high correlation with ratings)
    - Diversity (coverage of feature space)

    Attributes:
        mono_threshold_elite: Monosemanticity threshold for Tier 1
        mono_threshold_good: Monosemanticity threshold for Tier 2
        corr_threshold_elite: Correlation threshold for Tier 1
        corr_threshold_good: Correlation threshold for Tier 2
    """

    def __init__(
        self,
        mono_threshold_elite: float = DEFAULT_MONO_THRESHOLD_ELITE,
        mono_threshold_good: float = DEFAULT_MONO_THRESHOLD_GOOD,
        corr_threshold_elite: float = DEFAULT_CORR_THRESHOLD_ELITE,
        corr_threshold_good: float = DEFAULT_CORR_THRESHOLD_GOOD,
    ) -> None:
        """
        Initialize the selector with thresholds.

        Args:
            mono_threshold_elite: Monosemanticity >= this for elite tier
            mono_threshold_good: Monosemanticity >= this for good tier
            corr_threshold_elite: |Correlation| >= this for elite tier
            corr_threshold_good: |Correlation| >= this for good tier
        """
        self.mono_threshold_elite = mono_threshold_elite
        self.mono_threshold_good = mono_threshold_good
        self.corr_threshold_elite = corr_threshold_elite
        self.corr_threshold_good = corr_threshold_good

    def select_features(
        self,
        monosemanticity_results: list[MonosemanticityResult],
        correlation_results: CorrelationResults | None = None,
        n_per_tier: int = 50,
    ) -> dict[str, list[SelectedFeature]]:
        """
        Select features for each tier.

        Args:
            monosemanticity_results: Results from MonosemanticityScorer
            correlation_results: Optional results from CorrelationAnalyzer
            n_per_tier: Target number of features per tier

        Returns:
            Dict mapping tier name to list of SelectedFeature
        """
        # Get best correlation for each feature (if available)
        best_corr: dict[int, tuple[float, str]] = {}
        if correlation_results:
            for corr in correlation_results.correlations:
                feat_idx = corr.feature_idx
                abs_corr = abs(corr.correlation)
                if feat_idx not in best_corr or abs_corr > best_corr[feat_idx][0]:
                    best_corr[feat_idx] = (abs_corr, corr.rating_dimension)

        # Tier 1: Elite - high mono AND high correlation
        tier1: list[SelectedFeature] = []
        for mono in monosemanticity_results:
            if mono.score < self.mono_threshold_elite:
                continue
            feat_idx = mono.feature_idx
            if feat_idx in best_corr and best_corr[feat_idx][0] >= self.corr_threshold_elite:
                corr_val, rating = best_corr[feat_idx]
                tier1.append(
                    SelectedFeature(
                        feature_idx=feat_idx,
                        tier=1,
                        monosemanticity_score=mono.score,
                        strongest_correlation=corr_val,
                        strongest_rating=rating,
                        top_items=mono.top_items,
                        selection_reason=f"High mono ({mono.score:.2f}) + high corr ({corr_val:.2f})",
                    )
                )

        # Sort by combined score
        tier1.sort(
            key=lambda f: f.monosemanticity_score + f.strongest_correlation, reverse=True
        )
        tier1 = tier1[:n_per_tier]
        tier1_indices = {f.feature_idx for f in tier1}

        # Tier 2: Good - moderate mono OR high correlation (not in Tier 1)
        tier2: list[SelectedFeature] = []
        for mono in monosemanticity_results:
            feat_idx = mono.feature_idx
            if feat_idx in tier1_indices:
                continue

            corr_val, rating = best_corr.get(feat_idx, (0.0, ""))

            qualifies_mono = mono.score >= self.mono_threshold_good
            qualifies_corr = corr_val >= self.corr_threshold_good

            if qualifies_mono or qualifies_corr:
                reason_parts = []
                if qualifies_mono:
                    reason_parts.append(f"mono ({mono.score:.2f})")
                if qualifies_corr:
                    reason_parts.append(f"corr ({corr_val:.2f})")

                tier2.append(
                    SelectedFeature(
                        feature_idx=feat_idx,
                        tier=2,
                        monosemanticity_score=mono.score,
                        strongest_correlation=corr_val,
                        strongest_rating=rating,
                        top_items=mono.top_items,
                        selection_reason=" + ".join(reason_parts),
                    )
                )

        tier2.sort(
            key=lambda f: f.monosemanticity_score + f.strongest_correlation, reverse=True
        )
        tier2 = tier2[:n_per_tier]
        tier2_indices = {f.feature_idx for f in tier2}

        # Tier 3: Diverse - top by monosemanticity alone (not in Tier 1 or 2)
        tier3: list[SelectedFeature] = []
        sorted_by_mono = sorted(
            monosemanticity_results, key=lambda m: m.score, reverse=True
        )

        for mono in sorted_by_mono:
            feat_idx = mono.feature_idx
            if feat_idx in tier1_indices or feat_idx in tier2_indices:
                continue

            corr_val, rating = best_corr.get(feat_idx, (0.0, ""))

            tier3.append(
                SelectedFeature(
                    feature_idx=feat_idx,
                    tier=3,
                    monosemanticity_score=mono.score,
                    strongest_correlation=corr_val,
                    strongest_rating=rating,
                    top_items=mono.top_items,
                    selection_reason=f"Top mono ({mono.score:.2f})",
                )
            )

            if len(tier3) >= n_per_tier:
                break

        logger.info(
            f"Selected features: Tier1={len(tier1)}, Tier2={len(tier2)}, Tier3={len(tier3)}"
        )

        return {
            "tier1_elite": tier1,
            "tier2_good": tier2,
            "tier3_diverse": tier3,
        }

    def select_for_rating(
        self,
        monosemanticity_results: list[MonosemanticityResult],
        correlation_results: CorrelationResults,
        rating_dimension: str,
        n: int = 20,
    ) -> list[SelectedFeature]:
        """
        Select top features for a specific rating dimension.

        Args:
            monosemanticity_results: Monosemanticity scores
            correlation_results: Correlation results
            rating_dimension: Which rating to optimize for
            n: Number of features to select

        Returns:
            List of SelectedFeature most predictive of the rating
        """
        mono_by_idx = {r.feature_idx: r for r in monosemanticity_results}

        # Get correlations for this dimension
        dim_corrs = [
            c
            for c in correlation_results.correlations
            if c.rating_dimension == rating_dimension and c.feature_idx in mono_by_idx
        ]

        # Sort by |correlation|
        dim_corrs.sort(key=lambda c: abs(c.correlation), reverse=True)

        selected = []
        for corr in dim_corrs[:n]:
            mono = mono_by_idx[corr.feature_idx]
            selected.append(
                SelectedFeature(
                    feature_idx=corr.feature_idx,
                    tier=1 if mono.score >= self.mono_threshold_elite else 2,
                    monosemanticity_score=mono.score,
                    strongest_correlation=abs(corr.correlation),
                    strongest_rating=rating_dimension,
                    top_items=mono.top_items,
                    selection_reason=f"Top for {rating_dimension}",
                )
            )

        return selected
