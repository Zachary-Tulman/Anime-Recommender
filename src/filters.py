import pandas as pd

from dataclasses import dataclass

@dataclass(kw_only=True)
class MaskFilters:
    min_score: int | None = None
    max_episodes: int | None = None
    status: list[str] | None = None
    min_score_count: int | None = 20

def build_mask(df, filters: MaskFilters):
    """Create a boolean mask for filtering shows based on user query"""
    mask = pd.Series(True, index=df.index)

    if filters.min_score is not None:
        mask &= (df["score"] >= filters.min_score)
    
    if filters.max_episodes is not None:
        mask &= (df["num_episodes"] <= filters.max_episodes)

    if filters.status is not None:
        mask &= (df["status"].isin(filters.status))
    else:
        mask &= (df["status"].isin(["Finished Airing", "Currently Airing"]))
    
    if filters.min_score_count is not None:
        mask &= (df["score_count"] >= filters.min_score_count)

    return mask