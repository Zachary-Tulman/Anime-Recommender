import pandas as pd

def build_mask(df, min_score=None, max_episodes=None, status=None, source_type=None, min_scores=None):
    mask = pd.Series(True, index=df.index)

    if min_score is not None:
        mask &= (df["score"] >= min_score)
    
    if max_episodes is not None:
        mask &= (df["num_episodes"] <= max_episodes)

    if status is not None:
        mask &= (df["status"] == status)
    
    if source_type is not None:
        mask &= (df["source_type"] == source_type)
    
    if min_scores is not None:
        mask &= (df["score_count"] >= min_scores)

    return mask