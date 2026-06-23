import pandas as pd

def build_mask(df, min_score=None, max_episodes=None, status=None, min_score_count=20):
    mask = pd.Series(True, index=df.index)

    if min_score is not None:
        mask &= (df["score"] >= min_score)
    
    if max_episodes is not None:
        mask &= (df["num_episodes"] <= max_episodes)

    if status is not None:
        if isinstance(status, str): status = [status]
        mask &= (df["status"].isin(status))
    else:
        mask &= (df["status"].isin(["Finished Airing", "Currently Airing"]))
    
    if min_score_count is not None:
        mask &= (df["score_count"] >= min_score_count)

    return mask