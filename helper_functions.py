import numpy as np 
import pandas as pd 
import ast
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import plotly.express as px
import matplotlib.pyplot as plt

def user_vector(user_id,train_data,bid_to_row,X_items,like_threshold=4):
    
    liked_by_user = (
    train_data[train_data['stars_review'] >= like_threshold]
    .groupby('user_id')['business_id']
    .apply(list)
    .to_dict()
    )

    liked_ids = liked_by_user.get(user_id, [])
    rows = [bid_to_row[b] for b in liked_ids if b in bid_to_row]
    if not rows:
        return None      # cold-start user (no liked items)
    vec = X_items[rows].mean(axis=0)   # average vectors
    
    return np.asarray(vec)


def recommend_content(user_id,X_items,business_metadata,train_data,bid_to_row,prefer_same_city=True):
    u_vec = user_vector(user_id, train_data, bid_to_row, X_items)                       # 1) user taste vector
    if u_vec is None:
        return pd.DataFrame()                         

    sims = cosine_similarity(u_vec, X_items).ravel()   # 2) cosine similarity to every item

    seen = set(train_data.loc[train_data['user_id']==user_id, 'business_id'])
    for b in seen:                                     # 3) We don’t want to recommend a place the user already reviewed in train.
        idx = bid_to_row.get(b)
        if idx is not None:
            sims[idx] = -1

    rec = pd.DataFrame({'business_id': business_metadata['business_id'].values,
                        'score': sims})                # 4) attach scores
    rec = rec.merge(business_metadata[['business_id','name','city','is_open',
                           'cuisines','venue_type']],
                    on='business_id', how='left')

    rec = rec[rec['is_open'] == True]                  # 5) keep open places only

    if prefer_same_city:                               # 6)  user’s city
        b = train_data.loc[train_data['user_id']==user_id, 'business_id']
        cities = business_metadata.set_index('business_id').loc[b, 'city'].dropna()
        if not cities.empty:
            u_city = cities.mode().iat[0]              # modal (most frequent) city for the user
            same = rec[rec['city'] == u_city]
            
    return (rec.sort_values('score', ascending=False)  # 7) rank by similarity
               [['name','cuisines',
                         'venue_type','score']]
               .reset_index(drop=True))

def get_user_name(user_id,train_data):
    # Return the user_id itself if no 'name_user' column exists
    if 'name_user' not in train_data.columns:
        return f"User {user_id}"
    
    user_data = train_data[train_data['user_id'] == user_id]
    if not user_data.empty:
        return user_data['name_user'].iloc[0]


def user_history(user_id, train_data, business_metadata):
    user_activity = (train_data[train_data['user_id']==user_id]
            .sort_values('date', ascending=False)
            .merge(business_metadata[['business_id','name','city','cuisines','venue_type']],
                   on='business_id', how='left'))
    return user_activity

def user_recent_visited(user_id, train_data, business_metadata, n=5):
    hist = user_history(user_id, train_data, business_metadata)
    # last N unique places by most recent review
    recent = (hist.drop_duplicates('business_id', keep='first')
                 .head(n)[['name','cuisines','stars_review','date']])
    return recent

def rerank_with_cuisine_caps(df, k=10, cap_per_cuisine=3):
    # df: columns include ["name","score","cuisines" or "cuisine_list"]
    # pick a single primary cuisine for capping
    def primary_cuisine(row):
        lst = row.get("cuisines") or row.get("cuisines")
        if isinstance(lst, list) and lst: return lst[0]
        if isinstance(lst, str) and lst.startswith("["):  # stringified list
            try:
                import ast; L = ast.literal_eval(lst)
                return L[0] if L else None
            except Exception:
                pass
        return None

    df = df.copy()
    df["primary_cuisine"] = df.apply(primary_cuisine, axis=1)
    counts = {}
    out = []
    for _, r in df.sort_values("score", ascending=False).iterrows():
        c = r["primary_cuisine"]
        if c is None: c = "__unknown__"
        if counts.get(c, 0) < cap_per_cuisine:
            out.append(r)
            counts[c] = counts.get(c, 0) + 1
        if len(out) >= k:
            break
    return (pd.DataFrame(out)
              .drop(columns=["primary_cuisine"], errors="ignore")
              .reset_index(drop=True))

# Display related functions 
def _to_list(x):
    if isinstance(x, list):
        return x
    if pd.isna(x):
        return []
    s = str(x).strip()
    if s.startswith("[") and s.endswith("]"):
        try:
            return list(ast.literal_eval(s))
        except Exception:
            pass
    return [t.strip() for t in s.split(",") if t.strip()]

def _flatten(xs):
    out = []
    for x in xs or []:
        out.extend(x if isinstance(x, list) else [x])
    return [str(t).lower() for t in out if str(t).strip()]

def _list_to_text(x):
    return ", ".join(_flatten(_to_list(x)))

def rec_plain(score: float) -> str:
    if score >= 0.70:
        return "🌟 Top match - a great fit for your tastes!"
    if score >= 0.40:
        return "👍 Good match - you might enjoy this one!"
    return "Wildcard pick - worth exploring for a new experience"

def compute_user_snapshot(user_id, train_data, business_metadata):
    # Join user history with metadata you need
    hist = (train_data[train_data["user_id"] == user_id]
              .merge(business_metadata[["business_id", "cuisines", "food_type"]],
                     on="business_id", how="left"))

    # Totals
    total_places = int(hist["business_id"].nunique())
    avg_rating = float(hist["stars_review"].mean()) if len(hist) else np.nan

    # Most tried cuisine
    cuisines_flat = []
    for v in hist["cuisines"].dropna():
        cuisines_flat.extend(_flatten(_to_list(v)))
    top_cuisine = Counter(cuisines_flat).most_common(1)[0][0] if cuisines_flat else "—"

    # Favorite dish
    dishes_flat = []
    for v in hist["food_type"].dropna():
        dishes_flat.extend(_flatten(_to_list(v)))
    fav_dish = Counter(dishes_flat).most_common(1)[0][0] if dishes_flat else "—"

    fav_restaurant = "—"
    if len(hist):
        counts = hist.groupby("business_id").size().rename("n")
        means  = hist.groupby("business_id")["stars_review"].mean().rename("mean_rating")
        recency = hist.assign(date=pd.to_datetime(hist["date"], errors="coerce")) \
                      .groupby("business_id")["date"].max().rename("last_date")

        fav_df = (
            pd.concat([counts, means, recency], axis=1)
            .sort_values(["n", "mean_rating", "last_date"], ascending=[False, False, False])
        )
        top_bid = fav_df.index[0]
        # map to name
        name_map = business_metadata.set_index("business_id")["name"]
        fav_restaurant = name_map.get(top_bid, "—")

    # --- average visit frequency (visits/year)
    visits_per_month = np.nan
    if len(hist):
        dates = pd.to_datetime(hist["date"], errors="coerce").dropna()
        if len(dates):
            span_days = (dates.max() - dates.min()).days
            # scale to yearly even for short histories
            visits = len(dates)
            visits_per_month = visits * (30.44 / max(span_days, 1))
            visits_per_month = round(visits_per_month)

    return {
        "top_cuisine": top_cuisine,
        "fav_dish": fav_dish,
        "avg_rating": avg_rating,
        "total_places": total_places,
        "fav_restaurant": fav_restaurant,
        "visits_per_month": visits_per_month
    }


def plot_user_cuisine_distribution(user_id, train_data, business_metadata):
    """
    Plot top cuisines tried by a given user.
    """
    # Filter only user's records and merge cuisine info
    user_df = (
        train_data[train_data["user_id"] == user_id]
        .merge(business_metadata[["business_id", "cuisines"]], on="business_id", how="left")
    )

    if user_df.empty:
        return None

    # Ensure cuisines are list-like and flatten
    user_df["cuisines"] = user_df["cuisines"].apply(_to_list)
    user_df = user_df.explode("cuisines").dropna(subset=["cuisines"])
    if user_df.empty:
        return None

    # Count top cuisines
    cuisine_counts = (
        user_df["cuisines"]
        .str.title()
        .value_counts()
        .reset_index()
        .rename(columns={"index": "Cuisine", "cuisines": "Visits"})  # ensure proper labels
        .head(5)
    )

    cuisine_counts.columns = ["Cuisine", "Visits"]

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(cuisine_counts["Cuisine"], cuisine_counts["Visits"], color="#D7FF61D7",edgecolor="black")
    ax.set_xlabel("Number of Visits",fontsize=12)
    ax.set_ylabel("Cuisine",fontsize=12)
    ax.set_title(f"Your Top 5 Go-To Cuisines",fontsize=15,fontweight='bold')
    ax.tick_params(axis='x', labelsize=10)                                 
    ax.tick_params(axis='y', labelsize=10)
    #ax.invert_yaxis()
    plt.tight_layout()

    return fig


