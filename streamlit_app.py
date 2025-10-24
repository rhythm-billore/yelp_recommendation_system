

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity

from helper_functions import (
    user_vector,
    recommend_content,
    get_user_name,
    user_history,
    user_recent_visited,
    rerank_with_cuisine_caps,
    _to_list,
    _flatten,
    _list_to_text,
    rec_plain,
    compute_user_snapshot,
    plot_user_cuisine_distribution)

st.set_page_config(layout="wide")

art = "artifacts"

@st.cache_resource
def load_artifacts():
    business_metadata = pd.read_parquet(f"{art}/business_metadata.parquet")
    train_data = pd.read_parquet(f"{art}/train_data.parquet")
    liked_by_user = joblib.load(f"{art}/liked_by_user.joblib")
    tfidf = joblib.load(f"{art}/tfidf.joblib")
    X_items = sp.load_npz(f"{art}/X_items.npz")
    bid_to_row = joblib.load(f"{art}/bid_to_row.joblib")
    row_to_bid = np.load(f"{art}/row_to_bid.npy", allow_pickle=True)
    return business_metadata,train_data, liked_by_user, tfidf, X_items, bid_to_row, row_to_bid

business_metadata,train_data, liked_by_user, tfidf, X_items, bid_to_row, row_to_bid = load_artifacts()


st.markdown(
    "<h3 style='text-align: center; color: #333; font-weight: 700;'>🍽️ Yelp Restaurant Recommendation System</h3>",
    unsafe_allow_html=True
)

YELP_LOGO = "https://upload.wikimedia.org/wikipedia/commons/a/ad/Yelp_Logo.svg"   # or a URL to the logo

with st.sidebar:
    # Display the logo (centered)
    st.markdown(
        f"""
        <div style="text-align:center; padding-top:10px;">
            <h2 style="color:#FF0000; font-family:'Montserrat', sans-serif; margin-bottom:4px;">
                PHILADELPHIA
            </h2>
            <img src="{YELP_LOGO}" alt="Yelp Logo" width="100">
        </div>
        """,
        unsafe_allow_html=True
    )

    # Divider for a subtle separation
    st.markdown("---")

    # User section
    st.title("👤 User Details")
    user_id = st.text_input("Enter your User ID:")

    st.markdown("---")
     
    snap = compute_user_snapshot(user_id, train_data, business_metadata)

    st.markdown("## 🍽️ Your Dining Snapshot")

    # Snapshot metrics (now stacked vertically)
    st.metric("🌍  Most Enjoyed Cuisine", snap["top_cuisine"].title() if snap["top_cuisine"] != "—" else "—")
    st.metric("🥇  Go-To Dish", snap["fav_dish"].title() if snap["fav_dish"] != "—" else "—")
    st.metric("⭐  Your Average Rating", f'{snap["avg_rating"]:.1f}' if not np.isnan(snap["avg_rating"]) else "—")
    st.metric("📍  Total Venues Explored", f'{snap["total_places"]}')
    st.metric("❤️  Top-Rated Restaurant", snap["fav_restaurant"] if snap["fav_restaurant"] != "—" else "—")
    st.metric("⏱️  Monthly Dining Frequency", f"{snap['visits_per_month']} visits/month" if snap["visits_per_month"] == snap["visits_per_month"] else "—")

    st.markdown("---")

    st.markdown(
        """
        <style>
        [data-testid="stMetricValue"] {
            font-size: 1rem !important;
            font-weight: 600 !important;
            color: #262730 !important;
        }
        [data-testid="stMetricLabel"] {
            font-size: 0.85rem !important;
            color: #555 !important;
        }
        section[data-testid="stSidebar"] {
            width: 320px !important;   /* makes sidebar slightly wider */
        }
        </style>
        """,
        unsafe_allow_html=True
    )

user_name = get_user_name(user_id,train_data)
if user_name:
    st.subheader(f"👋 Welcome {user_name} !")
#st.markdown("Enter your **User ID** to see your top recommendations based on your preferences.")

col_a, col_b = st.columns([1, 1], gap="small")


# make the central block wider and reduce padding
st.markdown("""
<style>
.block-container {
    max-width: 100vw;
    padding-left: 3rem;
    padding-right: 3rem;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
table {
    font-size: 0.8rem !important;
    width: 100% !important;
}
th, td {
    white-space: nowrap !important;
}
</style>
""", unsafe_allow_html=True) 

with col_a:
    st.markdown("##### 🍴 Your 5 Most Recent Visits")
    recent_df = user_recent_visited(user_id, train_data, business_metadata, n=5)

    if len(recent_df):
        recent_df_disp = recent_df.copy()
        recent_df_disp["date"] = pd.to_datetime(recent_df_disp["date"]).dt.strftime("%Y-%m-%d")

        if "cuisines" in recent_df_disp.columns:
            recent_df_disp["cuisines"] = recent_df_disp["cuisines"].apply(_list_to_text)

        # Create a final tidy view
        recent5 = recent_df_disp.rename(
            columns={
                "name": "Restaurant",
                "cuisines": "Cuisines",
                "stars_review": "User Rating",
                "date": "Date of Visit",
            }
        )

        st.data_editor(
            recent5[["Restaurant", "Cuisines", "User Rating", "Date of Visit"]],
            hide_index=True,
            disabled=True,
            use_container_width=True,
            column_config={
                "Restaurant": st.column_config.TextColumn("Restaurant", width="small"),
                "Cuisines": st.column_config.TextColumn("Cuisines", width="small"),
                "User Rating": st.column_config.NumberColumn("User Rating", width="small"),
                "Date of Visit": st.column_config.TextColumn("Date of Visit", width="small"),
            },
        )
    else:
        st.caption("No recent visits in training history.")

with col_b:
    fig = plot_user_cuisine_distribution(user_id, train_data, business_metadata)
    if fig:
        st.pyplot(fig)

if user_id:
    with st.spinner("Generating recommendations..."):
        recs = recommend_content(
            user_id=user_id,
            X_items=X_items,
            train_data=train_data,
            bid_to_row=bid_to_row,
            business_metadata=business_metadata,
            prefer_same_city=True,
        )

        # Re-rank to limit duplicates from the same cuisine
        revisit_recs = rerank_with_cuisine_caps(recs, k=10, cap_per_cuisine=3)

    # --- display section ---
    if revisit_recs is not None and not revisit_recs.empty:
        st.markdown(
            """
            <div style="
                background-color:#D7FF6105;
                padding:15px 20px;
                border-radius:8px;
                border:1px solid #B2D8B2;
                font-size:16px;
                line-height:1.6;
                color:#155724;">
                🍽️ <strong>Your personalized restaurant recommendations:</strong><br><br>
                Food lovers with preferences like yours enjoyed these spots , you might like them too!
            </div>
            """,
            unsafe_allow_html=True
            )
        # Build a display-only copy
        show = revisit_recs.copy()

        # Prettify list-like columns
        for c in ["cuisines", "venue_type"]:
            if c in show.columns:
                show[c] = show[c].apply(_list_to_text)

        # Rename & compute display columns (native Streamlit, no HTML)
        show = show.assign(
            Restaurant=show["name"],
            Cuisines=show["cuisines"],
            Venue=show["venue_type"],
            **{"Recommendation Score": show["score"].round(2)},
            #Indicator=show["score"].apply(rec_dot),
            **{"Why You'll Love It": show["score"].apply(rec_plain)},
        )

        out = show[
            ["Restaurant", "Cuisines", "Venue", "Recommendation Score", "Why You'll Love It"]
        ]

        st.dataframe(
            out,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Restaurant": st.column_config.TextColumn("Restaurant", width="medium"),
                "Cuisines": st.column_config.TextColumn("Cuisines", width="small"),
                "Venue": st.column_config.TextColumn("Venue", width="small"),
                "Recommendation Score": st.column_config.NumberColumn(
                    "Recommendation Score", format="%.2f", min_value=0.0, max_value=1.0,width="small"
                ),
                #"Indicator": st.column_config.TextColumn("", width="small"),
                "Why You'll Love It": st.column_config.TextColumn("Why You'll Love It", width="stretch"),
            },
        )

    else:
        st.warning("No recommendations found for this user.")
else:
    st.caption("👆 Enter a valid User ID to begin.")
