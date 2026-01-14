#------------------
#Imports
#------------------
import streamlit as st
import pickle
import numpy as np
import requests

# -----------------------------
# Streamlit Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Hybrid Movie Recommendation System",
    layout="wide"
)

# -----------------------------
# Styling (minimal & safe)
# -----------------------------
st.markdown("""
<style>
.stButton>button {
    background-color: #E50914;
    color: white;
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)

st.title("🎬 Hybrid Movie Recommendation System")

# -----------------------------
# TMDB API (user must add key)
# -----------------------------
TMDB_API_KEY = "YOUR_TMDB_API_KEY"

# -----------------------------
# Helper: Fetch Movie Details
# -----------------------------
def fetch_movie_details(movie_title):
    url = "https://api.themoviedb.org/3/search/movie"
    params = {"api_key": TMDB_API_KEY, "query": movie_title}
    data = requests.get(url, params=params).json()

    if data["results"]:
        m = data["results"][0]
        poster = (
            "https://image.tmdb.org/t/p/w500" + m["poster_path"]
            if m["poster_path"] else None
        )
        rating = m.get("vote_average", "N/A")
        year = m.get("release_date", "")[:4]
        return poster, rating, year
    return None, None, None

# -----------------------------
# Load Precomputed Data
# -----------------------------
new_df = pickle.load(open('new_df.pkl', 'rb'))
similarity = pickle.load(open('similarity.pkl', 'rb'))
user_similarity = pickle.load(open('user_similarity.pkl', 'rb'))
user_movie_matrix = pickle.load(open('user_movie_matrix.pkl', 'rb'))
movies_cf = pickle.load(open('movies_cf.pkl', 'rb'))

# -----------------------------
# Content-Based Recommendation
# -----------------------------
def recommend(movie):
    movie = movie.lower()
    if movie not in new_df['title'].str.lower().values:
        return ["Movie not found."]
    
    index = new_df[new_df['title'].str.lower() == movie].index[0]
    distances = similarity[index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    return [new_df.iloc[i[0]].title for i in movies_list]

# -----------------------------
# Collaborative Filtering
# -----------------------------
def recommend_collaborative(user_id, n=5):
    user_idx = user_id - 1
    similar_users = list(enumerate(user_similarity[user_idx]))
    similar_users = sorted(similar_users, key=lambda x: x[1], reverse=True)[1:6]
    
    movie_scores = {}
    for user, sim in similar_users:
        user_ratings = user_movie_matrix.iloc[user]
        for movie_id, rating in user_ratings.items():
            if rating > 0:
                movie_scores[movie_id] = movie_scores.get(movie_id, 0) + sim * rating
    
    recommended = sorted(movie_scores.items(), key=lambda x: x[1], reverse=True)[:n]
    return [movies_cf[movies_cf['movieId'] == i[0]]['title'].values[0] for i in recommended]

# -----------------------------
# Hybrid Recommendation
# -----------------------------
def recommend_hybrid(movie, user_id=None, alpha=0.5, n=5):
    movie = movie.lower()
    
    # Content-based scores
    if movie in new_df['title'].str.lower().values:
        index = new_df[new_df['title'].str.lower() == movie].index[0]
        content_scores = similarity[index]
    else:
        content_scores = np.zeros(len(new_df))
    
    # Normalize content scores
    if content_scores.max() != 0:
        content_scores = content_scores / content_scores.max()
    
    # Collaborative scores
    collab_scores = np.zeros(len(new_df))
    
    if user_id is not None and user_id in user_movie_matrix.index:
        user_idx = user_movie_matrix.index.get_loc(user_id)
        similar_users = list(enumerate(user_similarity[user_idx]))
        similar_users = sorted(similar_users, key=lambda x: x[1], reverse=True)[1:6]
        
        for u, sim in similar_users:
            user_ratings = user_movie_matrix.iloc[u]
            for movie_id, rating in user_ratings.items():
                if rating > 0 and movie_id in new_df['movie_id'].values:
                    idx = new_df[new_df['movie_id'] == movie_id].index[0]
                    collab_scores[idx] += sim * rating
    
    # Normalize collaborative scores
    if collab_scores.max() != 0:
        collab_scores = collab_scores / collab_scores.max()
    
    # Hybrid score
    final_scores = alpha * content_scores + (1 - alpha) * collab_scores
    
    # Remove input movie
    if movie in new_df['title'].str.lower().values:
        final_scores[index] = -1
    
    top_indices = np.argsort(final_scores)[-n:][::-1]
    return new_df.iloc[top_indices]['title'].values

# -----------------------------
# Streamlit UI
# -----------------------------

option = st.selectbox(
    "Choose recommendation type",
    ("Content-Based", "Collaborative", "Hybrid")
)

if option == "Content-Based":
    movie = st.text_input("Enter movie name")
    if st.button("Recommend"):
        recs = recommend(movie)
        st.subheader("Recommendations:")

        cols = st.columns(5)
        for idx, movie_name in enumerate(recs):
            with cols[idx]:
                poster, rating, year = fetch_movie_details(movie_name)

                if poster:
                    st.image(poster, use_column_width=True)
                st.caption(f"{movie_name} ({year}) ⭐ {rating}")



elif option == "Collaborative":
    user_id = st.number_input("Enter user ID", min_value=1, step=1)
    if st.button("Recommend"):
        recs = recommend_collaborative(user_id)
        st.subheader("Recommendations:")

        cols = st.columns(5)
        for idx, movie_name in enumerate(recs):
            with cols[idx]:
                poster, rating, year = fetch_movie_details(movie_name)

                if poster:
                    st.image(poster, use_column_width=True)
                st.caption(f"{movie_name} ({year}) ⭐ {rating}")


elif option == "Hybrid":
    movie = st.text_input("Enter movie name")
    user_id = st.number_input("Enter user ID", min_value=1, step=1)

    alpha = st.slider(
        "Balance: Content vs Collaborative",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.1
    )

    if st.button("Recommend"):
        recs = recommend_hybrid(movie, user_id, alpha=alpha)
        st.subheader("Recommendations:")

        cols = st.columns(5)
        for idx, movie_name in enumerate(recs):
            with cols[idx]:
                poster, rating, year = fetch_movie_details(movie_name)

                if poster:
                    st.image(poster, use_column_width=True)
                st.caption(f"{movie_name} ({year}) ⭐ {rating}")


