import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
movies = pd.read_csv("dataset/tmdb_5000_movies.csv")
movies = movies[['title', 'overview', 'genres', 'keywords']]
movies.fillna('', inplace=True)
movies['combined_features'] = movies['overview'] + ' ' + movies['genres'] + ' ' + movies['keywords']

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['combined_features'])
similarity = cosine_similarity(tfidf_matrix, tfidf_matrix)

def recommend(movie_name):
    if movie_name not in movies['title'].values:
        return []
    idx = movies[movies['title'] == movie_name].index[0]
    distances = sorted(list(enumerate(similarity[idx])), reverse=True, key=lambda x: x[1])
    recommended = [movies.iloc[i[0]].title for i in distances[1:6]]
    return recommended

# Streamlit UI
st.title("🎬 Movie Recommendation System")
movie_name = st.text_input("Enter a movie name:")

if st.button("Recommend"):
    recommendations = recommend(movie_name)
    if recommendations:
        st.subheader(f"Movies similar to '{movie_name}':")
        for r in recommendations:
            st.write("👉", r)
    else:
        st.write("❌ Movie not found in the dataset.")
