import streamlit as st
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Page Configuration
st.set_page_config(page_title="CineMatch AI", page_icon="🍿", layout="wide")  # Set to wide for the grid!


# 2. API Setup for Posters
def fetch_poster(title):
    # NOTE: You will need to replace this with your own free API key from TMDB
    api_key = "2ad88702a28bc8e6756f5b1dfa137b26"
    url = f"https://api.themoviedb.org/3/search/multi?api_key={api_key}&query={title}"

    try:
        response = requests.get(url).json()
        poster_path = response['results'][0]['poster_path']
        if poster_path:
            return "https://image.tmdb.org/t/p/w500" + poster_path
        else:
            return "https://via.placeholder.com/500x750/1f1a13/deb887?text=No+Poster"
    except:
        # Fallback placeholder if the API fails or doesn't find the movie
        return "https://via.placeholder.com/500x750/1f1a13/deb887?text=No+Poster"


# 3. Load and process data (Cached)
@st.cache_data
def load_data_and_model():
    df = pd.read_csv('netflix_titles.csv')

    for feature in ['director', 'cast', 'listed_in', 'description']:
        df[feature] = df[feature].fillna('')

    def clean_data(x):
        return str.lower(x.replace(" ", "").replace(",", " ")) if isinstance(x, str) else ''

    for feature in ['director', 'cast', 'listed_in']:
        df[feature] = df[feature].apply(clean_data)

    def create_weighted_soup(x):
        return (x['director'] + ' ') * 2 + (x['cast'] + ' ') + (x['listed_in'] + ' ') * 3 + x['description']

    df['soup'] = df.apply(create_weighted_soup, axis=1)

    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['soup'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    return df, cosine_sim


df, cosine_sim = load_data_and_model()
indices = pd.Series(df.index, index=df['title']).drop_duplicates()


def get_recommendations(title):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return df['title'].iloc[movie_indices].tolist()


# 4. Streamlit UI Elements
st.title('🍿 CineMatch AI')
st.markdown(
    "An advanced content-based recommendation engine powered by **TF-IDF Vectorization** and **Cosine Similarity**.")
st.divider()

movie_list = df['title'].values
selected_movie = st.selectbox("Type or select a Netflix show/movie to get started:", movie_list)

if st.button('Generate Recommendations', type="primary"):
    with st.spinner('Calculating vector distances and fetching posters...'):
        recs = get_recommendations(selected_movie)

    st.success(f"Because you watched **{selected_movie}**, you should watch:")

    # 5. Create the Grid Layout (2 rows of 5 columns)
    for i in range(0, 10, 5):
        cols = st.columns(5)
        for j in range(5):
            if i + j < len(recs):
                movie_title = recs[i + j]
                poster_url = fetch_poster(movie_title)

                with cols[j]:
                    st.image(poster_url, use_container_width=True)
                    st.markdown(f"**{movie_title}**")