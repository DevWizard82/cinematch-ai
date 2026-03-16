# 🍿 CineMatch AI

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)

**CineMatch AI** is an advanced, content-based movie and TV show recommendation engine built with Python. It utilizes Natural Language Processing (NLP) techniques to analyze metadata from the Netflix catalog and serves recommendations through an interactive Streamlit web application.

🔗 **[Live Demo](https://cinematch-ai.streamlit.app)** *(Replace with your actual Streamlit link)*

## ✨ Features

* **Intelligent Recommendations:** Suggests 10 highly relevant movies/shows based on a user's selection.
* **Advanced Feature Engineering:** Custom "weighted soup" algorithm that prioritizes genres and directors over generic text.
* **Data Sanitization:** Cleans string data (e.g., merging first and last names) to ensure accurate vector mapping.
* **Real-Time Poster Fetching:** Integrates with the TMDB API to dynamically load high-quality movie posters for a premium UI experience.
* **Fast & Cached:** Computes vector distances efficiently and caches the ML model in memory for instant subsequent loads.

## 🧠 How It Works (The Data Science)

Unlike collaborative filtering (which relies on user ratings), this engine uses **Content-Based Filtering**. 

1. **Feature Soup Creation:** The algorithm combines critical metadata (`director`, `cast`, `listed_in`, `description`) into a single string. 
2. **Weighting:** Categorical importance is artificially multiplied (e.g., `listed_in` is repeated 3 times) to force the model to prioritize the movie's genre/vibe over ensemble voice casts.
3. **TF-IDF Vectorization:** Transforms the text soup into a mathematical matrix, penalizing common English stop words and assigning higher weights to unique identifiers.
4. **Cosine Similarity:** Calculates the mathematical angle (distance) between every movie vector. The closest vectors (highest cosine similarity scores) are returned as the top recommendations.

## 🛠️ Installation & Local Setup

### Prerequisites
* Python 3.8+
* A free API key from [The Movie Database (TMDB)](https://www.themoviedb.org/settings/api)

### 1. Clone the repository
```bash
git clone [https://github.com/DevWizard82/cinematch-ai.git](https://github.com/DevWizard82/cinematch-ai.git)
cd cinematch-ai
