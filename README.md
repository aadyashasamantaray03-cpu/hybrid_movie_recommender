# Hybrid Movie Recommendation System

This project implements a **hybrid movie recommendation system** that combines:
- **Content-based filtering** (movie similarity using textual features)
- **Collaborative filtering** (user-based recommendations from ratings)

The system is deployed as an **interactive Streamlit web application**.

---

## Features
- Content-based movie recommendations
- User-based collaborative filtering
- Weighted hybrid recommender with adjustable balance
- Interactive Streamlit UI with movie posters and ratings
- Graceful handling of cold-start cases

---
## Tech Stack
- Python
- Pandas, NumPy, scikit-learn
- NLTK (for text preprocessing)
- Streamlit
- TMDB API (for movie posters and metadata)


---

## How it Works
- Content-based filtering uses vectorized movie metadata and cosine similarity.
- Collaborative filtering leverages user–movie rating interactions.
- The hybrid model combines both approaches using a tunable weight parameter.

---

## Note
This repository focuses on demonstrating the **end-to-end design and implementation** of a recommender system.  
Large datasets and API keys are not included.

---

## Future Improvements
- Model-based collaborative filtering
- Improved scalability for large datasets
- Enhanced explainability and evaluation metrics
