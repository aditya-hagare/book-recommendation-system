import streamlit as st
import pickle
import pandas as pd
import numpy as np

# ---------------------------------------------------
# Page Configuration
# ---------------------------------------------------

st.set_page_config(
    page_title="Book Recommendation System",
    page_icon="📚",
    layout="wide"
)

# ---------------------------------------------------
# Premium Animated CSS (REAL GLOW WORKING)
# ---------------------------------------------------

st.markdown("""
<style>

/* Background */
body {
    background-color: #0e1117;
}

/* Title */
.big-title {
    font-size: 50px;
    font-weight: 800;
    text-align: center;
    color: #00ffcc;
    text-shadow: 0 0 10px rgba(0,255,200,0.2);
}

/* Subtitle */
.subtitle {
    text-align: center;
    font-size: 20px;
    color: #cccccc;
    margin-bottom: 30px;
}

/* Book Card */
.book-card {
    text-align: center;
    margin-bottom: 40px;
    animation: fadeIn 0.6s ease forwards;
}
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}
/* Book Image (FIXED SIZE + CLEAN ALIGNMENT) */
.book-card img {
    width: 180px;
    height: 260px;
    object-fit: cover;
    border-radius: 18px;
    transition: all 0.4s ease;
    box-shadow: 0 0 20px rgba(0,255,200,0.5);
    animation: float 4s ease-in-out infinite;
}

/* Hover Glow */
.book-card img:hover {
    transform: scale(1.08);
    box-shadow: 0 0 25px rgba(0,255,200,0.6);
}

/* Floating Animation */
@keyframes float {
    0% { transform: translateY(0px); }
    50% { transform: translateY(-8px); }
    100% { transform: translateY(0px); }
}

/* Book Title Alignment */
.book-title {
    margin-top: 12px;
    font-weight: 600;
    font-size: 14px;
    min-height: 50px;
}

/* Match Glow */
.match-text {
    font-weight: bold;
    color: #ff884d;
    text-shadow: 0 0 5px rgba(255,120,0,0.6);
    margin-top: 8px;
}

/* Footer */
.footer {
    text-align: center;
    color: gray;
    margin-top: 40px;
}

/* Dropdown cursor */
div[data-baseweb="select"] {
    cursor: pointer !important;
}
/* Better Selectbox Focus */
div[data-baseweb="select"] > div {
    border-radius: 12px !important;
}

div[data-baseweb="select"] > div:focus-within {
    box-shadow: 0 0 12px rgba(0,255,200,0.6);
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# Load Model
# ---------------------------------------------------

@st.cache_resource
def load_model():
    model = pickle.load(open("model_knn.pkl", "rb"))
    encoder = pickle.load(open("book_encoder.pkl", "rb"))
    matrix = pickle.load(open("sparse_matrix.pkl", "rb"))
    books_df = pd.read_csv("Books_clean.csv")
    return model, encoder, matrix, books_df

model_knn, book_encoder, sparse_matrix, books = load_model()

# ---------------------------------------------------
# Filter Books Used in Model
# ---------------------------------------------------

available_isbns = book_encoder.classes_
books = books[books["isbn"].isin(available_isbns)].copy()

# ---------------------------------------------------
# Sidebar Filter
# ---------------------------------------------------

st.sidebar.header("⚙ Filters")

min_year = int(books["year_of_publication"].min())
max_year = int(books["year_of_publication"].max())

year_range = st.sidebar.slider(
    "Publication Year",
    min_year,
    max_year,
    (min_year, max_year)
)

books = books[
    (books["year_of_publication"] >= year_range[0]) &
    (books["year_of_publication"] <= year_range[1])
]

# ---------------------------------------------------
# Title Section
# ---------------------------------------------------

st.markdown('<div class="big-title">📚 Book Recommendation System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Item-Based Collaborative Filtering (KNN)</div>', unsafe_allow_html=True)
st.markdown("---")

# ---------------------------------------------------
# Select Book (Searchable + Clean)
# ---------------------------------------------------

book_list = books["book_title"].sort_values().unique()

selected_book = st.selectbox(
    "📖 Search & Select a Book",
    book_list,
    index=None,
    placeholder="Type to search book title..."
)
# ---------------------------------------------------
# Recommendation Function
# ---------------------------------------------------

def recommend_books(book_title, n=5):

    isbn = books[books["book_title"] == book_title]["isbn"].values[0]
    book_idx = book_encoder.transform([isbn])[0]

    distances, indices = model_knn.kneighbors(
        sparse_matrix.T[book_idx],
        n_neighbors=n + 1
    )

    similar_indices = indices.flatten()[1:]
    similar_isbns = book_encoder.inverse_transform(similar_indices)
    similarity_scores = 1 - distances.flatten()[1:]

    recommended = books[books["isbn"].isin(similar_isbns)].copy()
    recommended["similarity"] = similarity_scores[:len(recommended)]

    return recommended

# ---------------------------------------------------
# ---------------------------------------------------
# Recommend Button
# ---------------------------------------------------

if st.button("✨ Recommend Books"):

    # Step 3: Prevent error if nothing selected
    if not selected_book:
        st.warning("Please select a book first.")
    else:
        with st.spinner("Finding best matches..."):
            recommendations = recommend_books(selected_book)

        if recommendations.empty:
            st.warning("No recommendations found.")
        else:
            st.subheader("📖 Recommended Books")

            cols = st.columns(5)

            for i, (_, row) in enumerate(recommendations.iterrows()):
                with cols[i % 5]:

                    image_url = row.get("image_url_m")

                    if pd.notna(image_url) and image_url.startswith("http"):
                        img_src = image_url
                    else:
                        img_src = "default_book.png"

                    similarity_percent = round(row["similarity"] * 100, 2)

                    st.markdown(f"""
                    <div class="book-card">
                        <img src="{img_src}">
                        <div class="book-title">{row['book_title']}</div>
                        <div class="match-text">🔥 {similarity_percent}% Match</div>
                    </div>
                    """, unsafe_allow_html=True)

# ---------------------------------------------------
# Footer
# ---------------------------------------------------

st.markdown("---")
st.markdown('<div class="footer">Developed by Aditya Hagare | KNN-Based Recommender</div>', unsafe_allow_html=True)