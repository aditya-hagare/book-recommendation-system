import streamlit as st
import pickle
import pandas as pd
import numpy as np

# ---------------------------------------------------
# Page Config
# ---------------------------------------------------
st.set_page_config(page_title="Book Recommender", layout="wide")

# ---------------------------------------------------
# CSS
# ---------------------------------------------------
st.markdown("""
<style>
body {background-color:#0e1117;}

.book-card {
    text-align:center;
    margin-bottom:30px;
}

.book-card img {
    width:160px;
    height:240px;
    border-radius:12px;
    transition:0.3s;
    box-shadow:0 0 15px rgba(0,255,200,0.4);
}

.book-card img:hover {
    transform:scale(1.05);
    box-shadow:0 0 25px rgba(0,255,200,1);
}

.book-title {
    margin-top:10px;
    font-size:14px;
    min-height:40px;
}

.book-score {
    color:#ff884d;
    font-weight:bold;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# Load Models
# ---------------------------------------------------
@st.cache_resource
def load_model():
    model = pickle.load(open("model_knn.pkl", "rb"))
    encoder = pickle.load(open("book_encoder.pkl", "rb"))
    matrix = pickle.load(open("sparse_matrix.pkl", "rb"))
    books = pd.read_csv("Books_clean.csv")

    user_model = pickle.load(open("user_model.pkl", "rb"))
    user_encoder = pickle.load(open("user_encoder.pkl", "rb"))

    return model, encoder, matrix, books, user_model, user_encoder

model_knn, book_encoder, sparse_matrix, books, user_model, user_encoder = load_model()

# ---------------------------------------------------
# Mode Selection
# ---------------------------------------------------
mode = st.radio("Select Mode", ["📖 Book Based", "👤 User Based"])

# ---------------------------------------------------
# IMAGE FIX FUNCTION
# ---------------------------------------------------
def get_image(row):
    img = row.get("image_url_m", "")
    if pd.notna(img) and str(img).startswith("http"):
        return img
    else:
        return f"https://covers.openlibrary.org/b/isbn/{row['isbn']}-M.jpg"

# ---------------------------------------------------
# BOOK BASED
# --------------------------------------------------
def recommend_books(book_title, n=5):

    selected_row = books[books["book_title"] == book_title]

    if selected_row.empty:
        st.error("Book not found")
        return pd.DataFrame()

    isbn = selected_row["isbn"].values[0]

    # 🔥 FIX FOR ERROR
    if isbn not in book_encoder.classes_:
        st.warning("This book is not in trained model")
        return pd.DataFrame()

    book_idx = book_encoder.transform([isbn])[0]

    distances, indices = model_knn.kneighbors(
        sparse_matrix.T[book_idx],
        n_neighbors=n+1
    )

    similar_indices = indices.flatten()[1:]
    similar_isbns = book_encoder.inverse_transform(similar_indices)
    similarity_scores = 1 - distances.flatten()[1:]

    rec = books[books["isbn"].isin(similar_isbns)].copy()
    rec["score"] = similarity_scores[:len(rec)]

    return rec

# ---------------------------------------------------
# USER BASED (WITH %)
# ---------------------------------------------------
def recommend_user(user_id, n=5):

    try:
        user_idx = user_encoder.transform([user_id])[0]
    except:
        return None

    distances, indices = user_model.kneighbors(
        sparse_matrix[user_idx],
        n_neighbors=6
    )

    similar_users = indices.flatten()[1:]
    similarity_scores = 1 - distances.flatten()[1:]

    recommended = {}

    for i, u in enumerate(similar_users):
        weight = similarity_scores[i]
        books_liked = sparse_matrix[u].nonzero()[1]

        for b in books_liked:
            if b not in recommended:
                recommended[b] = 0
            recommended[b] += weight

    recommended = sorted(recommended.items(), key=lambda x: x[1], reverse=True)

    results = []
    for book_idx, score in recommended[:n]:
        isbn = book_encoder.inverse_transform([book_idx])[0]
        percent = round(score / max(recommended[0][1],1) * 100, 2)

        row = books[books["isbn"] == isbn].iloc[0]
        row["score"] = percent
        results.append(row)

    return pd.DataFrame(results)

# ---------------------------------------------------
# TRENDING BOOKS
# ---------------------------------------------------
top_books = books.sample(10)

# ---------------------------------------------------
# UI
# ---------------------------------------------------

if mode == "📖 Book Based":

    selected_book = st.selectbox("Select Book", books["book_title"].unique())

    if st.button("Recommend"):

        rec = recommend_books(selected_book)

        cols = st.columns(5)

        for i, (_, row) in enumerate(rec.iterrows()):
            with cols[i % 5]:

                img = get_image(row)
                percent = round(row["score"] * 100, 2)

                st.markdown(f"""
                <div class="book-card">
                    <img src="{img}">
                    <div class="book-title">{row['book_title']}</div>
                    <div class="book-score">{percent}% match</div>
                </div>
                """, unsafe_allow_html=True)

# ---------------------------------------------------

elif mode == "👤 User Based":

    user_id = st.number_input("Enter User ID", min_value=1)

    if st.button("Recommend"):

        rec = recommend_user(user_id)

        if rec is None:
            st.error("User not found")
        else:
            cols = st.columns(5)

            for i, (_, row) in enumerate(rec.iterrows()):
                with cols[i % 5]:

                    img = get_image(row)
                    percent = row["score"]

                    st.markdown(f"""
                    <div class="book-card">
                        <img src="{img}">
                        <div class="book-title">{row['book_title']}</div>
                        <div class="book-score">{percent}% match</div>
                    </div>
                    """, unsafe_allow_html=True)

# ---------------------------------------------------
# TRENDING SECTION
# ---------------------------------------------------
st.subheader("🔥 Trending Books")

cols = st.columns(5)

for i, (_, row) in enumerate(top_books.iterrows()):
    with cols[i % 5]:

        img = get_image(row)

        st.markdown(f"""
        <div class="book-card">
            <img src="{img}">
            <div class="book-title">{row['book_title']}</div>
        </div>
        """, unsafe_allow_html=True)
