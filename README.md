🚀 Live Demo
👉 https://book-recommendation-system-main.streamlit.app/

📚 Book Recommendation System

A content-based + collaborative filtering Book Recommendation System built using Machine Learning (KNN) and deployed with Streamlit Cloud.

This system recommends similar books based on user-selected book using item-based collaborative filtering.

🧠 Problem Statement

With thousands of books available, users often struggle to find relevant books similar to their interests.

This project solves that problem by:

Analyzing book ratings data

Building a similarity model

Recommending top N similar books

⚙️ Tech Stack

Python

Pandas

NumPy

Scikit-learn

KNN (Nearest Neighbors)

Streamlit

Git & GitHub

🧮 Machine Learning Approach

We implemented Item-Based Collaborative Filtering using:

1️⃣ Data Preprocessing

Cleaned dataset

Removed duplicates

Handled missing values

Filtered active users & popular books

2️⃣ Feature Engineering

Created User-Book interaction matrix

Converted matrix to sparse format

Encoded ISBN using LabelEncoder

3️⃣ Model Building

Used KNN (Nearest Neighbors)

Cosine similarity metric

Trained on sparse matrix

4️⃣ Recommendation Logic

When user selects a book:

System finds its vector representation

Computes nearest neighbors

Returns top 5 most similar books

Displays similarity percentage

📊 Dataset Used

Books dataset

Ratings dataset

Processed into Books_clean.csv

🎨 Features

✔ Interactive UI
✔ Animated book cards
✔ Similarity percentage match
✔ Publication year filter
✔ Default fallback image support
✔ Fully deployed web app

🗂 Project Structure
app.py
Books_clean.csv
model_knn.pkl
book_encoder.pkl
sparse_matrix.pkl
requirements.txt
default_book.png

🎯 Future Improvements

Add user-based collaborative filtering

Add rating prediction

Deploy using Docker

Add login system

Improve scalability with database

👨‍💻 Developed By

Aditya Hagare
Machine Learning Enthusiast | Data Science Aspirant
