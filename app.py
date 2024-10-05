import streamlit as st
import numpy as np
import pickle
import json
from sklearn.metrics.pairwise import cosine_similarity

# Load the pre-trained vectorizer and data
with open("vectorizer_fited.pkl", 'rb') as file:
    loaded_vectorizer = pickle.load(file)

with open("description_fit_transformed.pkl", 'rb') as file:
    loaded_description = pickle.load(file)

with open("index_course_name.json", 'r') as file:
    loaded_dictionary = json.load(file)

# Recommendation function
def recommandation_deploy(title):
    # Process the input title
    title = title.split(" ")
    title = loaded_vectorizer.transform(title)
    # Calculate similarity
    similarities = cosine_similarity(title, loaded_description)
    # Get the top 10 recommended course indices
    indices_similaires = np.argsort(similarities[0])[::-1][:10]
    # Retrieve course names from the dictionary
    cours_recommandes = [loaded_dictionary[str(i)] for i in indices_similaires]
    return cours_recommandes

# Streamlit app UI
st.title("Course Recommendation System")

# Input from the user
user_input = st.text_input("Enter a course title to get recommendations:")

if st.button("Recommend"):
    if user_input:
        # Generate recommendations
        recommendations = recommandation_deploy(user_input)
        st.write("Recommended courses:")
        for course in recommendations:
            st.write(course)
    else:
        st.write("Please enter a valid course title.")
