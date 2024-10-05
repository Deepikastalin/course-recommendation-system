import gradio as gr
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Assuming embeddings, data, model, and tokenizer are already defined

def recommend_course(input_text):
    # Get embeddings for the input course
    input_embedding = get_embeddings(input_text, model, tokenizer)
    
    # Calculate cosine similarities
    similarities = cosine_similarity([input_embedding], embeddings)[0]
    
    # Get indices of the top_n most similar courses
    top_indices = np.argsort(similarities)[-5:][::-1]
    
    # Get the course titles of the most similar courses
    course_titles = data.iloc[top_indices]['Title'].tolist()
    
    # Format output as bullet points
    formatted_output = "\n".join([f"- {title}" for title in course_titles])
    
    return formatted_output

# Gradio Interface with styling
css = """
    #input_text {font-size: 16px; padding: 10px; border-radius: 5px;}
    #output_text {font-size: 16px; padding: 10px; border-radius: 5px; background-color: #f9f9f9;}
    .gr-button {font-size: 16px; padding: 10px 20px; border-radius: 5px;}
"""

iface = gr.Interface(
    fn=recommend_course,
    inputs=gr.Textbox(lines=2, placeholder="Enter course title or description here...", label="Course Input"),
    outputs=gr.Textbox(label="Recommended Courses"),
    title="Course Recommendation System",
    description="Enter a course title or description, and the system will recommend similar courses.",
    css=css
)

# Launch the app
iface.launch()
