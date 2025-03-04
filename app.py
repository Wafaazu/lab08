import streamlit as st
import requests
from bs4 import BeautifulSoup
import numpy as np
from mistralai import Mistral

# Define API key for Mistral API
api_key = "gQeOXgiZv5437TMVhKrChGMQ98rtNyfc"  # Replace with your actual API key

# Function to get text embeddings from Mistral
def get_text_embedding(list_txt_chunks):
    client = Mistral(api_key=api_key)
    embeddings_batch_response = client.embeddings.create(model="mistral-embed", inputs=list_txt_chunks)
    return embeddings_batch_response.data

# Fetching and parsing the UDST policies page based on the policy URL
def fetch_policies(url):
    response = requests.get(url)
    html_doc = response.text
    soup = BeautifulSoup(html_doc, "html.parser")
    # You can customize this parsing based on the structure of the policy page
    policies_text = soup.find("div").text  # Example to fetch policy text
    return policies_text

# Chunking the policies text into smaller chunks for processing
def chunk_text(text, chunk_size=512):
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    return chunks

# Initialize the FAISS index for similarity search
def initialize_faiss(embeddings):
    d = len(embeddings[0].embedding)
    import faiss
    index = faiss.IndexFlatL2(d)
    index.add(np.array([embedding.embedding for embedding in embeddings]))
    return index

# Function to handle user queries
def handle_query(query, chunks, index):
    question_embeddings = np.array([get_text_embedding([query])[0].embedding])
    D, I = index.search(question_embeddings, k=2)  # Searching for top 2 similar chunks
    retrieved_chunk = [chunks[i] for i in I[0]]
    prompt = f"""
    Context information is below.
    ---------------------
    {retrieved_chunk}
    ---------------------
    Given the context information and not prior knowledge, answer the query.
    Query: {query}
    Answer:
    """
    response = mistral(prompt)
    return response

# Function to interact with Mistral for generating answers
def mistral(user_message, model="mistral-large-latest"):
    client = Mistral(api_key=api_key)
    chat_response = client.chat.complete(
        model=model,
        messages=[
            {
                "role": "user",
                "content": user_message,
            },
        ]
    )
    return chat_response.choices[0].message.content

# Streamlit UI Setup
def main():
    st.title("UDST Policies Chatbot")

    # Define the policy links
    policy_links = {
        "Sport and Wellness Facilities and Services Usage Policy-V1": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/sport-and-wellness-facilities-and",
        "Student Conduct Policy-V2": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/student-conduct-policy",
        "Student Attendance Policy-V1": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/student-attendance-policy",
        "Academic Standing Policy-V4": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/academic-standing-policy",
        "Final Grade Policy-V2": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/final-grade-policy",
        "Registration Policy-V2": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/registration-policy",
        "Student Engagement Policy-V1": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/udst-policies-and-procedures/student-engagement-policy",
        "International Student Policy-V1": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/udst-policies-and-procedures/international-student-policy",
        "Graduation Policy-V1": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/udst-policies-and-procedures/graduation-policy",
        "Student Counselling Services Policy-V1": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/udst-policies-and-procedures/student-counselling-services-policy"
    }

    # Listbox for policy selection
    selected_policy = st.selectbox("Select a Policy", list(policy_links.keys()))
    
    # Fetch and chunk the selected policy text
    url = policy_links[selected_policy]
    policies_text = fetch_policies(url)
    chunks = chunk_text(policies_text)
    text_embeddings = get_text_embedding(chunks)
    index = initialize_faiss(text_embeddings)

    # Text box for entering the query
    user_query = st.text_input("Enter your query:")

    # Text area to display the response
    if user_query:
        answer = handle_query(user_query, chunks, index)
        st.text_area("Answer", value=answer, height=300)

if __name__ == "__main__":
    main()
