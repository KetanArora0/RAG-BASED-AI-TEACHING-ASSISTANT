import pandas as pd 
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np 
import joblib 
import requests
from groq import Groq
from config import groq_api_key   # <-- put your 


# Initialize Groq client
client = Groq(api_key=groq_api_key)


def create_embedding(text_list):
    # Ollama embeddings
    r = requests.post("http://localhost:11434/api/embed", json={
        "model": "bge-m3",
        "input": text_list
    })

    embedding = r.json()["embeddings"] 
    return embedding


def inference(prompt):
    r = requests.post("http://localhost:11434/api/generate", json={
        "model": "llama3.2",
        "prompt": prompt,
        "stream": False
    })

    response = r.json()
    print(response)
    return response


# 🔥 Replaced OpenAI with Groq (same function name kept)
def inference_openai(prompt):
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",  # can upgrade to llama3-70b-8192
        messages=[
            {
                "role": "system",
                "content": """You are a strict course assistant.
Answer ONLY from the provided video data.
Do NOT use any external knowledge.
If answer is not found, say: Not found in course."""
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.choices[0].message.content


# Load embeddings
df = joblib.load('embeddings.joblib')


# User input
incoming_query = input("Ask a Question: ")

# Create embedding for query
question_embedding = create_embedding([incoming_query])[0] 


# Compute similarity
similarities = cosine_similarity(
    np.vstack(df['embedding']), 
    [question_embedding]
).flatten()

top_results = 7
max_indx = similarities.argsort()[::-1][0:top_results]

new_df = df.loc[max_indx]
new_df = new_df.sort_values(by="start", ascending=True)  
new_df = new_df.dropna(subset=["text"])
new_df = new_df[new_df["text"].str.strip() != ""]
new_df = new_df[new_df["text"].str.len() > 50]
print("\n--- Retrieved Chunks ---")
print(new_df[["title", "number", "start", "end", "text"]].to_string())
print("------------------------\n")


# Prompt (slightly tightened for Groq)
prompt = f'''You are a strict course assistant for a Data Structures and Algorithms course, currently covering the topic of Arrays.

Below are relevant video subtitle chunks, each containing: video title, video number, start time (seconds), end time (seconds), and transcript text:

{new_df[["title", "number", "start", "end", "text"]].to_json(orient="records")}

---------------------------------

User Question:
"{incoming_query}"

---------------------------------

Instructions:
- Answer ONLY using the subtitle chunks provided above
- Do NOT use any external knowledge or assumptions
- Answer ONLY if the content is directly relevant to the question

Formatting Rules:
- Start your answer by mentioning the video number ONCE (e.g., "Video 13.10:")
- List ONLY timestamps where the content directly answers the question
- For each timestamp, write 1-2 sentences in your own words explaining what is taught
- Do NOT copy transcript text word for word — paraphrase clearly and simply
- Use a simple hyphen (-) for timestamp ranges, e.g., 0.0-22.96
- Do NOT invent or approximate timestamps — use only exact start and end values from the data
- Do NOT add any commentary, explanation, or notes about skipped or excluded chunks
- Do NOT use "Not found in course" as a section header — only write it as the entire response if nothing is relevant
- End your response immediately after the last timestamp entry — no closing remarks
- If nothing in the chunks answers the question, respond with exactly: "Not found in course"
'''

# Save prompt (optional)
with open("prompt.txt", "w") as f:
    f.write(prompt)


# 🔥 Get response from Groq
response = inference_openai(prompt)

print("\nFinal Answer:\n", response)


# Save response
with open("response.txt", "w") as f:
    f.write(response)