import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from groq import Groq
import os

# Load embeddings
df = joblib.load("embeddings.joblib")

# Load embedding model
embedding_model = SentenceTransformer("BAAI/bge-small-en-v1.5")

# Groq client
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def create_embedding(text_list):
    return [embedding_model.encode(text).tolist() for text in text_list]

def get_answer(query):
    # Step 1: Embedding
    question_embedding = create_embedding([query])[0]

    # Step 2: Cosine similarity
    similarities = cosine_similarity(
        np.vstack(df['embedding']),
        [question_embedding]
    ).flatten()

    # Step 3: Top 7 chunks
    top_results = 7
    max_indx = similarities.argsort()[::-1][0:top_results]
    new_df = df.loc[max_indx]

    # Step 4: Filter
    new_df = new_df[new_df["number"] != "13.00"]
    new_df = new_df.sort_values(by="start", ascending=True)
    new_df = new_df.dropna(subset=["text"])
    new_df = new_df[new_df["text"].str.strip() != ""]
    new_df = new_df[new_df["text"].str.len() > 50]

    # Step 5: Prompt
    prompt = f'''You are a strict course assistant for a Data Structures and Algorithms course, currently covering the topic of Arrays.

Below are relevant video subtitle chunks, each containing: video title, video number, start time (seconds), end time (seconds), and transcript text:

{new_df[["title", "number", "start", "end", "text"]].to_json(orient="records")}

---------------------------------

User Question:
"{query}"

---------------------------------

Instructions:
- Answer ONLY using the subtitle chunks provided above
- Do NOT use any external knowledge or assumptions
- Answer ONLY if the content is directly relevant to the question

Formatting Rules:
- Start with the video number ONCE (e.g., "Video 13.10:")
- Then for each relevant timestamp, follow this EXACT structure with new lines:

Video 13.10:

0.0-22.96
Your explanation here.

74.48-90.88
Your explanation here.

- Each timestamp MUST be on its own separate line
- Explanation MUST be on the next line after timestamp
- Leave a blank line between each timestamp block
- Only include timestamps where the content directly answers the question
- Do NOT copy transcript text word for word — paraphrase clearly
- Use a simple hyphen (-) for timestamp ranges
- Do NOT invent or approximate timestamps — use only exact values from the data
- Do NOT add any commentary or notes about skipped chunks
- End your response immediately after the last timestamp block
- If nothing answers the question, respond with exactly: "Not found in course"
'''

    # Step 6: Groq
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": """You are a strict course assistant.
Answer ONLY from the provided video data.
Do NOT use any external knowledge.
If answer is not found, say: Not found in course.
IMPORTANT: Always put each timestamp on a NEW LINE followed by explanation on the NEXT LINE.
Never put multiple timestamps on the same line."""
            },
            {"role": "user", "content": prompt}
        ]
    )

    raw = response.choices[0].message.content
    raw = re.sub(r'(\d+\.?\d*-\d+\.?\d*)', r'\n\n\1\n', raw)
    return raw


# ---- Streamlit UI ----
st.title("📚 AI Teaching Assistant")
st.caption("Ask anything from the DSA course")

query = st.text_input("Ask your question:")

if st.button("Submit"):
    if query:
        with st.spinner("Searching course videos..."):
            try:
                answer = get_answer(query)

                st.subheader("Answer:")

                for block in answer.strip().split("\n\n"):
                    block = block.strip()
                    if not block:
                        continue
                    if block.startswith("Video"):
                        st.markdown(f"### {block}")
                    elif re.match(r"^\d+\.?\d*-\d+\.?\d*", block):
                        lines = block.split("\n", 1)
                        timestamp = lines[0].strip()
                        explanation = lines[1].strip() if len(lines) > 1 else ""
                        st.markdown(f"**⏱ {timestamp}**")
                        st.write(explanation)
                        st.divider()
                    else:
                        st.write(block)

            except Exception as e:
                st.error(str(e))
    else:
        st.warning("Please enter a question!")