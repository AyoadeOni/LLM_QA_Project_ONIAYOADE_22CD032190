# app.py - Flask Web Application for LLM Question Answering
from flask import Flask, render_template, request
import os
import re
from groq import Groq
from dotenv import load_dotenv

# Load environment variables (our secret API key)
load_dotenv()

app = Flask(__name__)

# Initialize Groq client with your secret key
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Basic text preprocessing (lowercase + remove punctuation except ?)
def preprocess_question(text):
    text = text.strip()
    text = text.lower()
    text = re.sub(r'[^\w\s\?]', '', text)   # remove punctuation except ?
    text = re.sub(r'\s+', ' ', text)        # remove extra spaces
    return text.strip()

# Function to get answer from Groq LLM
def get_llm_answer(question):
    if not question:
        return "Please ask a question."

    prompt = f"""You are a helpful and accurate Question-Answering assistant.
Answer the question clearly and naturally in 1-3 sentences.

Question: {question}

Answer:"""

    try:
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",   # Fast & smart model (confirmed working Nov 2025)
            temperature=0.7,
            max_tokens=500
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error connecting to LLM: {str(e)}"

# Home page - shows the form
@app.route("/", methods=["GET", "POST"])
def index():
    original = ""
    processed = ""
    answer = ""

    if request.method == "POST":
        original = request.form["question"]
        processed = preprocess_question(original)
        answer = get_llm_answer(processed)

    return render_template("index.html",
                           original=original,
                           processed=processed,
                           answer=answer)

# Run the app (only when running locally)
if __name__ == "__main__":
    app.run(debug=True)