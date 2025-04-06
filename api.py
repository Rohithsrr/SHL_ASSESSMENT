from flask import Flask, request, jsonify
import pandas as pd
import google.generativeai as genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json

# Configure Gemini API
GEMINI_API_KEY = "AIzaSyDtY_3u6RvOWUebu54gDtgNmNxQKv1gh0Y"  # Replace with your actual Gemini API key
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")

# Load assessment data from JSON with validation
try:
    with open("shl_assessments_complete.json", "r") as f:
        data = json.load(f)
    
    # Ensure data is a list of dictionaries
    if isinstance(data, list):
        assessments_list = data
    elif isinstance(data, dict) and "assessments" in data:
        assessments_list = data["assessments"]
        if not isinstance(assessments_list, list):
            raise ValueError("The 'assessments' key must contain a list")
    else:
        raise ValueError("JSON must be a list or a dict with 'assessments' key")

    # Create DataFrame
    assessments_data = pd.DataFrame(assessments_list)

    # Add missing columns with default values
    for col in ['remote_testing', 'adaptive_irt', 'test_type']:
        if col not in assessments_data.columns:
            assessments_data[col] = 'N/A'

except FileNotFoundError:
    print("shl_assessments.json not found. Please ensure the file exists.")
    assessments_data = pd.DataFrame()
except json.JSONDecodeError as e:
    print(f"Failed to parse JSON: {e}")
    assessments_data = pd.DataFrame()
except ValueError as e:
    print(f"Data error: {e}")
    assessments_data = pd.DataFrame()

# Function to query Gemini API
def query_gemini(prompt):
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Gemini API error: {e}")
        return prompt

# Function to recommend assessments with relaxed filtering
def recommend_assessments(query, max_duration=None, top_k=10):
    if assessments_data.empty:
        return pd.DataFrame(columns=["name", "url", "remote_testing", "adaptive_irt", "duration", "test_type"])

    # Use Gemini to parse the query
    gemini_prompt = f"Extract key skills, requirements, and preferences from this job query: '{query}'. If no specific skills are mentioned, suggest general aptitude or language-related assessments."
    gemini_response = query_gemini(gemini_prompt)

    # Combine query and Gemini response for TF-IDF
    combined_text = query + " " + gemini_response
    tfidf = TfidfVectorizer(stop_words="english")
    assessment_tfidf = tfidf.fit_transform(assessments_data["description"].fillna(""))
    query_tfidf = tfidf.transform([combined_text])

    # Calculate similarity
    similarities = cosine_similarity(query_tfidf, assessment_tfidf).flatten()
    assessments_data["similarity"] = similarities

    # Filter by duration with relaxed logic (allow 'N/A' or non-numeric to pass if no max_duration)
    filtered_data = assessments_data.copy()
    if max_duration is not None:
        filtered_data = filtered_data[
            filtered_data["duration"].apply(lambda x: x == 'N/A' or (isinstance(x, (int, float)) and x <= max_duration))
        ]

    if filtered_data.empty:
        filtered_data = assessments_data.copy()  # Fallback to all assessments if filtering fails

    # Sort and select top recommendations
    recommendations = filtered_data.sort_values("similarity", ascending=False).head(top_k)
    return recommendations[["name", "url", "remote_testing", "adaptive_irt", "duration", "test_type"]]

# Flask app for API
flask_app = Flask(__name__)

@flask_app.route("/recommend", methods=["GET"])
def get_recommendations():
    query = request.args.get("query", "")
    max_duration = request.args.get("max_duration", type=int)
    top_k = request.args.get("top_k", default=10, type=int)

    if not query:
        return jsonify({"error": "Query parameter is required"}), 400

    results = recommend_assessments(query, max_duration, top_k)
    return jsonify(results.to_dict(orient="records"))

if __name__ == "__main__":
    flask_app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)