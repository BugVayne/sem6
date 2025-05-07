from flask import Flask, render_template, request, jsonify
from analyzer.text_analyzer import analyze_text
from lab5_6.llm.response_generation import ResponseGenerator
from rag.vector_db import VectorDB
from datetime import datetime
from flask import make_response
import json


app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

# Initialize components
db = VectorDB()
response_gen = ResponseGenerator()

# Chat history storage
chat_history = []


@app.route("/save_history", methods=["GET", "POST"])
def save_history():
    try:
        if request.method == "POST":
            # Get edited history from the request
            edited_history = request.json.get("history", chat_history)

            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"chat_history_{timestamp}.json"

            # Create response with JSON data
            response = make_response(json.dumps(edited_history, indent=2))
            response.headers["Content-Disposition"] = f"attachment; filename={filename}"
            response.headers["Content-type"] = "application/json"

            return response

        # For GET requests, return the current history as JSON
        return jsonify(chat_history)

    except Exception as e:
        app.logger.error(f"Error saving history: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/")
def home():
    return render_template("index.html", history=chat_history)

@app.route("/chat", methods=["POST"])
def chat():
    try:
        user_input = request.json["message"]

        # Analyze the user input
        analysis = analyze_text(user_input)

        # Prepare query terms
        query_terms = analysis["keywords"].copy()
        if analysis["entities"]:
            query_terms.append(analysis["entities"])

        weighted_query = " ".join(query_terms)

        # Get context from vector DB
        context_items = db.search(weighted_query)
        context_text = "\n".join([item["text"] for item in context_items[:2]])  # Limit context items

        # Determine sentiment
        sentiment_score = analysis["sentiment"]["compound"]
        sentiment = "positive" if sentiment_score >= 0.5 else "negative" if sentiment_score <= -0.5 else "neutral"

        # Prepare context dictionary for the response generator
        context = {
            "sentiment": sentiment,
            "keywords": analysis["keywords"],
            "weighted_query": weighted_query,
            "context": context_text
        }

        # Generate response using the response generator
        response = response_gen.generate_response(user_input, context)

        # Update history
        chat_history.append({
            "user": user_input,
            "bot": response,
            "time": datetime.now().strftime("%H:%M"),
            "analysis": {  # Store analysis data with the message
                "sentiment": analysis["sentiment"],
                "keywords": analysis["keywords"],
                "context": context_text
            }
        })

        return jsonify({
            "response": response,
            "sentiment": sentiment,
            "keywords": analysis["keywords"],
            "context": context_text,  # Add context to response
            "sentiment_score": sentiment_score  # Add raw score for visualization
        })

    except Exception as e:
        app.logger.error(f"Error in chat endpoint: {str(e)}")
        return jsonify({
            "response": "Let's talk music! What are you listening to these days?",
            "error": str(e)
        })

if __name__ == "__main__":
    app.run(debug=True)