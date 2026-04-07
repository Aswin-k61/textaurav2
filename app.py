from flask import Flask, request, jsonify, render_template
import requests
import os
from datetime import datetime

app = Flask(__name__)

# Replace local models with HF API
N8N_WEBHOOK_URL = "https://aswinn8n1.app.n8n.cloud/webhook/sentiment-alert"

API_URL = "https://router.huggingface.co/hf-inference/models/cardiffnlp/twitter-roberta-base-sentiment"
HEADERS = {
    "Authorization": f"Bearer {os.getenv('HF_TOKEN')}"
}

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    text = data.get("text")

    if not text:
        return jsonify({"error": "No text provided"}), 400

    response = requests.post(API_URL, headers=HEADERS, json={"inputs": text},timeout=10)
    result = response.json()

    print("HF RESPONSE:", result)  # 🔍 debug

    # ✅ HANDLE ERRORS (IMPORTANT)
    if isinstance(result, dict) and "error" in result:
        return jsonify({"error": result["error"]}), 500

    try:
        scores = result[0]

        label_map = {
            "LABEL_0": "negative",
            "LABEL_1": "neutral",
            "LABEL_2": "positive"
        }

        best = max(scores, key=lambda x: x['score'])

        sentiment = label_map.get(best['label'], best['label'])
        confidence = round(best['score'], 3)

        if sentiment.lower() == "negative":
            send_to_n8n(text, sentiment, confidence)

        return jsonify({
            "sentiment": sentiment,
            "confidence": confidence
        })

    except Exception as e:
        print("ERROR:", str(e))
        return jsonify({"error": "Processing failed"}), 500
@app.route("/health")
def health():
    return "OK", 200

def send_to_n8n(text, sentiment, confidence):
    data = {
        "text": text,
        "sentiment": sentiment,
        "score": confidence,
        "time": datetime.now().isoformat()
    }

    try:
        requests.post(N8N_WEBHOOK_URL, json=data, timeout=5)
        print("Sent to n8n")
    except Exception as e:
        print("n8n error:", e)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)