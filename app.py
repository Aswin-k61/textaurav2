from flask import Flask, request, jsonify, render_template
import requests
import os

app = Flask(__name__)

# Replace local models with HF API
API_URL = "https://api-inference.huggingface.co/models/cardiffnlp/twitter-roberta-base-sentiment"
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

    response = requests.post(API_URL, headers=HEADERS, json={"inputs": text})
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

        return jsonify({
            "sentiment": sentiment,
            "confidence": confidence
        })

    except Exception as e:
        print("ERROR:", str(e))
        return jsonify({"error": "Processing failed"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)