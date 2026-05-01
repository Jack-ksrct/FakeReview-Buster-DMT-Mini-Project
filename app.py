"""
FakeReview Buster v2 — Flask Backend
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd, io

from detector import detector

app = Flask(__name__)
CORS(app)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/analyze', methods=['POST'])
def analyze_single():
    data = request.get_json()
    if not data or 'review' not in data:
        return jsonify({"error": "No review text provided"}), 400
    review = data['review'].strip()
    if len(review) < 5:
        return jsonify({"error": "Review too short to analyze"}), 400
    return jsonify(detector.predict(review))


@app.route('/api/analyze-batch', methods=['POST'])
def analyze_batch():
    data = request.get_json()
    if not data or 'reviews' not in data:
        return jsonify({"error": "No reviews provided"}), 400
    reviews = data['reviews']
    if not isinstance(reviews, list) or not reviews:
        return jsonify({"error": "Reviews must be a non-empty list"}), 400
    return jsonify(detector.analyze_batch(reviews))


@app.route('/api/upload-csv', methods=['POST'])
def upload_csv():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files['file']
    if not file.filename.lower().endswith('.csv'):
        return jsonify({"error": "Only CSV files are supported"}), 400
    try:
        content = file.read().decode('utf-8')
        df = pd.read_csv(io.StringIO(content))
        review_col = None
        for col in df.columns:
            if col.lower() in ['review','reviews','text','comment','comments','body','content']:
                review_col = col; break
        if review_col is None:
            for col in df.columns:
                if df[col].dtype == object:
                    review_col = col; break
        if review_col is None:
            return jsonify({"error": "Could not find a review text column in CSV"}), 400
        reviews = df[review_col].dropna().astype(str).tolist()[:200]
        result  = detector.analyze_batch(reviews)
        result['column_used'] = review_col
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"Failed to parse CSV: {str(e)}"}), 400


@app.route('/api/model-stats', methods=['GET'])
def model_stats():
    return jsonify(detector.get_model_stats())


@app.route('/api/sample', methods=['GET'])
def get_samples():
    return jsonify({
        "fake_examples": [
            "BEST PRODUCT EVER!!!! I absolutely love this so much! Highly recommend to EVERYONE!!!",
            "Amazing quality, fast shipping, perfect in every way! My whole family loves it so much!!!",
            "This is the greatest purchase I have ever made in my entire life. Must buy!!!!!"
        ],
        "real_examples": [
            "Battery life is around 6 hours. Screen has minor glare outdoors but indoor use is fine.",
            "Fits true to size but runs slightly narrow. Color matches the photos well.",
            "Setup took about 20 minutes. Instructions could be clearer but works as expected after."
        ]
    })


if __name__ == '__main__':
    print("🌐 Server starting at http://localhost:5000")
    detector.train()
    app.run(debug=True, port=5000)
