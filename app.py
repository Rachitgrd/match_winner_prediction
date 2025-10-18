import os
import pandas as pd
from flask import Flask, request, jsonify, render_template
import joblib
import gdown  # ✅ new

# Initialize the Flask app
app = Flask(__name__)

# --- Google Drive File IDs ---
file_ids = {
    "encoders": "1zS5CY6IXtMZEbcH32loKPH2BTmlHQOr-",
    "score_predictor": "1YIw8qFJXEW8DfToFuRIbULlxu6dKer-N",
    "win_predictor": "1hJ3th1kBQbkENC20t1TyMXWiuN9fx_yK"
}

# --- Ensure 'models' folder exists ---
if not os.path.exists("models"):
    os.makedirs("models")

# --- Download Files if Not Present ---
def download_if_not_exists(filename, file_id):
    filepath = os.path.join("models", filename)
    if not os.path.exists(filepath):
        print(f"⬇️ Downloading {filename} from Google Drive...")
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, filepath, quiet=False)
    else:
        print(f"✅ {filename} already exists. Skipping download.")
    return filepath

encoders_path = download_if_not_exists("encoders.pkl", file_ids["encoders"])
score_model_path = download_if_not_exists("score_predictor.pkl", file_ids["score_predictor"])
win_model_path = download_if_not_exists("win_predictor.pkl", file_ids["win_predictor"])

# --- Load Models and Encoders ---
try:
    score_model = joblib.load(score_model_path)
    win_model = joblib.load(win_model_path)
    encoders = joblib.load(encoders_path)
    venue_avg_df = pd.read_csv('venue_avg.csv')
    venue_averages = venue_avg_df.set_index('venue')['avg_score'].to_dict()
except FileNotFoundError as e:
    print(f"Error loading model or data files: {e}")
    score_model = win_model = encoders = venue_averages = None

# --- Define Routes ---

@app.route('/')
def home():
    """Renders the main HTML page."""
    if not all([score_model, win_model, encoders, venue_averages]):
        return "Error: Model files not found. Please train and save the models first.", 500
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Handles prediction requests from the frontend."""
    if not all([score_model, win_model, encoders, venue_averages]):
        return jsonify({'error': 'Models not loaded'}), 500

    data = request.get_json()

    try:
        # --- Preprocess Input Data ---
        batting_team = data['batting_team']
        bowling_team = data['bowling_team']
        venue = data['venue']

        encoded_batting = encoders['batting_team'].transform([batting_team])[0]
        encoded_bowling = encoders['bowling_team'].transform([bowling_team])[0]
        encoded_venue = encoders['venue'].transform([venue])[0]
        ground_avg = venue_averages.get(venue, 165)

        innings = int(data['innings'])

        if innings == 1:
            current_score = int(data['current_score'])
            overs = float(data['overs'])
            wickets = int(data['wickets'])
            balls_bowled = int(overs) * 6 + int((overs - int(overs)) * 10)
            balls_left = 120 - balls_bowled
            wickets_left = 10 - wickets
            crr = (current_score / (balls_bowled / 6)) if balls_bowled > 0 else 0

            score_features = pd.DataFrame([[
                encoded_batting, encoded_bowling, encoded_venue,
                current_score, balls_left, wickets_left, crr,
                0, 0,
                balls_bowled / 120,
                1 if overs < 6 else 0,
                1 if overs >= 15 else 0,
                ground_avg
            ]], columns=[
                'batting_team', 'bowling_team', 'venue', 'total_runs', 'balls_left',
                'wickets_left', 'crr', 'runs_last_6', 'runs_last_12',
                'match_progress', 'is_powerplay', 'is_death_over', 'ground_avg_first_innings'
            ])

            projected_score = int(score_model.predict(score_features)[0])

            target = projected_score + 1
            rrr = target / 20.0

            win_features = pd.DataFrame([[
                encoded_bowling, encoded_batting, encoded_venue, 0,
                0, 120, 10, 0, 0, 0,
                0, 1, 0, ground_avg, rrr
            ]], columns=[
                'batting_team', 'bowling_team', 'venue', 'batsman_runs', 'total_runs',
                'balls_left', 'wickets_left', 'crr', 'runs_last_6', 'runs_last_12',
                'match_progress', 'is_powerplay', 'is_death_over',
                'ground_avg_first_innings', 'rrr'
            ])

            win_probabilities = win_model.predict_proba(win_features)[0]
            bowling_team_index = list(encoders['result'].classes_).index(bowling_team)
            win_prob_bowling = win_probabilities[bowling_team_index] * 100

            return jsonify({
                'projected_score': projected_score,
                'win_probability_batting': 100 - win_prob_bowling,
                'win_probability_bowling': win_prob_bowling
            })

        else:
            target = int(data['target'])
            current_score = int(data['current_score'])
            overs = float(data['overs'])
            wickets = int(data['wickets'])
            balls_bowled = int(overs) * 6 + int((overs - int(overs)) * 10)
            balls_left = 120 - balls_bowled
            wickets_left = 10 - wickets
            crr = (current_score / (balls_bowled / 6)) if balls_bowled > 0 else 0
            rrr = ((target - current_score) / (balls_left / 6)) if balls_left > 0 else 99

            win_features = pd.DataFrame([[
                encoded_batting, encoded_bowling, encoded_venue, 0,
                current_score, balls_left, wickets_left, crr, 0, 0,
                balls_bowled / 120,
                1 if overs < 6 else 0,
                1 if overs >= 15 else 0,
                ground_avg, rrr
            ]], columns=[
                'batting_team', 'bowling_team', 'venue', 'batsman_runs', 'total_runs',
                'balls_left', 'wickets_left', 'crr', 'runs_last_6', 'runs_last_12',
                'match_progress', 'is_powerplay', 'is_death_over',
                'ground_avg_first_innings', 'rrr'
            ])

            win_probabilities = win_model.predict_proba(win_features)[0]
            batting_team_index = list(encoders['result'].classes_).index(batting_team)
            win_prob_batting = win_probabilities[batting_team_index] * 100

            return jsonify({
                'projected_score': None,
                'win_probability_batting': win_prob_batting,
                'win_probability_bowling': 100 - win_prob_batting
            })

    except Exception as e:
        return jsonify({'error': str(e)}), 400


# --- Run the App ---
if __name__ == '__main__':
    app.run(debug=True)
