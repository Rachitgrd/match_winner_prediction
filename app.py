import pandas as pd
from flask import Flask, request, jsonify, render_template
import joblib

# Initialize the Flask app
app = Flask(__name__)

# --- Load Models and Encoders ---
try:
    score_model = joblib.load('models/score_predictor.pkl')
    win_model = joblib.load('models/win_predictor.pkl')
    encoders = joblib.load('models/encoders.pkl')
    venue_avg_df = pd.read_csv('venue_avg.csv')
    venue_averages = venue_avg_df.set_index('venue')['avg_score'].to_dict()
except FileNotFoundError as e:
    print(f"Error loading model or data files: {e}")
    print("Please ensure you have run the notebook to generate the .pkl model files and the CSVs are in the root directory.")
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

        # Encode categorical features
        encoded_batting = encoders['batting_team'].transform([batting_team])[0]
        encoded_bowling = encoders['bowling_team'].transform([bowling_team])[0]
        encoded_venue = encoders['venue'].transform([venue])[0]
        
        ground_avg = venue_averages.get(venue, 165) # Default average if venue not found

        innings = int(data['innings'])

        if innings == 1:
            # --- First Innings: Predict Score, then Win Probability ---
            
            # 1. Prepare data for Score Prediction
            current_score = int(data['current_score'])
            overs = float(data['overs'])
            wickets = int(data['wickets'])
            
            balls_bowled = int(overs) * 6 + int((overs - int(overs)) * 10)
            balls_left = 120 - balls_bowled
            wickets_left = 10 - wickets
            crr = (current_score / (balls_bowled / 6)) if balls_bowled > 0 else 0
            
            # This feature list must match the training script for the score model
            score_features = pd.DataFrame([[
                encoded_batting, encoded_bowling, encoded_venue,
                current_score, balls_left, wickets_left, crr, 
                0, 0, # runs_last_6, runs_last_12 (simplified for real-time)
                balls_bowled / 120, # match_progress
                1 if overs < 6 else 0, # is_powerplay
                1 if overs >= 15 else 0, # is_death_over
                ground_avg
            ]], columns=[
                'batting_team', 'bowling_team', 'venue', 'total_runs', 'balls_left',
                'wickets_left', 'crr', 'runs_last_6', 'runs_last_12',
                'match_progress', 'is_powerplay', 'is_death_over', 'ground_avg_first_innings'
            ])
            
            projected_score = int(score_model.predict(score_features)[0])

            # 2. Prepare data for Win Prediction (simulating start of 2nd innings)
            target = projected_score + 1
            rrr = target / 20.0
            
            # This feature list must match the training script for the win model
            win_features = pd.DataFrame([[
                encoded_bowling, encoded_batting, encoded_venue, 0, # batsman_runs
                0, 120, 10, 0, 0, 0, # score, balls, wickets, crr, runs_last...
                0, 1, 0, ground_avg, rrr # progress, pp, death, avg, rrr
            ]], columns=[
                'batting_team', 'bowling_team', 'venue', 'batsman_runs', 'total_runs',
                'balls_left', 'wickets_left', 'crr', 'runs_last_6', 'runs_last_12',
                'match_progress', 'is_powerplay', 'is_death_over',
                'ground_avg_first_innings', 'rrr'
            ])
            
            # Prediction gives probabilities for each class
            win_probabilities = win_model.predict_proba(win_features)[0]
            
            # Find the probability for the team batting first (now bowling)
            bowling_team_index = list(encoders['result'].classes_).index(bowling_team)
            win_prob_bowling = win_probabilities[bowling_team_index] * 100
            
            return jsonify({
                'projected_score': projected_score,
                'win_probability_batting': 100 - win_prob_bowling, # In 1st inn, we show prob of team batting now
                'win_probability_bowling': win_prob_bowling
            })

        else: # Innings == 2
            # --- Second Innings: Predict Win Probability Directly ---
            target = int(data['target'])
            current_score = int(data['current_score'])
            overs = float(data['overs'])
            wickets = int(data['wickets'])

            balls_bowled = int(overs) * 6 + int((overs - int(overs)) * 10)
            balls_left = 120 - balls_bowled
            wickets_left = 10 - wickets
            crr = (current_score / (balls_bowled / 6)) if balls_bowled > 0 else 0
            rrr = ((target - current_score) / (balls_left / 6)) if balls_left > 0 else 99
            
            # This feature list must match the training script for the win model
            win_features = pd.DataFrame([[
                encoded_batting, encoded_bowling, encoded_venue, 0, # batsman_runs (simplified)
                current_score, balls_left, wickets_left, crr, 0, 0, # runs_last.. (simplified)
                balls_bowled / 120, # match_progress
                1 if overs < 6 else 0, # is_powerplay
                1 if overs >= 15 else 0, # is_death_over
                ground_avg, rrr
            ]], columns=[
                'batting_team', 'bowling_team', 'venue', 'batsman_runs', 'total_runs',
                'balls_left', 'wickets_left', 'crr', 'runs_last_6', 'runs_last_12',
                'match_progress', 'is_powerplay', 'is_death_over',
                'ground_avg_first_innings', 'rrr'
            ])
            
            win_probabilities = win_model.predict_proba(win_features)[0]
            
            # Find the probability for the chasing team (current batting team)
            batting_team_index = list(encoders['result'].classes_).index(batting_team)
            win_prob_batting = win_probabilities[batting_team_index] * 100

            return jsonify({
                'projected_score': None, # No score projection in 2nd innings
                'win_probability_batting': win_prob_batting,
                'win_probability_bowling': 100 - win_prob_batting
            })

    except Exception as e:
        return jsonify({'error': str(e)}), 400


# --- Run the App ---
if __name__ == '__main__':
    app.run(debug=True)
