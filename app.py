from flask import Flask, render_template, request
import pickle
import numpy as np
from tensorflow import keras
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Initialize Flask app
app = Flask(__name__)

# Load saved artifacts
with open('artifacts/label_encoders.pkl', 'rb') as file:
    label_encoders = pickle.load(file)

with open('artifacts/scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

model = keras.models.load_model('artifacts/ipl_score_predictor.h5')

# Home route
@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    if request.method == 'POST':
        # Get form data
        bat_team = request.form['bat_team']
        bowl_team = request.form['bowl_team']
        venue = request.form['venue']
        batsman = request.form['batsman']
        bowler = request.form['bowler']
        runs = float(request.form['runs'])
        wickets = float(request.form['wickets'])
        overs = float(request.form['overs'])
        striker = float(request.form['striker'])

        # Encode categorical values
        encoded_bat = label_encoders['bat_team'].transform([bat_team])[0]
        encoded_bowl = label_encoders['bowl_team'].transform([bowl_team])[0]
        encoded_venue = label_encoders['venue'].transform([venue])[0]
        encoded_batsman = label_encoders['batsman'].transform([batsman])[0]
        encoded_bowler = label_encoders['bowler'].transform([bowler])[0]

        # Prepare input array
        input_features = np.array([[encoded_bat, encoded_bowl, encoded_venue,
                                    runs, wickets, overs, striker,
                                    encoded_batsman, encoded_bowler]])
        input_scaled = scaler.transform(input_features)

        # Predict total
        prediction = model.predict(input_scaled)[0][0]
        prediction = round(prediction)

    return render_template('index.html',
                           prediction=prediction,
                           venues=list(label_encoders['venue'].classes_),
                           bat_teams=list(label_encoders['bat_team'].classes_),
                           bowl_teams=list(label_encoders['bowl_team'].classes_),
                           batsmen=list(label_encoders['batsman'].classes_),
                           bowlers=list(label_encoders['bowler'].classes_)
                           )

if __name__ == '__main__':
    app.run(debug=True)
