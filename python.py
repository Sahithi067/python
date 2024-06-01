
 Step 1: Collect and Prepare Data
Assume we have a CSV file college_basketball_games.csv with columns like:
- team1: Name of the first team
- team2: Name of the second team
- team1_score: Score of the first team
- team2_score: Score of the second team
- team1_stats: Various statistics of the first team (e.g., average points, rebounds, assists)
- team2_stats: Various statistics of the second team

### Step 2: Load and Process the Data
python
import pandas as pd

# Load the dataset
data = pd.read_csv('college_basketball_games.csv')

# Create a target variable: 1 if team1 wins, 0 if team2 wins
data['team1_wins'] = (data['team1_score'] > data['team2_score']).astype(int)

# Drop unnecessary columns
data = data.drop(['team1_score', 'team2_score'], axis=1)

# Features and target
features = data.drop(['team1', 'team2', 'team1_wins'], axis=1)
target = data['team1_wins']


### Step 3: Train-Test Split
python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)


### Step 4: Model Selection and Training
python
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Create a pipeline with scaling and logistic regression
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('logreg', LogisticRegression())
])

# Train the model
pipeline.fit(X_train, y_train)


### Step 5: Evaluation
python
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Predictions
y_pred = pipeline.predict(X_test)

# Evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print('Confusion Matrix:')
print(conf_matrix)
print('Classification Report:')
print(class_report)


### Step 6: Prediction
To predict the outcome of a new game, you would format the input data similarly to the training data and use the pipeline.predict method.

python
# Example new game data (you would replace this with actual data)
new_game_data = pd.DataFrame({
    'team1_stats': [/* team1 statistics */],
    'team2_stats': [/* team2 statistics */]
})

# Predict the outcome
prediction = pipeline.predict(new_game_data)
if prediction[0] == 1:
    print("Team 1 is predicted to win")
else:
    print("Team 2 is predicted to win")
