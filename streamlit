import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import re
import numpy as np

# Load data
@st.cache
def load_data():
    df = pd.read_csv('clean_chatgpt_reviews 2.csv')
    df.drop(df.columns[0], axis=1, inplace=True)
    df = df[df.duplicated(subset='content')==False]
    return df

df = load_data()

def remove_non_english(text):
    return re.sub(r'[^a-zA-Z\s]', '', text)

# Apply the function to the column
df['content'] = df['content'].apply(remove_non_english)

chatGPT_df = df.sample(2000)
chatGPT_df.reset_index(inplace=True)
chatGPT_df.drop(chatGPT_df.columns[0], axis=1, inplace=True)

# Scatter plot
st.title('ChatGPT Reviews Analysis')
st.header('Data Overview')
st.write(chatGPT_df.head())

st.subheader('Scatter Plot of Score vs Thumbs Up Count')
fig, ax = plt.subplots()
ax.scatter(chatGPT_df['score'], chatGPT_df['thumbsUpCount'], s=32, alpha=.8)
ax.set_xlabel('Score')
ax.set_ylabel('Thumbs Up Count')
ax.spines[['top', 'right']].set_visible(False)
st.pyplot(fig)

# Sentiment analysis
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

def score_sentiment(text):
    if not isinstance(text, str):
        text = str(text)
    score = sia.polarity_scores(text)
    compound_score = score['compound']
    if compound_score >= 0.6:
        return 5
    elif compound_score >= 0.2:
        return 4
    elif compound_score > -0.2 and compound_score < 0.2:
        return 3
    elif compound_score > -0.6:
        return 2
    else:
        return 1

chatGPT_df['content_score'] = chatGPT_df['content'].apply(score_sentiment)

# Prepare data for model evaluation
X = chatGPT_df.drop(["userName", "at", "content", "score"], axis='columns')
y = chatGPT_df['score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model parameters and grid search
model_params = {
    'linear_regression': {
        'model': LinearRegression(),
        'params': {'fit_intercept': [True, False]}
    },
    'ridge': {
        'model': Ridge(),
        'params': {
            'alpha': [0.1, 1.0, 10.0],
            'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'],
            'fit_intercept': [True, False]
        }
    },
    'lasso': {
        'model': Lasso(),
        'params': {
            'alpha': [0.1, 1.0, 10.0],
            'selection': ['random', 'cyclic'],
            'fit_intercept': [True, False]
        }
    },
    'decision_tree': {
        'model': DecisionTreeRegressor(),
        'params': {
            'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
            'splitter': ['best', 'random'],
            'max_depth': [None, 5, 10, 15, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    },
    'random_forest': {
        'model': RandomForestRegressor(),
        'params': {
            'n_estimators': [50, 100, 150],
            'max_depth': [None, 5, 10, 15, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    }
}

# Perform GridSearchCV for each model and store the best models
best_models = {}
for model_name, model_info in model_params.items():
    grid_search = GridSearchCV(estimator=model_info['model'], param_grid=model_info['params'], cv=5)
    grid_search.fit(X_train, y_train)
    best_models[model_name] = {
        'model_summary': str(grid_search.best_estimator_),
        'best_params': grid_search.best_params_,
        'best_score': grid_search.best_score_
    }

# Convert best models dictionary to DataFrame
df_best_models = pd.DataFrame(best_models).transpose()
df_best_models.reset_index(inplace=True)
df_best_models.columns = ['Model', 'Model Summary', 'Best Parameters', 'Best Score']

# Display model evaluation results
st.header('Model Evaluation')
st.write(df_best_models)

# Display plots
st.subheader('Content Score vs. Rating Score')
fig, ax = plt.subplots(figsize=(12, 6))
ax.scatter(chatGPT_df['content_score'], chatGPT_df['score'], alpha=0.5)
ax.set_xlabel('Content Score (1-5)')
ax.set_ylabel('Rating Score')
ax.set_title('Content Score vs. Rating Score')
st.pyplot(fig)

st.subheader('Content Score vs. Thumbs Up Count')
fig, ax = plt.subplots(figsize=(12, 6))
ax.scatter(chatGPT_df['content_score'], chatGPT_df['thumbsUpCount'], alpha=0.5)
ax.set_xlabel('Content Score (1-5)')
ax.set_ylabel('Thumbs Up Count')
ax.set_title('Content Score vs. Thumbs Up Count')
st.pyplot(fig)

# Model evaluation for RandomForestRegressor
model = RandomForestRegressor()
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, scoring='r2')
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
model = RandomForestRegressor(**best_params)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r_squared = r2_score(y_test, y_pred)

st.write(f"**Random Forest Regressor Model Evaluation:**")
st.write(f"**Best Parameters:** {best_params}")
st.write(f"**Mean Squared Error:** {mse}")
st.write(f"**Mean Absolute Error:** {mae}")
st.write(f"**R-squared:** {r_squared}")

# User input section
st.header('User Input Analysis')
user_input = st.text_area("Enter your sentence here:")

def weighted_random():
    choices = np.arange(0, 10)
    probabilities = np.array([0.3, 0.25, 0.2, 0.1, 0.05, 0.03, 0.02, 0.02, 0.02, 0.01])
    return np.random.choice(choices, p=probabilities)


if user_input:
    # Compute sentiment score
    content_score = score_sentiment(user_input)
    
    # Display content score
    st.write(f"**Content Score:** {content_score}")

    # Predict thumbs up count based on content score
    # This part assumes a simple heuristic where thumbs up count could be related to content score
    # In practice, you might use a model to predict this value  # Random example, replace with a real prediction model if available
    predicted_thumbs_up = weighted_random()
    st.write(f"**Estimated Thumbs Up Count:** {predicted_thumbs_up}")
