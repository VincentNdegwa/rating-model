# Movie Ratings Prediction

This project consists of two Python notebooks for training a machine learning model to predict IMDb movie ratings and generating predictions for new movies. The model is built using **Linear Regression** from scikit-learn.

## Files in the Project

### `movie_model_training.ipynb`
This notebook handles:
1. **Data Preprocessing**:
   - Reads IMDb dataset (`imdb-movies-dataset.csv`).
   - Drops categorical and irrelevant columns.
   - Converts `Votes` and `Review Count` to numeric format and handles missing values.
2. **Model Training**:
   - Splits the data into training and testing sets.
   - Trains a **Linear Regression** model to predict movie ratings.
3. **Model Evaluation**:
   - Calculates performance metrics:
     - **R-squared (R²):** Measures model accuracy.
     - **Mean Absolute Error (MAE):** Evaluates prediction errors.
4. **Model Saving**:
   - Saves the trained model as `imdb_rating_prediction_model.pkl` using `joblib`.

### `movie_model_prediction.ipynb`
This notebook handles:
1. **Model Loading**:
   - Loads the saved Linear Regression model.
2. **Prediction on New Data**:
   - Accepts a dataset of new movies with features like `Year`, `Duration`, `Metascore`, `Votes`, and `Review Count`.
   - Processes the input data to match the trained model's format.
   - Predicts IMDb ratings for the new movies.
3. **Visualization**:
   - Displays predicted ratings using a **bar plot** for better visualization.

## Features Used for Prediction
- `Year`: The release year of the movie.
- `Duration (min)`: The duration of the movie in minutes.
- `Metascore`: The Metascore of the movie.
- `Votes`: The number of votes, converted to numerical format.
- `Review Count`: The number of reviews, converted to numerical format.

## Requirements
- Python 3.x
- Libraries:
  - `pandas`
  - `scikit-learn`
  - `joblib`
  - `matplotlib`
  - `seaborn`

## How to Use
1. **Train the Model**:
   - Open and run `movie_model_training.ipynb` with the provided dataset (`imdb-movies-dataset.csv`).
   - The trained model will be saved as `imdb_rating_prediction_model.pkl`.

2. **Predict Ratings for New Data**:
   - Open and run `movie_model_prediction.ipynb`.
   - Add new movie data with features like `Year`, `Duration`, `Metascore`, `Votes`, and `Review Count`.
   - The predictions will be displayed as a bar plot.

## Outputs
- A trained Linear Regression model saved as `imdb_rating_prediction_model.pkl`.
- Predicted ratings for new movies, visualized in a bar plot.

