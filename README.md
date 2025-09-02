# NFL Game Analysis

## Project Overview

This project analyzes National Football League (NFL) game data to determine whether machine learning algorithms can accurately predict key game outcomes using publicly available data. The analysis focuses on three main research questions:

1. **High-Scoring Games**: Can we predict whether a game will be high scoring (top 25% of total points)?
2. **Home Team Victories**: Can we predict whether the home team will win outright?
3. **Spread Coverage**: Can we predict whether the favorite team will cover the point spread?

## Motivation

As a former college athlete with a lifelong connection to sports, this project explores the intersection of sports analytics and machine learning. While not specifically a football player, the nationwide popularity of NFL provided access to comprehensive datasets that other sports couldn't match.

## Data Sources

The dataset was retrieved from Kaggle and includes:
- **Primary Dataset**: NFL scores and betting data by T. Crabtree
- **Team Metadata**: Mapping of team IDs to team names
- **Time Range**: Multiple decades of NFL seasons
- **Game Types**: Regular season games (primary focus) with playoff information available

### Data Features

- Game scheduling information (date, week, season)
- Team identities (home/away)
- Final scores
- Betting lines (point spreads, over/under)
- Weather conditions
- Stadium information
- Playoff designations

## Data Preprocessing

### Cleaning Steps

1. **Date Conversion**: Converted schedule dates to proper datetime format
2. **Numeric Conversion**: Converted point spreads, scores, and weather data from strings to numeric types
3. **Data Filtering**: Removed games with incomplete data or non-standard week designations
4. **Feature Engineering**:
   - Total points per game
   - Point differential calculations
   - Binary indicators for high-scoring games
   - Favorite team win margin calculations
   - Spread coverage indicators

### Feature Engineering

- **High-Scoring Classification**: Top 25% of games by total points
- **Weather Processing**: Temperature, wind speed, humidity, indoor/outdoor designation
- **Team Performance Metrics**: Recent win streaks (3-game rolling average)
- **Betting Line Analysis**: Spread coverage calculations
- **Home Field Advantage**: Binary indicators for home team status

## Exploratory Data Analysis

### Key Visualizations

1. **Scoring Trends**:
   - Playoff vs. regular season scoring patterns
   - Long-term scoring trends (1970-2020)
   - Team-specific scoring patterns

2. **Team Performance**:
   - Win distribution across teams
   - Historical performance analysis
   - Spread betting patterns by team

3. **Weather Impact**:
   - Temperature effects on scoring
   - Indoor vs. outdoor game analysis
   - Seasonal variations

### Notable Findings

- Steady increase in offensive output, particularly since 2010
- New England Patriots showing dominant win totals in analyzed periods
- Kansas City Chiefs and Green Bay Packers frequently favored in betting markets
- Buffalo Bills showing strong upward scoring trends in recent years

## Machine Learning Models

### Model 1: High-Scoring Game Prediction (Logistic Regression)

**Target Variable**: Binary classification of games in top 25% of total points

**Features**:
- Point spread
- Weather conditions (temperature, wind, humidity)
- Stadium information
- Team identities
- Schedule week

**Results**:
- Accuracy: 56.6%
- Precision (high-scoring): 0.30
- Recall (high-scoring): 0.50
- F1-score (high-scoring): 0.37

**Analysis**: Model shows limited predictive power for identifying high-scoring games, suggesting total points are difficult to predict from pre-game metadata alone.

### Model 2: Home Team Win Prediction (Random Forest)

**Target Variable**: Binary classification of home team victories

**Features**: Same as Model 1 plus home team win history

**Results**:
- Accuracy: 57.6%
- Precision (home wins): 0.61
- Recall (home wins): 0.76
- Weighted F1-score: 0.55

**Analysis**: Model captures home field advantage trends but struggles with class imbalance, performing better at predicting wins than losses.

### Model 3: Spread Coverage Prediction (XGBoost)

**Target Variable**: Binary classification of favorite covering the point spread

**Initial Features**: Same as previous models

**Enhanced Features** (for optimization):
- Recent win streaks (home, away, favorite, underdog teams)
- Binary indicator for favorite playing at home
- Advanced weather metrics

**Results**:
- Initial Accuracy: 72.6%
- Enhanced Accuracy: 73.5%
- Precision (favorite covers): 0.78
- Recall (favorite covers): 0.90
- F1-score (favorite covers): 0.84

**Analysis**: Most successful model, showing that betting market predictions can be modeled with reasonable accuracy when using sophisticated algorithms and rich feature sets.

## Technical Implementation

### Libraries Used
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as XGBClassifier
