from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

# Hyperparameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)  # Use your training data

print("🏆 Best Parameters:", grid_search.best_params_)
print("🎯 Best CV Score:", grid_search.best_score_)

# Save tuned model
joblib.dump(grid_search.best_estimator_, 'tuned_model.pkl')