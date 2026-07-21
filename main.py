import pandas as pd # type: ignore
import joblib # type: ignore
from sklearn.compose import ColumnTransformer # type: ignore
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder # type: ignore
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV # type: ignore
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error # type: ignore
from sklearn.pipeline import Pipeline # type: ignore    
from sklearn.linear_model import LinearRegression # type: ignore
from sklearn.tree import DecisionTreeRegressor # type: ignore
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor # type: ignore
import shap

df = pd.read_csv("data/Ds_Salaries.csv")

# encoder = LabelEncoder()
# model = LinearRegression()

# model = DecisionTreeRegressor(
#     max_depth=5,
#     min_samples_split=10,
#     min_samples_leaf=5,
#     random_state=42
# )

# model = RandomForestRegressor(
#     n_estimators=100,
#     random_state=42
# )

model = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=2,
    min_samples_split=2,
    min_samples_leaf=4,
    random_state=42
)


ordinal_cols = [
    "experience_level",
    "company_size"
]

onehot_cols = [
    "employment_type",
    "job_title",
    "employee_residence",
    "company_location"
]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "ordinal",
            OrdinalEncoder(
                categories=[
                    ["EN", "MI", "SE", "EX"],
                    ["S", "M", "L"]
                ]
            ),
            ordinal_cols
        ),
        (
            "onehot",
            OneHotEncoder(handle_unknown="ignore"),
            onehot_cols
        )
    ],
    remainder="passthrough"
)
# print("Preprocessor--->", preprocessor)


x = df[["work_year", "experience_level", "employment_type", "job_title", "remote_ratio", "company_location", "company_size", "employee_residence"]]
y = df["salary_in_usd"]

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])

param_grid = {
    "model__n_estimators":[50,100,150,200,300],
    "model__learning_rate":[0.01,0.03,0.05,0.1,0.2],
    "model__max_depth":[2,3,4,5],
    "model__min_samples_split":[2,5,10],
    "model__min_samples_leaf":[1,2,4]
}
grid = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=5,
    scoring="r2",
    n_jobs=-1
)

# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
scores = cross_val_score(pipeline, x, y, cv=5, scoring="r2")
# print("Cross Validation Scores:", scores)
print("Mean R2 Score:", scores.mean())
# pipeline.fit(x, y)

grid.fit(x, y)
# print("grid Score:", grid.best_score_)
# print(grid.best_params_)
best_pipeline = grid.best_estimator_

# model.fit(x_train, y_train)
# predictions = pipeline.predict(x_test)

# print("\nModel Score--->", pipeline.score(x_test, y_test)) 
# print("\nMean Absolute Error--->", mean_absolute_error(y_test, predictions))
# print("\nMean Squared Error--->", mean_squared_error(y_test, predictions)) 
# print("\nRoot Mean Squared Error--->", root_mean_squared_error(y_test, predictions)) 
# print ("\nR2 Score--->", r2_score(y_test, predictions))

# best_pipeline = grid.best_estimator_
# model = best_pipeline.named_steps["model"]
# preprocessor = best_pipeline.named_steps["preprocessor"]
# x_transformed = preprocessor.transform(x)

# # If sparse, convert to dense
# if hasattr(x_transformed, "toarray"):
#     x_transformed = x_transformed.toarray()

# # Force numeric type
# x_transformed = x_transformed.astype(float)
# explainer = shap.TreeExplainer(model)
# shap_values = explainer.shap_values(
#     x_transformed
# )
# shap.summary_plot(
#     shap_values,
#     x_transformed,
#     feature_names=preprocessor.get_feature_names_out(),
#     plot_type="bar"
# )


joblib.dump(best_pipeline, "models/gradient_boosting_model_v1.pkl")
