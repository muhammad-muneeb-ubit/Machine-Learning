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
import shap #type: ignore
from xgboost import XGBRegressor #type: ignore

df = pd.read_csv("data/Ds_Salaries.csv")

model = XGBRegressor(
    objective="reg:squarederror",
    random_state=42,
    tree_method="hist",
    eval_metric="rmse"
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


x = df[["work_year", "experience_level", "employment_type", "job_title", "remote_ratio", "company_location", "company_size", "employee_residence"]]
y = df["salary_in_usd"]

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])

param_grid = {
    "model__n_estimators": [200,300,500,700],
    "model__learning_rate": [0.01,0.03,0.05,0.1],
    "model__max_depth": [3,4,5,6,8],
    "model__subsample": [0.7,0.8,0.9,1],
    "model__colsample_bytree": [0.6,0.7,0.8,0.9,1],
    "model__gamma": [0,0.1,0.3,0.5,1],
    "model__reg_alpha": [0,0.1,0.5,1],
    "model__reg_lambda": [1,2,5,10],
    "model__min_child_weight": [1,3,5,7]
}

grid = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_grid,
    n_iter=100,
    cv=5,
    scoring="r2",
    random_state=42,
    n_jobs=-1
)



grid.fit(x, y)
best_pipeline = grid.best_estimator_

print("Best CV Score:", grid.best_score_)
print("Best Parameters:")
print(grid.best_params_)

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


joblib.dump(best_pipeline, "models/xgboost_model_v1.pkl")
