import os
import json
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import shap

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

warnings.filterwarnings("ignore")


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.join(BASE_DIR, "model")
DATA_DIR = os.path.join(BASE_DIR, "data")

MODEL_PATH = os.path.join(MODEL_DIR, "dropout_dnn.pt")
CONFIG_PATH = os.path.join(MODEL_DIR, "preprocessing_config.json")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")
SHAP_BACKGROUND_PATH = os.path.join(
    MODEL_DIR,
    "shap_background.npy"
)

STUDENTS_PATH = os.path.join(
    DATA_DIR,
    "students.csv"
)


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="Student Dropout Prediction API",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


# ============================================================
# MODEL
# ============================================================

class DropoutDNN(nn.Module):

    def __init__(self, input_dim):

        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.30),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.20),

            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.15),

            nn.Linear(32, 1)
        )

    def forward(self, x):

        return self.net(x)


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading model...")

checkpoint = torch.load(
    MODEL_PATH,
    map_location="cpu",
    weights_only=False
)

input_dim = checkpoint["input_dim"]

model = DropoutDNN(input_dim)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print(
    f"Model loaded successfully. Input dimension: {input_dim}"
)


# ============================================================
# LOAD PREPROCESSING CONFIG
# ============================================================

print("Loading preprocessing configuration...")

with open(CONFIG_PATH, "r") as f:

    config = json.load(f)


numeric_features = config[
    "numeric_features"
]

categorical_features = config[
    "categorical_features"
]


num_medians = dict(
    zip(
        numeric_features,
        config["num_medians"]
    )
)

num_mean = dict(
    zip(
        numeric_features,
        config["num_mean"]
    )
)

num_scale = dict(
    zip(
        numeric_features,
        config["num_scale"]
    )
)

categorical_info = config[
    "categorical_info"
]

feature_names = config[
    "feature_names"
]


print(
    f"Numerical features: {len(numeric_features)}"
)

print(
    f"Categorical features: {len(categorical_features)}"
)

print(
    f"Final features: {len(feature_names)}"
)


# ============================================================
# LOAD METRICS
# ============================================================

with open(METRICS_PATH, "r") as f:

    metrics = json.load(f)


# ============================================================
# LOAD STUDENT DATA
# ============================================================

print("Loading student dataset...")

students = pd.read_csv(
    STUDENTS_PATH
)

print(
    f"Students loaded: {len(students)}"
)


# ============================================================
# LOAD SHAP BACKGROUND
# ============================================================

print("Loading SHAP background...")

shap_background = np.load(
    SHAP_BACKGROUND_PATH
)

print(
    f"SHAP background shape: {shap_background.shape}"
)


# ============================================================
# DISPLAY NAMES
# ============================================================

display_names = {

    "num__num_of_prev_attempts":
        "Previous Attempts",

    "num__studied_credits":
        "Studied Credits",

    "num__date_registration":
        "Registration Date",

    "num__clicks_8w":
        "LMS Clicks (8 Weeks)",

    "num__active_days_8w":
        "Active Days",

    "num__active_weeks_8w":
        "Active Weeks",

    "num__unique_sites_8w":
        "Unique LMS Resources",

    "num__avg_daily_clicks":
        "Average Daily Activity",

    "num__clicks_w1_4":
        "Early Engagement (Weeks 1–4)",

    "num__clicks_w5_8":
        "Recent Engagement (Weeks 5–8)",

    "num__engagement_change":
        "Engagement Change",

    "num__engagement_ratio":
        "Engagement Ratio",

    "num__assessments_attempted":
        "Assessments Attempted",

    "num__mean_score_8w":
        "Average Assessment Score",

    "num__score_std_8w":
        "Assessment Score Variability",

    "num__late_submissions_8w":
        "Late Submissions",

    "num__avg_submission_delay":
        "Average Submission Delay"

}   

def humanize_feature_name(feature):

    if feature in display_names:
        return display_names[feature]

    if feature.startswith("cat__"):

        value = feature.replace(
            "cat__",
            ""
        )

        parts = value.split("_", 1)

        if len(parts) == 2:

            feature_name, category = parts

            readable = {

                "gender":
                    "Gender",

                "region":
                    "Region",

                "highest_education":
                    "Education Level",

                "imd_band":
                    "Socioeconomic Band",

                "age_band":
                    "Age Band",

                "disability":
                    "Disability Status",

                "code_module":
                    "Module",

                "code_presentation":
                    "Presentation"

            }.get(
                feature_name,
                feature_name.replace("_", " ").title()
            )

            return f"{readable}: {category}"

    return feature

# ============================================================
# PREPROCESSING
# ============================================================

def preprocess_student(row):

    transformed = []

    # --------------------------------------------------------
    # Numerical features
    # --------------------------------------------------------

    for feature in numeric_features:

        value = row.get(
            feature,
            np.nan
        )

        try:

            value = float(value)

        except:

            value = np.nan


        # Median imputation

        median = float(
            num_medians[feature]
        )

        if np.isnan(value):

            value = median


        # Standardization

        mean = float(
            num_mean[feature]
        )

        scale = float(
            num_scale[feature]
        )

        if scale == 0:

            scale = 1.0


        value = (
            value - mean
        ) / scale


        transformed.append(
            value
        )


    # --------------------------------------------------------
    # Categorical features
    # --------------------------------------------------------

    for feature in categorical_features:

        info = categorical_info[
            feature
        ]

        fill_value = info[
            "fill_value"
        ]

        categories = info[
            "categories"
        ]


        value = row.get(
            feature,
            None
        )


        if pd.isna(value):

            value = fill_value


        value = str(value)


        # Exact One-Hot Encoding
        # using the categories from training

        for category in categories:

            if value == str(category):

                transformed.append(
                    1.0
                )

            else:

                transformed.append(
                    0.0
                )


    result = np.array(
        transformed,
        dtype=np.float32
    )


    # Safety check

    if len(result) != input_dim:

        raise ValueError(
            f"Feature mismatch: "
            f"created {len(result)} features, "
            f"but model expects {input_dim}"
        )


    return result


# ============================================================
# PREDICTION
# ============================================================

def predict(X):

    tensor = torch.tensor(
        X,
        dtype=torch.float32
    )


    model.eval()


    with torch.no_grad():

        logits = model(
            tensor
        )

        probabilities = torch.sigmoid(
            logits
        ).numpy().flatten()


    return probabilities


# ============================================================
# RISK LEVEL
# ============================================================

def risk_level(probability):

    if probability >= 0.70:

        return "HIGH"

    elif probability >= 0.40:

        return "MEDIUM"

    return "LOW"


# ============================================================
# SHAP EXPLAINER
# ============================================================

print("Initializing SHAP...")

explainer = shap.DeepExplainer(

    model,

    torch.tensor(
        shap_background,
        dtype=torch.float32
    )

)

print("SHAP initialized successfully.")


# ============================================================
# SHAP EXPLANATION
# ============================================================

def explain_prediction(X):

    tensor = torch.tensor(
        X,
        dtype=torch.float32
    )


    values = explainer.shap_values(
        tensor
    )


    if isinstance(values, list):

        values = values[0]


    values = np.asarray(
        values
    )


    if values.ndim == 3:

        values = values.squeeze()


    values = values.flatten()


    results = []


    for feature, value in zip(
        feature_names,
        values
    ):

        results.append({

            "feature":
                feature,

            "name":
                humanize_feature_name(feature),

            "value":
                float(value)

        })


    results.sort(
        key=lambda x:
            abs(x["value"]),
        reverse=True
    )


    return results[:12]


# ============================================================
# REQUEST MODEL
# ============================================================

class StudentRequest(BaseModel):

    student_id: int


# ============================================================
# API STATUS
# ============================================================

@app.get("/api")
def api_status():

    return {

        "status":
            "running",

        "model":
            "Dropout DNN",

        "input_features":
            input_dim,

        "students":
            len(students)

    }


# ============================================================
# STUDENT LIST
# ============================================================

@app.get("/api/students")
def get_students():

    result = []


    unique_students = (

        students[
            [
                "id_student",
                "code_module",
                "code_presentation"
            ]
        ]

        .drop_duplicates()

    )


    for _, row in unique_students.iterrows():

        result.append({

            "student_id":
                int(row["id_student"]),

            "module":
                str(row["code_module"]),

            "presentation":
                str(row["code_presentation"])

        })


    return result


# ============================================================
# STUDENT PREDICTION
# ============================================================

@app.post("/api/predict")
def predict_student(
    request: StudentRequest
):

    student_rows = students[
        students["id_student"]
        == request.student_id
    ]


    if len(student_rows) == 0:

        raise HTTPException(

            status_code=404,

            detail="Student not found"

        )


    # For now use first course record
    # for the selected student

    row = student_rows.iloc[0]


    X = preprocess_student(
        row
    )


    X = X.reshape(
        1,
        -1
    )


    probability = float(
        predict(X)[0]
    )


    risk = risk_level(
        probability
    )


    shap_results = explain_prediction(
        X
    )


    positive = [

        x for x in shap_results

        if x["value"] > 0

    ]


    negative = [

        x for x in shap_results

        if x["value"] < 0

    ]


    def safe_float(value):

        if pd.isna(value):

            return None

        return float(value)


    return {

        "student": {

            "id":
                int(row["id_student"]),

            "module":
                str(row["code_module"]),

            "presentation":
                str(row["code_presentation"]),

            "age_band":
                str(row["age_band"]),

            "gender":
                str(row["gender"])

        },


        "prediction": {

            "probability":
                probability,

            "percentage":
                probability * 100,

            "risk":
                risk

        },


        "behaviour": {

            "clicks":
                safe_float(
                    row["clicks_8w"]
                ),

            "active_days":
                safe_float(
                    row["active_days_8w"]
                ),

            "active_weeks":
                safe_float(
                    row["active_weeks_8w"]
                ),

            "unique_sites":
                safe_float(
                    row["unique_sites_8w"]
                ),

            "assessments":
                safe_float(
                    row["assessments_attempted"]
                ),

            "mean_score":
                safe_float(
                    row["mean_score_8w"]
                ),

            "late_submissions":
                safe_float(
                    row["late_submissions_8w"]
                ),

            "engagement_change":
                safe_float(
                    row["engagement_change"]
                )

        },


        "shap": {

            "top":
                shap_results,

            "positive":
                positive[:5],

            "negative":
                negative[:5]

        }

    }


# ============================================================
# MODEL METRICS
# ============================================================

@app.get("/api/metrics")
def get_metrics():

    return {

        "metrics":
            metrics,

        "architecture": {

            "input":
                input_dim,

            "layers": [
                128,
                64,
                32,
                1
            ]

        }

    }


# ============================================================
# COHORT ANALYTICS
# ============================================================

@app.get("/api/cohort")
def cohort():

    # --------------------------------------------------------
    # Generate predictions for entire cohort
    # --------------------------------------------------------

    X = np.vstack(
        students.apply(
            preprocess_student,
            axis=1
        ).values
    )

    probabilities = predict(X)

    result = students.copy()

    result["dropout_probability"] = probabilities

    result["risk_level"] = (
        result["dropout_probability"]
        .apply(risk_level)
    )


    # --------------------------------------------------------
    # Overall statistics
    # --------------------------------------------------------

    total_students = len(result)

    high = int(
        (
            result["risk_level"] == "HIGH"
        ).sum()
    )

    medium = int(
        (
            result["risk_level"] == "MEDIUM"
        ).sum()
    )

    low = int(
        (
            result["risk_level"] == "LOW"
        ).sum()
    )


    observed_dropout = None

    if "dropout" in result.columns:

        observed_dropout = float(
            result["dropout"].mean() * 100
        )


    # --------------------------------------------------------
    # Module analytics
    # --------------------------------------------------------

    modules = []

    for module, group in result.groupby(
        "code_module"
    ):

        predicted_risk = (
            group["dropout_probability"].mean()
            * 100
        )

        module_data = {

            "module":
                str(module),

            "students":
                int(len(group)),

            "predicted_risk":
                round(
                    float(predicted_risk),
                    2
                )

        }


        if "dropout" in group.columns:

            module_data[
                "observed_dropout"
            ] = round(
                float(
                    group["dropout"].mean()
                    * 100
                ),
                2
            )

        else:

            module_data[
                "observed_dropout"
            ] = None


        modules.append(
            module_data
        )


    # --------------------------------------------------------
    # High-risk students
    # --------------------------------------------------------

    high_risk_df = result[
        result["risk_level"] == "HIGH"
    ].copy()


    high_risk_df = high_risk_df.sort_values(
        "dropout_probability",
        ascending=False
    )


    high_risk_students = []


    for _, row in high_risk_df.head(50).iterrows():

        high_risk_students.append({

            "student_id":
                int(row["id_student"]),

            "module":
                str(row["code_module"]),

            "presentation":
                str(row["code_presentation"]),

            "risk":
                round(
                    float(
                        row["dropout_probability"]
                        * 100
                    ),
                    2
                ),

            "clicks":
                int(
                    row["clicks_8w"]
                )
                if pd.notna(
                    row["clicks_8w"]
                )
                else 0,

            "active_days":
                int(
                    row["active_days_8w"]
                )
                if pd.notna(
                    row["active_days_8w"]
                )
                else 0,

            "mean_score":
                round(
                    float(
                        row["mean_score_8w"]
                    ),
                    1
                )
                if pd.notna(
                    row["mean_score_8w"]
                )
                else None

        })


    # --------------------------------------------------------
    # Return
    # --------------------------------------------------------

    return {

        "total_students":
            total_students,

        "high_risk":
            high,

        "medium_risk":
            medium,

        "low_risk":
            low,

        "observed_dropout":
            observed_dropout,

        "modules":
            modules,

        "high_risk_students":
            high_risk_students

    }


# ============================================================
# SERVE FRONTEND
# ============================================================

FRONTEND_DIR = os.path.join(

    os.path.dirname(
        BASE_DIR
    ),

    "frontend"

)


app.mount(

    "/",

    StaticFiles(

        directory=
            FRONTEND_DIR,

        html=True

    ),

    name="frontend"

)