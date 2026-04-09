import os
# root path
root_dir = os.path.dirname(os.path.dirname(__file__))


# ---------------------------
# path data
# ---------------------------
data_dir = os.path.join(root_dir,"data")
data_raw_dir = os.path.join(data_dir,"raw")
data_processed = os.path.join(data_dir,"processed")
# path data raw
data_raw_path = os.path.join(data_raw_dir,"Life Expectancy Data.csv")


# --------------------------
# path reports
# --------------------------
reports_dir = os.path.join(root_dir,"reports")
reports_eda_dir = os.path.join(reports_dir,"edu")
reports_results_dir = os.path.join(reports_dir,"results")
# name file report
file_name_reports = "report_life_expectancy.html"
#dir parameter
dir_parameter = os.path.join(reports_dir,"parameter")

# --------------------------
# random stage and test size
# --------------------------
random_state = 42
test_size = 0.2

# -------------------------
# columns required
# -------------------------
target_col = "Life expectancy"
numerical_cols = [
    "Year",
    "Adult Mortality",
    "Alcohol",
    "Hepatitis B",
    "Measles",
    "BMI",
    "under-five deaths",
    "Polio",
    "Total expenditure",
    "Diphtheria",
    "HIV/AIDS",
    "Income composition of resources",
    "Schooling"
]
nominal_cols = [
    "Country"
]
ordinal_cols = [
    "Status"
]
unnecessary_cols = [
    "infant deaths",
    "percentage expenditure",
    "GDP","Population",
    "thinness  1-19 years",
    "thinness 5-9 years"
]

# -----------------------------
# path model
#------------------------------
dir_model = os.path.join(root_dir,"models")


