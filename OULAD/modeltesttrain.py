import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# 1) Load data
info = pd.read_csv('studentInfo.csv')
assess = pd.read_csv('studentAssessment.csv')
reg = pd.read_csv('studentRegistration.csv')
courses = pd.read_csv('courses.csv')
ass_meta = pd.read_csv('assessments.csv')

# 2) Define target (binary)
risk_map = {'Fail': 1, 'Withdrawn': 1, 'Pass': 0, 'Distinction': 0}
info['at_risk'] = info['final_result'].map(risk_map)

# 3) Aggregate assessment features per student
ass_agg = assess.groupby('id_student').agg(
    score_mean=('score','mean'),
    score_std=('score','std'),
    score_min=('score','min'),
    score_max=('score','max'),
    score_count=('score','count'),
    date_mean=('date_submitted','mean'),
    date_std=('date_submitted','std'),
    banked_sum=('is_banked','sum')
).reset_index()

# 4) Registration features per student
reg2 = reg.copy()
reg2['date_unregistration'] = pd.to_numeric(reg2['date_unregistration'], errors='coerce')
reg_agg = reg2.groupby('id_student').agg(
    reg_date_min=('date_registration','min'),
    unreg_events=('date_unregistration', lambda x: np.isfinite(x).sum()),
    course_count=('code_module','count'),
).reset_index()

# 5) Course context per student (join registration to course lengths)
reg_courses = reg.merge(courses, on=['code_module','code_presentation'], how='left')
course_agg = reg_courses.groupby('id_student').agg(
    course_len_mean=('module_presentation_length','mean'),
    course_len_sum=('module_presentation_length','sum')
).reset_index()

# 6) Merge all features onto info
df = info.merge(ass_agg, on='id_student', how='left') \
         .merge(reg_agg, on='id_student', how='left') \
         .merge(course_agg, on='id_student', how='left')

# 7) Fill numeric NaNs
num_cols = df.select_dtypes(include=[np.number]).columns
df[num_cols] = df[num_cols].fillna(-1)

# 8) Select features and target
cat_features = ['gender','region','highest_education','imd_band','age_band','disability']
num_features = [
    'num_of_prev_attempts','studied_credits',
    'score_mean','score_std','score_min','score_max','score_count',
    'date_mean','date_std','banked_sum',
    'reg_date_min','unreg_events','course_count',
    'course_len_mean','course_len_sum'
]
X = df[cat_features + num_features]
y = df['at_risk'].astype(int)

# 9) Preprocess + two models
preprocess = ColumnTransformer(
    transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)],
    remainder='passthrough'
)

lr = Pipeline(steps=[('prep', preprocess),
                    ('clf', LogisticRegression(max_iter=1000, class_weight='balanced'))])

rf = Pipeline(steps=[('prep', preprocess),
                    ('clf', RandomForestClassifier(
                        n_estimators=300, max_depth=None, min_samples_split=10,
                        min_samples_leaf=3, random_state=42, class_weight='balanced'
                    ))])

# 10) Train/test split (stratified)
X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
    X, y, df['id_student'], test_size=0.2, random_state=42, stratify=y
)

# 11) Fit and evaluate both models
def fit_eval(pipe, name):
    pipe.fit(X_train, y_train)
    y_tr = pipe.predict(X_train)
    y_te = pipe.predict(X_test)
    print(f'{name} Train Acc: {accuracy_score(y_train, y_tr):.3f}')
    print(f'{name} Test  Acc: {accuracy_score(y_test, y_te):.3f}')
    return y_te, pipe

y_pred_lr, lr_model = fit_eval(lr, 'LogReg')
y_pred_rf, rf_model = fit_eval(rf, 'RandForest')

# 12) Select best model by test accuracy
acc_lr = accuracy_score(y_test, y_pred_lr)
acc_rf = accuracy_score(y_test, y_pred_rf)
best_model = rf_model if acc_rf >= acc_lr else lr_model
best_name = 'RandomForest' if acc_rf >= acc_lr else 'LogReg'
best_pred = y_pred_rf if acc_rf >= acc_lr else y_pred_lr
best_acc_train = accuracy_score(y_train, best_model.predict(X_train))
best_acc_test = accuracy_score(y_test, best_pred)
print(f'Best: {best_name} | Train: {best_acc_train:.3f} | Test: {best_acc_test:.3f}')

# 13) Predict for all students and export
all_preds = best_model.predict(X)
out = pd.DataFrame({
    'id_student': df['id_student'],
    'risk_label': np.where(all_preds==1, 'At Risk', 'Not at Risk')
})
out = out.drop_duplicates('id_student')
out.to_csv('student_risk_predictions.csv', index=False)
print('Saved student_risk_predictions.csv with id_student and risk_label')
