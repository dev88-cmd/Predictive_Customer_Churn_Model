import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, classification_report,
    confusion_matrix, RocCurveDisplay, PrecisionRecallDisplay
)

import joblib

df_raw = pd.read_csv("Churn.csv")

print(df_raw.shape)
print(df_raw.columns.tolist())
print(df_raw['Churn'].value_counts(normalize=True).rename("churn_rate"))
df=df_raw.copy()

df['Churn'] = df['Churn'].map({'No':0, 'Yes':1})

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors = 'coerce')

if 'customerID' in df.columns:
    df = df.drop(columns=['customerID'])

print(df.isna().sum().sort_values(ascending=False).head(10))

yes_no_cols=['PhoneService', 'MultipleLines', 'OnlineSecurity','OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV','StreamingMovies']

for c in yes_no_cols:
    if c in df.columns:
        df[c] = df[c].replace({'Yes':1, 'No':0, 'No phone service':0, 'No internet service':0})

if 'PaymentMethod' in df.columns:
    df['autopay'] = df['PaymentMethod'].str.contains('automatic', case=False, na=False).astype(int)
else:
    df['autopay']=0

if 'Contract' in df.columns:
    df['is_long_contract'] = df['Contract'].isin(['One year', 'Two year']).astype(int)
else:
    df['is_long_contract']=0        

if 'tenure' in df.columns and 'TotalCharges' in df.columns:
    df['charges_per_tenure'] = df['TotalCharges']/np.maximum(df['tenure'],1)

available_yes_no = [c for c in yes_no_cols if c in df.columns]
df['num_services']=df[available_yes_no].sum(axis=1) if available_yes_no else 0


y=df['Churn']
X=df.drop(columns=['Churn'])

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20, stratify=y, random_state=42)

print(y_train.mean(), y_test.mean())


num_selector = make_column_selector(dtype_include=np.number)
cat_selector = make_column_selector(dtype_include=object)

numeric_pipe = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipe = Pipeline(steps=[
    ("imputer",SimpleImputer(strategy="most_frequent")),
    ("ohe",OneHotEncoder(handle_unknown='ignore'))
])

preprocess = ColumnTransformer(transformers=[
    ('num', numeric_pipe, num_selector),
    ('cat', categorical_pipe, cat_selector)
])


dummy_clf = Pipeline([
    ('prep',preprocess),
    ('clf',DummyClassifier(strategy='most_frequent'))
])

logit_clf = Pipeline([
    ('prep', preprocess),
    ('clf', LogisticRegression(max_iter=1000, class_weight='balanced', n_jobs=None))
])

rf_clf = Pipeline([
    ('prep',preprocess),
    ('clf',RandomForestClassifier(n_estimators=400, max_depth=None, min_samples_leaf=2,random_state=42, class_weight='balanced',n_jobs=-1))
])


def evaluate_model(name, pipe, X_tr, y_tr, X_te, y_te, threshold=0.5,plot=False):
    pipe.fit(X_tr,y_tr)
    prob_tr=pipe.predict_proba(X_tr)[:,1]
    prob_te=pipe.predict_proba(X_te)[:,1]
    yhat_te=(prob_te>=threshold).astype(int)

    metrics = {
        'model': name,
        'accuracy':accuracy_score(y_te,yhat_te),
        'precision':precision_score(y_te,yhat_te,zero_division=0),
        'recall':recall_score(y_te,yhat_te,zero_division=0),
        'f1':f1_score(y_te,yhat_te,zero_division=0),
        'roc_auc':roc_auc_score(y_te,prob_te),
        'average_precision':average_precision_score(y_te,prob_te)
    }

    print(f"\n=={name}==")
    print(pd.DataFrame([metrics]).round(4))
    print("\nConfusion Matrix:\n")
    cm=confusion_matrix(y_te,yhat_te)
    print(pd.DataFrame(cm, index=['Actual:No(0)','Actual:Yes(1)'], columns=['Pred:No(0)','Pred:Yes(1)']))
    print("\nClassification Report:\n")
    print(classification_report(y_te,yhat_te,digits=3))

    if plot:
        RocCurveDisplay.from_predictions(y_te,prob_te)
        plt.title(f"ROC Curve - {name}")
        plt.show()

        PrecisionRecallDisplay.from_predictions(y_te,prob_te)
        plt.title(f"Precision-Recall Curve - {name}")
        plt.show()

    return metrics, pipe  

all_results=[]

m1, fitted_dummy = evaluate_model('Dummy', dummy_clf, X_train, y_train, X_test, y_test)
all_results.append(m1)

m2, fitted_logit = evaluate_model('Logistic Regression', logit_clf, X_train, y_train, X_test, y_test, plot=False)
all_results.append(m2)

m3, fitted_rf = evaluate_model("Random Forest", rf_clf, X_train, y_train, X_test, y_test, plot=False)
all_results.append(m3)

results_df = pd.DataFrame(all_results).sort_values(by='average_precision', ascending=False)
print('\n Summary(Sorted by Average Precision):')
print(results_df.round(4))

def get_feature_names_from_pipe(fitted_pipe):
    ct=fitted_pipe.named_steps['prep']
    return ct.get_feature_names_out()

def show_top_logit_coeffs(fitted_pipe, top_k=15):
    feat_name=get_feature_names_from_pipe(fitted_pipe)
    coef=fitted_pipe.named_steps['clf'].coef_[0]
    order=np.argsort(coef)
    low=list(zip(feat_name[order[:top_k]],coef[order[:top_k]]))
    high=list(zip(feat_name[order[-top_k:][::-1]],coef[order[-top_k:][::-1]]))

    print(f"\nTop Negative(reduce churn probability):")
    for f,v in low:
        print(f"{f:50s}{v: .4f}")
    print(f"\nTop Positive(increase churn probability):")
    for f,v in high:
        print(f"{f:50s}{v:.4f}")

show_top_logit_coeffs(fitted_logit)

def show_top_rf_importances(fitted_pipe, top_k=15):
    feat_names= get_feature_names_from_pipe(fitted_pipe)
    importances = fitted_pipe.named_steps['clf'].feature_importances_
    imp=pd.Series(importances, index=feat_names).sort_values(ascending=False).head(top_k)
    print(f"\nTop RF Feature Importances:")
    print(imp.to_string())

show_top_rf_importances(fitted_rf)    


def find_best_threshold_by_f1(fitted_pipe, X_val, y_val):
    prob = fitted_pipe.predict_proba(X_val)[:,1]
    ts = np.linspace(0.05, 0.95, 19)
    scores = [(t, f1_score(y_val, (prob >= t).astype(int))) for t in ts]
    best_t, best_f1 = max(scores, key=lambda x: x[1])
    return best_t, best_f1

best_t, best_f1 = find_best_threshold_by_f1(fitted_logit, X_test, y_test)
print(f"\nBest Threshold by F1 Score: {best_t:.2f} with F1: {best_f1:.3f}")


best_name = results_df.iloc[0]['model']
best_fitted = {'Dummy': fitted_dummy,
               'Logistic Regression': fitted_logit,
               'Random Forest': fitted_rf}[best_name]
joblib.dump(best_fitted, f'churn_{best_name.replace(" ", "_")}.joblib')
print(f'Saved: churn_{best_name.replace(" ", "_")}.joblib')
