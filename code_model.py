#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

import GPy
from lightgbm import LGBMClassifier

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score,
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    confusion_matrix,
)

import joblib


# In[ ]:


def to_minutes(hhmm):
    h, m = map(int, hhmm.split(':'))
    return h * 60 + m

def clean_static_block(df):
    stat_names = ['Age', 'Gender', 'Height', 'Weight', 'ICUType']
    df_stat = (
        df[df['parameter'].isin(stat_names)]
        .sort_values('time_min')
        .groupby(['recordid', 'parameter'])['value']
        .last()                       
        .unstack() 
    )

    df_stat['Age'] = pd.to_numeric(df_stat['Age'], errors='coerce')
    df_stat.loc[df_stat['Age'] > 130, 'Age'] = 91.4   

    df_stat['Gender'] = pd.to_numeric(df_stat['Gender'], errors='coerce')
    df_stat.loc[~df_stat['Gender'].isin([0, 1]), 'Gender'] = np.nan

    df_stat['Height'] = pd.to_numeric(df_stat['Height'], errors='coerce')
    df_stat.loc[df_stat['Height'] <= 0, 'Height'] = np.nan
    df_stat.loc[df_stat['Height'] > 250, 'Height'] = df_stat['Height'] / 10

    df_stat['Weight'] = pd.to_numeric(df_stat['Weight'], errors='coerce')
    df_stat.loc[df_stat['Weight'] <= 0, 'Weight'] = np.nan
    mask_lb = df_stat['Weight'] > 300
    df_stat.loc[mask_lb, 'Weight'] = df_stat.loc[mask_lb, 'Weight'] / 2.20462

    df_stat['ICUType'] = pd.to_numeric(df_stat['ICUType'], errors='coerce')
    df_stat.loc[~df_stat['ICUType'].isin([1, 2, 3, 4]), 'ICUType'] = np.nan

    return df_stat.reset_index()

def clean_dynamic_row(row):
    p, v = row['parameter'], row['value']
    try:
        v = float(v)
    except (ValueError, TypeError):
        return np.nan

    if p == 'HR':
        if v < 20 or v > 300:
            return np.nan

    elif p == 'RespRate':
        if v < 2 or v > 80:
            return np.nan

    elif p == 'Temp':
        if 80 <= v <= 120:
            v = (v - 32) * 5.0 / 9.0
        if v < 25 or v > 45:
            return np.nan

    elif p in ['NISysABP', 'SysABP']:
        if v < 50 or v > 300:
            return np.nan
    elif p in ['NIDiasABP', 'DiasABP']:
        if v < 20 or v > 200:
            return np.nan
    elif p in ['NIMAP', 'MAP']:
        if v < 30 or v > 250:
            return np.nan

    elif p == 'Glucose':
        if v < 1 or v > 100:          
            return np.nan

    elif p == 'GCS':
        if v < 3 or v > 15:
            return np.nan

    elif p == 'Urine':
        if v < 0:
            v = 0

    return v


def load_and_clean_physionet(data_dir):
    long_rows = []

    for fname in os.listdir(data_dir):
        if not fname.endswith('.txt'):
            continue

        rec_id = int(os.path.splitext(fname)[0])
        path = os.path.join(data_dir, fname)
        df = pd.read_csv(path)

        df['time_min'] = df['Time'].apply(to_minutes)

        df = df.rename(columns={'Parameter': 'parameter', 'Value': 'value'})
        df['recordid'] = rec_id

        df = df[df['parameter'] != 'RecordID']

        df['value'] = df.apply(clean_dynamic_row, axis=1)

        df = df.dropna(subset=['value'])

        long_rows.append(df[['recordid', 'time_min', 'parameter', 'value']])

    df_long = pd.concat(long_rows, ignore_index=True)
    df_long = df_long.sort_values(['recordid', 'parameter', 'time_min'])

    df_static = clean_static_block(df_long.copy())

    return df_static, df_long

df_static, df_long = load_and_clean_physionet(
    r'C:\Users\bogdy\Downloads\Ilya\predicting-mortality-of-icu-patients-the-physionet-computing-in-cardiology-challenge-2012-1.0.0\set-a'
)


# In[ ]:


STATIC_VARS = {'Age', 'Gender', 'Height', 'Weight', 'ICUType'}

def get_task_list(df_long, static_vars=STATIC_VARS, user_tasks=None):

    if user_tasks is not None:
        return list(user_tasks)

    tasks = sorted([p for p in df_long['parameter'].unique()
                    if p not in static_vars])
    return tasks

def prepare_patient_XY(df_patient, task_to_idx):

    X_parts, Y_parts = [], []

    for task, idx in task_to_idx.items():
        sub = df_patient[df_patient['parameter'] == task]
        if sub.empty:
            continue                       
        t = sub['time_min'].values.reshape(-1, 1)   
        y = sub['value'].values.reshape(-1, 1)      
        task_col = np.full_like(t, idx)             
        X_parts.append(np.hstack([t, task_col]))
        Y_parts.append(y)

    if not X_parts:
        return None, None                          

    X = np.vstack(X_parts).astype(float)
    Y = np.vstack(Y_parts).astype(float)
    return X, Y

def build_mtgp_dataset(df_long, tasks=None, min_points=10):

    tasks = get_task_list(df_long, user_tasks=tasks)
    task_to_idx = {t: i for i, t in enumerate(tasks)}

    X_dict, Y_dict = {}, {}
    for rid, df_p in df_long.groupby('recordid', sort=False):
        X, Y = prepare_patient_XY(df_p, task_to_idx)
        if X is None or len(Y) < min_points:
            continue                            
        X_dict[rid] = X
        Y_dict[rid] = Y

    return tasks, X_dict, Y_dict

tasks, X_dict, Y_dict = build_mtgp_dataset(df_long,
                                            tasks=['HR','RespRate','Temp',
                                                   'NIMAP','Urine','GCS'])

X_dict[132539].shape, Y_dict[132539].shape



# In[ ]:


def fit_mtgp_one(X, Y, num_tasks, time_scale=60.0, restarts=3):
    
    X_sc = X.copy()
    X_sc[:, 0] /= time_scale       

    k_time = GPy.kern.RBF(input_dim=1,
                          variance=1.0,
                          lengthscale=1.0,
                          active_dims=[0])

    k_task = GPy.kern.Coregionalize(input_dim=1,
                                    output_dim=num_tasks,
                                    rank=num_tasks,
                                    active_dims=[1])

    m = GPy.models.GPRegression(X_sc, Y, k_time * k_task)
    m.likelihood.variance = 1e-3    

    m.optimize_restarts(num_restarts=restarts,
                        optimizer='lbfgsb',
                        verbose=False)
    return m


def hypervector_from_model(m):
    k_rbf, k_coreg = m.kern.rbf, m.kern.coregion
    sigma2 = float(k_rbf.variance)
    ell    = float(k_rbf.lengthscale)
    W      = k_coreg.W.values                # (T, rank)
    kappa  = k_coreg.kappa.values.flatten()  # (T,)
    return [sigma2, ell] + W.flatten().tolist() + kappa.tolist()


def train_one_patient(rid, X_dict, Y_dict, num_tasks,
                      time_scale=60, restarts=3, min_points=10):
    X, Y = X_dict[rid], Y_dict[rid]
    if len(Y) < min_points:
        return None
    try:
        model = fit_mtgp_one(X, Y, num_tasks, time_scale, restarts)
        if not np.isfinite(model.param_array).all():
            return None
        return rid, hypervector_from_model(model)
    except Exception as e:
        print(f'{rid}: {e}')
        return None


def build_hyperparam_table_parallel(tasks, X_dict, Y_dict,
                                    time_scale=60, restarts=3,
                                    min_points=10, n_jobs=-1):
    T    = len(tasks)
    rank = T                               

    results = Parallel(n_jobs=n_jobs,
                       backend="loky",
                       verbose=5)(
        delayed(train_one_patient)(
            rid, X_dict, Y_dict, T,
            time_scale, restarts, min_points
        ) for rid in X_dict.keys()
    )

    results = [r for r in results if r]    
    if not results:
        raise RuntimeError("Ни один пациент не обучён")

    ids, rows = zip(*results)

    cols = (
        ['sigma2', 'ell'] +
        [f'W_{i}_{j}' for i in range(T) for j in range(rank)] +
        [f'kappa_{i}' for i in range(T)]
    )

    df_hyper = pd.DataFrame(rows, index=ids, columns=cols)
    df_hyper.index.name = 'recordid'
    return df_hyper.reset_index()

if __name__ == "__main__":
    try:
        df_hyper = build_hyperparam_table_parallel(
            tasks=tasks,
            X_dict=X_dict,
            Y_dict=Y_dict,
            time_scale=60,
            restarts=3,
            min_points=10,
            n_jobs=-1       # -1 - все доступные ядра
        )
        df_hyper.to_csv("mtgp_hyperparams.csv", index=False)
        print("Ok")
    except NameError:
        print("Not Ok")


# In[ ]:


outcome_path = r'C:\Users\bogdy\Downloads\Ilya\predicting-mortality-of-icu-patients-the-physionet-computing-in-cardiology-challenge-2012-1.0.0\Outcomes-a.txt'
df_hyper = pd.read_csv('mtgp_hyperparams.csv')


# In[15]:


def build_feature_table(df_static, df_hyper, outcome_path):

    y = (pd.read_csv(outcome_path, sep=',')
           .rename(columns={'RecordID':'recordid',
                            'In-hospital_death':'mortality'})
           [['recordid', 'mortality']])

    icu_ohe = (pd.get_dummies(df_static['ICUType'],
                              prefix='ICUType')
                 .reindex(df_static.index, fill_value=0))

    df_static_ohe = pd.concat(
        [df_static.drop(columns=['ICUType']), icu_ohe], axis=1)

    df_all = (df_static_ohe
              .merge(df_hyper, on='recordid', how='inner')
              .merge(y, on='recordid', how='inner'))

    return df_all

df_all = build_feature_table(df_static, df_hyper, outcome_path)
df_all.to_csv('full_feature_table.csv')


# In[ ]:


def train_mortality_model(df_all,
                          test_size=0.2,
                          random_state=42,
                          n_estimators=500,
                          learning_rate=0.05,
                          num_leaves=31,
                          cv_folds=5,
                          n_jobs=-1):

   #Разделяем X / y
    X = df_all.drop(columns=['recordid', 'mortality'])
    y = df_all['mortality']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    pipe = Pipeline(steps=[
        ('impute', SimpleImputer(strategy='median')),
        ('lgb',    LGBMClassifier(
            n_estimators = n_estimators,
            learning_rate= learning_rate,
            num_leaves   = num_leaves,
            objective    = 'binary',
            random_state = random_state,
            n_jobs       = n_jobs,
            class_weight='balanced'
        ))
    ])

    cv = StratifiedKFold(n_splits=cv_folds,
                         shuffle=True,
                         random_state=random_state)

    cv_auc = cross_val_score(pipe, X_train, y_train,
                             scoring='roc_auc',
                             cv=cv, n_jobs=n_jobs)

    print(f"CV ROC-AUC  ({cv_folds}-fold): "
          f"{cv_auc.mean():.3f} ± {cv_auc.std():.3f}")

    pipe.fit(X_train, y_train)

    y_prob = pipe.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.3).astype(int)

    test_auc = roc_auc_score(y_test, y_prob)
    acc      = accuracy_score(y_test, y_pred)

    print(f"ROC-AUC: {test_auc:.3f}")
    print(f"Accuracy: {acc:.3f}")
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

    return pipe   


def save_model(model, path="mortality_model.pkl"):
    joblib.dump(model, path)


def load_model(path="mortality_model.pkl"):
    return joblib.load(path)


if __name__ == "__main__":
    try:
        model = train_mortality_model(
            df_all,
            test_size   = 0.2,
            random_state= 42,
            n_estimators= 600,
            learning_rate=0.03,
            num_leaves  = 64,
            cv_folds    = 5,
            n_jobs      = -1
        )

        save_model(model, "mortality_model.pkl")

    except NameError:
        print("Error")

