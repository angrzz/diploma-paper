#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import io
import glob
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline


st.set_page_config(page_title="Аналитика пациентов", layout="wide")

PATHS = {
    "Пациент 1": "patient1_data.csv",
    "Пациент 2": "patient2_data.csv",
}
STATIC_PATH  = "static.csv"
DYNAMIC_PATH = "dynamic.csv"

@st.cache_data
def load_data(path):
    df = pd.read_csv(path, skipinitialspace=True)
    df = df[df["parameter"].notna() & (df["parameter"].str.strip() != "")]
    df["time_min"] = pd.to_numeric(df["time_min"], errors="coerce")
    df_piv = df.pivot_table(
        index="time_min",
        columns="parameter",
        values="value",
        aggfunc="mean"
    )
    return df_piv.apply(pd.to_numeric, errors="coerce")

@st.cache_data
def load_static(path):
    return pd.read_csv(path)

@st.cache_data
def load_dynamic(path):
    df = pd.read_csv(path, skipinitialspace=True)
    df = df[df["parameter"].notna() & (df["parameter"].str.strip() != "")]
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    # конвертация Time "HH:MM" → минуты
    df["time_min"] = pd.to_timedelta(df["Time"].str.strip() + ":00").dt.total_seconds() / 60
    df_piv = df.pivot_table(
        index="time_min",
        columns="parameter",
        values="value",
        aggfunc="mean"
    )
    return df_piv.apply(pd.to_numeric, errors="coerce")

data1      = load_data(PATHS["Пациент 1"])
data2      = load_data(PATHS["Пациент 2"])
static_df  = load_static(STATIC_PATH)
dynamic_df = load_dynamic(DYNAMIC_PATH)

st.sidebar.title("Меню")
main = st.sidebar.radio(
    "Выберите раздел",
    ["Общая информация", "Прогностическая модель", "Пациент 1", "Пациент 2"]
)

sub = None
if main in ("Пациент 1", "Пациент 2"):
    st.sidebar.markdown(f"**{main}:**")
    sub = st.sidebar.radio("", ["Статистика", "Визуализация"])

if main == "Общая информация":
    st.title("Общая информация по датасету")

    st.subheader("Статичные данные")
    miss = (static_df.isna().mean() * 100).round(2)
    st.write("Доля пропусков:")
    st.dataframe(miss.rename("missing_%").to_frame(), use_container_width=True)
    st.write("Типы данных:")
    st.dataframe(static_df.dtypes.rename("dtype").to_frame(), use_container_width=True)
    num_cols = ["Age", "Height", "Weight"]
    st.write("Описание:")
    st.dataframe(static_df[num_cols].describe().round(2), use_container_width=True)
    st.write("Квантили:")
    st.dataframe(static_df[num_cols].quantile([0.25, 0.5, 0.75]).round(2), use_container_width=True)

    st.subheader("Динамические данные")
    miss_d = (dynamic_df.isna().mean() * 100).round(2)
    st.write("Доля пропусков:")
    st.dataframe(miss_d.rename("missing_%").to_frame(), use_container_width=True)
    st.write("Типы данных:")
    st.dataframe(dynamic_df.dtypes.rename("dtype").to_frame(), use_container_width=True)
    st.write("Описание (describe):")
    st.dataframe(dynamic_df.describe().T.round(2), use_container_width=True)
    st.write("Квантили:")
    st.dataframe(
        dynamic_df.quantile([0.25, 0.5, 0.75]).T.rename(
            columns={0.25:"25%",0.5:"50%",0.75:"75%"}
        ).round(2),
        use_container_width=True
    )

    st.subheader("Корреляционная матрица")
    corr_dyn = dynamic_df.corr()
    st.dataframe(corr_dyn, use_container_width=True)

    st.write("**Вывод**")
    st.write("Статичные параметры")
    constraints_df = pd.DataFrame(
        [
            {
                "Признак": "Age",
                "Что считается «нормой»": "Число ≤ 130",
                "Что происходит с отклонениями": (
                    "нечисловые -> NaN\n"
                    "> 130 -> 91.4"
                ),
            },
            {
                "Признак": "Gender",
                "Что считается «нормой»": "Коды 0 или 1",
                "Что происходит с отклонениями": "другие коды -> NaN",
            },
            {
                "Признак": "Height",
                "Что считается «нормой»": "0 < рост ≤ 250 см",
                "Что происходит с отклонениями": (
                    "≤ 0 см -> NaN\n"
                    "> 250 см ÷ 10  (мм/дм -> см)"
                ),
            },
            {
                "Признак": "Weight",
                "Что считается «нормой»": "0 < вес ≤ 300 кг",
                "Что происходит с отклонениями": (
                    "≤ 0 кг -> NaN\n"
                    "> 300 кг / 2.20462  (фунты -> кг)"
                ),
            },
            {
                "Признак": "ICUType",
                "Что считается «нормой»": "Коды 1 – 4",
                "Что происходит с отклонениями": "другие коды -> NaN",
            },
        ]
    )
    st.write("Динамические параметры")
    st.dataframe(constraints_df, use_container_width=True)
    def dynamic_constraints_table() -> pd.DataFrame:
        rows = [
            {
                "Признак": "HR",
                "Что считается «нормой»": "20 ≤ HR ≤ 300",
                "Что происходит с отклонениями": "вне диапазона → NaN",
            },
            {
                "Признак": "RespRate",
                "Что считается «нормой»": "2 ≤ RespRate ≤ 80",
                "Что происходит с отклонениями": "вне диапазона → NaN",
            },
            {
                "Признак": "Temp",
                "Что считается «нормой»": "25 ≤ Temp ≤ 45 °C",
                "Что происходит с отклонениями": (
                    "80–120 °F → перевод в °C;\n"
                    "после конвертации / сразу\n"
                    "вне 25–45 °C → NaN"
                ),
            },
            {
                "Признак": "NISysABP / SysABP",
                "Что считается «нормой»": "50 ≤ SysABP ≤ 300",
                "Что происходит с отклонениями": "вне диапазона → NaN",
            },
            {
                "Признак": "NIDiasABP / DiasABP",
                "Что считается «нормой»": "20 ≤ DiasABP ≤ 200",
                "Что происходит с отклонениями": "вне диапазона → NaN",
            },
            {
                "Признак": "NIMAP / MAP",
                "Что считается «нормой»": "30 ≤ MAP ≤ 250",
                "Что происходит с отклонениями": "вне диапазона → NaN",
            },
            {
                "Признак": "Glucose",
                "Что считается «нормой»": "1 ≤ Glucose ≤ 100",
                "Что происходит с отклонениями": "вне диапазона → NaN",
            },
            {
                "Признак": "GCS",
                "Что считается «нормой»": "3 ≤ GCS ≤ 15",
                "Что происходит с отклонениями": "вне диапазона → NaN",
            },
            {
                "Признак": "Urine",
                "Что считается «нормой»": "Urine ≥ 0",
                "Что происходит с отклонениями": "значения < 0 → 0",
            },
            {
                "Признак": "Все параметры",
                "Что считается «нормой»": "значение должно быть числовым",
                "Что происходит с отклонениями": "нечисловые строки → NaN",
            },
        ]
        return pd.DataFrame(rows)
    
    st.dataframe(dynamic_constraints_table(), use_container_width=True)

    st.write(f"Итог: Приводим их к числовому типу и отсекаем/исправляем все значения, выходящие за указанные пределы. Всё, что не вписывается в эти правила, переводится в NaN, а подозрительные единицы измерения корректируются, чтобы обеспечить чистый и консистентный датасет для дальнейшего анализа.")


elif main == "Прогностическая модель":
    st.title("Прогностическая модель")
    st.write("Здесь можно разместить код и результаты вашей модели.")

    FEATURES_PATH = "full_feature_table.csv"
    MODEL_PATH    = "mortality_model.pkl"

    @st.cache_data
    def load_features(path):
        return pd.read_csv(path)

    @st.cache_data
    def load_model(path):
        import joblib, os
        return joblib.load(path) if os.path.exists(path) else None

    df_feat = load_features(FEATURES_PATH)
    model   = load_model(MODEL_PATH)

    st.subheader("📈 Итоговые метрики")
    col1, col2, col3 = st.columns(3)
    col1.metric("CV ROC-AUC (5-fold)",  "0.793", "± 0.023")
    col2.metric("Test ROC-AUC",        "0.758")
    col3.metric("Test Accuracy",       "0.845")

    st.write("Матрица ошибок (порог 0.3):")
    cm_df = pd.DataFrame([[645, 33], [89, 21]],
                         index=["0 — выжил", "1 — умер"],
                         columns=["Предсказано 0", "Предсказано 1"])
    st.dataframe(cm_df, use_container_width=True)


    def hyperparams_explanation() -> pd.DataFrame:
        rows = [
            {"Параметр": "`n_estimators = 600`",      "Значение": "600 деревьев",
             "Пояснение": "CV-AUC перестаёт расти после ≈550 → 600 — плато."},
            {"Параметр": "`learning_rate = 0.03`",    "Значение": "маленький шаг",
             "Пояснение": "много деревьев ⇒ нужен малый шаг; c 0.1 было переобучение."},
            {"Параметр": "`num_leaves = 64`",         "Значение": "глубина ≈ 6",
             "Пояснение": "32 — недообучение, 128 — переобучение; 64 — середина."},
            {"Параметр": "`test_size = 0.2`",         "Значение": "20 % в тест",
             "Пояснение": "≈ 600 «невидимых» пациентов, train при этом не худеет."},
            {"Параметр": "`cv_folds = 5`",            "Значение": "5-fold CV",
             "Пояснение": "баланс: ±0.02 AUC и ≈ 3 мин обучения."},
            {"Параметр": "`class_weight='balanced'`", "Значение": "автовеса классов",
             "Пояснение": "компенсирует дисбаланс: смертей всего 15 %."},
            {"Параметр": "`random_state = 42`",       "Значение": "фиксируем случайность",
             "Пояснение": "полная воспроизводимость эксперимента."},
            {"Параметр": "`n_jobs = -1`",             "Значение": "все ядра",
             "Пояснение": "ускоряем обучение ×2–3 за счёт многопоточности."},
        ]
        return pd.DataFrame(rows)
    
    st.subheader("Параметры при обучении")
    st.table(hyperparams_explanation())


    with st.expander("Как это работает (кликните)"):
        st.markdown("""
**Очистка → MTGP → LightGBM**

* ***Шаг 1.*** Чистим статические и динамические данные.  
* ***Шаг 2.*** Для сигналов **Multi-Task GP** восстанавливает
  пропуски и выдаёт гиперпараметры (σ², ℓ, W, κ).  
* ***Шаг 3.*** Склеиваем статистические + MTGP-параметры, где получаем фича-вектор.  
* ***Шаг 4.*** **LightGBM** (600 деревьев, class_weight = balanced)  
  обучается отделять умерших от выживших.  

| Метрика | Значение |
|---------|----------|
| 5-fold **ROC-AUC** | **0.793 ± 0.023** |
| Test **ROC-AUC**   | **0.758** |
| Test Accuracy      | **0.845** |
| Ложно-положит.     | 33 / 678 |
| Ложно-отрицат.     | 89 / 110 |

Топ-факторы риска — высокая волатильность сигналов, сильная связь HR ↔ RR,
низкий GCS и возраст.
        """)
else:
    data = data1 if main == "Пациент 1" else data2

    if sub == "Статистика":
        st.title(f"🧮 {main} — Статистика")
        desc = data.describe().T
        st.dataframe(desc, use_container_width=True)
        st.subheader("🔍 Интерпретация")
        for param, row in desc.iterrows():
            st.write(f"- **{param}**: mean={row['mean']:.2f}, σ={row['std']:.2f}, min={row['min']:.2f}, max={row['max']:.2f}")
        st.subheader("🔗 Корреляционная матрица")
        corr = data.corr()
        st.dataframe(corr, use_container_width=True)
        st.subheader("🏅 Топ-5 корреляций")
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        top = corr.where(mask).unstack().dropna().sort_values(ascending=False).head(5)
        st.write(top.to_frame("Correlation").reset_index().rename(columns={"level_0":"Param1","level_1":"Param2"}))

    elif sub == "Визуализация":
        st.title(f"📈 {main} — Визуализация")
        params = list(data.columns)
        sel = st.multiselect("Что показать", params, default=params)
        for p in sel:
            fig, ax = plt.subplots(figsize=(6,3))
            ax.plot(data.index, data[p], marker="o", linestyle="-", alpha=0.8)
            ax.set_title(p); ax.set_xlabel("Время (мин)"); ax.set_ylabel(p); ax.grid(True)
            st.pyplot(fig)

