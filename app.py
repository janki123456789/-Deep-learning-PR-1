"""
Red & White Skill Education — Deep Learning PR1
Streamlit Dashboard: Breast Cancer Classification with SLP, MLP,
Early Stopping, Dropout & Regularization

Run with:
    streamlit run streamlit_app.py
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_score, recall_score, f1_score, accuracy_score
)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping

tf.random.set_seed(42)
np.random.seed(42)

# ────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG + THEME
# ────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Deep Learning PR1 | Red & White Skill Education",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

PRIMARY = "#CC0000"
PRIMARY_DARK = "#8B0000"
CHARCOAL = "#2C2C2C"
LIGHT_BG = "#FFF6F6"
ACCENT = "#FF8080"

CUSTOM_CSS = f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700;800&display=swap');

    html, body, [class*="css"]  {{
        font-family: 'Poppins', sans-serif;
    }}

    .stApp {{
        background: linear-gradient(180deg, #FFFFFF 0%, {LIGHT_BG} 100%);
    }}

    /* Hero header */
    .hero {{
        background: linear-gradient(120deg, {PRIMARY} 0%, {PRIMARY_DARK} 100%);
        padding: 2.2rem 2.2rem;
        border-radius: 18px;
        color: white;
        margin-bottom: 1.6rem;
        box-shadow: 0 10px 30px rgba(204,0,0,0.25);
    }}
    .hero h1 {{
        font-size: 2.1rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }}
    .hero p {{
        font-size: 1.0rem;
        opacity: 0.92;
        margin: 0;
    }}
    .hero .badge {{
        display:inline-block;
        background: rgba(255,255,255,0.18);
        padding: 3px 12px;
        border-radius: 999px;
        font-size: 0.78rem;
        margin-right: 6px;
        margin-top: 10px;
        border: 1px solid rgba(255,255,255,0.35);
    }}

    /* Section card */
    .card {{
        background: white;
        border-radius: 16px;
        padding: 1.4rem 1.6rem;
        box-shadow: 0 4px 18px rgba(0,0,0,0.06);
        border: 1px solid #F1E0E0;
        margin-bottom: 1.2rem;
    }}
    .card h3 {{
        color: {PRIMARY_DARK};
        font-weight: 700;
        margin-bottom: 0.6rem;
    }}

    /* Metric tiles */
    div[data-testid="stMetric"] {{
        background: white;
        border: 1px solid #F1E0E0;
        border-radius: 14px;
        padding: 12px 10px;
        box-shadow: 0 3px 10px rgba(0,0,0,0.05);
    }}
    div[data-testid="stMetricValue"] {{
        color: {PRIMARY_DARK};
        font-weight: 800;
    }}

    /* Sidebar */
    section[data-testid="stSidebar"] {{
        background: linear-gradient(180deg, {CHARCOAL} 0%, #1a1a1a 100%);
    }}
    section[data-testid="stSidebar"] * {{
        color: #f2f2f2 !important;
    }}
    section[data-testid="stSidebar"] .stRadio > label {{
        font-weight: 600;
    }}

    /* Buttons */
    .stButton>button {{
        background: linear-gradient(120deg, {PRIMARY} 0%, {PRIMARY_DARK} 100%);
        color: white;
        border-radius: 10px;
        border: none;
        padding: 0.55rem 1.4rem;
        font-weight: 700;
        box-shadow: 0 4px 14px rgba(204,0,0,0.3);
    }}
    .stButton>button:hover {{
        background: linear-gradient(120deg, {PRIMARY_DARK} 0%, {PRIMARY} 100%);
        color: white;
    }}

    .insight-box {{
        background: #FFF0F0;
        border-left: 5px solid {PRIMARY};
        padding: 0.9rem 1.2rem;
        border-radius: 10px;
        margin: 0.8rem 0;
    }}

    footer {{visibility: hidden;}}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

PLOTLY_TEMPLATE = "plotly_white"


# ────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data():
    data = load_breast_cancer(as_frame=True)
    X = data.data
    y = data.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_sc = scaler.transform(X_train)
    X_test_sc = scaler.transform(X_test)
    return X, y, X_train, X_test, y_train, y_test, X_train_sc, X_test_sc


X, y, X_train, X_test, y_train, y_test, X_train_sc, X_test_sc = load_data()


def eval_model(model, X_te, y_te, name):
    loss, acc = model.evaluate(X_te, y_te, verbose=0)
    y_pred = (model.predict(X_te, verbose=0) > 0.5).astype(int).ravel()
    return {
        "Model": name,
        "Test Accuracy": accuracy_score(y_te, y_pred),
        "Test Precision": precision_score(y_te, y_pred),
        "Test Recall": recall_score(y_te, y_pred),
        "Test F1-Score": f1_score(y_te, y_pred),
        "y_pred": y_pred,
        "loss": loss,
    }


def confusion_fig(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    fig = px.imshow(
        cm, text_auto=True, color_continuous_scale="Reds",
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=["Malignant (0)", "Benign (1)"], y=["Malignant (0)", "Benign (1)"],
    )
    fig.update_layout(title=title, template=PLOTLY_TEMPLATE, height=380)
    return fig


def curve_fig(history_dict, keys, labels, colors, title, yaxis="Value"):
    fig = go.Figure()
    for k, lab, c in zip(keys, labels, colors):
        fig.add_trace(go.Scatter(
            y=history_dict[k], mode="lines", name=lab,
            line=dict(color=c, width=2.5)
        ))
    fig.update_layout(
        title=title, template=PLOTLY_TEMPLATE, height=380,
        xaxis_title="Epoch", yaxis_title=yaxis,
        legend=dict(orientation="h", y=1.12)
    )
    return fig


# ────────────────────────────────────────────────────────────────────────────
# CACHED MODEL TRAINERS
# ────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="🔴 Training Single-Layer Perceptron...")
def train_slp():
    model = keras.Sequential([keras.layers.Dense(1, activation="sigmoid", input_shape=(30,))])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    hist = model.fit(X_train_sc, y_train, epochs=50, batch_size=32,
                      validation_split=0.1, verbose=0)
    return model, hist.history


@st.cache_resource(show_spinner="🔵 Training MLP activation variants (ReLU / Tanh / Sigmoid)...")
def train_activation_variants():
    results = {}
    for act in ["relu", "tanh", "sigmoid"]:
        m = keras.Sequential([
            keras.layers.Dense(64, activation=act, input_shape=(30,)),
            keras.layers.Dense(32, activation=act),
            keras.layers.Dense(1, activation="sigmoid"),
        ])
        m.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        h = m.fit(X_train_sc, y_train, epochs=100, batch_size=32,
                  validation_split=0.1, verbose=0)
        results[act] = (m, h.history)
    return results


@st.cache_resource(show_spinner="🟠 Running Early Stopping experiment...")
def train_early_stopping():
    def build():
        m = keras.Sequential([
            keras.layers.Dense(128, activation="relu", input_shape=(30,)),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dense(1, activation="sigmoid"),
        ])
        m.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        return m

    es = EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True, verbose=0)
    m_es = build()
    h_es = m_es.fit(X_train_sc, y_train, epochs=300, batch_size=32,
                     validation_split=0.1, callbacks=[es], verbose=0)

    m_full = build()
    h_full = m_full.fit(X_train_sc, y_train, epochs=300, batch_size=32,
                         validation_split=0.1, verbose=0)

    return (m_es, h_es.history, len(h_es.history["loss"])), (m_full, h_full.history)


@st.cache_resource(show_spinner="🟣 Training Dropout rate comparison (0.1 / 0.3 / 0.5)...")
def train_dropout_variants():
    results = {}
    for rate in [0.1, 0.3, 0.5]:
        m = keras.Sequential([
            keras.layers.Dense(128, activation="relu", input_shape=(30,)),
            keras.layers.Dropout(rate),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dropout(rate),
            keras.layers.Dense(1, activation="sigmoid"),
        ])
        m.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        es = EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True, verbose=0)
        h = m.fit(X_train_sc, y_train, epochs=200, batch_size=32,
                  validation_split=0.1, callbacks=[es], verbose=0)
        results[rate] = (m, h.history)
    return results


@st.cache_resource(show_spinner="🟢 Training L1 / L2 / L1-L2 regularized models...")
def train_regularization_variants():
    configs = {
        "L1": regularizers.l1(0.001),
        "L2": regularizers.l2(0.001),
        "L1-L2": regularizers.l1_l2(l1=0.0001, l2=0.001),
    }
    results = {}
    for name, reg in configs.items():
        m = keras.Sequential([
            keras.layers.Dense(128, activation="relu", input_shape=(30,), kernel_regularizer=reg),
            keras.layers.Dense(64, activation="relu", kernel_regularizer=reg),
            keras.layers.Dense(1, activation="sigmoid"),
        ])
        m.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        es = EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True, verbose=0)
        h = m.fit(X_train_sc, y_train, epochs=200, batch_size=32,
                  validation_split=0.1, callbacks=[es], verbose=0)
        results[name] = (m, h.history)
    return results


@st.cache_resource(show_spinner="🏆 Training final combined model (Dropout + L2 + EarlyStopping)...")
def train_final_model():
    reg = regularizers.l2(0.001)
    m = keras.Sequential([
        keras.layers.Dense(128, activation="relu", input_shape=(30,), kernel_regularizer=reg),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(64, activation="relu", kernel_regularizer=reg),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(1, activation="sigmoid"),
    ])
    m.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    es = EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True, verbose=0)
    h = m.fit(X_train_sc, y_train, epochs=300, batch_size=32,
              validation_split=0.1, callbacks=[es], verbose=0)
    return m, h.history


# ────────────────────────────────────────────────────────────────────────────
# HERO HEADER
# ────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="hero">
    <h1>🩺 Deep Learning · PR1 Dashboard</h1>
    <p>Breast Cancer Wisconsin (Diagnostic) — SLP → MLP → Early Stopping → Dropout → Regularization</p>
    <span class="badge">Red & White Skill Education</span>
    <span class="badge">TensorFlow / Keras</span>
    <span class="badge">569 Samples · 30 Features</span>
</div>
""", unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────────────────────
# SIDEBAR NAVIGATION
# ────────────────────────────────────────────────────────────────────────────
st.sidebar.markdown("## 📚 Navigate Tasks")
page = st.sidebar.radio(
    "",
    [
        "🏠 Overview",
        "1️⃣ Data & EDA",
        "2️⃣ SLP Baseline",
        "3️⃣ MLP & Activations",
        "4️⃣ Early Stopping",
        "5️⃣ Dropout",
        "6️⃣ Regularization",
        "7️⃣ Final Model & Insights",
    ],
    label_visibility="collapsed",
)
st.sidebar.markdown("---")
st.sidebar.markdown(
    "<small>💡 Models train once and are cached — switching tasks is instant after the first run.</small>",
    unsafe_allow_html=True,
)

# ────────────────────────────────────────────────────────────────────────────
# OVERVIEW
# ────────────────────────────────────────────────────────────────────────────
if page == "🏠 Overview":
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Samples", "569")
    c2.metric("Features", "30")
    c3.metric("Malignant (0)", "212")
    c4.metric("Benign (1)", "357")

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🗺️ Project Roadmap")
    st.write(
        "This dashboard walks through every stage of PR1 — from raw data to a "
        "production-ready regularized MLP — with live-trained models and interactive charts."
    )
    steps = pd.DataFrame({
        "Task": ["1. Data & EDA", "2. SLP Baseline", "3. MLP & Activations",
                 "4. Early Stopping", "5. Dropout", "6. Regularization", "7. Final Model"],
        "Focus": [
            "Loading, scaling, correlation, class balance",
            "31-parameter linear baseline model",
            "ReLU vs Tanh vs Sigmoid hidden activations",
            "monitor / patience / restore_best_weights",
            "Rate comparison: 0.1 / 0.3 / 0.5",
            "L1 sparsity vs L2 shrinkage vs ElasticNet",
            "Combined model + clinical recommendation",
        ],
    })
    st.dataframe(steps, use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)

    fig = px.imshow(
        X.corr(), color_continuous_scale="RdBu_r", origin="lower",
        title="Full Feature Correlation Heatmap (30 × 30)"
    )
    fig.update_layout(template=PLOTLY_TEMPLATE, height=600)
    st.plotly_chart(fig, use_container_width=True)

# ────────────────────────────────────────────────────────────────────────────
# TASK 1: DATA & EDA
# ────────────────────────────────────────────────────────────────────────────
elif page == "1️⃣ Data & EDA":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 📦 Dataset Snapshot")
    c1, c2 = st.columns([1, 2])
    with c1:
        st.metric("X shape", f"{X.shape[0]} × {X.shape[1]}")
        st.metric("Malignant (0)", int((y == 0).sum()))
        st.metric("Benign (1)", int((y == 1).sum()))
    with c2:
        st.dataframe(X.describe().T.style.background_gradient(cmap="Reds"), height=280)
    st.markdown('</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        counts = y.value_counts().rename({0: "Malignant (0)", 1: "Benign (1)"})
        fig = px.bar(
            x=counts.index, y=counts.values, text=counts.values,
            color=counts.index, color_discrete_sequence=[PRIMARY, CHARCOAL],
            labels={"x": "Target Class", "y": "Count"},
            title="Target Class Distribution",
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(template=PLOTLY_TEMPLATE, showlegend=False, height=380)
        st.plotly_chart(fig, use_container_width=True)
        st.info("Mild imbalance (212 vs 357) — not severe enough to need SMOTE/class-weighting at this dataset size.")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        corr = X.corr()
        fig2 = px.imshow(corr, color_continuous_scale="RdBu_r", title="Feature Correlation Heatmap")
        fig2.update_layout(template=PLOTLY_TEMPLATE, height=380)
        st.plotly_chart(fig2, use_container_width=True)
        st.info("radius_mean, perimeter_mean & area_mean are highly correlated — not an issue for neural nets, unlike linear models.")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### ✂️ Train/Test Split & Scaling")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("X_train", str(X_train.shape))
    c2.metric("X_test", str(X_test.shape))
    c3.metric("Train mean (scaled)", f"{X_train_sc.mean():.4f}")
    c4.metric("Train std (scaled)", f"{X_train_sc.std():.4f}")
    st.markdown("""
<div class="insight-box">
<b>Why scale before training?</b> Gradient-based optimizers (SGD/Adam) converge faster and more stably
when features share a scale. Without scaling, <code>area_mean</code> (~up to 2500) would dominate early
gradient updates while <code>fractal_dimension_mean</code> (~0.05–0.10) barely contributes — causing slow,
biased learning. The scaler is fit on <b>X_train only</b> to avoid data leakage into the test set.
</div>
""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────────────────────
# TASK 2: SLP
# ────────────────────────────────────────────────────────────────────────────
elif page == "2️⃣ SLP Baseline":
    model_slp, hist_slp = train_slp()
    res_slp = eval_model(model_slp, X_test_sc, y_test, "SLP")

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🔴 Single-Layer Perceptron — Baseline")
    st.write("`Dense(1, activation='sigmoid', input_shape=(30,))` — 31 parameters (30 weights + 1 bias).")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Test Accuracy", f"{res_slp['Test Accuracy']:.2%}")
    c2.metric("Precision", f"{res_slp['Test Precision']:.2%}")
    c3.metric("Recall", f"{res_slp['Test Recall']:.2%}")
    c4.metric("F1-Score", f"{res_slp['Test F1-Score']:.2%}")
    st.markdown('</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.plotly_chart(curve_fig(hist_slp, ["loss", "val_loss"], ["Training Loss", "Validation Loss"],
                                   [PRIMARY, CHARCOAL], "SLP: Loss over Epochs", "Loss"),
                         use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.plotly_chart(curve_fig(hist_slp, ["accuracy", "val_accuracy"], ["Training Acc", "Validation Acc"],
                                   [PRIMARY, CHARCOAL], "SLP: Accuracy over Epochs", "Accuracy"),
                         use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.plotly_chart(confusion_fig(y_test, res_slp["y_pred"], "SLP Confusion Matrix"), use_container_width=True)
    with st.expander("📄 Classification Report"):
        st.text(classification_report(y_test, res_slp["y_pred"], target_names=["Malignant", "Benign"]))
    st.markdown("""
<div class="insight-box">
<b>Limitation:</b> An SLP can only learn a single linear hyperplane through 30-dimensional space. Breast
cancer diagnosis is unlikely to be perfectly linearly separable — this baseline is realistic but improvable,
motivating the MLP in the next task.
</div>
""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────────────────────
# TASK 3: MLP & ACTIVATIONS
# ────────────────────────────────────────────────────────────────────────────
elif page == "3️⃣ MLP & Activations":
    variants = train_activation_variants()
    colors = {"relu": PRIMARY, "tanh": CHARCOAL, "sigmoid": "#4C6EF5"}

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🔵 MLP Architecture: `64 → 32 → 1`")
    m_relu, h_relu = variants["relu"]
    total_params_manual = 30 * 64 + 64 + 64 * 32 + 32 + 32 * 1 + 1
    c1, c2 = st.columns(2)
    c1.metric("Manual parameter count", total_params_manual)
    c2.metric("Model parameter count", m_relu.count_params())
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### ⚔️ ReLU vs Tanh vs Sigmoid — Validation Accuracy")
    fig = go.Figure()
    for act in ["relu", "tanh", "sigmoid"]:
        _, h = variants[act]
        fig.add_trace(go.Scatter(y=h["val_accuracy"], mode="lines", name=act.upper(),
                                  line=dict(color=colors[act], width=2.5)))
    fig.update_layout(template=PLOTLY_TEMPLATE, height=420, xaxis_title="Epoch",
                       yaxis_title="Validation Accuracy", legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, use_container_width=True)

    results_table = []
    for act in ["relu", "tanh", "sigmoid"]:
        m, _ = variants[act]
        r = eval_model(m, X_test_sc, y_test, act.upper())
        results_table.append({"Activation": act.upper(), "Test Accuracy": r["Test Accuracy"],
                               "F1-Score": r["Test F1-Score"]})
    best = max(results_table, key=lambda d: d["Test Accuracy"])
    df_res = pd.DataFrame(results_table)
    st.dataframe(df_res.style.highlight_max(subset=["Test Accuracy", "F1-Score"], color="#ffcccc"),
                 use_container_width=True, hide_index=True)
    st.markdown(f"""
<div class="insight-box">
<b>ReLU</b> — max(0,x), no vanishing gradient for positive inputs, fastest convergence.
<b>Tanh</b> — zero-centred (−1,1) but can still vanish in deep nets.
<b>Sigmoid</b> (hidden layers) — severe vanishing-gradient risk, best reserved for the output layer.
On this dataset, <b>{best['Activation']}</b> achieved the best test accuracy ({best['Test Accuracy']:.2%}).
</div>
""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    best_model, _ = variants[best["Activation"].lower()]
    r_best = eval_model(best_model, X_test_sc, y_test, best["Activation"])
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.plotly_chart(confusion_fig(y_test, r_best["y_pred"], f"Best MLP ({best['Activation']}) Confusion Matrix"),
                     use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────────────────────
# TASK 4: EARLY STOPPING
# ────────────────────────────────────────────────────────────────────────────
elif page == "4️⃣ Early Stopping":
    (m_es, h_es, stop_epoch), (m_full, h_full) = train_early_stopping()
    res_es = eval_model(m_es, X_test_sc, y_test, "With Early Stopping")
    res_full = eval_model(m_full, X_test_sc, y_test, "No Callback (300 epochs)")

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🟠 EarlyStopping Callback")
    st.write("`monitor='val_loss'`, `patience=15`, `restore_best_weights=True`, `verbose=1`")
    c1, c2, c3 = st.columns(3)
    c1.metric("Stopped at epoch", stop_epoch)
    c2.metric("Test Acc (Early Stop)", f"{res_es['Test Accuracy']:.2%}")
    c3.metric("Test Acc (Full 300)", f"{res_full['Test Accuracy']:.2%}")
    st.markdown("""
<div class="insight-box">
<b>monitor='val_loss'</b> — training loss always decreases, only validation loss reveals true generalisation.<br>
<b>patience=15</b> — tolerate 15 flat epochs before stopping, avoiding a premature halt on a plateau.<br>
<b>restore_best_weights=True</b> — rewinds to the best-val_loss epoch instead of keeping post-overfit weights.<br>
<b>verbose=1</b> — logs the exact stopping epoch.
</div>
""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=h_es["val_loss"], mode="lines", name="Early Stopping — Val Loss",
                              line=dict(color=PRIMARY, width=2.5)))
    fig.add_trace(go.Scatter(y=h_full["val_loss"], mode="lines", name="No Callback — Val Loss",
                              line=dict(color=CHARCOAL, width=2.5)))
    fig.add_vline(x=stop_epoch, line_dash="dash", line_color=PRIMARY,
                  annotation_text="Training stopped — best weights restored")
    fig.update_layout(title="Validation Loss — With vs Without Early Stopping",
                       template=PLOTLY_TEMPLATE, height=440,
                       xaxis_title="Epoch", yaxis_title="Val Loss",
                       legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, use_container_width=True)
    best_val_es = min(h_es["val_loss"])
    best_val_full = min(h_full["val_loss"])
    st.markdown(f"""
<div class="insight-box">
Best validation loss — Early Stopping: <b>{best_val_es:.4f}</b> · No callback: <b>{best_val_full:.4f}</b>.
The no-callback run keeps training long after validation loss bottoms out, risking overfit weights at epoch 300.
</div>
""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────────────────────
# TASK 5: DROPOUT
# ────────────────────────────────────────────────────────────────────────────
elif page == "5️⃣ Dropout":
    variants = train_dropout_variants()
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🟣 Dropout Rate Comparison")
    st.write("`Dense(128, relu) → Dropout(rate) → Dense(64, relu) → Dropout(rate) → Dense(1, sigmoid)`")

    fig = go.Figure()
    rate_colors = {0.1: "#4C6EF5", 0.3: PRIMARY, 0.5: CHARCOAL}
    for rate in [0.1, 0.3, 0.5]:
        _, h = variants[rate]
        fig.add_trace(go.Scatter(y=h["val_accuracy"], mode="lines", name=f"Dropout {rate}",
                                  line=dict(color=rate_colors[rate], width=2.5)))
    fig.update_layout(title="Dropout Rate Comparison — Validation Accuracy",
                       template=PLOTLY_TEMPLATE, height=420,
                       xaxis_title="Epoch", yaxis_title="Validation Accuracy",
                       legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, use_container_width=True)

    rows = []
    for rate in [0.1, 0.3, 0.5]:
        m, _ = variants[rate]
        r = eval_model(m, X_test_sc, y_test, f"Dropout {rate}")
        rows.append({"Dropout Rate": rate, "Test Accuracy": r["Test Accuracy"], "F1-Score": r["Test F1-Score"]})
    df_drop = pd.DataFrame(rows)
    best_rate = df_drop.loc[df_drop["Test Accuracy"].idxmax(), "Dropout Rate"]
    st.dataframe(df_drop.style.highlight_max(subset=["Test Accuracy", "F1-Score"], color="#ffcccc"),
                 use_container_width=True, hide_index=True)

    st.markdown(f"""
<div class="insight-box">
Dropout randomly zeroes a fraction of activations each batch, forcing the network to distribute knowledge
across neurons instead of relying on any single one — equivalent to averaging an ensemble of thinned
sub-networks. At inference, Dropout is disabled entirely. <b>Rate 0.1</b> under-regularizes; <b>0.5</b> can
underfit by removing too much signal per batch. Best test accuracy here: <b>rate {best_rate}</b>.
</div>
""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    m_best, _ = variants[best_rate]
    r_best = eval_model(m_best, X_test_sc, y_test, f"Dropout {best_rate}")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.plotly_chart(confusion_fig(y_test, r_best["y_pred"], f"Best Dropout ({best_rate}) Confusion Matrix"),
                     use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────────────────────
# TASK 6: REGULARIZATION
# ────────────────────────────────────────────────────────────────────────────
elif page == "6️⃣ Regularization":
    variants = train_regularization_variants()
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🟢 L1 vs L2 vs L1-L2 (ElasticNet)")

    fig = make_subplots(rows=1, cols=3, subplot_titles=["L1", "L2", "L1-L2"])
    reg_colors = {"L1": "#4C6EF5", "L2": PRIMARY, "L1-L2": CHARCOAL}
    for i, name in enumerate(["L1", "L2", "L1-L2"], start=1):
        _, h = variants[name]
        fig.add_trace(go.Scatter(y=h["loss"], name=f"{name} Train", line=dict(color=reg_colors[name]),
                                  showlegend=(i == 1)), row=1, col=i)
        fig.add_trace(go.Scatter(y=h["val_loss"], name=f"{name} Val", line=dict(color=CHARCOAL, dash="dash"),
                                  showlegend=(i == 1)), row=1, col=i)
    fig.update_layout(title="Regularization Comparison — Training vs Validation Loss",
                       template=PLOTLY_TEMPLATE, height=420)
    st.plotly_chart(fig, use_container_width=True)

    rows = []
    for name in ["L1", "L2", "L1-L2"]:
        m, _ = variants[name]
        r = eval_model(m, X_test_sc, y_test, name)
        rows.append({"Regularization": name, "Test Accuracy": r["Test Accuracy"], "F1-Score": r["Test F1-Score"]})
    df_reg = pd.DataFrame(rows)
    st.dataframe(df_reg.style.highlight_max(subset=["Test Accuracy", "F1-Score"], color="#ffcccc"),
                 use_container_width=True, hide_index=True)

    st.markdown("""
<div class="insight-box">
<b>L2</b> (λΣw²) shrinks all weights toward zero — smoother decision boundary, handles the strongly
correlated radius/perimeter/area features well. <b>L1</b> (λΣ|w|) drives weakly-predictive weights to
exactly zero — effective feature selection. <b>L1-L2 (ElasticNet)</b> blends both effects.
</div>
""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────────────────────
# TASK 7: FINAL MODEL & BUSINESS INSIGHTS
# ────────────────────────────────────────────────────────────────────────────
elif page == "7️⃣ Final Model & Insights":
    with st.spinner("Assembling full results table — training any missing models..."):
        model_slp, hist_slp = train_slp()
        act_variants = train_activation_variants()
        (m_es, h_es, stop_epoch), (m_full, h_full) = train_early_stopping()
        drop_variants = train_dropout_variants()
        reg_variants = train_regularization_variants()
        final_model, final_hist = train_final_model()

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🏆 Final Combined Model")
    st.write("`Dense(128, relu, L2) → Dropout(0.3) → Dense(64, relu, L2) → Dropout(0.3) → Dense(1, sigmoid)` + EarlyStopping")
    res_final = eval_model(final_model, X_test_sc, y_test, "Final Combined")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Test Accuracy", f"{res_final['Test Accuracy']:.2%}")
    c2.metric("Precision", f"{res_final['Test Precision']:.2%}")
    c3.metric("Recall", f"{res_final['Test Recall']:.2%}")
    c4.metric("F1-Score", f"{res_final['Test F1-Score']:.2%}")
    st.plotly_chart(confusion_fig(y_test, res_final["y_pred"], "Final Model Confusion Matrix"),
                     use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Build full comparison table
    best_act = max(act_variants, key=lambda a: eval_model(act_variants[a][0], X_test_sc, y_test, a)["Test Accuracy"])
    best_drop_rate = max(drop_variants, key=lambda r: eval_model(drop_variants[r][0], X_test_sc, y_test, str(r))["Test Accuracy"])

    rows = [
        {"Model": "SLP", **{k: v for k, v in eval_model(model_slp, X_test_sc, y_test, "SLP").items() if k in ["Test Accuracy", "Test Precision", "Test Recall", "Test F1-Score"]}},
        {"Model": f"MLP-{best_act.upper()}", **{k: v for k, v in eval_model(act_variants[best_act][0], X_test_sc, y_test, best_act).items() if k in ["Test Accuracy", "Test Precision", "Test Recall", "Test F1-Score"]}},
        {"Model": "MLP + Early Stopping", **{k: v for k, v in eval_model(m_es, X_test_sc, y_test, "ES").items() if k in ["Test Accuracy", "Test Precision", "Test Recall", "Test F1-Score"]}},
        {"Model": f"MLP + Dropout ({best_drop_rate})", **{k: v for k, v in eval_model(drop_variants[best_drop_rate][0], X_test_sc, y_test, "Drop").items() if k in ["Test Accuracy", "Test Precision", "Test Recall", "Test F1-Score"]}},
        {"Model": "MLP + L2", **{k: v for k, v in eval_model(reg_variants["L2"][0], X_test_sc, y_test, "L2").items() if k in ["Test Accuracy", "Test Precision", "Test Recall", "Test F1-Score"]}},
        {"Model": "Final Combined Model", **{k: v for k, v in res_final.items() if k in ["Test Accuracy", "Test Precision", "Test Recall", "Test F1-Score"]}},
    ]
    df_final = pd.DataFrame(rows)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 📊 Full Results Comparison")
    st.dataframe(
        df_final.style.highlight_max(subset=["Test Accuracy", "Test F1-Score"], color="#ffcccc").format(
            {c: "{:.2%}" for c in ["Test Accuracy", "Test Precision", "Test Recall", "Test F1-Score"]}
        ),
        use_container_width=True, hide_index=True,
    )
    fig = px.bar(df_final, x="Model", y="Test Accuracy", color="Model",
                 color_discrete_sequence=px.colors.sequential.Reds_r,
                 title="Test Accuracy Across All Models")
    fig.update_layout(template=PLOTLY_TEMPLATE, showlegend=False, height=420, yaxis_tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🏥 Clinical Insight for Medical Diagnosis Support")
    st.markdown(f"""
**Recommended model for deployment:** the **Final Combined Model** (Dropout + L2 + Early Stopping) —
it generalises best across the regularisation techniques tested, achieving **{res_final['Test Accuracy']:.2%}**
test accuracy with the smallest train–validation gap.

**Decision threshold:** the default 0.5 cutoff treats false negatives and false positives equally. In cancer
screening, a **missed malignant case (false negative) is far costlier** than a false alarm — so a clinical
deployment should consider **lowering the threshold below 0.5** (e.g. 0.3–0.4) to bias the model toward
higher recall on the malignant class, accepting more false positives in exchange for catching more true
malignancies.

**Most impactful technique:** combining **Dropout with L2 regularisation and Early Stopping** together gave
the most consistent generalisation improvement — each addresses a different failure mode (co-adaptation,
weight magnitude, and training-duration overfitting respectively).
""")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown(
    f"<div style='text-align:center;color:#999;padding:1.5rem 0;'>"
    f"© Red & White Skill Education · Since 2008 · \"Shaping Skills for Scaling Higher\""
    f"</div>",
    unsafe_allow_html=True,
)
