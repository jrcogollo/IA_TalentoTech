import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Título principal
st.title("Exploración Interactiva de Datos con Streamlit")

# --- Barra lateral para controles ---
st.sidebar.header("Configuración de Datos")

# Selección del tipo de dataset
dataset = st.sidebar.selectbox(
    "Selecciona el tipo de datos:",
    ["Datos Aleatorios", "Senoidal", "Cosenoidal"]
)

# Slider para elegir el número de muestras
n_samples = st.sidebar.slider("Número de muestras:", 50, 500, 200)

# Opción para elegir el tipo de gráfico principal
chart_choice = st.sidebar.radio("Elige el tipo de gráfico:", ["Línea", "Barras"])

# --- Generación de datos según selección ---
if dataset == "Datos Aleatorios":
    data = np.random.randn(n_samples, 2)
    df = pd.DataFrame(data, columns=["X", "Y"])
elif dataset == "Senoidal":
    x = np.linspace(0, 10, n_samples)
    y = np.sin(x)
    df = pd.DataFrame({"X": x, "Y": y})
else:  # Cosenoidal
    x = np.linspace(0, 10, n_samples)
    y = np.cos(x)
    df = pd.DataFrame({"X": x, "Y": y})

# --- Visualización de la tabla de datos ---
st.subheader("Datos Generados")
st.dataframe(df)

# --- Visualización del gráfico principal ---
st.subheader("Visualización del Gráfico Principal")
if chart_choice == "Línea":
    st.line_chart(df.set_index("X"))
else:
    st.bar_chart(df.set_index("X"))

# --- Estadísticas descriptivas con checkbox ---
if st.checkbox("Mostrar estadísticas descriptivas"):
    st.write(df.describe())

# --- Histograma interactivo ---
st.subheader("Histograma de la columna Y")
bins = st.slider("Número de bins para el histograma:", min_value=5, max_value=50, value=20)
hist_values, bin_edges = np.histogram(df["Y"], bins=bins)
# Preparar DataFrame para el gráfico de barras del histograma
hist_df = pd.DataFrame({
    "Cuenta": hist_values,
    "Bordes": bin_edges[:-1]
})
st.bar_chart(hist_df.set_index("Bordes"))

# --- Gráfico de dispersión (Scatter Plot) usando Matplotlib ---
st.subheader("Gráfico de Dispersión (Scatter Plot)")
fig, ax = plt.subplots()
ax.scatter(df["X"], df["Y"], alpha=0.6)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_title("Dispersión de X vs Y")
st.pyplot(fig)
