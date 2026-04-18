import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(layout="wide")
st.title("Painel de Controle Quântico Interativo")

# Inicialização dos valores no session_state
if "theta_graus" not in st.session_state:
    st.session_state["theta_graus"] = 90
if "phi_graus" not in st.session_state:
    st.session_state["phi_graus"] = 0

# --- CONTROLES ---
st.sidebar.header("Controles do Estado Quântico")

# Funções para portas quânticas
def porta_h():
    st.session_state["theta_graus"] = 90
    st.session_state["phi_graus"] = 0

def porta_x():
    st.session_state["theta_graus"] = 180
    st.session_state["phi_graus"] = 0

def porta_y():
    st.session_state["theta_graus"] = 180
    st.session_state["phi_graus"] = 90

def porta_z():
    st.session_state["theta_graus"] = 0
    st.session_state["phi_graus"] = 180

# Sliders
theta_graus = st.sidebar.slider("Ângulo Theta (θ)", 0, 180, 90)
phi_graus = st.sidebar.slider("Ângulo Phi (ϕ)", 0, 360, 0)

# Botões de portas
col_h, col_x, col_y, col_z = st.sidebar.columns(4)
if col_h.button("H"):
    porta_h()
if col_x.button("X"):
    porta_x()
if col_y.button("Y"):
    porta_y()
if col_z.button("Z"):
    porta_z()

# --- CÁLCULOS ---
theta = np.deg2rad(st.session_state["theta_graus"])
phi = np.deg2rad(st.session_state["phi_graus"])
alpha = np.cos(theta / 2)
beta = np.exp(1j * phi) * np.sin(theta / 2)

# --- VISUALIZAÇÃO: ESFERA DE BLOCH ---
def plot_bloch(theta, phi):
    # Esfera
    u, v = np.mgrid[0:2*np.pi:100j, 0:np.pi:100j]
    x = np.cos(u)*np.sin(v)
    y = np.sin(u)*np.sin(v)
    z = np.cos(v)
    # Vetor de estado
    x_state = np.sin(theta) * np.cos(phi)
    y_state = np.sin(theta) * np.sin(phi)
    z_state = np.cos(theta)
    fig = go.Figure(data=[
        go.Surface(x=x, y=y, z=z, opacity=0.1, showscale=False),
        go.Scatter3d(x=[0, x_state], y=[0, y_state], z=[0, z_state],
                     marker=dict(size=4), line=dict(color="red", width=6))
    ])
    fig.update_layout(scene=dict(
        xaxis=dict(range=[-1,1]), yaxis=dict(range=[-1,1]), zaxis=dict(range=[-1,1]),
        aspectmode='cube'
    ), margin=dict(l=0, r=0, b=0, t=0))
    return fig

st.plotly_chart(plot_bloch(theta, phi), use_container_width=True)

# --- ÁREA MATEMÁTICA ---
st.header("Análise Matemática em Tempo Real")
col1, col2 = st.columns(2)
with col1:
    st.latex(r"|\psi\rangle = \cos(\frac{\theta}{2})|0\rangle + e^{i\phi}\sin(\frac{\theta}{2})|1\rangle")
    st.markdown(f"**α (alpha):** `{np.round(alpha, 3)}`")
    st.markdown(f"**β (beta):** `{np.round(beta, 3)}`")
with col2:
    prob_0 = np.abs(alpha) ** 2
    prob_1 = np.abs(beta) ** 2
    st.latex(fr"P(|0\rangle) = |\alpha|^2 = {prob_0:.2%}")
    st.latex(fr"P(|1\rangle) = |\beta|^2 = {prob_1:.2%}")

# (Opcional) Gráfico de barras para partes real e imaginária
st.subheader("Componentes α e β (Real e Imaginário)")
df = pd.DataFrame({
    "Real": [np.real(alpha), np.real(beta)],
    "Imaginário": [np.imag(alpha), np.imag(beta)]
}, index=["α", "β"])
st.bar_chart(df)
