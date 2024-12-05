import numpy as np
import plotly.graph_objs as go
from scipy.interpolate import interp1d
from static.styles import (
    LIGHT_BLUE_COLOR, 
    DARK_BLUE_COLOR, 
    DARK_PURPLE_COLOR,
    DARK_PURPLE_TEXT,
    PURE_BLACK_COLOR
)
import time
import streamlit as st
import matplotlib.pyplot as plt

Delta = 0.01

def generate_continuous_graphique(t, x_t, color, title, xlabel='Tiempo [s]', ylabel='Amplitud'):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=x_t, mode="lines", name=title, line=dict(color=color)))

    fig.update_layout(
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        showlegend=True,
    )

    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True)

    # Mostrar gráfico en la primera columna
    st.plotly_chart(fig, use_container_width=True)

def generate_continuous_graphique_pi2(t, x_t, color, title, xlabel='Tiempo [s]', ylabel='Amplitud'):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=x_t, mode="lines", name=title, line=dict(color=color)))

    fig.update_layout(
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        showlegend=True,
    )

    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True)

    # Mostrar gráfico en la primera columna
    st.plotly_chart(fig, use_container_width=True)

def invert_continous_signal(t, x_t):
    return -t[::-1], x_t[::-1]

def generate_continous_pyplot_graph(time, x_t, label, xlabel='Tiempo [s]', ylabel='Amplitud', title='Señal'):
    fig, ax = plt.subplots()
    ax.plot(time, x_t, color=DARK_PURPLE_COLOR, label=label)

    # Ajustes comunes al gráfico
    ax.legend()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    # Fondo blanco
    fig.patch.set_facecolor('white')  # Fondo de la figura
    ax.set_facecolor('white')         # Fondo del área de los ejes

    # Ajustes de los ejes
    ax.tick_params(colors=PURE_BLACK_COLOR)  # Color de los números en los ejes
    ax.spines['top'].set_color(PURE_BLACK_COLOR)
    ax.spines['bottom'].set_color(PURE_BLACK_COLOR)
    ax.spines['left'].set_color(PURE_BLACK_COLOR)
    ax.spines['right'].set_color(PURE_BLACK_COLOR)

    fig.patch.set_alpha(0.0)  # Fondo de la figura
    ax.patch.set_alpha(0.0)   # Fondo del área de los ejes
    st.pyplot(fig)

def generate_double_continous_pyplot_graph(time, x_t, x_t2, label, xlabel='Tiempo [s]', ylabel='Amplitud', title='Señal'):
    fig, ax = plt.subplots()
    ax.plot(time, x_t, color=DARK_PURPLE_COLOR, label=label)
    ax.plot(time, x_t2, color=LIGHT_BLUE_COLOR, label='Filtro pasa bajas')

    # Ajustes comunes al gráfico
    ax.legend()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    # Fondo blanco
    fig.patch.set_facecolor('white')  # Fondo de la figura
    ax.set_facecolor('white')         # Fondo del área de los ejes

    # Ajustes de los ejes
    ax.tick_params(colors=DARK_PURPLE_TEXT)  # Color de los números en los ejes
    ax.spines['top'].set_color(DARK_PURPLE_TEXT)
    ax.spines['bottom'].set_color(DARK_PURPLE_TEXT)
    ax.spines['left'].set_color(DARK_PURPLE_TEXT)
    ax.spines['right'].set_color(DARK_PURPLE_TEXT)

    fig.patch.set_alpha(0.0)  # Fondo de la figura
    ax.patch.set_alpha(0.0)   # Fondo del área de los ejes
    st.pyplot(fig)