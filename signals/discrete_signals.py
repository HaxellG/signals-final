import numpy as np
import plotly.graph_objs as go
from scipy.interpolate import interp1d
from static.styles import (
    LIGHT_BLUE_COLOR, 
    DARK_BLUE_COLOR, 
    DARK_PURPLE_COLOR,
)
import time
import streamlit as st

def generate_discrete_graphique(n, f, title, color):
    line_x = []
    line_y = []
    for x_val, y_val in zip(n, f):
        line_x.extend([x_val, x_val, None])  # Añadir None para separar las líneas
        line_y.extend([0, y_val, None])
    
    lines = go.Scatter(
        x=line_x,
        y=line_y,
        mode='lines',
        line=dict(color=color),
        showlegend=False
    )   
    markers = go.Scatter(
        x=n,
        y=f,
        mode='markers',
        marker=dict(color=color, size=10),
        name=title
    )
    
    fig = go.Figure(data=[lines, markers])
    fig.update_layout(
        title=title,
        xaxis=dict(tickmode="array", tickvals=n),
        xaxis_title="Tiempo",
        yaxis_title="Amplitud",
        showlegend=True,
        template='plotly_white'
    )
    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True)
    
    return fig

def invert_discrete_signal(x, y):
    return [-1 * i for i in x[::-1]], y[::-1]

def generate_discrete_conv(x, h, x_n, h_n):
    y_n = np.convolve(x_n, h_n)
    n_conv = np.arange(x[0] + h[0], x[0] + h[0] + len(y_n))
    h, h_n = invert_discrete_signal(h, h_n)
    interp_func = interp1d(n_conv, y_n, bounds_error=False, fill_value=0)

    n_min = min(min(x)-5, min(h)-1)
    n_max = max(max(x)+5, max(h)+1)

    fig_senales_fija = generate_discrete_graphique(x, x_n, "Señal Fija", DARK_BLUE_COLOR)
    fig_senales_fija.update_layout(
        xaxis=dict(showgrid=True, range=[n_min, n_max]),
        yaxis=dict(showgrid=True),
        xaxis_title="Tiempo",
        yaxis_title="Amplitud",
        title='Gráfica de la señal fija y la señal en movimiento'
    )

    shift_min = x[0] - 20 - h[0]
    shift_max = x[-1] + 20 - h[-1]

    x_full = np.arange(shift_min, shift_max)
    y_full = interp_func(x_full)

    fig_convolucion = generate_discrete_graphique(x_full, y_full, "Convolución", DARK_PURPLE_COLOR)

    col_1, col_2 = st.columns(2)
    plot_placeholder_1 = col_1.empty()
    plot_placeholder_2 = col_2.empty()
    
    plot_placeholder_1.plotly_chart(fig_senales_fija, use_container_width=True, key="signals_chart_initial")

    fig_senales_movil = go.Figure()
    
    # Crear una figura vacía para la convolución
    fig_convolucion = go.Figure()

    plot_placeholder_2.plotly_chart(fig_convolucion, use_container_width=True, key="convolution_chart_initial")
    
    # Desplazamiento de h_n_rev y actualización de las gráficas
    for shift in range(len(y_full)):
        # Calcular el desplazamiento actual
        new_h = h + x_full[shift]
        # Crear la gráfica de la señal móvil desplazada
        fig_senales_movil = generate_discrete_graphique(new_h, h_n, "Señal en Movimiento", LIGHT_BLUE_COLOR)
        
        # Combinar la señal fija y la señal móvil desplazada
        fig_combined = go.Figure(data=fig_senales_fija.data + fig_senales_movil.data)
        
        fig_combined.update_layout(
            title='Gráfica de la señal fija y la señal en movimiento',
            xaxis=dict(showgrid=True, range=[n_min, n_max]),
            yaxis=dict(showgrid=True),
            xaxis_title="Tiempo",
            yaxis_title="Amplitud",
            template='plotly_white'
        )
        
        # Actualizar la figura combinada
        plot_placeholder_1.plotly_chart(fig_combined, use_container_width=True, key=f"signals_chart_{shift}")
        
        # Actualizar la convolución parcial
        y_partial = y_full[:shift+1]
        n_partial = x_full[:shift+1]
        
        # Actualizar la traza de convolución parcial
        fig_convolucion = generate_discrete_graphique(n_partial, y_partial, "Convolución", DARK_PURPLE_COLOR)
        
        # Actualizar la figura de convolución
        plot_placeholder_2.plotly_chart(fig_convolucion, use_container_width=True, key=f"convolution_chart_{shift}")

        # Agregar un pequeño retardo para la animación final
        time.sleep(0.4)