import numpy as np
import streamlit as st
from scipy.io import wavfile
from static.styles import (
    CSS_TOPBAR_STYLES,
    CSS_SIDEBARD_STYLES,
    CSS_CREDITS_STYLES,
    LIGHT_BLUE_COLOR, 
    MEDIUM_BLUE_COLOR, 
    LIGHT_PURPLE_COLOR,
    DARK_BLUE_COLOR,
    DARK_PURPLE_COLOR,
    DARK_PURPLE_TEXT,
    PURE_BLACK_COLOR,
    build_custom_error,
)
from signals.continuous_signals import (
    generate_continuous_graphique,
    generate_continous_pyplot_graph,
    generate_double_continous_pyplot_graph,
)
from signals.discrete_signals import (
    generate_discrete_graphique,
)
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from sympy import *

Delta = 0.01


MENU_OPTIONS = [
    "Introducción", 
    "Series de Fourier", 
    "Transformada de Fourier y Modulación de Señales", 
    "Créditos"
]

def periodic_function(t):
    return (t + np.pi) % (2 * np.pi) - np.pi

def periodic_quadratic(t):
    return ((t + np.pi) % (2 * np.pi) - np.pi) ** 2

st.set_page_config(layout="wide")
st.markdown(CSS_TOPBAR_STYLES, unsafe_allow_html=True)
st.markdown(CSS_SIDEBARD_STYLES, unsafe_allow_html=True)
st.markdown(f"""
    <h1 style='text-align: center; color: {DARK_BLUE_COLOR};'>Interfaz de Fourier para Señales</h1>
""", unsafe_allow_html=True)
st.sidebar.title("MENU DE INTERACCION")
selected_option = st.sidebar.selectbox("Seleccione una opción", MENU_OPTIONS)

if selected_option == "Introducción":
    st.subheader("🌟 Bienvenido a la Interfaz Gráfica de Series y Transformadas de Fourier")
    st.markdown("""
    En esta aplicación interactiva, **explorarás y comprenderás** los conceptos fundamentales de las **Series y Transformadas de Fourier**. 
    ¡Sumérgete en el fascinante mundo de la descomposición de señales y el análisis de frecuencia! 🎯
    """)

    column_1, column_2 = st.columns(2)

    with column_1:
        st.markdown(f"""
            <h3 style='text-align: center; color: {DARK_PURPLE_COLOR};'>📈 ¿Qué son las Series de Fourier?</h3>
        """, unsafe_allow_html=True)

        st.write("""
        Las Series de Fourier permiten **representar señales periódicas** como una suma de funciones sinusoidales 
        con diferentes frecuencias, amplitudes y fases.
        """)
        with st.expander("Componentes de las Series de Fourier"):
            st.write("""
            Principales componentes:
            - **Fundamental**: Frecuencia base de la señal periódica
            - **Armónicos**: Múltiplos enteros de la frecuencia fundamental
            - **Coeficientes**: Amplitudes de cada componente sinusoidal
            Las Series de Fourier revelan la estructura interna de señales periódicas complejas.
            """)

    with column_2:
        st.markdown(f"""
            <h3 style='text-align: center; color: {DARK_PURPLE_COLOR};'>🔄 Transformada de Fourier</h3>
        """, unsafe_allow_html=True)
        st.write("""
        La **Transformada de Fourier** extiende el concepto de Series de Fourier a señales no periódicas, 
        permitiendo el análisis de su contenido frecuencial.
        """)
        st.write("**Tipos fundamentales**:")
        st.markdown("""
        - **Transformada de Fourier Continua**: Análisis de señales continuas en frecuencia
        - **Transformada de Fourier Discreta (DFT)**: Análisis de señales discretas
        - **Transformada Rápida de Fourier (FFT)**: Algoritmo eficiente para calcular la DFT
        """)

    st.markdown("""
    Utilice el **menú de la izquierda** para navegar entre las diferentes secciones y explorar todas las funcionalidades de esta aplicación. 
    ¡Descubre los secretos del análisis de frecuencia! 🚀
    """)

elif selected_option == "Series de Fourier":
    def u(t):
        return np.where(t >= 0, 1, 0)
    Delta = 0.01

    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown(f"""
            <h3 style='color: {DARK_PURPLE_COLOR};'>Seleccione la señal a graficar</h3>
        """, unsafe_allow_html=True)
    selected_signal = st.selectbox("Señal", ["Seleccione la señal", "3.6.1", "3.6.2", "3.6.3", "3.6.4"])

    if selected_signal == "Seleccione la señal":
        CSS_CUSTOM_ERROR_STYLES = build_custom_error('⚠️ Debe seleccionar una señal para continuar')
        st.markdown(CSS_CUSTOM_ERROR_STYLES, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <h3 style='text-align: center;color: {DARK_BLUE_COLOR};'>Señal original</h3>
            """, unsafe_allow_html=True)
        
        if selected_signal == "3.6.1":
            T = 4  # Periodo de la señal
            delta = 1

            # Definir el ciclo fundamental
            t1 = np.arange(-T/2, 0, delta)
            t2 = np.arange(0, T/2 + delta, delta)
            x1 = 1 + 4 * t1 / T
            x2 = 1 - 4 * t2 / T
            x_cycle = np.concatenate((x1, x2))
            t_cycle = np.concatenate((t1, t2))

            # Crear señal periódica
            num_periods = 1  # Número de periodos a repetir
            t_periodic = []
            x_periodic = []

            for k in range(-num_periods // 2, num_periods // 2 + 1):
                t_periodic.append(t_cycle + k * T)  # Desplazar tiempo por múltiplos de T
                x_periodic.append(x_cycle)  # Repetir el mismo ciclo


            t_periodic = np.concatenate(t_periodic)
            x_periodic = np.concatenate(x_periodic)
            generate_continuous_graphique(t_periodic, x_periodic, MEDIUM_BLUE_COLOR, "x(t)")

            st.markdown(f"""
                <h4 style='text-align: left;color: {DARK_BLUE_COLOR};'>Ingrese el número de armónicos para reconstruir la señal</h4>
                """, unsafe_allow_html=True)
            num_armonicos = st.number_input(
                " ", 
                value=0, 
                step=1, 
                min_value=0,
                max_value=100,
                format="%d",
            )

            if num_armonicos == 0:
                CSS_CUSTOM_ERROR_STYLES = build_custom_error('⚠️ Ingrese el número de armónicos para continuar (número entero)')
                st.markdown(CSS_CUSTOM_ERROR_STYLES, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <h3 style='text-align: center;color: {DARK_PURPLE_COLOR};'>Señal Reconstruida</h3>
                """, unsafe_allow_html=True)
                t= sp.Symbol('t')
                n= sp.Symbol('n')
                x=sp.Symbol('x')
                N=num_armonicos
                T=4
                f1=1+4*(t/T)
                f2=1-4*(t/T)
                af11=(2/T)*f1*sp.cos(n*(2*sp.pi/T)*t)
                af22=(2/T)*f2*sp.cos(n*(2*sp.pi/T)*t)
                aI= integrate(af11,(t,-T/2,0))+integrate(af22,(t,0,T/2))
                a0=aI.subs(n,0).evalf()
                bf11=(2/T)*f1*sp.sin(n*(2*sp.pi/T)*t)
                bf22=(2/T)*f2*sp.sin(n*(2*sp.pi/T)*t)
                bI= integrate(bf11,(t,-T/2,0))+integrate(bf22,(t,0,T/2))
                i=0
                xn=0
                while (i<=N):
                    if(i==0):
                        xn+=a0/2
                    else:
                        xn+= aI.subs(n,i).evalf()*cos(x*2*i*sp.pi/T)+bI.subs(n,i).evalf()*sin(x*2*i*sp.pi/T)
                    i+=1


                li = -6
                lf = 2
                delta = 0.01
                tn = np.arange(li, lf+delta, delta)  # Crear puntos para el eje x

                f_lambda = sp.lambdify(x, xn, modules=["numpy"])  # Convertir función simbólica a una evaluable
                generate_continuous_graphique(tn, f_lambda(tn), DARK_PURPLE_COLOR, "Señal Reconstruida")


                st.markdown(f"""
                <h3 style='text-align: center;color: {DARK_BLUE_COLOR};'>Espectro de Frecuencia</h3>
                """, unsafe_allow_html=True)

                Cn=sqrt(aI**2+bI**2)
                Sp=np.zeros(N+N+1)
                ts=np.arange(-N,N+1)
                k=0
                for i in range (-N,N+1):
                    if(i==0):
                        Sp[k]=a0/2
                    else:
                        Sp[k]=Cn.subs(n,i).evalf()
                    k=k+1
                graf = generate_discrete_graphique(ts, Sp, 'Espectro de Magnitud', LIGHT_PURPLE_COLOR)
                st.plotly_chart(graf, use_container_width=True)

        elif selected_signal == "3.6.2":
            t = np.linspace(-2 * np.pi, 2 * np.pi, 1000)
            x_t = periodic_function(t)
            generate_continuous_graphique(t, x_t, MEDIUM_BLUE_COLOR, "x(t)")

            st.markdown(f"""
                <h4 style='text-align: left;color: {DARK_BLUE_COLOR};'>Ingrese el número de armónicos para reconstruir la señal</h4>
                """, unsafe_allow_html=True)
            num_armonicos = st.number_input(
                " ", 
                value=0, 
                step=1, 
                min_value=0,
                max_value=100,
                format="%d",
            )

            if num_armonicos == 0:
                CSS_CUSTOM_ERROR_STYLES = build_custom_error('⚠️ Ingrese el número de armónicos para continuar (número entero)')
                st.markdown(CSS_CUSTOM_ERROR_STYLES, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <h3 style='text-align: center;color: {DARK_PURPLE_COLOR};'>Señal Reconstruida</h3>
                """, unsafe_allow_html=True)

                t= sp.Symbol('t')
                n= sp.Symbol('n')
                x=sp.Symbol('x')
                N=num_armonicos
                T=2*sp.pi
                f1=t
                af11=(2/T)*f1*sp.cos(n*(2*sp.pi/T)*t)
                aI= integrate(af11,(t,-sp.pi,sp.pi))
                a0=aI.subs(n,0).evalf()
                bf11=(2/T)*f1*sp.sin(n*(2*sp.pi/T)*t)
                bI= integrate(bf11,(t,-sp.pi,sp.pi))
                i=0
                xn=0
                while (i<=N):
                    if(i==0):
                        xn+=a0/2
                    else:
                        xn+= aI.subs(n,i).evalf()*cos(x*2*i*sp.pi/T)+bI.subs(n,i).evalf()*sin(x*2*i*sp.pi/T)
                    i+=1
                li = -2*np.pi
                lf = 2*np.pi
                delta = 0.01
                tn = np.arange(li, lf+delta, delta)  # Crear puntos para el eje x

                # Graficar la función simbólica
                f_lambda = sp.lambdify(x, xn, modules=["numpy"])  # Convertir función simbólica a una evaluable
                generate_continuous_graphique(tn, f_lambda(tn), DARK_PURPLE_COLOR, "Señal Reconstruida")


                st.markdown(f"""
                <h3 style='text-align: center;color: {DARK_BLUE_COLOR};'>Espectro de Frecuencia</h3>
                """, unsafe_allow_html=True)

                Cn=sqrt(aI**2+bI**2)
                Sp=np.zeros(N+1)
                ts=np.arange(N+1)
                k=0
                for i in range (0,N+1):
                    if(i==0):
                        Sp[k]=a0/2
                    else:
                        Sp[k]=Cn.subs(n,i).evalf()
                    k=k+1
                graf = generate_discrete_graphique(ts, Sp, 'Espectro de Magnitud', LIGHT_PURPLE_COLOR)
                st.plotly_chart(graf, use_container_width=True)
        
        elif selected_signal == "3.6.3":
            t = np.linspace(-3 * np.pi, 3 * np.pi, 1000)
            x_t = periodic_quadratic(t)
            generate_continuous_graphique(t, x_t, MEDIUM_BLUE_COLOR, "x(t)")

            st.markdown(f"""
                <h4 style='text-align: left;color: {DARK_BLUE_COLOR};'>Ingrese el número de armónicos para reconstruir la señal</h4>
                """, unsafe_allow_html=True)
            num_armonicos = st.number_input(
                " ", 
                value=0, 
                step=1, 
                min_value=0,
                max_value=100,
                format="%d",
            )

            if num_armonicos == 0:
                CSS_CUSTOM_ERROR_STYLES = build_custom_error('⚠️ Ingrese el número de armónicos para continuar (número entero)')
                st.markdown(CSS_CUSTOM_ERROR_STYLES, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <h3 style='text-align: center;color: {DARK_PURPLE_COLOR};'>Señal Reconstruida</h3>
                """, unsafe_allow_html=True)

                t= sp.Symbol('t')
                n= sp.Symbol('n')
                x=sp.Symbol('x')
                N=num_armonicos
                T=2*sp.pi
                f1=t**2
                af11=(2/T)*f1*sp.cos(n*(2*sp.pi/T)*t)
                aI= integrate(af11,(t,-sp.pi,sp.pi))
                a0= aI.subs(n,0).evalf()
                bf11=(2/T)*f1*sp.sin(n*(2*sp.pi/T)*t)
                bI= integrate(bf11,(t,-sp.pi,sp.pi))
                i=0
                xn=0
                while (i<=N):
                    if(i==0):
                        xn+=a0/2
                    else:
                        xn+= aI.subs(n,i).evalf()*cos(x*2*i*sp.pi/T)+bI.subs(n,i).evalf()*sin(x*2*i*sp.pi/T)
                    i+=1
                li = -2*np.pi
                lf = 2*np.pi
                delta = 0.01
                tn = np.arange(li, lf+delta, delta)  # Crear puntos para el eje x

                # Graficar la función simbólica
                f_lambda = sp.lambdify(x, xn, modules=["numpy"])  # Convertir función simbólica a una evaluable
                generate_continuous_graphique(tn, f_lambda(tn), DARK_PURPLE_COLOR, "Señal Reconstruida")


                st.markdown(f"""
                <h3 style='text-align: center;color: {DARK_BLUE_COLOR};'>Espectro de Frecuencia</h3>
                """, unsafe_allow_html=True)

                Cn=sqrt(aI**2+bI**2)
                Sp=np.zeros(N+1)
                ts=np.arange(N+1)
                k=0
                for i in range (0,N+1):
                    if(i==0):
                        Sp[k]=a0/2
                    else:
                        Sp[k]=Cn.subs(n,i).evalf()
                    k=k+1
                graf = generate_discrete_graphique(ts, Sp, 'Espectro de Magnitud', LIGHT_PURPLE_COLOR)
                st.plotly_chart(graf, use_container_width=True)

        elif selected_signal == "3.6.4":
            T=2  #Definir el periodo de esta señal
            delta = 0.01
            t1b = np.arange(-T/2, 0, delta)
            t2b = np.arange(0, T/2 + delta, delta)
            x1tb = t1b
            x2tb = np.ones(len(t2b))
            x_ciclo = np.concatenate((x1tb, x2tb))
            t_ciclo = np.concatenate((t1b, t2b))

            periodo = 2  # Número de periodos a repetir
            t_periodo = []
            x_periodo= []
            for k in range(-periodo // 2, periodo // 2 + 1):
                t_periodo.append(t_ciclo + k * T)  # Desplazar tiempo por múltiplos de T
                x_periodo.append(x_ciclo)  # Repetir el mismo ciclo

            t_periodo = np.concatenate(t_periodo)
            x_periodo = np.concatenate(x_periodo)
            generate_continuous_graphique(t_periodo, x_periodo, MEDIUM_BLUE_COLOR, "x(t)")

            st.markdown(f"""
                <h4 style='text-align: left;color: {DARK_BLUE_COLOR};'>Ingrese el número de armónicos para reconstruir la señal</h4>
                """, unsafe_allow_html=True)
            num_armonicos = st.number_input(
                " ", 
                value=0, 
                step=1, 
                min_value=0,
                max_value=100,
                format="%d",
            )

            if num_armonicos == 0:
                CSS_CUSTOM_ERROR_STYLES = build_custom_error('⚠️ Ingrese el número de armónicos para continuar (número entero)')
                st.markdown(CSS_CUSTOM_ERROR_STYLES, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <h3 style='text-align: center;color: {DARK_PURPLE_COLOR};'>Señal Reconstruida</h3>
                """, unsafe_allow_html=True)

                t= sp.Symbol('t')
                n= sp.Symbol('n')
                x=sp.Symbol('x')
                N=num_armonicos
                T=2
                f1=t
                f2=1
                af11=(2/T)*f1*sp.cos(n*(2*sp.pi/T)*t)
                af22=(2/T)*f2*sp.cos(n*(2*sp.pi/T)*t)
                aI= integrate(af11,(t,-T/2,0))+integrate(af22,(t,0,T/2))
                a0= aI.subs(n,0).evalf()
                bf11=(2/T)*f1*sp.sin(n*(2*sp.pi/T)*t)
                bf22=(2/T)*f2*sp.sin(n*(2*sp.pi/T)*t)
                bI= integrate(bf11,(t,-T/2,0))+integrate(bf22,(t,0,T/2))
                i=0
                xn=0
                while (i<=N):
                    if(i==0):
                        xn+=a0/2
                    else:
                        xn+= aI.subs(n,i).evalf()*cos(x*2*i*sp.pi/T)+bI.subs(n,i).evalf()*sin(x*2*i*sp.pi/T)
                    i+=1


                li = -2
                lf = 2
                delta = 0.01
                tn = np.arange(li, lf+delta, delta)  # Crear puntos para el eje x

                # Graficar la función simbólica
                f_lambda = sp.lambdify(x, xn, modules=["numpy"])  # Convertir función simbólica a una evaluable
                generate_continuous_graphique(tn, f_lambda(tn), DARK_PURPLE_COLOR, "Señal Reconstruida")


                st.markdown(f"""
                <h3 style='text-align: center;color: {DARK_BLUE_COLOR};'>Espectro de Frecuencia</h3>
                """, unsafe_allow_html=True)

                Cn=sqrt(aI**2+bI**2)
                Sp=np.zeros(N+1)
                ts=np.arange(N+1)
                k=0
                for i in range (0,N+1):
                    if(i==0):
                        Sp[k]=a0/2
                    else:
                        Sp[k]=Cn.subs(n,i).evalf()
                    k=k+1
                graf = generate_discrete_graphique(ts, Sp, 'Espectro de Magnitud', LIGHT_PURPLE_COLOR)
                st.plotly_chart(graf, use_container_width=True)

elif selected_option == "Transformada de Fourier y Modulación de Señales":
    files = st.file_uploader(" ", accept_multiple_files=True, type=["wav"])
    if files:
        for file in files:
            fs, data = wavfile.read(file)
            length = data.shape[0] / fs


        is_stereo = len(data.shape) == 2 and data.shape[1] > 1

        # Seleccionamos el primer canal si es estéreo
        if is_stereo:
            data = data[:, 0]

        x_t = data / np.max(data)
        n = len(x_t)
        tiempo = n / fs       # Duración del audio
        Delta_t = 1 / fs      # Tiempo de muestreo
        t = np.arange(n) * Delta_t

        st.markdown(f"""
            <h4 style='text-align: center;color: {DARK_BLUE_COLOR};'>Audio cargado</h4>
            """, unsafe_allow_html=True
        )

        st.audio(x_t, sample_rate=fs)

        st.markdown(f"""
            <h3 style='text-align: left;color: {PURE_BLACK_COLOR};'>Señal del audio cargado</h3>
            """, unsafe_allow_html=True
        )
        
        generate_continous_pyplot_graph(t, x_t, "DEP Canal 1" if is_stereo else "Mono Canal")


        w = np.linspace(-len(x_t)/2, len(x_t)/2, len(x_t))

        X_w = np.fft.fft(x_t)            # Transformada de Fourier de x(t)
        X_w_cent = np.fft.fftshift(X_w)  # Centramos el espectro
        Delta_f = 1 / (n * Delta_t)
        f = np.arange(-n / 2, n / 2) * Delta_f
        #magnitud = np.abs(X_w_cent) / np.max
        magnitud = np.abs(X_w_cent) / n
        
        fpb = np.abs(f) <= 10000  # Frecuencia de corte, en este caso 1000 Hz

        st.markdown(f"""
            <h3 style='text-align: left;color: {PURE_BLACK_COLOR};'>Filtro pasa bajas ideal</h3>
            """, unsafe_allow_html=True
        )
        generate_continous_pyplot_graph(f, fpb, "Filtro Pasa Bajas", " ", " ")

        st.markdown(f"""
            <h3 style='text-align: left;color: {PURE_BLACK_COLOR};'>Espectro Filtrado</h3>
            """, unsafe_allow_html=True
        )
        X_w_fil = X_w_cent * fpb
        generate_continous_pyplot_graph(f, np.abs(X_w_fil) / np.max(np.abs(X_w_fil)), "Espectro Filtrado", "Espectro Filtrado", "Frequencia [Hz]")

        X_w_filt_corrida = np.fft.ifftshift(X_w_fil)    # Corrimiento del espectro
        x_t_filt2 = np.fft.ifft(X_w_filt_corrida)       # Transformada inversa
        st.markdown(f"""
            <h5 style='text-align: left;color: {PURE_BLACK_COLOR};'>Audio original</h5>
            """, unsafe_allow_html=True
        )
        st.audio(x_t, sample_rate=fs)
        st.markdown(f"""
            <h5 style='text-align: left;color: {PURE_BLACK_COLOR};'>Audio filtrado</h5>
            """, unsafe_allow_html=True
        )
        st.audio(x_t_filt2, sample_rate=fs)

        carrier_amplitude = st.number_input("Ingrese la amplitud de la portadora: ", value=1.0)
        modulating_frequency = st.number_input("Ingrese la frecuencia de la señal moduladora (Hz): ", value=10000.0)
        carrier_frequency = st.number_input("Ingrese la frecuencia de la portadora (Hz): ", value=100000.0)

        if carrier_frequency < (10*modulating_frequency):
            carrier_frequency = st.number_input("Ingrese la frecuencia de la portadora (Hz):", value=carrier_frequency)
            CSS_CUSTOM_ERROR_STYLES = build_custom_error('⚠️ La frecuencia de la portadora debe ser al menos 10 veces la frecuencia de la señal moduladora')
            st.markdown(CSS_CUSTOM_ERROR_STYLES, unsafe_allow_html=True)

        else: 
            w = 2*np.pi* carrier_frequency
            portadora = carrier_amplitude * np.cos(w*t)
            T = 1/ carrier_frequency
            t2 = np.arange(0,10*T+(1/100*T), (1/100*T))
            portadora2 = carrier_amplitude * np.cos(w*t2)
            yt = x_t_filt2 * portadora

            pt=carrier_amplitude*np.cos(w*t)
            Y_w = np.fft.fft(yt)            # Transformada de Fourier de x(t)
            Y_w_cent = np.fft.fftshift(Y_w)  # Centramos el espectro
            Delta_f = 1 / (n * Delta_t)
            w_p = np.linspace(-len(pt)/2,len(pt)/2,len(pt))
            magnitud = np.abs(Y_w_cent) / n


            st.subheader("Gráfica de la portadora")

            # Comparación de señales en el dominio del tiempo
            generate_continous_pyplot_graph(
                t2, portadora2,
                "Gráfica de la portadora", 
                "Tiempo [s]", 
                "Amplitud"
            )

            # Cálculo de la FFT de la señal modulada
            st.subheader("Espectro Modulado")

            generate_continous_pyplot_graph(
                w_p, np.abs(Y_w_cent) / np.max(np.abs(X_w_fil)),
                "Espectro Modulado", 
                "Frecuencia [Hz]", 
                "Magnitud (dB)"
            )

            st.subheader("Espectro Demodulado")

            xtdem=yt*pt

            # Cálculo del espectro
            X_wdem = np.fft.fft(xtdem)            # Transformada de Fourier de x(t)
            X_w_demcent = np.fft.fftshift(X_wdem)  # Centramos el espectro
            Delta_f = 1 / (n * Delta_t)
            magnitud = np.abs(X_w_demcent) / n
            f = np.arange(-len(xtdem)/2, len(xtdem)/2) * Delta_f

            generate_continous_pyplot_graph(
                f, np.abs(X_w_demcent) / np.max(np.abs(X_w_demcent)),
                "Espectro Demodulado", 
                "Frecuencia [Hz]", 
                "Magnitud (dB)"
            )

            X_w_fil2 = X_w_demcent * fpb
            st.subheader("Espectro Recuperado")
            f = np.arange(-n / 2, n / 2) * Delta_f
            XWrecuperada= X_w_demcent*fpb
            XWrecup = np.fft.ifftshift(X_w_fil2)    # Corrimiento del espectro
            x_t_filt2= np.fft.ifft(XWrecup)       # Transformada inversa
            generate_continous_pyplot_graph(
                f, np.abs(X_w_fil) / np.max(np.abs(X_w_fil)),
                "Espectro Recuperado", 
                "Frecuencia [Hz]", 
                "Magnitud (dB)"
            )
            st.markdown(f"""
                <h5 style='text-align: left;color: {PURE_BLACK_COLOR};'>Audio original</h5>
                """, unsafe_allow_html=True
            )
            st.audio(x_t, sample_rate=fs)
            st.markdown(f"""
                <h5 style='text-align: left;color: {PURE_BLACK_COLOR};'>Audio recuperado</h5>
                """, unsafe_allow_html=True
            )
            st.audio(x_t_filt2, sample_rate=fs)
    else:
        CSS_CUSTOM_ERROR_STYLES = build_custom_error('⚠️ Debe cargar un archivo .wav para continuar')
        st.markdown(CSS_CUSTOM_ERROR_STYLES, unsafe_allow_html=True)


elif selected_option == "Créditos":
    st.markdown(CSS_CREDITS_STYLES, unsafe_allow_html=True)
    st.markdown("<br><br>", unsafe_allow_html=True)

    column_1, column_2 = st.columns(2)
    with column_1:
        st.markdown("""
        <div class="custom-column">
        <h3 class="custom-header">Desarrolladores</h3>
        - Emmanuel Cabrera Janer<br>
        - Haxell Gómez Lara<br>
        - Nikolas Pedraza Wilson
        </div>
        """, unsafe_allow_html=True)

    with column_2:
        st.markdown("""
        <div class="custom-column custom-offset"> <!-- Aplicar el margen superior aquí -->
        <h3 class="custom-header">Profesor Supervisor</h3>
        - PhD Juan Tello Portillo
        </div>
        """, unsafe_allow_html=True)

    column_3, column_4 = st.columns(2)
    with column_3:
        st.markdown("""
        <div class="custom-column custom-offset">
        <h3 class="custom-header">Universidad del Norte</h3>
        - Departamento de Ingeniería Eléctrica y Electrónica
        </div>
        """, unsafe_allow_html=True)

    with column_4:
        st.markdown("""
        <div class="custom-column"> <!-- Aplicar el margen superior aquí -->
        <h3 class="custom-header">Tecnologías Utilizadas</h3>
        - Python: Lenguaje de programación<br>
        - Streamlit: Framework para la creación de interfaces gráficas web<br>
        - HTML: Lenguaje para crear el esquema básico de la página<br>
        - CSS: Lenguaje para personalizar los estilos de la interfaz
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown(f"""
        <div class="custom-footer">
        Esta interfaz fue presentada como parte del laboratorio final del curso de Señales y Sistemas para el año académico 2024-03
        </div>
    """, unsafe_allow_html=True)
