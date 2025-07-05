import streamlit as st
import joblib
import pandas as pd

# Carga el modelo entrenado
filename = 'modelo_naive_bayes.pkl'
model = joblib.load(filename)

# Mapeo de categorías a números y viceversa (debe coincidir con el entrenamiento)
mapeo_horas = {"Alta": 1, "Baja": 0}
mapeo_asistencia = {"Buena": 1, "Mala": 0}
mapeo_resultado = {"Sí": 1, "No": 0}
mapeo_resultado_inverso = {v: k for k, v in mapeo_resultado.items()}

st.title("Predicción de clase")
st.markdown("<h3 style='color: red;'>Elaborado por: Shirley Ortega</h3>", unsafe_allow_html=True)

st.write("Por favor, seleccione los valores de las variables de entrada:")

# Crear selectbox para la entrada del usuario
horas_estudio = st.selectbox("Horas de Estudio:", list(mapeo_horas.keys()))
asistencia = st.selectbox("Asistencia:", list(mapeo_asistencia.keys()))

# Codificar la entrada del usuario
nueva_observacion_codificada = pd.DataFrame([[mapeo_horas[horas_estudio], mapeo_asistencia[asistencia]]],
                                            columns=['Horas de Estudio', 'Asistencia'])

# Realizar la predicción
prediccion_numerica = model.predict(nueva_observacion_codificada)

# Decodificar la predicción
prediccion_etiqueta = mapeo_resultado_inverso[prediccion_numerica[0]]

# Mostrar el resultado con emojis
if prediccion_etiqueta == "Sí":
    st.success("¡Felicitaciones, aprueba! 😊")
else:
    st.error("No aprueba 😞")

