import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

st.set_page_config(page_title="AI диагностика пневмонии", layout="centered")

st.title("🩺 Интеллектуальная система диагностики пневмонии")

st.markdown("### 🧾 Клинические данные пациента")

age = st.slider("Возраст", 18, 90, 40)
temperature = st.slider("Температура (°C)", 36.0, 41.0, 37.0, 0.1)
spo2 = st.slider("SpO₂ (%)", 80, 100, 96)
dyspnea = st.radio("Одышка", ["Нет", "Да"])

st.markdown("---")
st.markdown("### 🩻 Загрузка рентгеновского снимка")

# Загружаем модель
model = load_model(r"C:\Users\NewUser\Desktop\pneumonia_model.h5")

uploaded_file = st.file_uploader("Загрузите рентген грудной клетки", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:

    # Показываем изображение
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Загруженный рентген", use_container_width=True)

    # Подготовка изображения
    img_resized = img.resize((224, 224))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Предсказание модели
    prediction = model.predict(img_array)
    prob = float(prediction[0][0])

    st.markdown("### 📊 Результат анализа изображения")
    percent = round(prob * 100, 2)
    st.write(f"Вероятность пневмонии по рентгену: {percent}%")
    st.progress(int(percent))

    # --- Клинический модуль ---
    clinical_score = 0

    if temperature > 38:
        clinical_score += 0.1
    if spo2 < 94:
        clinical_score += 0.1
    if dyspnea == "Да":
        clinical_score += 0.1
    if age > 60:
        clinical_score += 0.05

    final_risk = min(prob + clinical_score, 1.0)
    final_percent = round(final_risk * 100, 2)

    st.markdown("### 🧠 Итоговая оценка (изображение + клинические данные)")
    st.write(f"Общий риск: {final_percent}%")
    st.progress(int(final_percent))

    # Интерпретация
    if final_risk > 0.8:
        st.error("🔴 Высокий риск пневмонии. Рекомендуется срочная консультация врача.")
    elif final_risk > 0.5:
        st.warning("🟡 Средний риск. Требуется дополнительное обследование.")
    else:
        st.success("🟢 Низкий риск пневмонии.")

    st.markdown("---")
    st.caption("⚠️ Система является вспомогательным инструментом и не заменяет медицинскую диагностику.")