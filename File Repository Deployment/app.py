import numpy as np
import pickle
from flask import Flask, request, render_template
import os

# Inisialisasi Aplikasi Flask
app = Flask(__name__)

# --- Konfigurasi Path Model dan Scaler ---
MODEL_FILE_NAME = "random_forest_model.pkl"
SCALER_FILE_NAME = "scaler.pkl"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, MODEL_FILE_NAME)
SCALER_PATH = os.path.join(BASE_DIR, SCALER_FILE_NAME)

# Variabel global
model = None
scaler = None

# Fungsi untuk memuat model dan scaler
def load_resources():
    global model, scaler

    if not os.path.exists(MODEL_PATH):
        print(f"🚫 File model tidak ditemukan: {MODEL_PATH}")
        return False

    if not os.path.exists(SCALER_PATH):
        print(f"🚫 File scaler tidak ditemukan: {SCALER_PATH}")
        return False

    try:
        with open(MODEL_PATH, "rb") as f_model:
            model = pickle.load(f_model)
        with open(SCALER_PATH, "rb") as f_scaler:
            scaler = pickle.load(f_scaler)
        print("✅ Model dan Scaler berhasil dimuat.")
        return True
    except Exception as e:
        print(f"❌ Gagal memuat resource: {e}")
        return False

# Muat model dan scaler saat aplikasi mulai
if not load_resources():
    print("❗ Aplikasi tidak dapat dijalankan. Pastikan file model dan scaler tersedia.")

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html", prediction_result=None, result_class=None)

@app.route("/predict", methods=["POST"])
def predict():
    prediction_result = None
    result_class = "error"

    if model is None or scaler is None:
        prediction_result = "Model atau scaler belum dimuat."
        return render_template("index.html", prediction_result=prediction_result, result_class=result_class)

    try:
        features = [
            'FastingBloodSugar', 'HbA1c', 'SleepQuality', 'CholesterolHDL', 
            'FatigueLevels', 'CholesterolLDL', 'MedicationAdherence', 
            'QualityOfLifeScore', 'DiastolicBP', 'Age'
        ]

        input_data = []
        for feature in features:
            value = request.form.get(feature, '')
            if not value:
                raise ValueError(f"Input untuk '{feature}' tidak boleh kosong.")
            input_data.append(float(value))

        # Konversi ke NumPy array dan scaling
        input_array = np.array(input_data).reshape(1, -1)
        input_scaled = scaler.transform(input_array)

        # Prediksi probabilitas
        prediction_proba = model.predict_proba(input_scaled)[0]
        pred_class = np.argmax(prediction_proba)
        confidence = prediction_proba[pred_class] * 100

        if pred_class == 1:
            output = f"Hasil: Berisiko Tinggi Terkena Diabetes (Keyakinan: {confidence:.2f}%)"
            result_class = "positive"
        else:
            output = f"Hasil: Risiko Rendah Terkena Diabetes (Keyakinan: {confidence:.2f}%)"
            result_class = "negative"

        prediction_result = output

    except ValueError as ve:
        prediction_result = f"Input tidak valid: {ve}"
        result_class = "error"
    except Exception as e:
        prediction_result = f"Terjadi error: {e}"
        result_class = "error"

    return render_template("index.html", prediction_result=prediction_result, result_class=result_class)

if __name__ == "__main__":
    app.run(debug=True)