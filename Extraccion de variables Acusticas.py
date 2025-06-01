import os
import librosa
import numpy as np
import pandas as pd
import soundfile as sf

# === Ruta base ===
base_dir = r"C:\Users\Josia\OneDrive\Escritorio\Dataset\Dataset"
metadata_path = r"C:\Users\Josia\Downloads\Dataset.xlsx"

# === Cargar Excel ===
df = pd.read_excel(metadata_path)

# === Parámetros de extracción ===
N_FFT = 512
HOP_LENGTH = 256

# === Función para extraer características ===
def extract_features(file_path):
    try:
        with sf.SoundFile(file_path) as f:
            _ = f.frames
        y, sr = librosa.load(file_path, sr=None)
        if len(y) < N_FFT:
            y = np.pad(y, (0, N_FFT - len(y)), mode='constant')
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=N_FFT,
                                    hop_length=HOP_LENGTH, n_mels=20, fmax=8000)
        delta = librosa.feature.delta(mfcc)
        f0 = librosa.yin(y, fmin=librosa.note_to_hz("C2"),
                         fmax=librosa.note_to_hz("C7"), sr=sr)

        return {
            **{f"MFCC_{i+1}": round(mfcc[i].mean(), 4) for i in range(13)},
            "f0 (Hz)": round(np.nanmean(f0), 4) if np.any(~np.isnan(f0)) else 0,
            "Centroid (Hz)": round(librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH).mean(), 4),
            "ZCR (Zero Crossing Rate)": round(librosa.feature.zero_crossing_rate(y, frame_length=N_FFT, hop_length=HOP_LENGTH).mean(), 6),
            "Spectral rolloff": round(librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH).mean(), 4),
            "Spectral flatness": round(librosa.feature.spectral_flatness(y=y, n_fft=N_FFT, hop_length=HOP_LENGTH).mean(), 6),
            "Crest factor": round(np.max(np.abs(y)) / (np.sqrt(np.mean(y**2)) + 1e-6), 4),
            "Spectral Bandwidth": round(librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH).mean(), 4),
            "RMS Energy": round(librosa.feature.rms(y=y, frame_length=N_FFT, hop_length=HOP_LENGTH).mean(), 6),
            "Delta MFCCs": round(delta.mean(), 4)
        }
    except Exception as e:
        print(f"❌ Error con {file_path}: {e}")
        return {col: None for col in [f"MFCC_{i+1}" for i in range(13)] + [
            "f0 (Hz)", "Centroid (Hz)", "ZCR (Zero Crossing Rate)", "Spectral rolloff",
            "Spectral flatness", "Crest factor", "Spectral Bandwidth", "RMS Energy", "Delta MFCCs"
        ]}

# === Procesar cada fila ===
features_list = []
for _, row in df.iterrows():
    nota = row["ID_archivo"].split("_")[0]
    intensidad = row["Símbolo_dinámica"]
    duracion = row["Clasificación_duración"]
    archivo = row["ID_archivo"]

    # Asegurarse de que tenga extensión .wav
    filename = archivo if archivo.endswith(".wav") else f"{archivo}.wav"

    # Construir ruta completa
    subfolder = f"{nota}_dinamica_{intensidad}"
    duracion_folder = f"{nota}_dinamica_{duracion}"
    full_path = os.path.join(base_dir, nota, subfolder, duracion_folder, filename)

    # Extraer características
    feats = extract_features(full_path)
    features_list.append(feats)

# === Unir los features al dataset original ===
df_features = pd.DataFrame(features_list)
df_final = pd.concat([df.iloc[:, :8], df_features, df["tonalidad_soplido"]], axis=1)

# === Guardar resultados ===
df_final.to_excel(r"C:\Users\Josia\OneDrive\Escritorio\Dataset_completo.xlsx", index=False)
df_final.to_csv(r"C:\Users\Josia\OneDrive\Escritorio\Dataset_completo.csv", index=False)

print("¡Listo! Archivo guardado como 'Dataset_completo.xlsx' y 'Dataset_completo.csv'")
