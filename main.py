import os
import cv2
import numpy as np
import streamlit as st
import urllib.request

# --- 1. DOWNLOAD CASCADE DETECTOR ---
CASCADE_FILE = "haarcascade_frontalface_default.xml"
if not os.path.exists(CASCADE_FILE):
    url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
    try:
        urllib.request.urlretrieve(url, CASCADE_FILE)
    except Exception as e:
        st.error(f"Gagal mengunduh file cascade: {e}")

face_cascade = cv2.CascadeClassifier(CASCADE_FILE)

# --- 2. CONFIG DIREKTORI & TRAINING ENGINE ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KNOWN_FACE_DIR = os.path.join(BASE_DIR, "known_face")
if not os.path.exists(KNOWN_FACE_DIR):
    os.makedirs(KNOWN_FACE_DIR)

recognizer = cv2.face.LBPHFaceRecognizer_create()

def train_known_faces():
    faces = []
    labels = []
    names_map = {}
    current_id = 0
    if os.path.exists(KNOWN_FACE_DIR):
        for filename in os.listdir(KNOWN_FACE_DIR):
            if filename.lower().endswith((".jpg", ".png", ".jpeg")):
                path = os.path.join(KNOWN_FACE_DIR, filename)
                name = os.path.splitext(filename)[0]
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if img is None: continue
                detected_faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5)
                for (x, y, w, h) in detected_faces:
                    faces.append(img[y:y+h, x:x+w])
                    labels.append(current_id)
                    names_map[current_id] = name
                current_id += 1
    if len(faces) > 0:
        recognizer.train(faces, np.array(labels))
    return names_map

NAMES_MAP = train_known_faces()

# --- 3. INTERFACE WEB STREAMLIT ---
st.set_page_config(page_title="Smart CCTV", layout="centered")
st.title("SISTEM KEAMANAN CCTV")

if "audio_enabled" not in st.session_state: 
    st.session_state["audio_enabled"] = False

menu = st.sidebar.selectbox("Menu", ["Monitoring Live CCTV", "Daftarkan Wajah Baru"])

if menu == "Monitoring Live CCTV":
    if not st.session_state["audio_enabled"]:
        if st.button("🔊 Klik Untuk Mengaktifkan Suara Alarm"):
            st.session_state["audio_enabled"] = True
            st.rerun()
    else:
        st.success("🔊 Suara alarm aktif!")

    # Menggunakan camera_input bawaan Streamlit yang jauh lebih stabil di server cloud
    img_file = st.camera_input("Nyalakan Kamera CCTV")

    if img_file is not None:
        # Konversi gambar dari browser ke format OpenCV (BGR)
        file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
        has_intruder = False
        
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            name = "PENYUSUP"
            color = (0, 0, 255)
            
            if len(NAMES_MAP) > 0:
                try:
                    label_id, confidence = recognizer.predict(roi_gray)
                    if confidence < 90:
                        name = NAMES_MAP.get(label_id, "PENYUSUP")
                        color = (0, 255, 0)
                    else:
                        has_intruder = True
                except:
                    has_intruder = True
            else:
                has_intruder = True
            
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 3)
            cv2.putText(img, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # Tampilkan hasil pemrosesan CCTV ke layar
        st.image(img, channels="BGR", caption="Live Monitoring Hasil Deteksi")

        # Bunyikan alarm jika terdeteksi penyusup
        if has_intruder:
            st.error("🚨 PENYUSUP TERDETEKSI!")
            if st.session_state.get("audio_enabled"):
                st.audio("https://actions.google.com/sounds/v1/alarms/digital_watch_alarm_long.ogg", autoplay=True)

elif menu == "Daftarkan Wajah Baru":
    st.header("Form Pendaftaran Wajah Baru")
    new_name = st.text_input("Masukkan Nama Lengkap:")
    img_file = st.camera_input("Ambil Foto Wajah")
    
    if img_file is not None and new_name:
        file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
        opencv_img = cv2.imdecode(file_bytes, 1)
        
        if st.button(f"Simpan Wajah {new_name}"):
            file_path = os.path.join(KNOWN_FACE_DIR, f"{new_name.strip().replace(' ', '_')}.jpg")
            cv2.imwrite(file_path, opencv_img)
            st.success(f"Berhasil mendaftarkan wajah: {new_name}!")
            NAMES_MAP = train_known_faces()
            st.rerun()
    elif img_file and not new_name:
        st.warning("Silakan isi nama terlebih dahulu sebelum mengambil foto.")
