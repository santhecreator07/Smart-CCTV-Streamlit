import os
import cv2
import numpy as np
import streamlit as st
import urllib.request
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

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

# --- KONFIGURASI WebRTC (STUN SERVER) ---
# Wajib agar WebRTC bisa berjalan lancar di server cloud / hosting publik
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# --- 3. ENGINE MONITORING VIDEO ---
class CCTVVideoProcessor(VideoProcessorBase):
    def __init__(self, names_map):
        self.names_map = names_map
        # Menggunakan properti internal class, BUKAN session_state agar aman dari crash
        self.has_intruder = False 

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
        
        self.has_intruder = False
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            name = "PENYUSUP"
            color = (0, 0, 255)
            
            if len(self.names_map) > 0:
                try:
                    label_id, confidence = recognizer.predict(roi_gray)
                    if confidence < 90:
                        name = self.names_map.get(label_id, "PENYUSUP")
                        color = (0, 255, 0)
                    else:
                        self.has_intruder = True
                except:
                    self.has_intruder = True
            else:
                self.has_intruder = True
            
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 3)
            cv2.putText(img, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        return frame.from_ndarray(img, format="bgr24")

# --- 4. INTERFACE WEB STREAMLIT ---
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

    # Memanggil streamer dengan konfigurasi RTC dan menyimpan konteksnya
    ctx = webrtc_streamer(
        key="cctv", 
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=lambda: CCTVVideoProcessor(NAMES_MAP),
        media_stream_constraints={"video": True, "audio": False}, # Mematikan mic agar tidak feedback sound
    )

    # Cara aman membaca status deteksi dari video processor thread ke UI utama
    if ctx.video_processor:
        if ctx.video_processor.has_intruder:
            st.error("🚨 PENYUSUP TERDETEKSI!")
            if st.session_state.get("audio_enabled"):
                st.audio("https://actions.google.com/sounds/v1/alarms/digital_watch_alarm_long.ogg", autoplay=True)

elif menu == "Daftarkan Wajah Baru":
    # (Kode registrasi kamu tetap sama di sini)
    st.info("Halaman pendaftaran wajah baru.")
