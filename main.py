import os
import cv2
import numpy as np
import streamlit as st
import urllib.request
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

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
                if img is None:
                    continue
                    
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

# --- 3. ENGINE MONITORING VIDEO ---
class CCTVVideoProcessor(VideoProcessorBase):
    def __init__(self, names_map):
        self.names_map = names_map

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
        has_intruder = False

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            name = "PENYUSUP"
            color = (0, 0, 255)  # Merah
            display_text = "PENYUSUP"
            
            if len(self.names_map) > 0:
                try:
                    label_id, confidence_value = recognizer.predict(roi_gray)
                    similarity = round(max(0, (100 - confidence_value)), 1)
                    
                    if confidence_value < 90: 
                        name = self.names_map.get(label_id, "PENYUSUP")
                        color = (0, 255, 0)  # Hijau jika aman
                        display_text = f"{name} ({similarity}%)"
                    else:
                        has_intruder = True
                except:
                    has_intruder = True
            else:
                has_intruder = True

            cv2.rectangle(img, (x, y), (x + w, y + h), color, 3)
            cv2.putText(img, display_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        return frame.from_ndarray(img, format="bgr24")

# --- 4. INTERFACE WEB STREAMLIT ---
st.set_page_config(page_title="Smart CCTV Live Web", layout="centered")
st.title("SISTEM KEAMANAN - LIVE MONITORING")

menu = st.sidebar.selectbox("Pilih Menu", ["Monitoring Live CCTV", "Daftarkan Wajah Baru"])

if menu == "Monitoring Live CCTV":
    st.subheader("Live Feed Kamera Pengawas")
    st.write("Sistem mendeteksi pergerakan wajah dan mencocokkan kemiripan database secara real-time.")

    # TOMBOL PEMANCING SUARA (Trik Bypass Kebijakan Keamanan Browser)
    st.warning("⚠️ KLIK TOMBOL DI BAWAH INI SATU KALI SEBELUM MEMULAI KAMERA AGAR ALARM BISA BERBUNYI!")
    tombol_aktif = st.button("🔊 Aktifkan Sistem Audio Alarm")

    # Injeksi Pemutar Suara Konstan berbasis HTML5 Audio dengan Audio Context Generator murni
    st.components.v1.html(
        """
        <div style="background-color: #f1f3f4; padding: 10px; border-radius: 5px; text-align: center;">
            <p style="margin: 0; font-family: sans-serif; font-size: 14px; color: #3c4043;">Status Alarm Browser: <b>Siap Menembak Suara</b></p>
            <button id="test-btn" onclick="initAudio()" style="margin-top: 5px; padding: 5px 10px; border: none; background: #1a73e8; color: white; border-radius: 4px; cursor: pointer;">Tes Koneksi Suara Browser</button>
        </div>

        <script>
        var audioCtx = null;

        function initAudio() {
            if (!audioCtx) {
                audioCtx = new (window.AudioContext || window.webkitAudioContext)();
            }
            // Bunyi tes konfirmasi klik berhasil
            var osc = audioCtx.createOscillator();
            var gain = audioCtx.createGain();
            osc.connect(gain);
            gain.connect(audioCtx.destination);
            osc.frequency.value = 1000;
            gain.gain.setValueAtTime(0.2, audioCtx.currentTime);
            osc.start();
            osc.stop(audioCtx.currentTime + 0.1);
        }

        // Loop independen di sisi klien: Memaksa bunyi jika ada elemen gambar canvas/video aktif dari WebRTC
        setInterval(function() {
            if (audioCtx) {
                // Mencari elemen video streaming dari webrtc yang sedang aktif di halaman induk
                var frames = window.parent.document
