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
    intruder_flag = False

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

        # Update status global
        CCTVVideoProcessor.intruder_flag = has_intruder

        return frame.from_ndarray(img, format="bgr24")

# --- 4. INTERFACE WEB STREAMLIT ---
st.set_page_config(page_title="Smart CCTV Live Web", layout="centered")
st.title("SISTEM KEAMANAN - LIVE MONITORING")

menu = st.sidebar.selectbox("Pilih Menu", ["Monitoring Live CCTV", "Daftarkan Wajah Baru"])

if menu == "Monitoring Live CCTV":
    st.subheader("Live Feed Kamera Pengawas")
    
    # Elemen placeholder audio diletakkan di paling atas halaman web utama
    sound_placeholder = st.empty()

    ctx = webrtc_streamer(
        key="cctv-final-cloud",
        video_processor_factory=lambda: CCTVVideoProcessor(NAMES_MAP),
        media_stream_constraints={
            "video": {"width": 640, "height": 480, "frameRate": 15},
            "audio": False
        },
        rtc_configuration={
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        }
    )

    # Indikator Status di Web Utama
    status_placeholder = st.empty()

    # Selama kamera streaming aktif, dengarkan perubahan variabel kelas secara konstan
    if ctx.state.playing:
        while True:
            if CCTVVideoProcessor.intruder_flag:
                status_placeholder.error("🚨 PERINGATAN: TERDETEKSI PENYUSUP DI AREA MONITORING!")
                # Langsung suntikkan elemen audio murni ke HTML browser tanpa nunggu siklus rerun
                sound_placeholder.markdown(
                    """
                    <iframe src="https://actions.google.com/sounds/v1/alarms/digital_watch_alarm_long.ogg" allow="autoplay" style="display:none" id="iframeAudio"></iframe>
                    <audio autoplay loop hidden>
                        <source src="https://actions.google.com/sounds/v1/alarms/digital_watch_alarm_long.ogg" type="audio/ogg">
                    </audio>
                    """, 
                    unsafe_allow_html=True
                )
            else:
                status_placeholder.success("✅ Situasi Aman. Tidak ada penyusup terdeteksi.")
                sound_placeholder.empty()

elif menu == "Daftarkan Wajah Baru":
    st.subheader("Registrasi Pemilik Wajah Baru")
    nama_baru = st.text_input("Masukkan Nama Pemilik Wajah:")
    upload_foto = st.camera_input("Ambil Foto Lewat Kamera")

    if st.button("Simpan ke Database") and nama_baru and upload_foto:
        file_bytes = np.asarray(bytearray(upload_foto.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, 1)
        
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces_detected = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)
        
        if len(faces_detected) == 0:
            st.warning("Wajah tidak terdeteksi dengan jelas. Coba lagi.")
        else:
            path_simpan = os.path.join(KNOWN_FACE_DIR, f"{nama_baru}.jpg")
            cv2.imwrite(path_simpan, frame)
            st.success(f"Berhasil disimpan! Silakan klik menu Monitoring kembali.")
