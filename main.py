import streamlit as st
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

# Setup detektor wajah (Pastikan file cascade ada)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

class CCTVProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)
        
        # Jika ada wajah, gambar kotak merah
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
            # Simpan status di session state agar bisa dibaca di luar class
            st.session_state["alert"] = True
            
        return frame.from_ndarray(img, format="bgr24")

st.title("Smart CCTV")
if "alert" not in st.session_state:
    st.session_state["alert"] = False

# Tombol untuk memulai kamera
webrtc_streamer(key="cctv", video_processor_factory=CCTVProcessor)

# Peringatan visual jika deteksi aktif
if st.session_state.get("alert"):
    st.error("PENYUSUP TERDETEKSI!")
    # Bunyikan alarm lewat browser menggunakan audio tag sederhana
    st.audio("https://actions.google.com/sounds/v1/alarms/digital_watch_alarm_long.ogg", autoplay=True)
    st.session_state["alert"] = False
