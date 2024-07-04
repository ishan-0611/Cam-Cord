import numpy as np
import streamlit as st
import os
import pandas as pd
import face_recognition
import uuid
import cv2

# Paths
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
VISITOR_DB = os.path.join(ROOT_DIR, "visitor_database")
file_db = 'visitors_db.csv'
file_encodings = 'face_encodings.csv'
COLS_INFO = ['ID', 'Name']
COLS_ENCODE = [f'v{i}' for i in range(128)]
allowed_image_type = ['.png', 'jpg', '.jpeg']


def initialize_data():
    if not os.path.exists(VISITOR_DB):
        os.mkdir(VISITOR_DB)

    if not os.path.exists(os.path.join(VISITOR_DB, file_db)):
        pd.DataFrame(columns=COLS_INFO).to_csv(os.path.join(VISITOR_DB, file_db), index=False)

    if not os.path.exists(os.path.join(VISITOR_DB, file_encodings)):
        pd.DataFrame(columns=['ID'] + COLS_ENCODE).to_csv(os.path.join(VISITOR_DB, file_encodings), index=False)


def add_data_db(pid, name, face_encoding):
    try:
        # Add to visitors_db.csv
        df_info = pd.read_csv(os.path.join(VISITOR_DB, file_db))
        df_info = pd.concat([df_info, pd.DataFrame([[pid, name]], columns=COLS_INFO)], ignore_index=True)
        df_info.to_csv(os.path.join(VISITOR_DB, file_db), index=False)

        # Add to face_encodings.csv
        df_encodings = pd.read_csv(os.path.join(VISITOR_DB, file_encodings))
        df_encoding = pd.DataFrame([[pid] + face_encoding.tolist()], columns=['ID'] + COLS_ENCODE)
        df_encodings = pd.concat([df_encodings, df_encoding], ignore_index=True)
        df_encodings.to_csv(os.path.join(VISITOR_DB, file_encodings), index=False)

        st.success('Details Added Successfully!')
    except Exception as e:
        st.error(e)


def database_page():
    initialize_data()

    col1, col2 = st.columns(2)

    face_name = col1.text_input('Enter Name')
    pic_option = col1.radio('Choose', options=['Upload Picture', 'Click Picture'])

    file_bytes = None
    img_file_buffer = None
    if pic_option == 'Upload Picture':
        img_file_buffer = col2.file_uploader('Upload Picture', type=allowed_image_type)

        if img_file_buffer is not None:
            file_bytes = np.asarray(bytearray(img_file_buffer.read()), dtype=np.uint8)

    elif pic_option == 'Click Picture':
        img_file_buffer = col2.camera_input('Click Picture')

        if img_file_buffer is not None:
            file_bytes = np.frombuffer(img_file_buffer.getvalue(), np.uint8)

    if img_file_buffer is not None and face_name:
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        face_encodings = face_recognition.face_encodings(image)

        if face_encodings:
            face_encoding = face_encodings[0]
            unique_id = str(uuid.uuid4())
            add_data_db(unique_id, face_name, face_encoding)
        else:
            st.error("No face detected. Please upload a clear image.")
