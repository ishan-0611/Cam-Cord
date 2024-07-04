import streamlit as st
import pandas as pd
import numpy as np
import cv2
import os
import face_recognition


def load_face_encodings(encodings_file):
    if os.path.exists(encodings_file):
        data = pd.read_csv(encodings_file)
        ids = data['ID'].tolist()
        face_encodings = data.drop(columns=['ID']).values
        return ids, face_encodings
    else:
        st.error("Encodings file not found.")
        return [], []


def validation_page():
    # Load face encodings and IDs
    encodings_file = 'visitor_database/face_encodings.csv'
    ids, known_face_encodings = load_face_encodings(encodings_file)

    # Load visitor database
    visitors_db = 'visitor_database/visitors_db.csv'
    if os.path.exists(visitors_db):
        df_visitors = pd.read_csv(visitors_db)
    else:
        st.error("Visitors database not found.")
        return

    validate = st.button('Start Webcam')
    not_validate = st.button('Stop Webcam')

    if validate:
        stframe = st.empty()
        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture video.")
                break

            # Resize frame for faster face recognition processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            # Find all face locations and face encodings in the frame
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"

                # If a match was found, use the first one
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    visitor_id = ids[best_match_index]
                    name = df_visitors.loc[df_visitors['ID'] == visitor_id, 'Name'].values[0]

                face_names.append(name)

            # Display the results
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                # Scale back up face locations since the frame was scaled to 1/4 size
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                # Draw a box around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

                # Draw a label with a name below the face
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

            # Display the resulting frame
            stframe.image(frame, channels="BGR")

            if not_validate:
                break

        cap.release()
