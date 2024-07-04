import cv2
import face_recognition
import numpy
import numpy as np
import pandas as pd


def load_face_encodings(encod_file):
    data = pd.read_csv(encod_file)
    ids = data['ID'].tolist()
    face_encodings = data.drop(columns=['ID']).values
    return ids, face_encodings


def recognize_faces(path, encod_file):
    ids, known_face_encodings = load_face_encodings(encod_file)

    # Open the input movie file
    input_movie = cv2.VideoCapture(path)
    length = int(input_movie.get(cv2.CAP_PROP_FRAME_COUNT))

    while input_movie.isOpened():
        # Grab a single frame of video
        ret, frame = input_movie.read()
        if not ret:
            break

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_frame = numpy.ascontiguousarray(frame[:, :, ::-1])

        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        # Loop through each face found in the frame to see if it's someone we know
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # Use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = ids[best_match_index]

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        # Display the resulting image
        cv2.imshow('Video', frame)

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release handle to the webcam
    input_movie.release()
    cv2.destroyAllWindows()


video_path = '/Users/ishanchaturvedi/Downloads/Movie on 03-07-24 at 7.28 PM.mov'
encodings_file = 'visitor_database/face_encodings.csv'

recognize_faces(video_path, encodings_file)
