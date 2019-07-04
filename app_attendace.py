# Nishant Chaudhary


import face_recognition   #uses dlib which have ML algorithms like Deep Learning
import cv2
import numpy as np
import os

# Read photos and process them using face_recognition from photos folder
known_face_names = []
known_face_encodings = []
a = os.walk("photos")
for r, d, f in a:
    for ff in f:
        print(ff)
        known_face_names.append(ff.split(".")[0])
        face = face_recognition.load_image_file("photos/" + ff)
        known_face_encodings.append(face_recognition.face_encodings(face)[0])

# Read camera feed
camera_live_feed = cv2.VideoCapture(0)

# Frame processing.
face_in_frame_positions = []  # Box for drawing over detected faces
extracted_features_from_images = []  # Face info per frame
face_names = []  # Add to detected faces
should_process = True  # For processing speed up, do only one frame

# Save who are attending
attended = {}

for name in known_face_names:
    attended[name] = 0

# continue for each frame
while True:
    ret, frame = camera_live_feed.read()

    # convert from BGR to RGB for ndarray
    rgb_small_frame = frame[:, :, ::-1]

    # Should process this frame ?
    if should_process:
        # Let face_recognition module do its magic
        face_in_frame_positions = face_recognition.face_locations(rgb_small_frame)
        extracted_features_from_images = face_recognition.face_encodings(rgb_small_frame, face_in_frame_positions)

        # Detect faces from the magic's output
        face_names = []
        for face_encoding in extracted_features_from_images:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.55)
            name = None

            # face_distances is the measure of how similar the detected face is from known faces
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            # get the lowest measure, i.e., best matched face from known faces
            best_match_index = np.argmin(face_distances)

            # get name from the image and array of known face names
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
            # increment count for attended dictionary
            if name:
                if attended[name] != 1:
                    attended[name] += 1
            face_names.append(name)

    should_process = not should_process

    for (top, right, bottom, left), name in zip(face_in_frame_positions, face_names):
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # render frame with boxes around faces
    cv2.imshow('Video', frame)

    # wait for key and exit on 'q' press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        in_class = []
        not_in_class = []
        in_class = [x for x in attended if attended[x] == 1]
        not_in_class = [x for x in attended if attended[x] == 0]
        print("In class = ", in_class)
        print("Not in class = ", not_in_class)
        break

# release resources
camera_live_feed.release()
cv2.destroyAllWindows()