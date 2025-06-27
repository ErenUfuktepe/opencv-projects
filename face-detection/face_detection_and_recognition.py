import os
import cv2
import numpy as np
import concurrent.futures
import pandas as pd
import tkinter as tk
import face_recognition
from tkinter import filedialog as fd
from PIL import Image, ImageDraw, ImageFont


def face_recognition_from_camera():
    """
    Launches the webcam and performs real-time face recognition.

    Uses known face encodings from the global DataFrame `df` to identify faces.
    Draws rectangles and labels around recognized faces in the video stream.

    Globals used:
    - camera: window title for OpenCV display
    - unknown: label used when face is not recognized
    - df: DataFrame containing 'name' and 'encoding' columns
    """
    global camera, unknown, df
    try:
        video_cap = cv2.VideoCapture(0)

        while video_cap.isOpened():
            _, frame = video_cap.read()
            frame = cv2.flip(frame, 1)
            # opencv captures images in bgr format by default.
            # Reversing the channels turn it into rgb format.
            rgb_frame = np.ascontiguousarray(frame[:, :, ::-1])

            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                matches = face_recognition.compare_faces(df['encoding'].tolist(), face_encoding)
                name = df['name'].iloc[matches.index(True)] if True in matches else unknown
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.rectangle(frame, (left, bottom - 25), (right, bottom), (0, 0, 255), cv2.FILLED)
                cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow(camera, frame)
            cv2.setWindowProperty(camera, cv2.WND_PROP_TOPMOST, 1)
            cv2.waitKey(1)
            if cv2.getWindowProperty(camera, cv2.WND_PROP_VISIBLE) < 1:
                break

        video_cap.release()
        cv2.destroyAllWindows()
    except Exception as exception:
        raise Exception(exception)


def detection_with_haarcascade():
    """
    Launches the webcam and performs real-time detection using Haar cascades.

    Depending on the global `haarcascade_mode`, detects faces, eyes, or smiles.
    Draws rectangles around detected regions in the video stream.

    Globals used:
    - haarcascade_mode: one of 'face', 'eye', or 'smile'
    - camera: OpenCV window title
    - face_classifier, eye_classifier, smile_classifier: loaded Haar cascades
    """
    global window, camera, haarcascade_mode, face_classifier, eye_classifier, smile_classifier
    try:
        video_cap = cv2.VideoCapture(0)
        while video_cap.isOpened():
            objects = None
            _, frame = video_cap.read()
            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            match haarcascade_mode:
                case 'face':
                    objects = face_classifier.detectMultiScale(gray, 1.3, 5)
                case 'eye':
                    objects = eye_classifier.detectMultiScale(gray, 1.1, 20)
                case 'smile':
                    objects = smile_classifier.detectMultiScale(gray, 1.7, 22)

            for x_, y_, width_, height_ in objects:
                cv2.rectangle(frame, (x_, y_), (x_ + width_, y_ + height_), (255, 0, 0), 2)

            cv2.imshow(camera, frame)
            cv2.setWindowProperty(camera, cv2.WND_PROP_TOPMOST, 1)
            cv2.waitKey(1)
            if cv2.getWindowProperty(camera, cv2.WND_PROP_VISIBLE) < 1:
                break
        video_cap.release()
        cv2.destroyAllWindows()
    except Exception as exception:
        raise Exception(exception)


def change_haarcascade_mode(mode):
    """
    Updates the Haar cascade detection mode.

    Args:
        mode (str): The mode to switch to. Expected values: 'face', 'eye', or 'smile'.

    Returns:
        str: The new mode that was set.

    Globals modified:
        - haarcascade_mode
    """
    global haarcascade_mode
    try:
        haarcascade_mode = mode
        return haarcascade_mode
    except Exception as exception:
        raise Exception(exception)


def create_haarcascade_setting_window():
    """
    Creates a Tkinter interface with buttons to switch between Haar cascade modes.

    Submits the Haar cascade detection function to a thread pool and displays UI
    buttons for selecting face, smile, or eye detection.

    Globals used:
    - window: the main Tkinter window
    - pool: a concurrent.futures.ThreadPoolExecutor used for running detection
    """
    global window, pool
    try:
        pool.submit(detection_with_haarcascade)

        # Remove all the buttons
        for widget in window.winfo_children():
            if isinstance(widget, tk.Button):
                widget.destroy()

        face_detection_button = tk.Button(window, text="Face Detection", width=20,
                                          command=lambda: change_haarcascade_mode('face'))
        face_detection_button.place(x=55, y=30)

        smile_detection_button = tk.Button(window, text="Smile Detection", width=20,
                                           command=lambda: change_haarcascade_mode('smile'))
        smile_detection_button.place(x=55, y=60)

        eye_detection_button = tk.Button(window, text="Eye Detection", width=20,
                                         command=lambda: change_haarcascade_mode('eye'))
        eye_detection_button.place(x=55, y=90)
    except Exception as exception:
        raise Exception(exception)


def learn_new_face():
    """
    Opens a file dialog for selecting an image, detects faces, and collects encodings.

    Displays the detected faces with bounding boxes and allows the user to label them.
    Displays each face one-by-one for review and collection.

    Globals used:
    - window: Tkinter main window
    - file_types: file filter for the open file dialog
    - df: DataFrame holding existing face encodings and names
    - unknown: label used for unrecognized faces
    """
    global window, file_types, df, unknown
    try:
        # Remove all the buttons
        for widget in window.winfo_children():
            if isinstance(widget, tk.Button):
                widget.destroy()

        # Labels and entries
        name_ = tk.StringVar()
        # name_.set("5")
        name_label = tk.Label(window, text="Person Name")
        name_label.place(x=10, y=50)
        name_entry = tk.Entry(window, width=20, textvariable=name_)
        name_entry.place(x=90, y=50)

        faces = []
        submit_button = tk.Button(window, text="Submit", width=20, command=lambda: submit_button_action(faces, name_))
        submit_button.place(x=55, y=90)

        filename_ = fd.askopenfilename(
            title='Select a image file',
            initialdir='/',
            filetypes=file_types)

        if not filename_:
            window.destroy()
            return

        image = face_recognition.load_image_file(filename_)
        face_locations = face_recognition.face_locations(image)
        face_encodings = face_recognition.face_encodings(image, face_locations)

        counter = 1
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(df['encoding'].tolist(), face_encoding, tolerance=0.5)
            name = df['name'].iloc[matches.index(True)] if True in matches else unknown

            pil_image = Image.fromarray(image)
            draw = ImageDraw.Draw(pil_image)

            draw.rectangle(((left, top), (right, bottom)), outline=(255, 255, 0))
            font = ImageFont.truetype("arial.ttf", 36)
            _, top, _, _ = draw.textbbox((0, 0), name, font=font)
            draw.rectangle(((left, bottom - top - 10), (right, bottom)), fill=(255, 255, 0),
                           outline=(255, 255, 0))
            draw.text((left + 6, bottom - top - 5), name, fill=(0, 0, 0))
            if len(face_encodings) == counter:
                pil_image.show()
            faces.append((pil_image, face_encoding))
            counter += 1
        if len(face_encodings) != 0:
            del draw
    except Exception as exception:
        raise Exception(exception)


def submit_button_action(faces, name=None):
    """
    Handles the logic for saving a newly labeled face and updating the DataFrame.

    Args:
        faces (list): A list of (PIL image, face encoding) tuples pending labeling.
        name (tk.StringVar): Tkinter variable containing the user input name.

    Saves the collected face encodings into the global DataFrame `df`, and writes it
    to a Parquet file once all images have been processed.

    Globals used:
    - df: DataFrame to store names and encodings
    - path: directory to save the parquet file
    - filename: name of the parquet file
    - window: Tkinter main window
    """
    global window, df, path, filename
    try:
        if len(faces) != 0:
            image, encoding = faces.pop()
            data = {
                'name': name.get(),
                'encoding': encoding
            }
            name.set("")
            df.loc[len(df)] = data

            if len(faces) == 0:
                df.to_parquet(f"{path}/{filename}")
                window.destroy()
                return

            next_image, next_encoding = faces[0]
            next_image.show()
    except Exception as exception:
        raise Exception(exception)


file_types = (
    ('Image files', '*.jpg'),
    ('Image files', '*.jpeg'),
    ('Image files', '*.png')
)

pool = concurrent.futures.ThreadPoolExecutor(max_workers=2)

unknown = "Unknown"
camera = "Camera"
haarcascade_mode = 'face'
filename = 'faces.parquet'

path = os.path.join(os.getcwd(), 'data')
os.makedirs(path, exist_ok=True)

# Create a Parquet file if it doesn't exist to store the face encodings; otherwise, read the existing Parquet file.
if filename in os.listdir(path):
    df = pd.read_parquet(f"{path}/{filename}")
else:
    df = pd.DataFrame(columns=['name', 'encoding'])
    df.to_parquet(f"{path}/{filename}")

# https://github.com/opencv/opencv/tree/master/data/haarcascades
face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

eye_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)

smile_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_smile.xml"
)

window = tk.Tk()
window.title("Face Detection and Recognition")

width, height = 250, 200
ws = window.winfo_screenwidth()  # width of the screen
hs = window.winfo_screenheight()  # height of the screen
# Setting the window to open in the middle of the screen
x = int((ws / 2) - int(width / 2))
y = int((hs / 2) - int(height / 2))
window.geometry('%dx%d+%d+%d' % (width, height, x, y))

face_recognition_camera_button = tk.Button(window, text="Face Recognition Camera", width=20,
                                           command=face_recognition_from_camera)
face_recognition_camera_button.place(x=55, y=30)

haarcascade_detection_button = tk.Button(window, text="Haarcascade Detection", width=20,
                                         command=create_haarcascade_setting_window)
haarcascade_detection_button.place(x=55, y=60)

learn_new_face_button = tk.Button(window, text="Learn New Face", width=20, command=learn_new_face)
learn_new_face_button.place(x=55, y=90)

window.attributes('-topmost', 'true')  # keep it on top
window.mainloop()
