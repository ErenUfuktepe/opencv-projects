import cv2
import concurrent.futures
import tkinter as tk
import numpy as np
from tkinter.colorchooser import askcolor


def close_settings():
    """
    This function closes the settings window and sets the close flag to True, which signals the OpenCV window to close.
    :return:
    """
    global settings, close_flag
    close_flag = True
    settings.destroy()


def change_drawing_mode(mode):
    """
    This function switches the drawing mode and disables its associated button.
    :param mode: (str) drawing mode
    :return: (str) Returns the new drawing mode.
    """
    global drawing_mode, pen_button, rectangle_button, circle_button
    try:
        drawing_mode = mode
        match mode:
            case 'pen':
                pen_button.config(state="disabled")
                rectangle_button.config(state="normal")
                circle_button.config(state="normal")
            case 'rectangle':
                pen_button.config(state="normal")
                rectangle_button.config(state="disabled")
                circle_button.config(state="normal")
            case 'circle':
                pen_button.config(state="normal")
                rectangle_button.config(state="normal")
                circle_button.config(state="disabled")
            case _:
                raise Exception("Invalid drawing mode.")
        return drawing_mode
    except Exception as exception:
        raise Exception(exception)


def change_color():
    """
    This function changes the drawing color.
    :return: (tuple) Returns RGB color code.
    """
    global color_button, drawing_color
    try:
        color_picker = askcolor(color=drawing_color, title="Colors")
        drawing_color = color_picker[0]
        color_button.configure(bg=color_picker[1])
        return drawing_color
    except Exception as exception:
        raise Exception(exception)


def change_background_color():
    """
    This function changes the board’s color to either white or black.
    :return: (ndarray) Returns N-dimensional array
    """
    global background_color, board
    try:
        if background_color == 0:
            board[board > -1] = 255
        else:
            board[board > -1] = 0
        background_color = 0 if background_color == 255 else 255
        return background_color
    except Exception as exception:
        raise Exception(exception)


def draw_line(event, x_, y_, is_drawing, param):
    global cursor_x, cursor_y, board, drawing_color, pen_thickness
    """
    This function enables pen drawing mode to write or draw lines.
    """
    try:
        pen_thickness_ = int(pen_thickness.get())
    except ValueError:
        pen_thickness_ = 5

    # Left click button action.
    if event == cv2.EVENT_LBUTTONDOWN:
        cursor_x, cursor_y = x_, y_
    # Mouse movement action when clicked.
    elif event == cv2.EVENT_MOUSEMOVE:
        if is_drawing:
            cv2.line(board, (cursor_x, cursor_y), (x_, y_), tuple(reversed(drawing_color)), pen_thickness_)
            cursor_x, cursor_y = x_, y_
    # Left button release action.
    elif event == cv2.EVENT_LBUTTONUP:
        cv2.line(board, (cursor_x, cursor_y), (x_, y_), tuple(reversed(drawing_color)), pen_thickness_)


def draw_rectangle(event, x_, y_, is_drawing, param):
    """
    This function enables rectangle drawing mode to draw rectangle.
    """
    global cursor_x, cursor_y, board, drawing_color
    # Left button click action.
    if event == cv2.EVENT_LBUTTONDOWN:
        cursor_x, cursor_y = x_, y_
    # Mouse movement action when clicked.
    elif event == cv2.EVENT_MOUSEMOVE:
        if is_drawing:
            cv2.rectangle(board, (cursor_x, cursor_y), (x_, y_), tuple(reversed(drawing_color)), -1)
    # Left button release action.
    elif event == cv2.EVENT_LBUTTONUP:
        cv2.rectangle(board, (cursor_x, cursor_y), (x_, y_), tuple(reversed(drawing_color)), -1)


def draw_circle(event, x_, y_, is_drawing, param):
    """
    This function enables circle drawing mode to draw circle.
    """
    global cursor_x, cursor_y, board, drawing_color, circle_size

    try:
        circle_size_ = int(circle_size.get())
    except ValueError:
        circle_size_ = 10

    # Left click event.
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(board, (x_, y_), circle_size_, tuple(reversed(drawing_color)), 0)
    # Right click event.
    if event == cv2.EVENT_RBUTTONDOWN:
        cv2.circle(board, (x_, y_), circle_size_, tuple(reversed(drawing_color)), -1)


def create_board():
    """
    Opens an OpenCV window called 'Paint Board' to enable interactive drawing.
    """
    global close_flag, board, drawing_mode
    title = "Paint Board"
    cv2.namedWindow(title)
    # Place the board to left top corner
    cv2.moveWindow(title, 0, 0)
    while True:
        match drawing_mode:
            case 'pen':
                cv2.setMouseCallback(title, draw_line)
            case 'rectangle':
                cv2.setMouseCallback(title, draw_rectangle)
            case 'circle':
                cv2.setMouseCallback(title, draw_circle)
            case _:
                raise Exception("Invalid drawing mode.")
        cv2.imshow(title, board)
        if close_flag or (cv2.waitKey(1) & 0xFF == ord('q')):
            cv2.destroyAllWindows()
            break


# 0 is black and 255 is white
background_color = 0
drawing_color = (255, 255, 255)
drawing_mode = "pen"
board = np.zeros((700, 700, 3), np.uint8)

# For tracking close event activity
close_flag = False
cursor_x, cursor_y = -1, -1

settings = tk.Tk()
settings.title("Settings")

settings_width, settings_height = 450, 100
ws = settings.winfo_screenwidth()  # width of the screen
hs = settings.winfo_screenheight()  # height of the screen
x = int((ws / 2) - int(settings_width / 2))
y = int((hs / 2) - int(settings_height / 2))

settings.geometry('%dx%d+%d+%d' % (settings_width, settings_height, x, y))
settings.attributes('-topmost', 'true')  # keep it on top
settings.protocol("WM_DELETE_WINDOW", close_settings)

# Buttons
rectangle_button = tk.Button(settings, text="Rectangle", command=lambda: change_drawing_mode('rectangle'))
rectangle_button.place(x=10, y=10)
circle_button = tk.Button(settings, text="Circle", command=lambda: change_drawing_mode('circle'))
circle_button.place(x=80, y=10)
pen_button = tk.Button(settings, text="Pen", command=lambda: change_drawing_mode('pen'))
pen_button.place(x=130, y=10)
pen_button.config(state="disabled")
color_button = tk.Button(settings, text="Change Color", command=lambda: change_color())
color_button.place(x=165, y=10)
background_button = tk.Button(settings, text="Change Background Color", command=lambda: change_background_color())
background_button.place(x=255, y=10)

# Labels and entries
pen_thickness = tk.StringVar()
pen_thickness.set("5")
pen_thickness_label = tk.Label(settings, text="Pen Thickness")
pen_thickness_label.place(x=10, y=50)
pen_thickness_entry = tk.Entry(settings, width=5, textvariable=pen_thickness)
pen_thickness_entry.place(x=100, y=50)

circle_size = tk.StringVar()
circle_size.set("10")
circle_size_label = tk.Label(settings, text="Circle Size")
circle_size_label.place(x=150, y=50)
circle_size_entry = tk.Entry(settings, width=5, textvariable=circle_size)
circle_size_entry.place(x=220, y=50)

# Thread pool for open cv window
pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
pool.submit(create_board)

settings.mainloop()
pool.shutdown(wait=True)
