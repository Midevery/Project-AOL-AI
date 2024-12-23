import cv2
import pickle
import cvzone
import numpy as np
from tkinter import Tk, Label, Button, Scale, HORIZONTAL, filedialog


class ParkingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Parking Space Detection")
        self.root.geometry("500x450")
        self.video_path = None

        # Default values
        self.default_val1 = 25
        self.default_val2 = 16
        self.default_val3 = 5
        self.val1 = self.default_val1
        self.val2 = self.default_val2
        self.val3 = self.default_val3

        # UI Elements
        self.title_label = Label(root, text="Parking Space Detection App", font=("Arial", 18, "bold"))
        self.title_label.pack(pady=10)

        self.select_button = Button(root, text="Select Video", command=self.select_video, bg="#4CAF50", fg="white",
                                    font=("Arial", 12, "bold"))
        self.select_button.pack(pady=5)

        self.slider_label1 = Label(root, text="Threshold Val1 (Block Size)", font=("Arial", 12))
        self.slider_label1.pack()
        self.slider1 = Scale(root, from_=11, to=51, resolution=2, orient=HORIZONTAL, command=self.update_val1)
        self.slider1.set(self.val1)
        self.slider1.pack()

        self.slider_label2 = Label(root, text="Threshold Val2 (Constant C)", font=("Arial", 12))
        self.slider_label2.pack()
        self.slider2 = Scale(root, from_=1, to=50, orient=HORIZONTAL, command=self.update_val2)
        self.slider2.set(self.val2)
        self.slider2.pack()

        self.slider_label3 = Label(root, text="Threshold Val3 (Median Blur)", font=("Arial", 12))
        self.slider_label3.pack()
        self.slider3 = Scale(root, from_=1, to=15, resolution=2, orient=HORIZONTAL, command=self.update_val3)
        self.slider3.set(self.val3)
        self.slider3.pack()

        self.start_button = Button(root, text="Start Detection", command=self.start_detection, bg="#2196F3", fg="white",
                                   font=("Arial", 12, "bold"))
        self.start_button.pack(pady=10)

        self.default_button = Button(root, text="Set to Default", command=self.set_to_default, bg="#FFC107", fg="black",
                                     font=("Arial", 12, "bold"))
        self.default_button.pack(pady=5)

        self.quit_button = Button(root, text="Quit", command=root.quit, bg="#F44336", fg="white",
                                  font=("Arial", 12, "bold"))
        self.quit_button.pack(pady=10)

    def select_video(self):
        self.video_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.avi")])
        if self.video_path:
            self.title_label.config(text=f"Selected Video: {self.video_path.split('/')[-1]}")

    def update_val1(self, val):
        self.val1 = int(val)

    def update_val2(self, val):
        self.val2 = int(val)

    def update_val3(self, val):
        self.val3 = int(val)

    def set_to_default(self):
        """Reset all sliders to default values."""
        self.slider1.set(self.default_val1)
        self.slider2.set(self.default_val2)
        self.slider3.set(self.default_val3)

    def start_detection(self):
        if not self.video_path:
            self.title_label.config(text="Please select a video first!")
            return

        # Load parking position data
        try:
            with open('CarParkPos', 'rb') as f:
                pos_list = pickle.load(f)
        except FileNotFoundError:
            self.title_label.config(text="CarParkPos file not found!")
            return

        # Start Detection in a new window
        self.run_detection(pos_list)

    def run_detection(self, pos_list):
        cap = cv2.VideoCapture(self.video_path)
        width, height = 103, 43

        def check_spaces(img_pro, img):
            spaces = 0
            for pos in pos_list:
                x, y = pos
                img_crop = img_pro[y:y + height, x:x + width]
                count = cv2.countNonZero(img_crop)

                if count < 900:
                    color = (0, 255, 0)
                    thickness = 5
                    spaces += 1
                else:
                    color = (0, 0, 255)
                    thickness = 2

                cv2.rectangle(img, (x, y), (x + width, y + height), color, thickness)
                cvzone.putTextRect(img, str(count), (x, y + height - 3), scale=1, thickness=2, offset=0, colorR=color)

            cvzone.putTextRect(img, f'Free: {spaces}/{len(pos_list)}', (50, 50), scale=2, thickness=4,
                               offset=10, colorR=(0, 200, 0))

        while True:
            success, img = cap.read()
            if not success:
                break

            # Convert to grayscale and blur
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_blur = cv2.GaussianBlur(img_gray, (3, 3), 1)

            # Threshold parameters
            val1 = self.val1
            if val1 % 2 == 0:
                val1 += 1
            img_thresh = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                               cv2.THRESH_BINARY_INV, val1, self.val2)
            img_thresh = cv2.medianBlur(img_thresh, self.val3)
            kernel = np.ones((3, 3), np.uint8)
            img_thresh = cv2.dilate(img_thresh, kernel, iterations=1)

            check_spaces(img_thresh, img)

            # Show video
            cv2.imshow("Parking Space Detection", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    root = Tk()
    app = ParkingApp(root)
    root.mainloop()
