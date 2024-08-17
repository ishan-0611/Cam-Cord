
# Cam-Cord

Cam-Cord is a real-time face recognition application that utilizes OpenCV and the `face_recognition` library to capture, detect, and recognize faces from live video streams. This project is designed to identify individuals by comparing detected faces against pre-stored face encodings, making it suitable for applications like surveillance, authentication, and more.

## Features

- **Real-Time Face Detection:** Capture and detect faces from live video streams using OpenCV.
- **Face Recognition:** Identify or verify faces by comparing them with stored face encodings.
- **Efficient Frame Processing:** Optimized for real-time processing, ensuring quick and accurate face recognition.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/ishan-0611/Cam-Cord.git
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   python cam_cord.py
   ```

## Usage

- Ensure your webcam is connected and functioning.
- Run the application to start capturing and recognizing faces.
- The recognized faces will be displayed with corresponding labels, if matched with stored encodings.

## Technologies Used

- Python
- OpenCV
- face_recognition library
- Numpy

## License

This project is licensed under the MIT License.
