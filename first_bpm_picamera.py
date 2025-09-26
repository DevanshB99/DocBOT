import sys
import cv2
import dlib
import numpy as np
from scipy import signal
from scipy.signal import butter, filtfilt, find_peaks
from filterpy.kalman import KalmanFilter
import time
import requests
import threading
from queue import Queue, Empty
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QPushButton
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
import pyqtgraph as pg

class NetworkCameraThread(QThread):
    """Improved threaded camera reader for Pi MJPEG stream"""
    frame_ready = pyqtSignal(np.ndarray)
    
    def __init__(self, pi_ip, port=8080):
        super().__init__()
        self.url = f"http://{pi_ip}:{port}/video_feed"#/stream.mjpg"
        self.session = None
        self.running = False
        self.connected = False
        
    def connect(self):
        """Connect to Pi camera stream"""
        try:
            print(f"Connecting to Pi camera at {self.url}")
            self.session = requests.Session()
            # Test connection first
            response = self.session.get(self.url, stream=True, timeout=5)
            if response.status_code == 200:
                self.connected = True
                self.stream = response
                print("Successfully connected to Pi camera")
                return True
            else:
                print(f"Failed to connect: HTTP {response.status_code}")
                return False
        except Exception as e:
            print(f"Connection error: {e}")
            return False
    
    def run(self):
        """Main thread loop for reading MJPEG frames"""
        self.running = True
        buffer = b''
        
        try:
            for chunk in self.stream.iter_content(chunk_size=1024):
                if not self.running:
                    break
                    
                if chunk:
                    buffer += chunk
                    
                    # Look for complete JPEG frames
                    while True:
                        # Find JPEG start and end markers
                        start = buffer.find(b'\xff\xd8')  # JPEG SOI
                        if start == -1:
                            break
                            
                        end = buffer.find(b'\xff\xd9', start + 2)  # JPEG EOI
                        if end == -1:
                            break
                            
                        # Extract complete JPEG
                        jpeg_data = buffer[start:end + 2]
                        buffer = buffer[end + 2:]
                        
                        # Decode and emit frame
                        try:
                            if len(jpeg_data) > 1000:  # Valid JPEG size
                                nparr = np.frombuffer(jpeg_data, np.uint8)
                                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                                if frame is not None and frame.shape[0] > 100:
                                    # Resize to match your app's expected resolution
                                    if frame.shape[1] > 800:
                                        scale = 800 / frame.shape[1]
                                        new_width = 800
                                        new_height = int(frame.shape[0] * scale)
                                        frame = cv2.resize(frame, (new_width, new_height))
                                    
                                    self.frame_ready.emit(frame)
                                    
                        except Exception as e:
                            print(f"Frame decode error: {e}")
                            continue
                        
                        # Limit buffer size
                        if len(buffer) > 100000:
                            buffer = buffer[-50000:]
                            
        except Exception as e:
            print(f"Stream reading error: {e}")
        finally:
            self.connected = False
            print("Camera thread stopped")
    
    def stop(self):
        """Stop the camera thread"""
        self.running = False
        self.connected = False
        if self.session:
            try:
                self.session.close()
            except:
                pass
        self.quit()
        self.wait(3000)  # Wait up to 3 seconds for clean shutdown

class HeartRateFilter:
    def __init__(self):
        self.kf = KalmanFilter(dim_x=2, dim_z=1)
        self.kf.x = np.array([[65.], [0.]])
        self.kf.F = np.array([[1., 0.5], [0., 1.]])
        self.kf.H = np.array([[1., 0.]])
        self.kf.P *= 100.
        self.kf.R = 25
        self.kf.Q = np.array([[0.05, 0.05], [0.05, 0.05]])

    def update(self, measurement):
        self.kf.predict()
        self.kf.update(np.array([[measurement]]))
        return self.kf.x[0][0]

class App(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Heart Rate Monitor - Pi Camera Stream')
        self.camera_active = False

        # Configuration - Update this with your Pi's IP address
        self.PI_IP = "192.168.0.242"  # CHANGE THIS TO YOUR PI'S IP ADDRESS
        self.PI_PORT = 5000  # Pi camera server port

        # Performance tracking
        self.last_frame_time = time.time()
        self.frame_count = 0
        self.fps_counter = 0

        # Buffers
        self.data_buffer = []
        self.times = []
        self.red_buffer = []
        self.green_buffer = []
        self.blue_buffer = []
        self.bpm_history = []
        self.buffer_size = 300
        self.last_bpm = 0
        self.fps = 0

        # Current frame with thread safety
        self.current_frame = None
        self.frame_lock = threading.Lock()

        self.hr_filter = HeartRateFilter()
        self.face_detector = dlib.get_frontal_face_detector()

        self.camera_thread = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_display)

        self.initUI()

    def initUI(self):
        # Connection info label
        self.connection_label = QLabel(f'Pi Camera: {self.PI_IP}:{self.PI_PORT}', self)
        self.connection_label.setStyleSheet("color: blue; font-weight: bold;")
        
        # Camera button
        self.camera_button = QPushButton('Connect to Pi Camera', self)
        self.camera_button.clicked.connect(self.toggle_camera)

        # Camera display
        self.camera_label = QLabel(self)
        self.camera_label.setScaledContents(True)
        self.camera_label.setStyleSheet("border: 2px solid gray;")
        self.camera_label.setMinimumSize(640, 480)

        # BPM label
        self.bpm_label = QLabel('BPM: --', self)
        self.bpm_label.setStyleSheet("font-size: 24px; font-weight: bold; color: red;")

        # Status label
        self.status_label = QLabel('Status: Disconnected', self)
        self.status_label.setStyleSheet("color: red;")

        # FPS label
        self.fps_label = QLabel('FPS: --', self)
        self.fps_label.setStyleSheet("color: green;")

        # RGB plot
        self.rgb_plot = pg.PlotWidget(self)
        self.rgb_plot.setYRange(0, 255)
        self.rgb_plot.showGrid(x=True, y=True)
        self.rgb_plot.setLabel('left', 'RGB Value')
        self.rgb_plot.setLabel('bottom', 'Time')
        self.rgb_plot.setTitle('RGB Signal')
        self.red_line = self.rgb_plot.plot(pen='r')
        self.green_line = self.rgb_plot.plot(pen='g')
        self.blue_line = self.rgb_plot.plot(pen='b')

        # Green frequency plot
        self.green_freq_plot = pg.PlotWidget(self)
        self.green_freq_plot.showGrid(x=True, y=True)
        self.green_freq_plot.setLabel('left', 'Amplitude')
        self.green_freq_plot.setLabel('bottom', 'Frequency (Hz)')
        self.green_freq_plot.setTitle('Frequency Domain')
        self.freq_line = self.green_freq_plot.plot(pen='g')

        # Layout
        self.resizeEvent(None)
        self.show()

    def resizeEvent(self, event):
        w = self.width()
        h = self.height()

        connection_h = 30
        btn_h = 40
        status_h = 30
        fps_h = 25
        cam_h = int(h * 0.4)
        bpm_h = 40
        plot_h = h - connection_h - btn_h - status_h - fps_h - cam_h - bpm_h - 70

        y_pos = 10
        
        # Connection info
        self.connection_label.setGeometry(10, y_pos, w - 20, connection_h)
        y_pos += connection_h + 5
        
        # Camera button
        self.camera_button.setGeometry(10, y_pos, w - 20, btn_h)
        y_pos += btn_h + 5
        
        # Status
        self.status_label.setGeometry(10, y_pos, (w - 20) // 2, status_h)
        self.fps_label.setGeometry(10 + (w - 20) // 2, y_pos, (w - 20) // 2, status_h)
        y_pos += status_h + 5
        
        # Camera display
        self.camera_label.setGeometry(10, y_pos, w - 20, cam_h)
        y_pos += cam_h + 10
        
        # BPM label
        self.bpm_label.setGeometry(10, y_pos, 200, bpm_h)
        y_pos += bpm_h + 10
        
        # Plots side by side
        plot_w = (w - 30) // 2
        self.rgb_plot.setGeometry(10, y_pos, plot_w, plot_h)
        self.green_freq_plot.setGeometry(20 + plot_w, y_pos, plot_w, plot_h)

    def on_frame_ready(self, frame):
        """Handle new frame from camera thread"""
        with self.frame_lock:
            self.current_frame = frame.copy()
        
        # Update FPS counter
        current_time = time.time()
        self.frame_count += 1
        if current_time - self.last_frame_time > 1.0:
            self.fps_counter = self.frame_count / (current_time - self.last_frame_time)
            self.fps_label.setText(f'FPS: {self.fps_counter:.1f}')
            self.frame_count = 0
            self.last_frame_time = current_time

    def butter_bandpass(self, lowcut, highcut, fs, order=4):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a

    def butter_bandpass_filter(self, data, lowcut, highcut, fs, order=4):
        b, a = self.butter_bandpass(lowcut, highcut, fs, order=order)
        y = filtfilt(b, a, data)
        return y

    def toggle_camera(self):
        if not self.camera_active:
            # Connect to Pi camera
            self.camera_thread = NetworkCameraThread(self.PI_IP, self.PI_PORT)
            self.camera_thread.frame_ready.connect(self.on_frame_ready)
            
            if not self.camera_thread.connect():
                self.status_label.setText("Status: Failed to connect to Pi camera")
                self.status_label.setStyleSheet("color: red;")
                return
            
            self.camera_thread.start()
            self.timer.start(33)  # ~30 FPS display update
            self.camera_active = True
            self.camera_button.setText('Disconnect from Pi Camera')
            self.status_label.setText("Status: Connected and streaming")
            self.status_label.setStyleSheet("color: green;")
            
            # Clear buffers
            self.data_buffer.clear()
            self.red_buffer.clear()
            self.green_buffer.clear()
            self.blue_buffer.clear()
            self.times.clear()
            self.bpm_history.clear()
            self.last_frame_time = time.time()
            self.frame_count = 0
        else:
            # Disconnect
            self.timer.stop()
            if self.camera_thread:
                self.camera_thread.stop()
                self.camera_thread = None
            self.camera_active = False
            self.camera_button.setText('Connect to Pi Camera')
            self.status_label.setText("Status: Disconnected")
            self.status_label.setStyleSheet("color: red;")
            self.fps_label.setText('FPS: --')
            self.camera_label.clear()
            self.bpm_label.setText('BPM: --')

    def update_display(self):
        """Update display and process heart rate - called by timer"""
        with self.frame_lock:
            if self.current_frame is None:
                return
            frame = self.current_frame.copy()

        # Process frame for heart rate detection
        display_frame = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector(gray)

        for face in faces:
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            forehead_h = h // 3
            cheek_h = h // 4

            # Define ROIs (Region of Interest)
            forehead_roi = self.get_safe_roi(frame, x, y, w, forehead_h)
            left_cheek_roi = self.get_safe_roi(frame, x, y + forehead_h, w // 3, cheek_h)
            right_cheek_roi = self.get_safe_roi(frame, x + 2 * w // 3, y + forehead_h, w // 3, cheek_h)

            # Calculate weighted average of ROIs
            weights = [0.6, 0.2, 0.2]  # Forehead gets more weight
            mean_green = np.average([
                np.mean(forehead_roi[:, :, 1]) if forehead_roi.size > 0 else 0,
                np.mean(left_cheek_roi[:, :, 1]) if left_cheek_roi.size > 0 else 0,
                np.mean(right_cheek_roi[:, :, 1]) if right_cheek_roi.size > 0 else 0
            ], weights=weights)

            mean_red = np.average([
                np.mean(forehead_roi[:, :, 2]) if forehead_roi.size > 0 else 0,
                np.mean(left_cheek_roi[:, :, 2]) if left_cheek_roi.size > 0 else 0,
                np.mean(right_cheek_roi[:, :, 2]) if right_cheek_roi.size > 0 else 0
            ], weights=weights)

            mean_blue = np.average([
                np.mean(forehead_roi[:, :, 0]) if forehead_roi.size > 0 else 0,
                np.mean(left_cheek_roi[:, :, 0]) if left_cheek_roi.size > 0 else 0,
                np.mean(right_cheek_roi[:, :, 0]) if right_cheek_roi.size > 0 else 0
            ], weights=weights)

            # Store RGB values if valid
            if mean_green > 0:
                self.green_buffer.append(mean_green)
                self.red_buffer.append(mean_red)
                self.blue_buffer.append(mean_blue)
                self.times.append(time.time())

                # Maintain buffer size
                if len(self.green_buffer) > self.buffer_size:
                    self.red_buffer = self.red_buffer[-self.buffer_size:]
                    self.green_buffer = self.green_buffer[-self.buffer_size:]
                    self.blue_buffer = self.blue_buffer[-self.buffer_size:]
                    self.times = self.times[-self.buffer_size:]
                    self.calculate_heart_rate()

            # Draw ROI rectangles on display frame
            cv2.rectangle(display_frame, (x, y), (x + w, y + forehead_h), (0, 255, 0), 2)
            cv2.rectangle(display_frame, (x, y + forehead_h), (x + w // 3, y + forehead_h + cheek_h), (0, 255, 0), 2)
            cv2.rectangle(display_frame, (x + 2 * w // 3, y + forehead_h), (x + w, y + forehead_h + cheek_h), (0, 255, 0), 2)
            
            # Add face rectangle
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Show camera feed
        rgb_display = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        h_img, w_img, ch = rgb_display.shape
        qt_image = QImage(rgb_display.data, w_img, h_img, ch * w_img, QImage.Format_RGB888)
        self.camera_label.setPixmap(QPixmap.fromImage(qt_image))

        # Update RGB plots (less frequently to save processing)
        if len(self.red_buffer) > 0 and self.frame_count % 3 == 0:
            self.red_line.setData(self.red_buffer)
            self.green_line.setData(self.green_buffer)
            self.blue_line.setData(self.blue_buffer)

    def get_safe_roi(self, frame, x, y, w, h):
        """Extract safe ROI from frame with bounds checking"""
        x = max(0, x)
        y = max(0, y)
        w = min(w, frame.shape[1] - x)
        h = min(h, frame.shape[0] - y)
        if w <= 0 or h <= 0:
            return np.array([])
        return frame[y:y + h, x:x + w]

    def calculate_heart_rate(self):
        """Calculate heart rate from green channel data"""
        if len(self.green_buffer) < self.buffer_size // 2:
            return

        # Signal processing
        detrended = signal.detrend(self.green_buffer)
        normalized = (detrended - np.mean(detrended)) / np.std(detrended)

        # Calculate actual FPS
        time_elapsed = np.array(self.times) - self.times[0]
        if len(time_elapsed) > 1:
            self.fps = len(time_elapsed) / (time_elapsed[-1] - time_elapsed[0])
        else:
            self.fps = 30  # Default fallback

        # Apply bandpass filter (0.75-2.0 Hz corresponds to 45-120 BPM)
        try:
            filtered = self.butter_bandpass_filter(normalized, 0.75, 2.0, self.fps, order=4)
        except:
            return

        # Frequency domain analysis
        freqs = np.fft.fftfreq(len(filtered)) * self.fps
        fft_vals = np.fft.fft(filtered)

        # Focus on heart rate frequency range
        positive_idx = np.where((freqs > 0) & (freqs * 60 >= 45) & (freqs * 60 <= 120))
        
        if len(positive_idx[0]) == 0:
            return
            
        amplitudes = np.abs(fft_vals[positive_idx]) * 2

        if len(amplitudes) > 0:
            # Find peak frequency
            max_idx = np.argmax(amplitudes)
            bpm = freqs[positive_idx][max_idx] * 60
            
            # Apply Kalman filter for smoothing
            filtered_bpm = self.hr_filter.update(bpm)

            # Only update if change is reasonable or it's the first measurement
            if abs(filtered_bpm - self.last_bpm) < 15 or self.last_bpm == 0:
                self.last_bpm = filtered_bpm
                self.bpm_history.append(filtered_bpm)

                # Maintain BPM history
                if len(self.bpm_history) > 10:
                    self.bpm_history = self.bpm_history[-100:]

                # Use median for final BPM to reduce noise
                avg_bpm = np.median(self.bpm_history)
                self.bpm_label.setText(f'BPM: {int(avg_bpm)}')

        # Update frequency plot (less frequently)
        if len(amplitudes) > 0 and len(positive_idx[0]) > 0 and self.frame_count % 5 == 0:
            self.freq_line.setData(freqs[positive_idx], amplitudes)

    def closeEvent(self, event):
        """Clean up when closing the application"""
        if self.camera_thread:
            self.camera_thread.stop()
        event.accept()

if __name__ == '__main__':
    print("Heart Rate Monitor - Optimized Pi Camera Stream")
    print("Make sure your camera_gui4.py is running on the Pi")
    
    app = QApplication(sys.argv)
    ex = App()
    ex.resize(1200, 900)
    sys.exit(app.exec_())