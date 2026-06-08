import tkinter as tk                          # Giao diện người dùng cơ bản
from tkinter import ttk, messagebox, filedialog  # Các widget và hộp thoại nâng cao
from PIL import Image, ImageTk                  # Xử lý và hiển thị ảnh trong tkinter
import cv2                                      # Xử lý ảnh và webcam
import numpy as np                              # Tính toán ma trận ảnh
import threading                                # Đa luồng nếu cần chạy song song
import json                                     # Đọc file JSON chứa class indices
import os                                       # Kiểm tra, tạo thư mục, đường dẫn
import pyttsx3                                  # Chuyển văn bản thành giọng nói (Text-to-Speech)
from tensorflow.keras.models import load_model  # Load mô hình đã huấn luyện
import winsound                                 # Phát âm thanh (ding) trên Windows

# --- Load file class indices ---
def load_class_indices(path):
    if not os.path.exists(path):
        return {}
    with open(path, 'r') as f:
        return json.load(f)

# --- Lấy danh sách nhãn từ class_indices (dạng list theo thứ tự) ---
def get_label_list(class_indices):
    labels = [None] * len(class_indices)
    for k, v in class_indices.items():
        labels[v] = k
    return labels

# --- Load mô hình huấn luyện và nhãn mặc định ---
def load_default_model():
    model_path = 'D:/PyNet/fruit_recognition/nn/fruit_model.h5'
    class_indices_path = 'D:/PyNet/fruit_recognition/dataset/images/class_indices.json'
    model = load_model(model_path)
    class_indices = load_class_indices(class_indices_path)
    return model, get_label_list(class_indices)

model, labels = load_default_model()
img_size = (64, 64)  # Kích thước ảnh đầu vào của mô hình

# --- Hàm dự đoán top 3 trái cây từ ảnh ---
def predict_image_array(img_array):
    img_input = img_array.astype('float32') / 255.0
    img_input = np.expand_dims(img_input, axis=0)
    pred = model.predict(img_input)[0]
    top3_indices = pred.argsort()[-3:][::-1]  # Lấy 3 chỉ số có xác suất cao nhất
    results = [(labels[i], pred[i]) for i in top3_indices]
    return results

# --- Lớp chính: Giao diện ứng dụng ---
class FruitApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🍏 Hệ Thống Nhận Diện Trái Cây - Nền Tối 🍇")
        self.root.geometry("1000x750")
        self.root.configure(bg="#121212")  # Giao diện nền tối

        self.cap = None             # Đối tượng Webcam
        self.running = False        # Trạng thái webcam
        self.current_frame = None   # Khung hình hiện tại
        self.engine = pyttsx3.init()  # Engine phát giọng nói
        self.fruit_info = {         # CSDL mô tả trái cây
            "Táo": "Táo chứa nhiều vitamin C và chất xơ tốt cho sức khỏe.",
            "Chuối": "Chuối giàu kali, giúp cân bằng điện giải cơ thể.",
            "Cam": "Cam có nhiều vitamin C và chất chống oxy hóa.",
            "Lê": "Lê giúp cải thiện tiêu hóa và chứa nhiều chất chống oxy hóa.",
            "Dưa hấu": "Dưa hấu giúp giải nhiệt, cung cấp nước và vitamin A, C."
        }

        self.build_ui()  # Khởi tạo giao diện

    # --- Hiển thị thông tin trái cây ---
    def show_fruit_info(self, label):
        # Lấy thông tin chi tiết của trái cây dựa trên nhãn đã nhận diện
        info = self.fruit_info.get(label, "Không có thông tin về loại quả này.")
        self.info_text.configure(state="normal")  # Cho phép chỉnh sửa Text box
        self.info_text.delete("1.0", "end")  # Xóa nội dung cũ
        self.info_text.insert("1.0", info)  # Hiển thị thông tin mới
        self.info_text.configure(state="disabled")  # Vô hiệu hóa chỉnh sửa sau khi cập nhật

    def build_ui(self):
        # Tiêu đề
        title = tk.Label(self.root, text="🍎 HỆ THỐNG NHẬN DIỆN TRÁI CÂY THÔNG MINH 🍌",
                         font=("Segoe UI", 26, "bold"), bg="#1f1f1f", fg="#80cbc4", pady=20)
        title.pack(fill="x")

        # Khung điều khiển
        control_frame = tk.Frame(self.root, bg="#121212")
        control_frame.pack(pady=15)

        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TButton", padding=10, relief="flat", background="#263238", foreground="#e0f7fa",
                        font=("Segoe UI", 12, "bold"))
        style.map("TButton",
                  background=[('active', '#80cbc4')],
                  foreground=[('active', '#004d40')])

        ttk.Button(control_frame, text="🎥 Bắt đầu Webcam", command=self.start_webcam).grid(row=0, column=0, padx=15)
        ttk.Button(control_frame, text="🛑 Dừng Webcam", command=self.stop_webcam).grid(row=0, column=1, padx=15)
        ttk.Button(control_frame, text="🖼️ Chọn Ảnh Nhận Diện", command=self.open_image).grid(row=0, column=2, padx=15)
        ttk.Button(control_frame, text="💾 Lưu Ảnh Webcam", command=self.save_frame).grid(row=0, column=3, padx=15)

        # Khung hiển thị video/ảnh và kết quả
        display_frame = tk.Frame(self.root, bg="#121212")
        display_frame.pack(pady=10)

        self.video_label = tk.Label(display_frame, bg="#000000", relief="sunken", bd=3, width=640, height=480)
        self.video_label.grid(row=0, column=0, padx=20)

        result_frame = tk.Frame(display_frame, bg="#1f1f1f", bd=2, relief="groove")
        result_frame.grid(row=0, column=1, padx=20, sticky="n")

        tk.Label(result_frame, text="Kết quả nhận diện", font=("Segoe UI", 16, "bold"),
                 bg="#1f1f1f", fg="#80cbc4").pack(pady=10)

        self.result_label = tk.Label(result_frame, text="", font=("Segoe UI", 14),
                                     bg="#1f1f1f", fg="#e0f7fa", wraplength=280, justify="left")
        self.result_label.pack(pady=10, padx=10)

        # Thông tin chi tiết trái cây
        self.info_text = tk.Text(result_frame, height=8, width=35, bg="#263238", fg="#e0f7fa", font=("Segoe UI", 10))
        self.info_text.pack(pady=10, padx=10)
        self.info_text.configure(state="disabled")

        # Lịch sử nhận diện
        tk.Label(result_frame, text="📜 Lịch sử", font=("Segoe UI", 12, "bold"), bg="#1f1f1f", fg="#80cbc4").pack(pady=(10, 5))
        self.history_listbox = tk.Listbox(result_frame, width=35, height=8, bg="#263238", fg="#e0f7fa",
                                          font=("Segoe UI", 10), selectbackground="#80cbc4")
        self.history_listbox.pack(padx=10, pady=5)

        # Footer
        footer = tk.Label(self.root, text="Nhấn 'Dừng' để ngừng webcam hoặc đóng cửa sổ để thoát.",
                          font=("Segoe UI", 10), bg="#121212", fg="#666")
        footer.pack(side="bottom", pady=10)

    def start_webcam(self):
        if self.running:
            return
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Lỗi", "Không thể mở webcam.")
            return
        self.running = True
        self.update_video()

    def stop_webcam(self):
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        # Xóa ảnh webcam hiện tại
        self.video_label.configure(image='')
        self.current_frame = None

    def update_video(self):
        if not self.running:
            return

        ret, frame = self.cap.read()
        if not ret:
            messagebox.showerror("Lỗi", "Không thể lấy khung hình từ webcam.")
            self.stop_webcam()
            return

        frame = cv2.flip(frame, 1)
        self.current_frame = frame.copy()

        img = cv2.resize(frame, img_size)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = predict_image_array(img_rgb)
        label, conf = results[0]

        display_frame = frame.copy()
        cv2.putText(display_frame, f"{label} ({conf*100:.1f}%)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        img_display = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_display)
        imgtk = ImageTk.PhotoImage(image=img_pil)

        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

        # Hiển thị thông tin chi tiết từ self.fruit_info
        result_text = "\n".join([f"{label} - {conf * 100:.2f}%" for label, conf in results])
        self.result_label.configure(text=f"Nhận diện Top 3:\n{result_text}")
        self.show_fruit_info(label)

        self.root.after(30, self.update_video)

    def open_image(self):
        file_path = filedialog.askopenfilename(
            title="Chọn ảnh để nhận diện",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
        if not file_path:
            return

        img = cv2.imread(file_path)
        if img is None:
            messagebox.showerror("Lỗi", "Không thể đọc ảnh.")
            return

        img_rgb = cv2.cvtColor(cv2.resize(img, img_size), cv2.COLOR_BGR2RGB)
        results = predict_image_array(img_rgb)
        label, conf = results[0]

        # Hiển thị ảnh trong GUI
        img_rgb_show = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb_show)
        img_pil = img_pil.resize((640, 480), Image.Resampling.LANCZOS)

        imgtk = ImageTk.PhotoImage(img_pil)

        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

        # Hiển thị kết quả và thông tin trái cây
        result_text = "\n".join([f"{label} - {conf * 100:.2f}%" for label, conf in results])
        self.result_label.configure(text=f"Ảnh Top 3:\n{result_text}")
        self.show_fruit_info(label)

        # Đọc bằng giọng nói (chỉ ở đây)
        self.engine.say(f"Đây là {label}, độ tin cậy {int(conf*100)} phần trăm")
        self.engine.runAndWait()

        # Thêm vào lịch sử
        self.history_listbox.insert(0, f"{label} - {conf*100:.2f}%")
        if self.history_listbox.size() > 20:
            self.history_listbox.delete(20, "end")

        # Tự động lưu ảnh
        os.makedirs("results", exist_ok=True)
        filename = os.path.join("results", f"{label}.png")
        cv2.imwrite(filename, img)

        if self.running:
            self.stop_webcam()

    def save_frame(self):
        if self.current_frame is None:
            messagebox.showwarning("Cảnh báo", "Chưa có khung hình để lưu!")
            return

        filename = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg *.jpeg")],
            title="Lưu ảnh khung hình webcam"
        )
        if filename:
            cv2.imwrite(filename, self.current_frame)
            messagebox.showinfo("Thôngi báo", f"Đã lưu ảnh tại: {filename}")


if __name__ == '__main__':
    root = tk.Tk()
    app = FruitApp(root)
    root.mainloop()
