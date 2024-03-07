from flask import Flask, render_template, request
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

app = Flask(__name__)

DATA_FILE = 'pisangmas.csv'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

resize = (200, 200)

## Mengecek apakah ekstensi file gambar diizinkan
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

#normalisasi RGB
def normalize_rgb(image):
    return image / 255.0

def calculate_hsv(normalized_r, normalized_g, normalized_b):
    v = np.max([normalized_r, normalized_g, normalized_b])

    s = (v - np.min([normalized_r, normalized_g, normalized_b])) / v if v != 0 else 0

    if v == 0:
        h = 0
    elif v == normalized_r:
        h = 60 * (normalized_g - normalized_b) / (v - np.min([normalized_r, normalized_g, normalized_b]))
    elif v == normalized_g:
        h = 120 + 60 * (normalized_b - normalized_r) / (v - np.min([normalized_r, normalized_g, normalized_b]))
    else:
        h = 240 + 60 * (normalized_r - normalized_g) / (v - np.min([normalized_r, normalized_g, normalized_b]))

    h = h if h >= 0 else h + 360
    return h, s, v

def create_knn_model(data_file):
    data = pd.read_csv(data_file)
    X = data[['MeanR', 'MeanG', 'MeanB', 'MeanH', 'MeanS', 'MeanV']]
    y = data['Kematangan']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Inisialisasi model KNN dengan k=7 dan menggunakan Euclidean distance
    knn = KNeighborsClassifier(n_neighbors=7, metric='euclidean')

    # Melatih model KNN
    knn.fit(X_train, y_train)

    # Melakukan prediksi pada data uji
    y_pred = knn.predict(X_test)
    knn.fit(X, y)

    return knn

# Mengembalikan keterangan klasifikasi berdasarkan hasil prediksi
def get_klasifikasi_keterangan(hasil_klasifikasi):
    if hasil_klasifikasi == 1:
        return "Mentah"
    elif hasil_klasifikasi == 2:
        return "Matang"
    elif hasil_klasifikasi == 3:
        return "Sangat Matang"
    else:
        return "Upload Gambar Terlebih Dahulu"

@app.route('/')
def index(): # Menampilkan halaman utama
    return render_template('index.html')

@app.route('/klasifikasi', methods=['GET', 'POST'])
def klasifikasi(): #Fungsi untuk menangani klasifikasi gambar
    hasil_klasifikasi = None #Variabel untuk menyimpan hasil klasifikasi gambar, diinisialisasi sebagai None

    if request.method == 'POST':
        if 'image' not in request.files:
            return render_template('klasifikasi.html', hasil_klasifikasi="Upload file terlebih dahulu")

        image = request.files['image']
        if image.filename == '' or not allowed_file(image.filename):
            return render_template('klasifikasi.html', hasil_klasifikasi="Pilih file gambar dengan ekstensi JPEG atau JPG atau PNG")

        # Membaca gambar 
        image_cv2_bgr = cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_UNCHANGED) #Membaca dan mendekode gambar menjadi format BGR
        image_cv2_rgb = cv2.cvtColor(image_cv2_bgr, cv2.COLOR_BGR2RGB)

        # Resize gambar
        image_cv2_resized = cv2.resize(image_cv2_rgb, resize)
        normalized_image = normalize_rgb(image_cv2_resized)
        hsv_values = calculate_hsv(*normalized_image.mean(axis=(0, 1)))

        # Bentuk data baru untuk prediksi
        new_data = np.array([*normalized_image.mean(axis=(0, 1)), *hsv_values]).reshape(1, -1)

        # Memanggil model KNN yang sudah dibuat
        knn_model = create_knn_model(DATA_FILE)

        # Melakukan prediksi
        hasil_klasifikasi = knn_model.predict(new_data)[0]

    keterangan_klasifikasi = get_klasifikasi_keterangan(hasil_klasifikasi)
    return render_template('klasifikasi.html', hasil_klasifikasi=keterangan_klasifikasi)

if __name__ == '__main__':
    app.run(debug=True)