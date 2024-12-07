import cv2
import mediapipe as mp
import numpy as np

# Inisialisasi MediaPipe dan OpenCV
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Warna dan pengaturan menggambar
colors = {
    'blue': (255, 0, 0),
    'red': (0, 0, 255), 
    'green': (0, 255, 0),
    'yellow': (0, 255, 255),
    'eraser': (0, 0, 0)  # Tambahkan warna penghapus
}
current_color = colors['blue']
brush_thickness = 5
eraser_thickness = 50
eraser_color = colors['eraser']
is_eraser_mode = False  # Tambahkan status mode penghapus

# Buka kamera
cap = cv2.VideoCapture(0)
ret, frame = cap.read()

if not ret:
    print("Gagal membuka kamera.")
    cap.release()
    exit()

# Dapatkan ukuran frame awal
image_height, image_width, _ = frame.shape

# Kanvas kosong
canvas = np.zeros((image_height, image_width, 3), dtype="uint8")
prev_point = None

# Fungsi untuk mendapatkan koordinat ujung jari
def get_finger_positions(hand_landmarks, image_width, image_height):
    # Dapatkan posisi ujung telunjuk (8) dan ujung jempol (4)
    index_finger_tip = hand_landmarks.landmark[8]
    thumb_tip = hand_landmarks.landmark[4]
    middle_finger_tip = hand_landmarks.landmark[12]
    ring_finger_tip = hand_landmarks.landmark[16]  # Tambahkan deteksi jari manis
    
    # Konversi ke koordinat pixel
    index_x = int(index_finger_tip.x * image_width)
    index_y = int(index_finger_tip.y * image_height)
    thumb_x = int(thumb_tip.x * image_width)
    thumb_y = int(thumb_tip.y * image_height)
    middle_x = int(middle_finger_tip.x * image_width)
    middle_y = int(middle_finger_tip.y * image_height)
    ring_x = int(ring_finger_tip.x * image_width)
    ring_y = int(ring_finger_tip.y * image_height)
    
    return (index_x, index_y), (thumb_x, thumb_y), (middle_x, middle_y), (ring_x, ring_y)

def calculate_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip frame untuk mirror efek
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    # Tampilkan warna yang sedang aktif dan mode penghapus
    if is_eraser_mode:
        cv2.rectangle(frame, (10, 10), (30, 30), eraser_color, -1)
        cv2.putText(frame, "ERASER", (35, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    else:
        cv2.rectangle(frame, (10, 10), (30, 30), current_color, -1)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Dapatkan posisi jari
            index_pos, thumb_pos, middle_pos, ring_pos = get_finger_positions(hand_landmarks, image_width, image_height)
            
            # Hitung jarak antara jempol dan telunjuk
            thumb_index_distance = calculate_distance(thumb_pos, index_pos)
            
            # Toggle mode penghapus dengan mengangkat jari manis
            if ring_pos[1] < index_pos[1] - 50:  # Jika jari manis diangkat
                is_eraser_mode = not is_eraser_mode
            
            # Mode menggambar: jika jarak jempol-telunjuk kecil
            if thumb_index_distance < 40:  # Jarak threshold untuk mendeteksi "mencubit"
                if prev_point is not None:
                    if is_eraser_mode:
                        cv2.line(canvas, prev_point, index_pos, eraser_color, eraser_thickness)
                    else:
                        cv2.line(canvas, prev_point, index_pos, current_color, brush_thickness)
                prev_point = index_pos
            else:
                prev_point = None
            
            # Ganti warna dengan mengangkat jari tengah dan mendekatkan jempol-telunjuk
            if middle_pos[1] < index_pos[1] - 50 and thumb_index_distance < 20 and not is_eraser_mode:
                # Rotasi warna (skip eraser color)
                color_list = [c for c in colors.values() if c != colors['eraser']]
                current_idx = color_list.index(current_color) if current_color in color_list else 0
                current_color = color_list[(current_idx + 1) % len(color_list)]

    # Gabungkan frame kamera dengan kanvas
    frame = cv2.addWeighted(frame, 0.7, canvas, 0.3, 0)

    # Tampilkan instruksi
    cv2.putText(frame, "Dekatkan jempol & telunjuk untuk menggambar", (10, image_height-80), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    cv2.putText(frame, "Angkat jari manis untuk mode penghapus", (10, image_height-60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    cv2.putText(frame, "Angkat jari tengah & dekatkan jempol-telunjuk untuk ganti warna", (10, image_height-40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    cv2.putText(frame, "Tekan 'c' untuk membersihkan kanvas, 'q' untuk keluar", (10, image_height-20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    # Tampilkan hasil
    cv2.imshow("Drawing App", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):  # Tekan 'c' untuk membersihkan kanvas
        canvas = np.zeros((image_height, image_width, 3), dtype="uint8")

# Lepaskan sumber daya
cap.release()
cv2.destroyAllWindows()
