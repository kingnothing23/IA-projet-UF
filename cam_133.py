import cv2
import time


# =========================================================
# ⚙️ CONFIGURATION GSTREAMER (Version Pi Cam V1.3)
# =========================================================
def gstreamer_pipeline(
        sensor_id=0,
        capture_width=1280,
        capture_height=720,
        display_width=1280,
        display_height=720,
        framerate=30,
        flip_method=0,
):
    """
    Pipeline GStreamer pour OV5647 (Pi Cam V1.3).
    Note : Le capteur V1.3 supporte max 30fps en pleine résolution,
    mais 1280x720 à 30fps est le mode le plus stable.
    """
    return (
            "nvarguscamerasrc sensor-id=%d ! "
            "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, format=(string)NV12, framerate=(fraction)%d/1 ! "
            "nvvidconv flip-method=%d ! "
            "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
            "videoconvert ! "
            "video/x-raw, format=(string)BGR ! appsink"
            % (
                sensor_id,
                capture_width,
                capture_height,
                framerate,
                flip_method,
                display_width,
                display_height,
            )
    )


# =========================================================
# 📸 FONCTION PRISE DE PHOTO
# =========================================================
def test_photo_v1():
    print("🚀 Démarrage du test Caméra Pi V1.3 (OV5647)...")

    # 1. Pipeline
    pipeline = gstreamer_pipeline(flip_method=0)  # Mets 2 si l'image est à l'envers

    # 2. Ouverture
    print("⏳ Ouverture du flux vidéo...")
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

    if not cap.isOpened():
        print("❌ ERREUR : Impossible d'ouvrir la caméra !")
        print("👉 Vérifie :")
        print("   1. La nappe (Argent vers Radiateur).")
        print("   2. Que la Jetson supporte bien le capteur OV5647 (Vérifie /dev/video0).")
        return

    # 3. Stabilisation (La V1.3 a besoin d'un peu plus de temps pour la lumière)
    print("💡 Caméra ouverte ! Stabilisation (3 secondes)...")
    time.sleep(3)

    # 4. Lecture
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("❌ ERREUR : Caméra détectée mais image vide (écran noir/vert ?).")
    else:
        filename = "photo_test_v1.jpg"
        cv2.imwrite(filename, frame)
        print(f"✅ SUCCÈS ! Photo enregistrée sous : {filename}")


# =========================================================
# ▶️ EXÉCUTION
# =========================================================
if __name__ == "__main__":
    test_photo_v1()