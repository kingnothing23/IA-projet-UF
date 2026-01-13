import cv2
import time


# =========================================================
# ⚙️ CONFIGURATION GSTREAMER (C'est la partie magique)
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
    Crée la chaîne de connexion pour la Pi Cam V2 via le GPU de la Jetson.
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
def test_photo_simple():
    print("🚀Démarrage du test Caméra Pi V2...")

    # 1. On construit le pipeline
    pipeline = gstreamer_pipeline(flip_method=0)
    print(f" Pipeline GStreamer : \n{pipeline}")

    # 2. Ouverture de la caméra
    print(" Ouverture du flux vidéo...")
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

    if not cap.isOpened():
        print(" ERREUR CRITIQUE : Impossible d'ouvrir la caméra !")
        print(" Vérifie que la nappe est bien branchée (Côté argenté vers le radiateur).")
        return

    # 3. Laisser la caméra 'chauffer' (Important pour la balance des blancs)
    print("💡 Caméra ouverte ! Stabilisation de l'image (2 secondes)...")
    time.sleep(2)

    # 4. Lecture d'une frame
    ret, frame = cap.read()

    # 5. Fermeture immédiate
    cap.release()

    if not ret:
        print(" ERREUR : La caméra est ouverte mais n'envoie pas d'image.")
    else:
        filename = "photo_test_pi.jpg"
        cv2.imwrite(filename, frame)
        print(f" SUCCÈS ! Photo enregistrée sous : {filename}")
        print(f" Taille de l'image : {frame.shape[1]}x{frame.shape[0]} px")


# =========================================================
# ▶️ EXÉCUTION
# =========================================================
if __name__ == "__main__":
    test_photo_simple()