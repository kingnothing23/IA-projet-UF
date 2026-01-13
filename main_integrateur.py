#!/usr/bin/python3
import cv2
import numpy as np
import sys
import time
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from opcua import Client

# =========================================================================
#  CONFIGURATION GLOBALE
# =========================================================================

# --- OPC UA (AUTOMATE) ---
PLC_URL = "opc.tcp://192.168.0.1:4840"  # IP de l'automate
NODE_TRIGGER = "ns=4;s=MAIN.iCommande"  # Variable d'entrée (0, 1, 2)
NODE_RES_CLASSE = "ns=4;s=MAIN.iResultatClasse"  # Sortie Forme (0=Carré, 1=Cercle, 2=Autre)
NODE_RES_ROUILLE = "ns=4;s=MAIN.bResultatRouille"  # Sortie Rouille (0=OK, 1=NOK)

# --- PARAMÈTRES IA & VISION ---
ENGINE_PATH = "model_rouille.engine"
CONF_THRESHOLD_ROUILLE = 0.50
COOLDOWN_DELAY = 10  # Secondes avant de pouvoir relancer une analyse


# =========================================================================
#  MODULE CAMÉRA (MODULABLE)
# =========================================================================

def gstreamer_pipeline(sensor_id=0, capture_width=1280, capture_height=720, framerate=30, flip_method=0):
    """Pipeline spécifique pour la Pi Cam V2 sur Jetson Nano"""
    return (
            "nvarguscamerasrc sensor-id=%d ! "
            "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, format=(string)NV12, framerate=(fraction)%d/1 ! "
            "nvvidconv flip-method=%d ! "
            "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
            "videoconvert ! "
            "video/x-raw, format=(string)BGR ! appsink"
            % (sensor_id, capture_width, capture_height, framerate, flip_method, 640, 360)
    )


def prendre_photo():
    """
    Fonction unique pour capturer une image.
    C'est ICI qu'il faudra modifier si on change de caméra.
    Retourne : une image OpenCV (numpy array) ou None si erreur.
    """
    print(" Tentative de prise de photo...")

    # 1. Essai avec GStreamer (Pi Cam)
    cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)

    # 2. Si ça rate, essai Webcam standard (USB) - Utile pour les tests
    if not cap.isOpened():
        print(" GStreamer échoué, essai USB standard...")
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print(" ERREUR : Impossible d'ouvrir une caméra.")
        return None

    # On lit une image
    ret, frame = cap.read()
    cap.release()  # On libère la caméra tout de suite

    if not ret:
        print(" ERREUR : Impossible de lire l'image.")
        return None

    print(" Photo capturée avec succès.")
    return frame


# =========================================================================
#  MODULE 1 : ANALYSE FORME (OPENCV CLASSIQUE)
# =========================================================================

def analyse_forme(img):
    """
    Retourne : 0 (Carré), 1 (Cercle), 2 (Ambigu)
    """
    print("📐 Démarrage Analyse Forme...")
    try:
        h, w = img.shape[:2]
        center_x, center_y = w // 2, h // 2
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Prétraitement
        kernel_tophat = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel_tophat)
        gray_ready = cv2.addWeighted(gray, 0.6, tophat, 0.4, 0)

        # Masque Radial
        Y, X = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)
        max_dist = min(h, w) * 0.65
        radial_mask = np.clip(1.0 - (dist_from_center / max_dist), 0, 1)

        # Gradient + Sobel
        blur = cv2.GaussianBlur(gray_ready, (5, 5), 0)
        grad_x = cv2.Sobel(blur, cv2.CV_16S, 1, 0, ksize=3)
        grad_y = cv2.Sobel(blur, cv2.CV_16S, 0, 1, ksize=3)
        gradient = cv2.addWeighted(cv2.convertScaleAbs(grad_x), 0.5, cv2.convertScaleAbs(grad_y), 0.5, 0)
        weighted_gradient = (gradient.astype(np.float32) * radial_mask).astype(np.uint8)

        # Binarisation
        _, thresh = cv2.threshold(weighted_gradient, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        closed_mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
                                       iterations=2)

        # Contours
        contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_cnt = None
        best_score = 0
        for c in contours:
            area = cv2.contourArea(c)
            if area < 200: continue
            M = cv2.moments(c)
            if M["m00"] == 0: continue
            cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
            dist = np.sqrt((cx - center_x) ** 2 + (cy - center_y) ** 2)
            score = area / (1 + (dist / (min(h, w) * 0.2)) ** 2)
            if score > best_score:
                best_score = score
                best_cnt = c

        if best_cnt is None:
            print(" Aucun contour valide trouvé.")
            return 2  # Ambigu

        # Circularité
        area = cv2.contourArea(best_cnt)
        perimeter = cv2.arcLength(best_cnt, True)
        if perimeter == 0: return 2
        circularity = 4 * np.pi * (area / (perimeter * perimeter))

        print(f"   -> Circularité mesurée : {circularity:.3f}")

        if circularity >= 0.89:
            return 1  # CERCLE
        elif circularity <= 0.82:
            return 0  # CARRE
        else:
            return 2  # AMBIGU

    except Exception as e:
        print(f" Erreur Algo Forme: {e}")
        return 2


# =========================================================================
#  MODULE 2 : ANALYSE ROUILLE (IA TENSORRT)
# =========================================================================

class RustDetector:
    def __init__(self, engine_path):
        print(f" Chargement Moteur IA: {engine_path}")
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.inputs, self.outputs, self.bindings, self.stream = [], [], [], cuda.Stream()

        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding))
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(device_mem))
            if self.engine.binding_is_input(binding):
                self.inputs.append({'host': host_mem, 'device': device_mem})
            else:
                self.outputs.append({'host': host_mem, 'device': device_mem})

    def detect(self, img):
        """ Retourne 1 (Rouille) ou 0 (Propre) """
        print(" Démarrage Analyse Rouille...")
        # Resize 640x640
        img_resized = cv2.resize(img, (640, 640))
        img_in = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB).transpose((2, 0, 1)).astype(np.float32)
        img_in /= 255.0
        img_in = np.expand_dims(img_in, axis=0)

        # Inférence
        np.copyto(self.inputs[0]['host'], img_in.ravel())
        cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(self.outputs[0]['host'], self.outputs[0]['device'], self.stream)
        self.stream.synchronize()

        # Parsing result
        output = self.outputs[0]['host'].reshape(1, 5, 8400)
        scores = output[0, 4, :]
        max_score = np.max(scores)

        print(f"   -> Confiance Rouille Max : {max_score:.2%}")
        return 1 if max_score > CONF_THRESHOLD_ROUILLE else 0


# =========================================================================
#  BOUCLE PRINCIPALE
# =========================================================================

def main():
    # 1. INITIALISATION
    print("--- DÉMARRAGE DU SYSTÈME ---")

    # Init IA
    try:
        ai_engine = RustDetector(ENGINE_PATH)
    except Exception as e:
        print(f"CRITIQUE: Impossible de charger l'IA ({e})")
        sys.exit(1)

    # Init PLC
    client = Client(PLC_URL)
    try:
        client.connect()
        print(" PLC Connecté.")
        node_trigger = client.get_node(NODE_TRIGGER)
        node_res_classe = client.get_node(NODE_RES_CLASSE)
        node_res_rouille = client.get_node(NODE_RES_ROUILLE)
    except Exception as e:
        print(f" ATTENTION: PLC non connecté ({e}). Mode Simulation.")
        client = None

    last_execution_time = 0

    print("--- EN ATTENTE DE SIGNAL PLC (0=Attente, 1=Forme, 2=Rouille) ---")

    # 2. BOUCLE INFINIE
    while True:
        try:
            # A. Lecture Trigger
            trigger = 0
            if client:
                try:
                    trigger = node_trigger.get_value()
                except:
                    print(" Perte connexion PLC, tentative reconnexion...")
                    try:
                        client.connect()
                    except:
                        pass

            # Simulation clavier pour test sans PLC (Optionnel)
            # if trigger == 0: pass

            # B. Logique de déclenchement
            current_time = time.time()

            if trigger in [1, 2]:
                if (current_time - last_execution_time) > COOLDOWN_DELAY:
                    print(f"\n SIGNAL REÇU : {trigger}")

                    # 1. Prendre Photo
                    img = prendre_photo()

                    if img is not None:
                        # 2. Traitement selon le signal
                        if trigger == 1:
                            # --- ANALYSE FORME ---
                            resultat = analyse_forme(img)
                            print(f" Envoi Résultat CLASSE : {resultat}")
                            if client:
                                try:
                                    node_res_classe.set_value(resultat, varianttype=client.get_node(
                                        NODE_RES_CLASSE).get_data_value().Value.VariantType)
                                except:
                                    print(" Erreur écriture PLC")

                        elif trigger == 2:
                            # --- ANALYSE ROUILLE ---
                            resultat = ai_engine.detect(img)
                            print(f" Envoi Résultat ROUILLE : {resultat}")
                            if client:
                                try:
                                    node_res_rouille.set_value(bool(resultat))
                                except:
                                    print(" Erreur écriture PLC")

                        # Mise à jour du timer
                        last_execution_time = time.time()
                        print(f" Pause de {COOLDOWN_DELAY}s activée...")
                    else:
                        print(" Échec Photo, pas de traitement.")

                else:
                    # On est encore dans les 10s de blocage
                    # On ne fait rien, on ignore le signal
                    pass

            # Petite pause pour ne pas saturer le CPU
            time.sleep(0.1)

        except KeyboardInterrupt:
            print("\nArrêt manuel.")
            break
        except Exception as e:
            print(f"Erreur inattendue dans la boucle : {e}")
            time.sleep(1)

    if client: client.disconnect()


if __name__ == "__main__":
    main()