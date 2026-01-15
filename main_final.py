#!/usr/bin/python3
import cv2
import numpy as np
import sys
import time
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from opcua import Client
from opcua import ua  # Important pour le typage des données Siemens

# =========================================================================
#  CONFIGURATION GLOBALE
# =========================================================================

# --- OPC UA (AUTOMATE) ---
# Mise à jour avec tes infos réelles trouvées précédemment
PLC_URL = "opc.tcp://192.168.0.26:4840"

# NodeIDs exacts trouvés avec le scan
NODE_TRIGGER = 'ns=3;s="MAIN"."iCommande"'  # Entrée (1=Forme, 2=Rouille)
NODE_RES_CLASSE = 'ns=3;s="MAIN"."iResultatClasse"'  # Sortie Forme (Int)
NODE_RES_ROUILLE = 'ns=3;s="MAIN"."bResultatRouille"'  # Sortie Rouille (Bool)

# --- PARAMÈTRES IA & VISION ---
ENGINE_PATH = "model_rouille.engine"
CONF_THRESHOLD_ROUILLE = 0.50
COOLDOWN_DELAY = 10  # Secondes avant de pouvoir relancer une analyse


# =========================================================================
#  MODULE CAMÉRA (WEBCAM USB)
# =========================================================================

def prendre_photo():
    """
    Capture une image via la Webcam USB.
    Plus simple et plus robuste que GStreamer pour ce projet.
    """
    print(" Tentative de prise de photo (USB)...")

    # 1. Essai sur l'index 0 (Première caméra USB)
    cap = cv2.VideoCapture(0)

    # 2. Si échec, essai index 1 (cas où l'index 0 est réservé par le système)
    if not cap.isOpened():
        print(" Index 0 échoué, essai Index 1...")
        cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        print(" ERREUR CRITIQUE : Aucune Webcam USB détectée.")
        return None

    # Configuration standard (640x480 est suffisant et rapide)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # ⏳ TEMPS DE CHAUFFE IMPORTANT
    # Les webcams ont besoin de 1 à 2 sec pour régler la lumière (Auto-Exposure)
    # Sinon la photo sera noire ou très sombre.
    time.sleep(1.0)

    # Lecture de l'image
    ret, frame = cap.read()
    cap.release()  # On libère la caméra immédiatement après

    if not ret:
        print(" ERREUR : La caméra est ouverte mais n'envoie pas d'image.")
        return None

    print(f" Photo capturée ({frame.shape[1]}x{frame.shape[0]}px).")
    return frame


# =========================================================================
#  MODULE 1 : ANALYSE FORME (OPENCV CLASSIQUE)
# =========================================================================

def analyse_forme(img):
    """
    Retourne : 0 (Carré), 1 (Cercle), 2 (Ambigu)
    """
    print(" Démarrage Analyse Forme...")
    try:
        # Conversion en gris
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # Prétraitement (Réduction bruit + éclairage)
        kernel_tophat = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel_tophat)
        gray_ready = cv2.addWeighted(gray, 0.6, tophat, 0.4, 0)

        # Flou pour lisser les bords
        blur = cv2.GaussianBlur(gray_ready, (5, 5), 0)

        # Binarisation (Noir et Blanc) avec méthode Otsu
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Fermeture pour combler les petits trous
        closed_mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
                                       iterations=2)

        # Recherche des contours
        contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_cnt = None
        best_area = 0

        # On cherche le plus grand objet
        for c in contours:
            area = cv2.contourArea(c)
            if area < 500: continue  # On ignore le bruit (poussière)
            if area > best_area:
                best_area = area
                best_cnt = c

        if best_cnt is None:
            print("️ Aucun objet détecté.")
            return 2  # Ambigu

        # Calcul Circularité
        perimeter = cv2.arcLength(best_cnt, True)
        if perimeter == 0: return 2
        circularity = 4 * np.pi * (best_area / (perimeter * perimeter))

        print(f"   -> Circularité mesurée : {circularity:.3f}")

        if circularity >= 0.85:
            return 1  # CERCLE
        elif circularity <= 0.82:
            return 0  # CARRE
        else:
            return 2  # AMBIGU

    except Exception as e:
        print(f" Erreur Algo Forme: {e}")
        return 2


# =========================================================================
# 🧠 MODULE 2 : ANALYSE ROUILLE (IA TENSORRT)
# =========================================================================

class RustDetector:
    def __init__(self, engine_path):
        print(f"🤖 Chargement Moteur IA: {engine_path}")
        self.logger = trt.Logger(trt.Logger.WARNING)
        try:
            with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
                self.engine = runtime.deserialize_cuda_engine(f.read())
        except FileNotFoundError:
            print(f" ERREUR: Le fichier '{engine_path}' est introuvable !")
            sys.exit(1)

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
        # Resize 640x640 pour YOLO
        img_resized = cv2.resize(img, (640, 640))
        img_in = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB).transpose((2, 0, 1)).astype(np.float32)
        img_in /= 255.0
        img_in = np.expand_dims(img_in, axis=0)

        # Inférence GPU
        np.copyto(self.inputs[0]['host'], img_in.ravel())
        cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(self.outputs[0]['host'], self.outputs[0]['device'], self.stream)
        self.stream.synchronize()

        # Lecture résultats
        output = self.outputs[0]['host'].reshape(1, 5, 8400)
        # On suppose que la classe Rouille est la plus probable si le score est haut
        scores = output[0, 4, :]
        max_score = np.max(scores)

        print(f"   -> Confiance Rouille Max : {max_score:.2%}")
        return 1 if max_score > CONF_THRESHOLD_ROUILLE else 0


# =========================================================================
#  BOUCLE PRINCIPALE
# =========================================================================

def main():
    print("--- DÉMARRAGE DU SYSTÈME (Mode Webcam USB) ---")

    # 1. Chargement de l'IA
    rust_detector = RustDetector(ENGINE_PATH)

    # 2. Connexion Automate
    client = Client(PLC_URL)
    connected = False

    try:
        print(f" Connexion à l'automate {PLC_URL}...")
        client.connect()
        print(" PLC CONNECTÉ.")
        node_trigger = client.get_node(NODE_TRIGGER)
        node_res_classe = client.get_node(NODE_RES_CLASSE)
        node_res_rouille = client.get_node(NODE_RES_ROUILLE)
        connected = True
    except Exception as e:
        print(f" ÉCHEC CONNEXION PLC ({e}). Mode Simulation/Attente.")
        client = None

    last_execution_time = 0

    print("\n--- SYSTÈME PRÊT. EN ATTENTE... ---")

    while True:
        try:
            trigger = 0

            # A. Lecture OPC UA (Si connecté)
            if connected:
                try:
                    trigger = int(node_trigger.get_value())
                except:
                    print("️ Perte connexion PLC, tentative reconnexion...")
                    try:
                        client.disconnect()
                        client.connect()
                        print(" Reconnecté.")
                    except:
                        pass

            # --- ZONE DE TEST MANUEL (A COMMENTER EN PROD) ---
            # Si pas de PLC, tu peux décommenter les lignes ci-dessous pour tester au clavier
            # if not connected:
            #     t = input("Simulation (1=Forme, 2=Rouille) : ")
            #     if t == '1': trigger = 1
            #     elif t == '2': trigger = 2
            # -------------------------------------------------

            # B. Logique de déclenchement
            current_time = time.time()

            if trigger in [1, 2]:
                # Vérification du délai (Cooldown)
                if (current_time - last_execution_time) > COOLDOWN_DELAY:
                    print(f"\n⚡ ORDRE REÇU : {trigger}")

                    # 1. Prise de photo (Webcam USB)
                    img = prendre_photo()

                    if img is not None:
                        # 2. Traitement
                        if trigger == 1:
                            # --- ANALYSE FORME ---
                            res = analyse_forme(img)
                            print(f" Résultat CLASSE : {res}")
                            if connected:
                                try:
                                    # Envoi en Int16 pour Siemens
                                    dv = ua.DataValue(ua.Variant(int(res), ua.VariantType.Int16))
                                    node_res_classe.set_value(dv)
                                except:
                                    print(" Erreur écriture PLC")

                        elif trigger == 2:
                            # --- ANALYSE ROUILLE ---
                            res = rust_detector.detect(img)
                            print(f" Résultat ROUILLE : {res}")
                            if connected:
                                try:
                                    # Envoi en Boolean pour Siemens
                                    dv = ua.DataValue(ua.Variant(bool(res), ua.VariantType.Boolean))
                                    node_res_rouille.set_value(dv)
                                except:
                                    print(" Erreur écriture PLC")

                        # Reset Timer
                        last_execution_time = time.time()
                        print(f" Pause de {COOLDOWN_DELAY}s...")
                    else:
                        print(" Échec Webcam, abandon.")

                else:
                    # On ignore l'ordre car on est en pause
                    pass

            time.sleep(0.1)  # Petite pause CPU

        except KeyboardInterrupt:
            print("\nArrêt manuel.")
            break
        except Exception as e:
            print(f"Erreur boucle: {e}")
            time.sleep(1)

    if connected: client.disconnect()


if __name__ == "__main__":
    main()