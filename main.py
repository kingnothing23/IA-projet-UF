#!/usr/bin/python3
import cv2
import numpy as np
import sys
import time
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from opcua import Client
from opcua import ua  # Nécessaire pour forcer les types Siemens (Int16, Boolean)

# =========================================================================
#  CONFIGURATION GLOBALE
# =========================================================================

# --- OPC UA (AUTOMATE SIEMENS) ---
PLC_URL = "opc.tcp://192.168.0.26:4840"

# NodeIDs (Adresses des variables dans l'automate)
# Vérifie bien ces ID avec UaExpert si jamais ça ne connecte pas.
NODE_TRIGGER = 'ns=3;s="MAIN"."iCommande"'  # Entrée (1=Forme, 2=Rouille)
NODE_RES_CLASSE = 'ns=3;s="MAIN"."iResultatClasse"'  # Sortie Forme (Int)
NODE_RES_ROUILLE = 'ns=3;s="MAIN"."bResultatRouille"'  # Sortie Rouille (Bool)

# --- PARAMÈTRES VISION & IA ---
ENGINE_PATH = "model_rouille.engine"
CONF_THRESHOLD_ROUILLE = 0.50  # 50% de confiance minimum
COOLDOWN_DELAY = 10  # Pause en secondes entre deux analyses


# =========================================================================
#  MODULE CAMÉRA (WEBCAM USB)
# =========================================================================

def prendre_photo():
    """ Capture une image via la Webcam USB (0 ou 1). """
    print(" Tentative de prise de photo (USB)...")

    # Essai index 0
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print(" Index 0 échoué, essai Index 1...")
        cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        print(" ERREUR CRITIQUE : Aucune Webcam USB détectée.")
        return None

    # Config 640x480
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Temps de chauffe (Auto-exposure)
    time.sleep(1.0)

    ret, frame = cap.read()
    cap.release()

    if not ret:
        print(" ERREUR : La caméra est ouverte mais l'image est vide.")
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
        # 1. Niveaux de gris
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 2. Top-Hat (Correction éclairage)
        kernel_tophat = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
        # CORRECTION SYNTAXE ICI
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel_tophat)
        gray_ready = cv2.addWeighted(gray, 0.6, tophat, 0.4, 0)

        # 3. Flou
        blur = cv2.GaussianBlur(gray_ready, (5, 5), 0)

        # 4. Otsu (Seuillage auto)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 5. Fermeture (Combler les trous)
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        # CORRECTION SYNTAXE ICI
        closed_mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_close, iterations=2)

        # 6. Contours
        contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_cnt = None
        best_area = 0

        for c in contours:
            area = cv2.contourArea(c)
            if area < 500: continue  # Filtre bruit
            if area > best_area:
                best_area = area
                best_cnt = c

        if best_cnt is None:
            print(" Aucun objet détecté.")
            return 2  # Ambigu

        # 7. Calcul Circularité
        perimeter = cv2.arcLength(best_cnt, True)
        if perimeter == 0: return 2

        circularity = 4 * np.pi * (best_area / (perimeter * perimeter))
        print(f"   -> Circularité mesurée : {circularity:.3f}")

        # Seuils de décision
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
#  MODULE 2 : ANALYSE ROUILLE (IA TENSORRT)
# =========================================================================

class RustDetector:
    def __init__(self, engine_path):
        print(f" Chargement Moteur IA: {engine_path}")
        self.logger = trt.Logger(trt.Logger.WARNING)
        try:
            with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
                self.engine = runtime.deserialize_cuda_engine(f.read())
        except FileNotFoundError:
            print(f" ERREUR: Le fichier '{engine_path}' est introuvable !")
            sys.exit(1)

        self.context = self.engine.create_execution_context()
        self.inputs, self.outputs, self.bindings, self.stream = [], [], [], cuda.Stream()

        # Allocation mémoire GPU
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

        # Preprocessing YOLO (640x640, RGB, Normalize)
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

        # Post-processing
        output = self.outputs[0]['host'].reshape(1, 5, 8400)
        scores = output[0, 4, :]

        # CORRECTION SYNTAXE ICI
        max_score = np.max(scores)

        print(f"   -> Confiance Rouille Max : {max_score:.2%}")
        return 1 if max_score > CONF_THRESHOLD_ROUILLE else 0


# =========================================================================
#  BOUCLE PRINCIPALE (MAIN)
# =========================================================================

def main():
    print("--- DÉMARRAGE DU SYSTÈME OPC UA ---")

    # 1. Chargement IA
    rust_detector = RustDetector(ENGINE_PATH)

    # 2. Connexion Automate
    client = Client(PLC_URL)
    connected = False

    try:
        print(f" Connexion à l'automate {PLC_URL}...")
        client.connect()
        print(" PLC CONNECTÉ.")

        # Récupération des noeuds (Pointeurs vers les variables)
        node_trigger = client.get_node(NODE_TRIGGER)
        node_res_classe = client.get_node(NODE_RES_CLASSE)
        node_res_rouille = client.get_node(NODE_RES_ROUILLE)
        connected = True
    except Exception as e:
        print(f" ÉCHEC CONNEXION PLC : {e}")
        print(" -> Vérifiez câble, IP, ou réglage 'No Security' dans TIA Portal.")
        # On continue quand même pour tester la caméra si besoin, ou on quitte :
        # sys.exit(1)

    last_execution_time = 0

    print("\n--- EN ATTENTE D'ORDRES (Polling) ---")

    while True:
        try:
            trigger = 0

            # A. Lecture OPC UA
            if connected:
                try:
                    val = node_trigger.get_value()
                    trigger = int(val)
                except Exception as e:
                    print(f" Perte connexion PLC : {e}")
                    connected = False
                    # Tentative reconnexion rapide
                    try:
                        client.disconnect()
                        client.connect()
                        connected = True
                        print(" Reconnecté.")
                    except:
                        pass

            current_time = time.time()

            # B. Traitement si ordre reçu (1 ou 2)
            if trigger in [1, 2]:
                # Anti-spam (Cooldown)
                if (current_time - last_execution_time) > COOLDOWN_DELAY:
                    print(f"\n ORDRE REÇU : {trigger}")

                    # 1. Photo
                    img = prendre_photo()

                    if img is not None:
                        # 2. Analyse
                        if trigger == 1:
                            # FORME
                            res = analyse_forme(img)
                            print(f" Résultat CLASSE à envoyer : {res}")
                            if connected:
                                try:
                                    # Ecriture Int16
                                    dv = ua.DataValue(ua.Variant(int(res), ua.VariantType.Int16))
                                    node_res_classe.set_value(dv)
                                    print(" Donnée envoyée (iResultatClasse).")
                                except Exception as e:
                                    print(f" Erreur écriture PLC: {e}")

                        elif trigger == 2:
                            # ROUILLE
                            res = rust_detector.detect(img)
                            print(f" Résultat ROUILLE à envoyer : {res}")
                            if connected:
                                try:
                                    # Ecriture Boolean
                                    dv = ua.DataValue(ua.Variant(bool(res), ua.VariantType.Boolean))
                                    node_res_rouille.set_value(dv)
                                    print(" Donnée envoyée (bResultatRouille).")
                                except Exception as e:
                                    print(f" Erreur écriture PLC: {e}")

                        last_execution_time = time.time()
                    else:
                        print(" Erreur Caméra (Image None).")

                else:
                    # On est en pause (Cooldown), on ignore l'ordre
                    pass

            # Petite pause pour ne pas surcharger le CPU
            time.sleep(0.1)

        except KeyboardInterrupt:
            print("\n Arrêt manuel (CTRL+C).")
            break
        except Exception as e:
            print(f" Erreur inattendue dans la boucle : {e}")
            time.sleep(1)

    if connected:
        client.disconnect()


if __name__ == "__main__":
    main()