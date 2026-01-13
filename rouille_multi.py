import cv2
import numpy as np
import sys
import os

# =========================================================================
# ⚙️ CONFIGURATION (C'est ici que tu touches)
# =========================================================================

# Sur quelle machine es-tu ? "PC" ou "NANO"
MODE = "NANO"

# Chemin de l'image à tester
IMG_PATH = "rou_1.jpg"

# Chemins des modèles
MODEL_PATH_PC = "best.pt"  # Pour le PC
MODEL_PATH_NANO = "model_rouille.engine"  # Pour la Nano

# Paramètres de détection
CONF_THRESHOLD = 0.50  # 50% de confiance minimum

# Affichage (True = ouvre une fenêtre, False = mode silencieux)
AFFICHAGE_ACTIF = True


# =========================================================================
# 🔧 PARTIE TECHNIQUE (Ne pas toucher sauf si expert)
# =========================================================================

class Detector:
    def __init__(self, mode):
        self.mode = mode
        self.model = None
        self.context = None
        self.inputs = None
        self.outputs = None
        self.bindings = None
        self.stream = None

        if self.mode == "PC":
            print("💻 Initialisation Mode PC (Ultralytics)...")
            try:
                from ultralytics import YOLO
                self.model = YOLO(MODEL_PATH_PC)
            except ImportError:
                print("❌ ERREUR: Librairie 'ultralytics' manquante. Fais 'pip install ultralytics'")
                sys.exit()
            except Exception as e:
                print(f"❌ ERREUR Chargement Modèle PC: {e}")
                sys.exit()

        elif self.mode == "NANO":
            print("🔌 Initialisation Mode NANO (TensorRT)...")
            try:
                import tensorrt as trt
                import pycuda.driver as cuda
                import pycuda.autoinit
            except ImportError:
                print("❌ ERREUR: Librairies Nano (tensorrt/pycuda) manquantes.")
                sys.exit()

            try:
                self.logger = trt.Logger(trt.Logger.WARNING)
                with open(MODEL_PATH_NANO, "rb") as f, trt.Runtime(self.logger) as runtime:
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
            except Exception as e:
                print(f"❌ ERREUR Chargement Modèle NANO: {e}")
                sys.exit()

    def detect(self, img):
        """
        Retourne : (rouille_int, max_score, annotated_img)
        rouille_int : 0 ou 1
        """
        rouille = 0
        max_score = 0.0
        annotated_img = img.copy()

        # --- LOGIQUE PC ---
        if self.mode == "PC":
            results = self.model.predict(img, conf=CONF_THRESHOLD, verbose=False)
            result = results[0]

            # Récupérer les infos
            if len(result.boxes) > 0:
                rouille = 1
                max_score = float(result.boxes.conf.max())
                # Ultralytics dessine déjà très bien les boîtes
                annotated_img = result.plot()
            else:
                rouille = 0
                max_score = 0.0

        # --- LOGIQUE NANO ---
        elif self.mode == "NANO":
            import pycuda.driver as cuda

            # 1. Preprocessing
            input_h, input_w = 640, 640
            img_resized = cv2.resize(img, (input_w, input_h))
            img_in = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            img_in = img_in.transpose((2, 0, 1)).astype(np.float32)
            img_in /= 255.0
            img_in = np.expand_dims(img_in, axis=0)

            # 2. Inférence
            np.copyto(self.inputs[0]['host'], img_in.ravel())
            cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
            self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
            cuda.memcpy_dtoh_async(self.outputs[0]['host'], self.outputs[0]['device'], self.stream)
            self.stream.synchronize()

            # 3. Post-processing (Simplifié : on cherche juste le score max)
            output = self.outputs[0]['host']
            output = output.reshape(1, 5, 8400)  # [Batch, 4coords+1conf, Anchors]
            scores = output[0, 4, :]  # Ligne des scores
            max_score = float(np.max(scores))

            if max_score > CONF_THRESHOLD:
                rouille = 1
                # Dessin manuel simple (Juste le statut, pas les boîtes compliquées)
                text = f"NOK: ROUILLE ({max_score:.0%})"
                color = (0, 0, 255)
            else:
                rouille = 0
                text = f"OK: PROPRE ({max_score:.0%})"
                color = (0, 255, 0)

            cv2.putText(annotated_img, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        return rouille, max_score, annotated_img


# =========================================================================
# 🚀 EXÉCUTION
# =========================================================================

if __name__ == "__main__":
    # 1. Vérif image
    if not os.path.exists(IMG_PATH):
        print(f"❌ Image introuvable : {IMG_PATH}")
        print("Mets une image test dans le dossier !")
        sys.exit()

    # 2. Chargement image
    img_bgr = cv2.imread(IMG_PATH)

    # 3. Init Détecteur
    detector = Detector(MODE)

    # 4. Détection
    print(f"🔍 Analyse de {IMG_PATH}...")
    var_rouille, score, img_result = detector.detect(img_bgr)

    # 5. Résultats dans la console
    print("-" * 30)
    print(f"📝 RÉSULTAT FINAL :")
    print(f"   Variable 'rouille' = {var_rouille}")
    print(f"   Confiance max      = {score:.2%}")
    print("-" * 30)

    # 6. Affichage (Si activé)
    if AFFICHAGE_ACTIF:
        window_name = f"Resultat ({MODE})"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        # Resize intelligent pour ne pas dépasser l'écran
        h, w = img_result.shape[:2]
        if h > 800:
            ratio = 800 / h
            cv2.resizeWindow(window_name, int(w * ratio), 800)

        cv2.imshow(window_name, img_result)
        print("Appuie sur une touche pour fermer...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()