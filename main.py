#!/usr/bin/python3
import cv2
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
import time
from opcua import Client

# ================= CONFIGURATION =================
ENGINE_PATH = "model_rouille.engine"
CONF_THRESHOLD = 0.50  # 50% de confiance pour déclencher
INPUT_SIZE = 640

# CONFIG OPC UA (SIEMENS)
PLC_URL = "opc.tcp://192.168.0.1:4840" # <-- Mets l'IP de ton automate ici
NODE_ID = "ns=4;s=MAIN.bRouille"       # <-- Mets l'adresse de ta variable ici

# ================= FONCTION PI CAMERA (GSTREAMER) =================
def gstreamer_pipeline(
    sensor_id=0,
    capture_width=1280,
    capture_height=720,
    display_width=640,
    display_height=360,
    framerate=30,
    flip_method=0,
):
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

# ================= CLASSE D'INFERENCE TENSORRT =================
class TRTInference:
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        print(f"🔄 Chargement du moteur {engine_path}...")
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

    def infer(self, image):
        # 1. Prétraitement (Resize + Normalisation 0-1)
        img_resized = cv2.resize(image, (INPUT_SIZE, INPUT_SIZE))
        img_in = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_in = img_in.transpose((2, 0, 1)).astype(np.float32)
        img_in /= 255.0
        img_in = np.expand_dims(img_in, axis=0)
        
        # 2. Copie CPU -> GPU
        np.copyto(self.inputs[0]['host'], img_in.ravel())
        cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
        
        # 3. Exécution IA
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        
        # 4. Copie GPU -> CPU
        cuda.memcpy_dtoh_async(self.outputs[0]['host'], self.outputs[0]['device'], self.stream)
        self.stream.synchronize()
        
        return self.outputs[0]['host']

# ================= PROGRAMME PRINCIPAL =================
def main():
    # 1. Init IA
    try:
        trt_model = TRTInference(ENGINE_PATH)
        print("✅ Moteur IA prêt.")
    except Exception as e:
        print(f"❌ Erreur chargement moteur: {e}")
        return

    # 2. Init Caméra
    print("📷 Démarrage Caméra Pi V2...")
    # On tente d'ouvrir avec GStreamer (pour la Pi Cam)
    cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
    
    if not cap.isOpened():
        print("⚠️ Echec GStreamer, tentative Webcam USB standard...")
        cap = cv2.VideoCapture(0) # Fallback USB
        if not cap.isOpened():
            print("❌ Aucune caméra trouvée.")
            return

    # 3. Init Automate (PLC)
    client = None
    var_rouille = None
    try:
        client = Client(PLC_URL)
        client.connect()
        var_rouille = client.get_node(NODE_ID)
        print(f"✅ Connecté à l'automate Siemens ({PLC_URL})")
    except:
        print("⚠️ Automate non détecté -> Mode Simulation")

    print("\n--- INSTRUCTION : Appuie sur 'q' pour quitter ---\n")

    while True:
        ret, frame = cap.read()
        if not ret: break

        start_time = time.time()
        
        # --- ANALYSE IA ---
        output = trt_model.infer(frame)
        
        # --- INTERPRETATION ---
        # YOLOv8 sort un tableau [1, 5, 8400] -> (cx, cy, w, h, proba_rouille)
        output = output.reshape(1, 5, 8400)
        
        # On récupère la ligne 4 (Probabilité de rouille)
        rust_probs = output[0, 4, :] 
        max_score = np.max(rust_probs)
        
        is_rusty = max_score > CONF_THRESHOLD
        
        fps = 1.0 / (time.time() - start_time)

        # --- AFFICHAGE RESULTAT ---
        if is_rusty:
            text = f"NOK: ROUILLE ({max_score:.0%})"
            color = (0, 0, 255) # Rouge
        else:
            text = f"OK: PROPRE ({max_score:.0%})"
            color = (0, 255, 0) # Vert
            
        cv2.putText(frame, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

        # Indicateur PLC
        plc_status = "PLC: CONNECTÉ" if client else "PLC: OFF"
        cv2.putText(frame, plc_status, (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow("Inspection IA", frame)

        # --- ENVOI VERS AUTOMATE ---
        if client and var_rouille:
            try:
                # Si rouille -> True, Sinon -> False
                var_rouille.set_value(bool(is_rusty))
            except:
                pass # On évite de faire planter la vidéo si le câble réseau bouge

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    if client: client.disconnect()

if __name__ == "__main__":
    main()
