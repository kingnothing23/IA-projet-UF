import cv2
import numpy as np
import os

# === PARAMÈTRES À AJUSTER ===
CIRCULARITY_THRESHOLD = 0.88     # <--- plus haut = plus strict (forme)
GRADIENT_VARIANCE_THRESHOLD = 20000  # <--- plus bas = plus sensible (surface)
IMG_PATH = "google2.jpg"       # <--- chemin vers ton image à tester


#FONCTIONS





# === LECTURE ET PRÉTRAITEMENT ===
img = cv2.imread(IMG_PATH)
if img is None:
    raise ValueError("❌ Impossible de charger l'image. Vérifie le chemin.")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# AJOUT : Utilisation de CLAHE pour améliorer le contraste local
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)) # Ajustez clipLimit si besoin
contrasted_gray = clahe.apply(blur) # Appliquer CLAHE au flou


# AJOUT : Utilisation de CLAHE pour améliorer le contraste local
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)) # Ajustez clipLimit si besoin
contrasted_gray = clahe.apply(blur) # Appliquer CLAHE au flou

# Seuil adaptatif pour isoler la pièce
thresh = cv2.adaptiveThreshold(
    contrasted_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 8
)
# Nettoyage du masque
kernel = np.ones((5,5), np.uint8)
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

# === DÉTECTION DE CONTOUR ===
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
if not contours:
    raise ValueError("❌ Aucun contour détecté.")

# Sélection du contour principal
contour = max(contours, key=cv2.contourArea)
area = cv2.contourArea(contour)
perimeter = cv2.arcLength(contour, True)

# === CALCUL DE LA CIRCULARITÉ ===
if perimeter == 0:
    circularity = 0
else:
    circularity = 4 * np.pi * (area / (perimeter * perimeter))

# === DÉTECTION DES PLIS / BOSSAGES ===
mask = np.zeros_like(gray)
cv2.drawContours(mask, [contour], -1, 255, -1)
masked_gray = cv2.bitwise_and(gray, gray, mask=mask)

# Carte de gradient (Sobel)
grad_x = cv2.Sobel(masked_gray, cv2.CV_64F, 1, 0, ksize=3)
grad_y = cv2.Sobel(masked_gray, cv2.CV_64F, 0, 1, ksize=3)
gradient_magnitude = cv2.magnitude(grad_x, grad_y)

# Calcul de la variance du gradient
gradient_variance = np.var(gradient_magnitude[mask == 255])

# === DÉCISION ===
is_shape_ok = circularity >= CIRCULARITY_THRESHOLD
is_surface_ok = gradient_variance <= GRADIENT_VARIANCE_THRESHOLD
piece_ok = is_shape_ok and is_surface_ok

# === AFFICHAGE ===
output = img.copy()
color = (0, 255, 0) if piece_ok else (0, 0, 255)
cv2.drawContours(output, [contour], -1, color, 2)

text = "PIECE OK" if piece_ok else "DEFAUT"
cv2.putText(output, f"{text}", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
cv2.putText(output, f"Circ: {circularity:.2f}  Var: {gradient_variance:.1f}",
            (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (41, 110, 122), 2)

cv2.imshow("Inspection Capsule", output)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Optionnel : sauvegarde du résultat
os.makedirs("results", exist_ok=True)
cv2.imwrite(f"results/result_{os.path.basename(IMG_PATH)}", output)
print(f"✅ Analyse terminée. Résultat enregistré dans /results/")
print(f"Circularité = {circularity:.2f}")
print(f"Variance du gradient = {gradient_variance:.1f}")
