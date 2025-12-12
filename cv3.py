import cv2
import numpy as np
import os

# === PARAMÈTRES À AJUSTER ===
CIRCULARITY_THRESHOLD = 0.88     # <--- plus haut = plus strict (forme)
GRADIENT_VARIANCE_THRESHOLD = 20000  # <--- plus bas = plus sensible (surface)
IMG_PATH = "hexa2.png"       # <--- chemin vers ton image à tester

# === LECTURE ET PRÉTRAITEMENT ===
img = cv2.imread(IMG_PATH)
if img is None:
    raise ValueError("❌ Impossible de charger l'image. Vérifie le chemin.")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# --- DÉBUT MODIFICATION : UTILISATION DE CLAHE ---
# CLAHE (Contrast Limited Adaptive Histogram Equalization) pour améliorer le contraste local
# Ajustez clipLimit (e.g., de 1.5 à 4.0) si nécessaire pour le contraste
clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(8,8))
contrasted_gray = clahe.apply(blur)
# --- FIN MODIFICATION ---
if contrasted_gray is None:
    print("----------------------------------------------------------------------")
    print("ERREUR FATALE: L'image contrastée est vide. L'image source est-elle bien chargée ?")
    # Vérifiez la source de contrasted_gray
    if img is None:
        print(f"CAUSE: L'image d'origine ({IMG_PATH}) n'a pas été trouvée ou chargée.")
    elif gray is None:
        print("CAUSE: La conversion en niveaux de gris a échoué.")
    elif blur is None:
        print("CAUSE: Le flou gaussien a échoué.")
    print("----------------------------------------------------------------------")
    exit()

# Seuil adaptatif pour isoler la pièce
# On utilise l'image contrastée par CLAHE
# Seuil adaptatif pour isoler la pièce
_, thresh = cv2.threshold(
    contrasted_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
)

if thresh is None:
    print("----------------------------------------------------------------------")
    print("DIAGNOSTIC CRITIQUE: 'thresh' est NONE après cv2.threshold.")
    print("----------------------------------------------------------------------")
    exit()
# === FIN DU NOUVEAU BLOC ===
# Nettoyage du masque
kernel = np.ones((5,5), np.uint8)
# Fermeture pour combler les petits trous, Ouverture pour lisser et supprimer le bruit
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=20)
thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

# === AFFICHAGE INTERMÉDIAIRE DU MASQUE BINAIRE ===
cv2.imshow("1. Masque Binaire (Thresh) pour Contour", thresh)
# ================================================


# === DÉTECTION DE CONTOUR ===
# CORRECTION IMPORTANTE : Utiliser CHAIN_APPROX_NONE pour un périmètre précis des courbes
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
if not contours:
    raise ValueError("❌ Aucun contour détecté après seuillage.")

# Sélection du contour principal (le plus grand)
contour = max(contours, key=cv2.contourArea)
area = cv2.contourArea(contour)
perimeter = cv2.arcLength(contour, True)


# === CALCUL DE LA CIRCULARITÉ ===
if perimeter == 0:
    circularity = 0
else:
    # La formule est correcte pour mesurer la circularité
    circularity = 4 * np.pi * (area / (perimeter * perimeter))

# === DÉTECTION DES PLIS / BOSSAGES ===
# Ce calcul utilise toujours l'image en NIVEAUX DE GRIS (gray) pour détecter la texture
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

# === AFFICHAGE DU RÉSULTAT FINAL ===
output = img.copy()
color = (0, 255, 0) if piece_ok else (0, 0, 255)
cv2.drawContours(output, [contour], -1, color, 2)

text = "PIECE OK" if piece_ok else "DEFAUT"
cv2.putText(output, f"{text}", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
cv2.putText(output, f"Circ: {circularity:.2f}  Var: {gradient_variance:.1f}",
            (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (41, 110, 122), 2)

# Affichage du résultat final
cv2.imshow("2. Inspection Capsule (Resultat)", output)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Optionnel : sauvegarde du résultat
os.makedirs("results", exist_ok=True)
cv2.imwrite(f"results/result_{os.path.basename(IMG_PATH)}", output)
print(f"✅ Analyse terminée. Résultat enregistré dans /results/")
print(f"Circularité = {circularity:.2f}")
print(f"Variance du gradient = {gradient_variance:.1f}")