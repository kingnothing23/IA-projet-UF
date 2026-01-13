import cv2
import numpy as np
import sys

# === PARAMÈTRES ===
IMG_PATH = "g.jpg"
CIRCULARITY_THRESHOLD_CIRCLE = 0.89
CIRCULARITY_THRESHOLD_SQUARE = 0.82

# === CONFIGURATION AFFICHAGE (Booleens) ===
# Mettre à False sur la Nano si pas d'écran branché
affichage_masque_radial = True
affichage_gradient = True
affichage_binaire = True
affichage_contours = True
affichage_resultat = True


def smart_show(window_name, image, width=800):
    """Affiche l'image seulement si la fenêtre est active"""
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    h, w = image.shape[:2]
    ratio = width / float(w)
    cv2.resizeWindow(window_name, width, int(h * ratio))
    cv2.imshow(window_name, image)


# === 1. LECTURE & PRÉPARATION ===
img = cv2.imread(IMG_PATH)
if img is None:
    print(f"Erreur: Image {IMG_PATH} introuvable")
    sys.exit()

h, w = img.shape[:2]
center_x, center_y = w // 2, h // 2
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Top-Hat léger
kernel_tophat = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel_tophat)
gray_ready = cv2.addWeighted(gray, 0.6, tophat, 0.4, 0)

# === 2. MASQUE RADIAL ===
Y, X = np.ogrid[:h, :w]
dist_from_center = np.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)
max_dist = min(h, w) * 0.65

radial_mask = 1.0 - (dist_from_center / max_dist)
radial_mask = np.clip(radial_mask, 0, 1)

if affichage_masque_radial:
    smart_show("0. Masque Radial Doux", (radial_mask * 255).astype(np.uint8))

# === 3. SOBEL (DÉTECTION D'ÉNERGIE) ===
blur = cv2.GaussianBlur(gray_ready, (5, 5), 0)

grad_x = cv2.Sobel(blur, cv2.CV_16S, 1, 0, ksize=3)
grad_y = cv2.Sobel(blur, cv2.CV_16S, 0, 1, ksize=3)
abs_grad_x = cv2.convertScaleAbs(grad_x)
abs_grad_y = cv2.convertScaleAbs(grad_y)
gradient_magnitude = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

# Application du masque radial
weighted_gradient = gradient_magnitude.astype(np.float32) * radial_mask
weighted_gradient = weighted_gradient.astype(np.uint8)

if affichage_gradient:
    smart_show("1. Gradient Pondere", weighted_gradient)

# === 4. SEUILLAGE ET NETTOYAGE ===
_, thresh = cv2.threshold(weighted_gradient, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
closed_mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_close, iterations=2)

if affichage_binaire:
    smart_show("2. Masque Binaire", closed_mask)

# === 5. SÉLECTION INTELLIGENTE (LE SCORE) ===
contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

best_contour = None
best_score = 0
debug_img = img.copy()

for c in contours:
    area = cv2.contourArea(c)
    if area < 200: continue

    M = cv2.moments(c)
    if M["m00"] == 0: continue
    cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])

    dist = np.sqrt((cx - center_x) ** 2 + (cy - center_y) ** 2)
    penalty_factor = 1 + (dist / (min(h, w) * 0.2)) ** 2
    score = area / penalty_factor

    color = (0, 0, 255)
    if score > best_score:
        best_score = score
        best_contour = c
        color = (0, 255, 0)

    if affichage_contours:
        cv2.drawContours(debug_img, [c], -1, color, 2)

if affichage_contours:
    smart_show("3. Tri des contours (Vert=Gagnant)", debug_img)

if best_contour is None:
    print("ECHEC : Rien de probant trouve.")
    # On peut définir une classe d'erreur ici si besoin (ex: -1)
    sys.exit()

# === 6. ANALYSE ET RÉSULTAT ===
contour_final = best_contour
area = cv2.contourArea(contour_final)
perimeter = cv2.arcLength(contour_final, True)

if perimeter == 0:
    circularity = 0
else:
    circularity = 4 * np.pi * (area / (perimeter * perimeter))

# Détermination de la CLASSE et du TYPE
shape_type = "INCONNU"
classe = 2  # Par défaut : Ambigu

if circularity >= CIRCULARITY_THRESHOLD_CIRCLE:
    shape_type = "CERCLE"
    classe = 1
elif circularity <= CIRCULARITY_THRESHOLD_SQUARE:
    shape_type = "CARRE/AUTRE"
    classe = 0
else:
    shape_type = "AMBIGU"
    classe = 2

# === 7. SORTIE ===
print(f"Termine. Resultat: {shape_type} (Circularite: {circularity:.3f})")
print(f"Classe ID: {classe}")  # 0=Carré, 1=Cercle, 2=Ambigu

if affichage_resultat:
    output = img.copy()

    # Couleur selon la classe
    if classe == 1:
        color_res = (0, 255, 0)  # Vert pour Cercle
    elif classe == 0:
        color_res = (255, 0, 0)  # Bleu pour Carré
    else:
        color_res = (0, 165, 255)  # Orange pour Ambigu

    cv2.drawContours(output, [contour_final], -1, color_res, 3)

    label = f"{shape_type} ({circularity:.3f}) [Cls:{classe}]"
    cv2.putText(output, label, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color_res, 2)

    smart_show("4. Resultat Final", output)

    # Attendre une touche uniquement si on affiche quelque chose
    cv2.waitKey(0)
    cv2.destroyAllWindows()