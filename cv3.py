import cv2
import numpy as np

# === PARAMÈTRES ===
IMG_PATH = "80.jpg"  # Testez sur le carré ET sur le bruit
CIRCULARITY_THRESHOLD_CIRCLE = 0.89
CIRCULARITY_THRESHOLD_SQUARE = 0.82


def smart_show(window_name, image, width=800):
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    h, w = image.shape[:2]
    ratio = width / float(w)
    cv2.resizeWindow(window_name, width, int(h * ratio))
    cv2.imshow(window_name, image)


# === 1. LECTURE & PRÉPARATION ===
img = cv2.imread(IMG_PATH)
if img is None: raise ValueError("Image introuvable")
h, w = img.shape[:2]
center_x, center_y = w // 2, h // 2
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Top-Hat léger (juste pour égaliser, sans être destructif)
kernel_tophat = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel_tophat)
# On mélange l'original et le tophat pour garder de la substance sur le carré
gray_ready = cv2.addWeighted(gray, 0.6, tophat, 0.4, 0)

# === 2. MASQUE RADIAL (LINÉAIRE - Version Douce) ===
# On revient au masque linéaire qui ne "mange" pas le carré
Y, X = np.ogrid[:h, :w]
dist_from_center = np.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)
max_dist = min(h, w) * 0.65  # Rayon un peu plus large pour ne pas couper le carré

radial_mask = 1.0 - (dist_from_center / max_dist)
radial_mask = np.clip(radial_mask, 0, 1)  # Garde entre 0 et 1

smart_show("0. Masque Radial Doux", (radial_mask * 255).astype(np.uint8))

# === 3. SOBEL (DÉTECTION D'ÉNERGIE) ===
# Flou léger
blur = cv2.GaussianBlur(gray_ready, (5, 5), 0)

grad_x = cv2.Sobel(blur, cv2.CV_16S, 1, 0, ksize=3)
grad_y = cv2.Sobel(blur, cv2.CV_16S, 0, 1, ksize=3)
abs_grad_x = cv2.convertScaleAbs(grad_x)
abs_grad_y = cv2.convertScaleAbs(grad_y)
gradient_magnitude = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

# Application du masque radial
weighted_gradient = gradient_magnitude.astype(np.float32) * radial_mask
weighted_gradient = weighted_gradient.astype(np.uint8)

smart_show("1. Gradient Pondéré", weighted_gradient)

# === 4. SEUILLAGE ET NETTOYAGE ===
# Otsu standard
_, thresh = cv2.threshold(weighted_gradient, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Morphologie : Une fermeture modérée pour relier les traits sans fusionner le bruit
kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
closed_mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_close, iterations=2)

smart_show("2. Masque Binaire", closed_mask)

# === 5. SÉLECTION INTELLIGENTE (LE SCORE) ===
contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

best_contour = None
best_score = 0

debug_img = img.copy()  # Pour voir ce qui est rejeté

for c in contours:
    area = cv2.contourArea(c)
    # Filtre de bruit de base
    if area < 200: continue

    # Calcul du centre du contour
    M = cv2.moments(c)
    if M["m00"] == 0: continue
    cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])

    # Distance au centre de l'image
    dist = np.sqrt((cx - center_x) ** 2 + (cy - center_y) ** 2)

    # === LA FORMULE MAGIQUE ===
    # Plus c'est gros, mieux c'est.
    # Plus c'est loin, moins c'est bon (pénalité exponentielle)
    penalty_factor = 1 + (dist / (min(h, w) * 0.2)) ** 2
    score = area / penalty_factor

    # Visualisation du score (Optionnel)
    color = (0, 0, 255)  # Rouge par défaut (rejeté)
    if score > best_score:
        best_score = score
        best_contour = c
        color = (0, 255, 0)  # Vert (candidat actuel)

    cv2.drawContours(debug_img, [c], -1, color, 2)

smart_show("3. Tri des contours (Vert=Gagnant)", debug_img)

if best_contour is None:
    print("❌ ECHEC : Rien de probant trouvé.")
    exit()

# Création du masque final propre
final_mask = np.zeros_like(gray)
cv2.drawContours(final_mask, [best_contour], -1, 255, cv2.FILLED)

# === 6. ANALYSE ET RÉSULTAT ===
contour_final = best_contour
area = cv2.contourArea(contour_final)
perimeter = cv2.arcLength(contour_final, True)

if perimeter == 0:
    circularity = 0
else:
    circularity = 4 * np.pi * (area / (perimeter * perimeter))

shape_type = "INCONNU"
if circularity >= CIRCULARITY_THRESHOLD_CIRCLE:
    shape_type = "CERCLE"
elif circularity <= CIRCULARITY_THRESHOLD_SQUARE:
    shape_type = "CARRE/AUTRE"
else:
    shape_type = "AMBIGU"

# Affichage
output = img.copy()
cv2.drawContours(output, [contour_final], -1, (0, 255, 0), 3)
label = f"{shape_type} ({circularity:.3f})"
cv2.putText(output, label, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

smart_show("4. Resultat Final", output)
print(f"✅ Terminé. {label}")
cv2.waitKey(0)
cv2.destroyAllWindows()