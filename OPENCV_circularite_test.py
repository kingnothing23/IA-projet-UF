import cv2
import numpy as np
import os
import zipfile
import random

# === CONFIGURATION ===
zip_path = r"C:\Users\yugst\Downloads\archive.zip"
extract_path = r"C:\Users\yugst\Documents\analyse_capsules\dataset"



# === FONCTION ANALYSE CIRCULARITÉ ===
def circularite_image(path):
    img = cv2.imread(path)
    if img is None:
        return None, None, None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, None

    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return None, None, None

    circularite = 4 * np.pi * (area / (perimeter * perimeter))
    return img, contour, circularite

# === PARCOURS DU DATASET ===
results = []
for root, _, files in os.walk(extract_path):
    for file in files:
        if file.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp', '.tif')):
            path = os.path.join(root, file)
            img, contour, circ = circularite_image(path)
            if circ is not None:
                results.append((path, img, contour, circ))

print(f"\n🔍 {len(results)} images analysées.")

# === FILTRAGE 0.85–0.9 ===
filtered = [r for r in results if 0.75 <= r[3] <= 0.9]
print(f"📸 {len(filtered)} images ont une circularité entre 0.85 et 0.9.")

# === AFFICHAGE ALÉATOIRE (max 15 images) ===
to_show = random.sample(filtered, min(len(filtered), 15))

for path, img, contour, circ in to_show:
    cv2.drawContours(img, [contour], -1, (0, 255, 0), 2)
    cv2.putText(img, f"Circularite: {circ:.3f}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow(f"{os.path.basename(path)}", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
