import pandas as pd
import struct
import io


def float_to_mid_big_endian_32bit(value: float) -> bytes:
    """
    Convertit un nombre réel (float) en une séquence de 4 octets
    en format 32 bits (simple précision) mid-big endian (ordre 2, 1, 4, 3).
    """
    # 1. Convertir le float en 4 octets big-endian (>) format 'f' (float 32-bit).
    # L'ordre big-endian est (octet 1, octet 2, octet 3, octet 4).
    big_endian_bytes = struct.pack('>f', value)

    # 2. Réorganiser les octets pour obtenir le format mid-big endian (2, 1, 4, 3).
    # [0] = Octet 1, [1] = Octet 2, [2] = Octet 3, [3] = Octet 4
    # Le nouveau format sera : Octet 2, Octet 1, Octet 4, Octet 3
    # Le format PDP-11/mid-big-endian est non standard.
    byte_2 = big_endian_bytes[1:2]
    byte_1 = big_endian_bytes[0:1]
    byte_4 = big_endian_bytes[3:4]
    byte_3 = big_endian_bytes[2:3]

    mid_big_endian_bytes = byte_2 + byte_1 + byte_4 + byte_3

    return mid_big_endian_bytes


def convertir_excel_mid_big_endian(chemin_excel: str, nom_feuille: str = 0) -> dict:
    """
    Lit un fichier Excel, convertit les valeurs des colonnes A à H (lignes 1 à 7)
    en 32-bit mid-big endian et retourne un dictionnaire contenant les
    valeurs originales et les octets convertis.
    """

    # 1. Lecture du fichier Excel (inchangée)
    try:
        df = pd.read_excel(
            chemin_excel,
            sheet_name=nom_feuille,
            header=None,
            skiprows=0,
            nrows=7,
            usecols='A:H'
        )
    except FileNotFoundError:
        print(f"Erreur : Le fichier {chemin_excel} n'a pas été trouvé.")
        return {}
    except ValueError as e:
        print(f"Erreur lors de la lecture de la feuille Excel : {e}")
        return {}
    except Exception as e:
        print(f"Une erreur inattendue est survenue : {e}")
        return {}

    # 2. Traiter et convertir les valeurs
    # Le dictionnaire stockera: {(ligne, col): {'original': float, 'mid_big_endian': bytes}}
    resultats = {}

    for i in df.index:  # i est l'index de ligne (0 à 6)
        for j in df.columns:  # j est l'index de colonne (0 à 7)
            valeur_lue = df.iloc[i, j]

            # Assurer que la valeur est un nombre réel (float)
            try:
                # Tente de convertir la valeur en float
                valeur_float = float(valeur_lue)

                # Effectuer la conversion non standard mid-big endian
                octets_mid_big_endian = float_to_mid_big_endian_32bit(valeur_float)

                # Stocker les DEUX valeurs
                resultats[(i + 1, j + 1)] = {
                    'original': valeur_float,
                    'mid_big_endian': octets_mid_big_endian
                }

            except ValueError:
                col_lettre = chr(ord('A') + j)
                print(
                    f"Avertissement : La valeur à la ligne {i + 1}, colonne {col_lettre} ({valeur_lue}) n'est pas un nombre réel valide et sera ignorée.")
                # Stocker un indicateur d'erreur
                resultats[(i + 1, j + 1)] = {
                    'original': valeur_lue,
                    'mid_big_endian': b'ERR!'
                }

    return resultats




# Sauvegarder le DataFrame dans un fichier Excel
fichier_test = r"C:\Users\yugst\Downloads\postions_dem_siens.xlsx"



## 🚀 Exécution du Script

# **Étape 2 : Appeler la fonction de conversion**

resultats_complets = convertir_excel_mid_big_endian(fichier_test)

# **Étape 3 : Afficher les résultats**

if resultats_complets:
    print("\n✅ Conversion terminée. Résultats affichés colonne par colonne :\n")

    # Colonnes 1 à 8 = A à H
    for colonne in range(1, 9):
        col_lettre = chr(ord('A') + colonne - 1)
        print(f"--- Colonne {col_lettre} ---")

        # Lignes 1 à 7
        for ligne in range(1, 8):
            data = resultats_complets.get((ligne, colonne))

            if data is None:
                print(f"  Ligne {ligne}: (aucune donnée)")
                continue

            valeur_originale = data['original']
            octets = data['mid_big_endian']
            octets_hex = octets.hex()

            print(f"  ({ligne}, {col_lettre}) - Valeur: {valeur_originale:<10} -> Octets: {octets} (Hex: {octets_hex})")

        print()  # Ligne vide entre les colonnes
