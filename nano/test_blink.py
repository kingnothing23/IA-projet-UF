from opcua import Client, ua
import time

# --- CONFIGURATION ---
url = "opc.tcp://192.168.0.26:4840"
NODE_ID_VAR = 'ns=3;s="OPC_UA_1"."bon"'

print(f"Connexion à {url}...")

try:
    client = Client(url)
    client.connect()
    print(" Connecté !")

    var_bon = client.get_node(NODE_ID_VAR)

    print("\nLancement de la boucle (Ctrl+C pour arrêter)...")

    while True:
        # --- ETAPE 1 : Mettre à TRUE ---
        print("Envoi de TRUE...")

        # 1. On prépare la valeur (Booléen)
        valeur = ua.Variant(True, ua.VariantType.Boolean)

        # 2. On l'emballe dans un DataValue
        data_value = ua.DataValue(valeur)

        # 3. CRUCIAL : On supprime explicitement le temps et le statut
        # Pour que l'automate ne reçoive QUE la valeur
        data_value.SourceTimestamp = None
        data_value.ServerTimestamp = None
        data_value.StatusCode = None

        # 4. On envoie
        var_bon.set_value(data_value)

        time.sleep(2)

        # --- ETAPE 2 : Mettre à FALSE ---
        print("Envoi de FALSE...")

        valeur = ua.Variant(False, ua.VariantType.Boolean)
        data_value = ua.DataValue(valeur)

        # Pareil, on nettoie tout
        data_value.SourceTimestamp = None
        data_value.ServerTimestamp = None
        data_value.StatusCode = None

        var_bon.set_value(data_value)

        time.sleep(2)

except KeyboardInterrupt:
    print("\nArrêt manuel.")

except Exception as e:
    print(f"\n Erreur : {e}")

finally:
    client.disconnect()
    print("Déconnecté.")