import asyncio
import json
import websockets
import cv2
import threading
import numpy as np
import socket
import sys
from hand_analysis import HandDetector

# Configuration globale
DEFAULT_PORT = 8765
MAX_PORT_ATTEMPTS = 10  # Essaiera jusqu'à 10 ports différents

# Fonction pour trouver un port disponible
def find_available_port(start_port):
    for port_offset in range(MAX_PORT_ATTEMPTS):
        port = start_port + port_offset
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind(('localhost', port))
            sock.close()
            return port
        except OSError:
            continue
    print(f"Impossible de trouver un port disponible après {MAX_PORT_ATTEMPTS} tentatives.")
    sys.exit(1)

# Trouver un port disponible
PORT = find_available_port(DEFAULT_PORT)
print(f"Port sélectionné: {PORT}")

detector = HandDetector(min_detection_confidence=0.7)

# Tenter d'ouvrir la webcam
try:
    webcam = cv2.VideoCapture(1)  # Essayez d'abord l'index 1
    if not webcam.isOpened():
        print("La caméra 1 n'est pas disponible, utilisation de la caméra 0")
        webcam = cv2.VideoCapture(0)  # Si 1 ne fonctionne pas, essayez 0
    
    if not webcam.isOpened():
        print("Aucune caméra disponible")
        sys.exit(1)
        
except Exception as e:
    print(f"Erreur lors de l'initialisation de la webcam: {e}")
    sys.exit(1)

# Variables globales pour stocker les données de mouvement
movement_data = {
    "direction": [],
    "vector": [0, 0],
    "size": 1.0,
    "is_size_gesture": False
}

# Flag pour contrôler l'affichage
DISPLAY_WINDOW = False  # Mettre à True seulement si vous êtes sûr que l'affichage GUI fonctionne

# Variable pour contrôler l'exécution du thread vidéo
running = True

# Fonction pour traiter le flux vidéo et détecter les mouvements
def process_video():
    global movement_data, running
    
    while running:
        success, image = webcam.read()
        if not success:
            print("Erreur lors de la lecture de la webcam - tentative de reconnexion")
            time.sleep(1)
            continue
            
        try:
            hand_landmarks = detector.findHandLandMarks(image=image, draw=False)
            
            if len(hand_landmarks) > 0:
                direction, movement_vector = detector.analyzeMovement(image, hand_landmarks)
                is_size_gesture = bool(detector.isSizeGesture(hand_landmarks))  # Convert to Python bool
                
                if is_size_gesture:
                    size = detector.changeBubbleSize(hand_landmarks)
                    hand_size = float(np.linalg.norm(np.array([hand_landmarks[0][1], hand_landmarks[0][2]]) - 
                                              np.array([hand_landmarks[5][1], hand_landmarks[5][2]])))
                    if hand_size > 0:
                        bubble_size = float(size / hand_size)  # Convert to Python float
                    else:
                        bubble_size = float(movement_data["size"])
                else:
                    bubble_size = float(movement_data["size"])
                
                # Convert all values to native Python types
                movement_data = {
                    "direction": list(direction),  # Convert to Python list
                    "vector": [float(movement_vector[0]), float(movement_vector[1])],  # Convert to Python float
                    "size": float(bubble_size),  # Convert to Python float
                    "is_size_gesture": bool(is_size_gesture)  # Convert to Python bool
                }
                
                print(f"Direction: {direction}, Vector: {movement_vector[0]:.1f}, {movement_vector[1]:.1f}, Size: {bubble_size:.2f}")
            
            # Afficher l'image uniquement si l'option est activée
            if DISPLAY_WINDOW:
                cv2.imshow("Hand Tracking", image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    running = False
                    break
                    
        except Exception as e:
            print(f"Erreur lors du traitement de l'image: {e}")
    
    # Libérer les ressources
    webcam.release()
    if DISPLAY_WINDOW:
        cv2.destroyAllWindows()
    print("Thread vidéo terminé")

# Gestion des connexions WebSocket
async def websocket_handler(websocket):
    client_id = id(websocket)
    print(f"Client connecté: {client_id}")
    try:
        while running:
            # Envoyer les données au client
            await websocket.send(json.dumps(movement_data))
            await asyncio.sleep(0.03)  # ~30 fps
    except websockets.exceptions.ConnectionClosed:
        print(f"Client déconnecté: {client_id}")
    except Exception as e:
        print(f"Erreur WebSocket: {e}")

# Démarrer le serveur
async def start_server():
    try:
        server = await websockets.serve(websocket_handler, "localhost", PORT)
        print(f"Serveur démarré sur ws://localhost:{PORT}")
        
        # Mettre à jour le client HTML avec le bon port
        update_client_port(PORT)
        
        await server.wait_closed()
    except Exception as e:
        print(f"Erreur lors du démarrage du serveur: {e}")
        global running
        running = False

# Mettre à jour le fichier HTML client avec le bon port
def update_client_port(port):
    try:
        html_file = "client.html"
        with open(html_file, "r") as f:
            content = f.read()
        
        # Remplacer le port dans le fichier HTML
        updated_content = content.replace("const wsUrl = 'ws://localhost:8765';", 
                                        f"const wsUrl = 'ws://localhost:{port}';")
        
        with open(html_file, "w") as f:
            f.write(updated_content)
        
        print(f"Le fichier client HTML a été mis à jour avec le port {port}")
    except Exception as e:
        print(f"Erreur lors de la mise à jour du fichier HTML: {e}")
        print("Vous devrez mettre à jour manuellement le port dans le fichier HTML.")

# Gestion propre de l'arrêt
def cleanup():
    global running
    running = False
    print("Arrêt du serveur...")
    # Donner au thread vidéo le temps de se terminer proprement
    time.sleep(1)

if __name__ == "__main__":
    import time
    
    try:
        # Démarrer le traitement vidéo dans un thread séparé
        video_thread = threading.Thread(target=process_video)
        video_thread.daemon = True
        video_thread.start()
        
        # Démarrer le serveur WebSocket
        print("Démarrage du serveur...")
        asyncio.run(start_server())
    except KeyboardInterrupt:
        print("Interruption de l'utilisateur")
    except Exception as e:
        print(f"Erreur: {e}")
    finally:
        cleanup()