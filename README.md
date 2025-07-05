# detection_visage_emotion.py
Ce script permet une analyse des émotions en temps réel à partir du flux vidéo d'une webcam. En combinant OpenCV pour la détection faciale , il identifie les expressions comme la joie, la colère ou la tristesse, tout en ajoutant des nuances plus subtiles (ennui, excitation, anxiété).
  Fonctionnalités principales
- Détection des visages
- Utilise haarcascade_frontalface_default.xml (OpenCV) pour localiser les visages dans l'image.
- Reconnaissance des émotions
- Utilise la bibliothèque FER (Facial Expression Recognition) pour analyser 7 émotions de base :
happy (joie), sad (tristesse), angry (colère), disgust (dégoût), fear (peur), surprise, neutral.

Émotions étendues

   Ajoute des interprétations personnalisées pour des émotions complexes :
excited (excitation), bored (ennui), anxious (anxiété), frustrated (frustration), melancholy (mélancolie), amazed (émerveillement).

Affichage visuel

Encadre les visages avec une couleur associée à l’émotion (ex: jaune pour la joie, rouge pour la colère).

Affiche le nom de l’émotion au-dessus du visage.

-  Fonctionnement technique
Entrée : Flux vidéo de la webcam (via OpenCV).

Traitement :
- Détection des visages avec Haar Cascade.
- Analyse des émotions avec FER (basé sur MTCNN pour une meilleure précision).
- Interprétation des scores d’émotions pour affiner les résultats.

Optimisations :
- Un cache des émotions (2 secondes) pour éviter des calculs répétitifs.

Réglage de la résolution (640x480) et du format vidéo (MJPG) pour fluidité.

- Palette de couleurs des émotions
Émotion	Couleur associée
Joie	Jaune
Tristesse	Bleu
Colère	Rouge
Peur	Violet
Surprise	Orange
Neutre	Gris
Excitement	Or
Anxiété	Indigo


- Comment l’exécuter ?
Installer les dépendances :

bash
pip install opencv-python fer matplotlib

- Lancer le script :

bash
python script.py

Quitter : Appuyer sur la touche q dans la fenêtre vidéo.

Fonctionnalités clés :
✅ Détection visage + reconnaissance de 7 émotions de base
✅ Interprétation d'émotions complexes (ex: frustration, mélancolie)
✅ Visualisation colorée et temps réel
✅ Utilisation simple (appuyez sur Q pour quitter)
