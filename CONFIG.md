# Configuration de l'Enregistreur Audio

## Installation rapide

1. Copiez le fichier d'exemple de configuration :
   ```
   cp config.example.json config.json
   ```

2. Éditez `config.json` avec vos vraies valeurs :
   - `api.url` : L'URL de votre serveur API
   - `api.token` : Votre token d'authentification API
   - `api.agent_token` : Votre token d'agent

## Structure de la configuration

### Section `api`
- **url** : URL complète de l'endpoint d'upload
- **token** : Token d'authentification pour l'API
- **agent_token** : Token spécifique à l'agent
- **timeout** : Timeout en secondes pour les requêtes HTTP

### Section `recording`
- **sample_rate** : Fréquence d'échantillonnage (Hz)
- **channels** : Nombre de canaux audio (1 = mono, 2 = stéréo)
- **blocksize** : Taille des blocs audio pour le traitement
- **threshold_db** : Seuil de déclenchement en décibels
- **silence_duration** : Durée de silence avant arrêt (secondes)
- **max_buffer_duration** : Durée max du buffer de pré-enregistrement
- **output_prefix** : Préfixe des fichiers d'enregistrement

### Section `audio_processing`
- **quality** : Qualité de compression ('low', 'medium', 'high')
- **remove_silence** : Supprimer automatiquement les silences
- **silence_threshold** : Seuil de détection de silence (dB)

### Section `schedule`
- **daily_merge_time** : Heure de fusion quotidienne (format HH:MM)
- **to_send_folder** : Dossier de destination des fichiers fusionnés

## Sécurité

⚠️ **Important** : Le fichier `config.json` contient des informations sensibles et ne doit jamais être commité dans Git. Il est automatiquement ignoré par `.gitignore`.

## Dépannage

Si vous obtenez une erreur "Fichier de configuration non trouvé", assurez-vous que :
1. Le fichier `config.json` existe dans le même dossier que `main.py`
2. Le fichier contient un JSON valide
3. Toutes les sections requises sont présentes