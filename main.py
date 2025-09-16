import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write, read
import time
import sys
import os
import subprocess
import threading
import datetime
import glob
from pathlib import Path

# Paramètres
sample_rate = 16000
channels = 1
chunk_duration = 0.5      # durée de chaque trame en secondes
threshold_db = 30         # seuil en "dB positifs" arbitraires
silence_duration = 10     # secondes de silence avant d'arrêter l'enregistrement
output_prefix = "recording"

# Paramètres de fusion quotidienne
daily_merge_time = "23:59"  # Heure de fusion quotidienne (format HH:MM)
to_send_folder = "toSend"   # Dossier de destination

def positive_db(signal):
    """Convertit un signal int16 en dB positifs arbitraires"""
    rms = np.sqrt(np.mean(np.square(signal)))
    if rms == 0:
        return 0
    return 20 * np.log10(rms)

def format_size(size_bytes):
    """Convertit la taille en octets en format lisible"""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"

def format_duration(seconds):
    """Convertit la durée en secondes en format MM:SS"""
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes:02d}:{seconds:02d}"

def estimate_file_size(samples_count, sample_rate, channels, dtype='int16'):
    """Estime la taille du fichier WAV en octets"""
    # En-tête WAV ~ 44 octets
    # Données: nombre d'échantillons * taille d'un échantillon * nombre de canaux
    bytes_per_sample = 2 if dtype == 'int16' else 4  # int16 = 2 bytes, float32 = 4 bytes
    data_size = samples_count * bytes_per_sample * channels
    return 44 + data_size

def create_to_send_folder():
    """Crée le dossier toSend s'il n'existe pas"""
    if not os.path.exists(to_send_folder):
        os.makedirs(to_send_folder)
        print(f"📁 Dossier '{to_send_folder}' créé")

def merge_audio_files():
    """Fusionne tous les fichiers audio en un seul fichier quotidien"""
    # Trouver tous les fichiers audio dans le répertoire courant
    audio_files = glob.glob(f"{output_prefix}_*.wav")
    
    if not audio_files:
        print("📭 Aucun fichier audio à fusionner")
        return
    
    print(f"\n📦 Début de la fusion de {len(audio_files)} fichiers...")
    
    # Trier les fichiers par timestamp
    audio_files.sort()
    
    try:
        # Lire et concaténer tous les fichiers audio
        merged_data = []
        total_duration = 0
        
        for file_path in audio_files:
            try:
                rate, data = read(file_path)
                if rate == sample_rate:
                    merged_data.append(data)
                    total_duration += len(data) / sample_rate
                    print(f"   ✓ {file_path} ajouté")
                else:
                    print(f"   ⚠️ {file_path} ignoré (fréquence différente)")
            except Exception as e:
                print(f"   ❌ Erreur lecture {file_path}: {e}")
        
        if merged_data:
            # Concaténer tous les données audio
            final_audio = np.concatenate(merged_data, axis=0)
            
            # Créer le nom du fichier final avec la date
            today = datetime.datetime.now().strftime("%Y-%m-%d")
            merged_filename = f"merged_{today}.wav"
            merged_path = os.path.join(to_send_folder, merged_filename)
            
            # Sauvegarder le fichier fusionné
            write(merged_path, sample_rate, final_audio)
            
            # Calculer la taille du fichier fusionné
            merged_size = os.path.getsize(merged_path)
            
            print(f"✅ Fusion terminée !")
            print(f"   Fichier: {merged_path}")
            print(f"   Durée totale: {format_duration(total_duration)}")
            print(f"   Taille: {format_size(merged_size)}")
            
            # Supprimer les fichiers individuels
            for file_path in audio_files:
                try:
                    os.remove(file_path)
                    print(f"   🗑️ {file_path} supprimé")
                except Exception as e:
                    print(f"   ❌ Erreur suppression {file_path}: {e}")
            
            print(f"🎯 Fichier fusionné déplacé vers '{to_send_folder}'")
            
    except Exception as e:
        print(f"❌ Erreur lors de la fusion: {e}")

def schedule_daily_merge():
    """Programme la fusion quotidienne à l'heure spécifiée"""
    def run_daily_merge():
        while True:
            now = datetime.datetime.now()
            merge_hour, merge_minute = map(int, daily_merge_time.split(':'))
            
            # Calculer le prochain moment de fusion
            next_merge = now.replace(hour=merge_hour, minute=merge_minute, second=0, microsecond=0)
            
            # Si l'heure est déjà passée aujourd'hui, programmer pour demain
            if next_merge <= now:
                next_merge += datetime.timedelta(days=1)
            
            # Calculer le temps d'attente
            wait_seconds = (next_merge - now).total_seconds()
            
            print(f"\n⏰ Prochaine fusion programmée pour {next_merge.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"   (dans {format_duration(wait_seconds)})")
            
            # Attendre jusqu'au moment de fusion
            time.sleep(wait_seconds)
            
            # Exécuter la fusion
            print(f"\n🕐 {daily_merge_time} - Début de la fusion quotidienne...")
            global recording_active, recording
            
            # Sauvegarder l'enregistrement en cours si nécessaire
            if recording_active and len(recording) > 0:
                print("   💾 Sauvegarde de l'enregistrement en cours...")
                full_recording = np.concatenate(recording, axis=0)
                timestamp = int(time.time())
                file_name = f"{output_prefix}_{timestamp}.wav"
                write(file_name, sample_rate, full_recording)
                recording_active = False
                recording = []
                print(f"   ✓ Enregistrement sauvegardé: {file_name}")
            
            # Effectuer la fusion
            merge_audio_files()
            
            print("⏰ Fusion quotidienne terminée, reprise de l'écoute...")
    
    # Lancer le thread de programmation
    merge_thread = threading.Thread(target=run_daily_merge, daemon=True)
    merge_thread.start()

print("Écoute en continu... Appuyez sur Ctrl+C pour arrêter")

# Créer le dossier toSend et démarrer la programmation de fusion
create_to_send_folder()
schedule_daily_merge()

recording = []
recording_active = False
silence_counter = silence_duration
recording_start_time = 0
last_status_update = 0
status_update_interval = 0.1  # Mise à jour du statut toutes les 100ms

def update_status(force=False):
    """Met à jour l'affichage du statut"""
    global last_status_update
    current_time = time.time()
    
    if not force and current_time - last_status_update < status_update_interval:
        return
    
    last_status_update = current_time
    
    if recording_active:
        duration = current_time - recording_start_time
        samples_count = sum(len(chunk) for chunk in recording)
        file_size = estimate_file_size(samples_count, sample_rate, channels)
        
        status = f"\r🔴 ENREGISTREMENT - Durée: {format_duration(duration)} | Taille: {format_size(file_size)} | Silence restant: {silence_counter:.1f}s"
    else:
        status = f"\r⚪ EN ATTENTE - Écoute en cours..."
    
    print(status, end='', flush=True)

try:
    # Affichage initial
    update_status(force=True)
    
    while True:
        # Lire un petit chunk
        chunk = sd.rec(int(chunk_duration * sample_rate), samplerate=sample_rate,
                       channels=channels, dtype='int16')
        sd.wait()

        level_db = positive_db(chunk)
        # print(level_db)  # pour calibrer le seuil

        if level_db > threshold_db:
            if not recording_active:
                print(f"\n🎤 Démarrage de l'enregistrement (niveau {level_db:.1f} dB)...")
                recording_active = True
                recording = []  # reset
                recording_start_time = time.time()
                silence_counter = silence_duration
            recording.append(chunk)
            silence_counter = silence_duration  # reset si son revient
        else:
            if recording_active:
                recording.append(chunk)
                silence_counter -= chunk_duration
                if silence_counter <= 0:
                    # Arrêt et sauvegarde
                    full_recording = np.concatenate(recording, axis=0)
                    timestamp = int(time.time())
                    file_name = f"{output_prefix}_{timestamp}.wav"
                    write(file_name, sample_rate, full_recording)
                    
                    # Calculer la taille réelle du fichier sauvegardé
                    actual_size = os.path.getsize(file_name)
                    duration = time.time() - recording_start_time
                    
                    print(f"\n✅ Enregistrement terminé et sauvegardé : {file_name}")
                    print(f"   Durée: {format_duration(duration)} | Taille: {format_size(actual_size)}")
                    
                    recording_active = False
                    recording = []
        
        # Mettre à jour l'affichage du statut
        update_status()

except KeyboardInterrupt:
    if recording_active and len(recording) > 0:
        full_recording = np.concatenate(recording, axis=0)
        timestamp = int(time.time())
        file_name = f"{output_prefix}_{timestamp}.wav"
        write(file_name, sample_rate, full_recording)
        
        # Calculer la taille réelle du fichier sauvegardé
        actual_size = os.path.getsize(file_name)
        duration = time.time() - recording_start_time
        
        print(f"\n💾 Enregistrement sauvegardé après interruption : {file_name}")
        print(f"   Durée: {format_duration(duration)} | Taille: {format_size(actual_size)}")
    
    # Proposer une fusion manuelle avant de quitter
    print("\n🛑 Arrêt du script.")
    response = input("Voulez-vous fusionner les enregistrements avant de quitter ? (o/N): ")
    if response.lower() in ['o', 'oui', 'y', 'yes']:
        print("🔄 Fusion en cours...")
        merge_audio_files()
        print("✅ Fusion terminée !")
except Exception as e:
    print(f"\n❌ Erreur inattendue: {e}")
    print("🛑 Arrêt du script.")