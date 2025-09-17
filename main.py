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
import queue
import collections

# Paramètres optimisés pour un enregistrement léger et de qualité
sample_rate = 16000
channels = 1
blocksize = 512
threshold_db = 30         # seuil en decibels
silence_duration = 10     # secondes de silence avant d'arrêter l'enregistrement
output_prefix = "recording"
max_buffer_duration = 60  # Durée max du buffer en secondes (augmenté)

# Paramètres de compression audio
audio_quality = 'medium'  # 'low', 'medium', 'high'
remove_silence = False
silence_threshold = -30   # Seuil de silence en dB (plus bas = plus strict)

# Paramètres de fusion quotidienne
daily_merge_time = "23:59"
to_send_folder = "toSend"

# Variables globales pour l'enregistrement continu
audio_queue = queue.Queue(maxsize=1000)  # Queue plus grande pour éviter les pertes
recording_buffer = collections.deque(maxlen=int(max_buffer_duration * sample_rate / blocksize))
recording_active = False
recording_data = []
recording_start_time = 0
silence_counter = 0
stop_recording = False

def positive_db(signal):
    """Convertit un signal int16 en dB positifs arbitraires"""
    # Éviter les warnings avec des valeurs nulles
    signal_safe = signal.astype(float)
    rms = np.sqrt(np.mean(np.square(signal_safe)))
    if rms == 0 or np.isnan(rms):
        return 0
    return 20 * np.log10(rms)

def remove_silence_from_audio(audio_data, sample_rate, min_silence_len=1000, silence_thresh=-30):
    """Supprime les silences longs d'un enregistrement audio avec numpy"""
    if not remove_silence:
        return audio_data
    
    try:
        # Convertir le seuil en amplitude (approximation)
        silence_amplitude = 32767 * (10 ** (silence_thresh / 20))
        
        # Calculer l'énergie locale par fenêtres
        window_size = int(sample_rate * 0.1)
        
        if len(audio_data) < window_size:
            return audio_data
        
        # Calculer l'amplitude RMS par fenêtre
        num_windows = len(audio_data) // window_size
        silence_mask = []
        
        for i in range(num_windows):
            start = i * window_size
            end = start + window_size
            window_data = audio_data[start:end]
            
            # Calculer RMS
            rms = np.sqrt(np.mean(np.square(window_data.astype(float))))
            is_silence = rms < silence_amplitude
            silence_mask.append(is_silence)
        
        # Identifier les segments de silence longs
        min_silence_windows = max(1, int(min_silence_len / 1000 * 10))  # Convertir ms en nombre de fenêtres
        
        # Marquer les silences longs pour suppression
        to_remove = []
        silence_start = None
        
        for i, is_silent in enumerate(silence_mask):
            if is_silent:
                if silence_start is None:
                    silence_start = i
            else:
                if silence_start is not None:
                    silence_length = i - silence_start
                    if silence_length >= min_silence_windows:
                        # Garder une petite portion du silence (200ms)
                        keep_windows = min(2, silence_length // 2)
                        to_remove.append((silence_start + keep_windows, i - keep_windows))
                    silence_start = None
        
        # Traiter le dernier segment si c'est du silence
        if silence_start is not None:
            silence_length = len(silence_mask) - silence_start
            if silence_length >= min_silence_windows:
                keep_windows = min(2, silence_length // 2)
                to_remove.append((silence_start + keep_windows, len(silence_mask)))
        
        # Construire le nouveau signal sans les silences longs
        if to_remove:
            keep_segments = []
            last_end = 0
            
            for remove_start, remove_end in to_remove:
                # Garder la partie avant le silence
                keep_segments.append(audio_data[last_end * window_size:remove_start * window_size])
                last_end = remove_end
            
            # Garder la dernière partie
            keep_segments.append(audio_data[last_end * window_size:])
            
            # Concaténer tous les segments gardés
            cleaned_data = np.concatenate([seg for seg in keep_segments if len(seg) > 0])
            
            reduction = (1 - len(cleaned_data) / len(audio_data)) * 100
            if reduction > 5:  # Si réduction significative
                print(f"   🔇 Silences supprimés : {reduction:.1f}% de réduction")
                return cleaned_data
        
        return audio_data
        
    except Exception as e:
        print(f"   ⚠️ Erreur suppression silence: {e}")
        return audio_data

def save_optimized_audio(audio_data, sample_rate, file_path, quality='medium'):
    """Sauvegarde l'audio en format WAV optimisé"""
    base_name = os.path.splitext(file_path)[0]
    
    # Supprimer les silences si activé
    if remove_silence:
        audio_data = remove_silence_from_audio(audio_data, sample_rate)
    
    # Sauvegarder en WAV optimisé
    final_path = f"{base_name}.wav"
    write(final_path, sample_rate, audio_data)
    file_size = os.path.getsize(final_path)
    
    return final_path, file_size

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
    bytes_per_sample = 2 if dtype == 'int16' else 4
    data_size = samples_count * bytes_per_sample * channels
    return 44 + data_size

def create_to_send_folder():
    """Crée le dossier toSend s'il n'existe pas"""
    if not os.path.exists(to_send_folder):
        os.makedirs(to_send_folder)
        print(f"📁 Dossier '{to_send_folder}' créé")

def merge_audio_files():
    """Fusionne tous les fichiers audio WAV en un seul fichier quotidien optimisé"""
    # Chercher uniquement les fichiers WAV
    wav_files = glob.glob(f"{output_prefix}_*.wav")
    
    if not wav_files:
        print("📭 Aucun fichier audio à fusionner")
        return
    
    print(f"\n📦 Début de la fusion de {len(wav_files)} fichiers...")
    wav_files.sort()
    
    try:
        # Fusion WAV classique
        merged_data = []
        total_duration = 0
        
        for file_path in wav_files:
            try:
                rate, data = read(file_path)
                if rate == sample_rate:
                    merged_data.append(data)
                    total_duration += len(data) / sample_rate
                    print(f"   ✓ {file_path} ajouté")
                else:
                    print(f"   ⚠️ {file_path} ignoré (fréquence différente: {rate}Hz vs {sample_rate}Hz)")
            except Exception as e:
                print(f"   ❌ Erreur lecture {file_path}: {e}")
        
        if merged_data:
            final_audio = np.concatenate(merged_data, axis=0)
            today = datetime.datetime.now().strftime("%Y-%m-%d")
            merged_base = os.path.join(to_send_folder, f"merged_{today}")
            
            final_path, merged_size = save_optimized_audio(
                final_audio, sample_rate, merged_base, 'high'
            )
            
            print(f"✅ Fusion terminée !")
            print(f"   Fichier: {final_path}")
            print(f"   Durée totale: {format_duration(total_duration)}")
            print(f"   Taille: {format_size(merged_size)}")
        
        # Supprimer les fichiers individuels après fusion réussie
        for file_path in wav_files:
            try:
                os.remove(file_path)
                print(f"   🗑️ {file_path} supprimé")
            except Exception as e:
                print(f"   ❌ Erreur suppression {file_path}: {e}")
        
        print(f"🎯 Fichier fusionné optimisé déplacé vers '{to_send_folder}'")
        
    except Exception as e:
        print(f"❌ Erreur lors de la fusion: {e}")

def audio_callback(indata, frames, time, status):
    """Callback pour l'enregistrement audio continu - exécuté en temps réel"""
    global recording_buffer, recording_active, recording_data
    
    if status:
        print(f"\n⚠️ Audio Status: {status}")
    
    # Convertir en int16 et aplatir si nécessaire
    audio_data = indata.copy()
    if audio_data.dtype != np.int16:
        audio_data = (audio_data * 32767).astype(np.int16)
    
    if len(audio_data.shape) > 1:
        audio_data = audio_data.flatten()
    
    # Ajouter au buffer circulaire (toujours actif)
    recording_buffer.append(audio_data)
    
    # Si enregistrement actif, ajouter aussi aux données d'enregistrement
    if recording_active:
        recording_data.append(audio_data)
    
    # Envoyer à la queue pour le traitement principal avec gestion des erreurs
    try:
        audio_queue.put_nowait(audio_data)
    except queue.Full:
        # Queue pleine - c'est un problème critique, essayer de vider
        print(f"\n⚠️ Queue audio pleine ! Perte de données possible.")
        try:
            # Vider quelques éléments anciens
            for _ in range(min(10, audio_queue.qsize())):
                audio_queue.get_nowait()
            audio_queue.put_nowait(audio_data)
        except:
            pass  # Si on ne peut toujours pas, on perd ce chunk

def process_audio():
    """Thread principal de traitement audio"""
    global recording_active, recording_data, recording_start_time, silence_counter, stop_recording
    
    last_status_update = 0
    status_update_interval = 0.5  # Mise à jour toutes les 500ms
    
    print("🎧 Traitement audio démarré...")
    
    while not stop_recording:
        try:
            # Récupérer les données audio avec timeout
            audio_data = audio_queue.get(timeout=1.0)
            
            # Calculer le niveau audio
            level_db = positive_db(audio_data)
            
            current_time = time.time()
            
            if recording_active:
                # Gérer le compteur de silence
                if level_db > threshold_db:
                    silence_counter = silence_duration
                else:
                    silence_counter -= (len(audio_data) / sample_rate)
                
                # Vérifier si on doit arrêter l'enregistrement
                if silence_counter <= 0:
                    # Sauvegarder l'enregistrement avec optimisation
                    if recording_data:
                        full_recording = np.concatenate(recording_data, axis=0)
                        timestamp = int(time.time())
                        file_base = f"{output_prefix}_{timestamp}"
                        
                        # Logs de debug
                        actual_duration = len(full_recording) / sample_rate
                        expected_duration = current_time - recording_start_time
                        print(f"\n📊 Debug enregistrement:")
                        print(f"   Durée calculée: {format_duration(actual_duration)}")
                        print(f"   Durée attendue: {format_duration(expected_duration)}")
                        print(f"   Échantillons: {len(full_recording)}")
                        print(f"   Chunks capturés: {len(recording_data)}")
                        
                        print(f"\n💾 Sauvegarde optimisée en cours...")
                        final_path, actual_size = save_optimized_audio(
                            full_recording, sample_rate, file_base, audio_quality
                        )
                        
                        print(f"✅ Enregistrement terminé : {os.path.basename(final_path)}")
                        print(f"   Durée audio: {format_duration(actual_duration)} | Taille: {format_size(actual_size)}")
                    
                    recording_active = False
                    recording_data = []
                    silence_counter = 0
            else:
                # Vérifier si on doit démarrer un enregistrement
                if level_db > threshold_db:
                    print(f"\n🎤 Démarrage enregistrement (niveau {level_db:.1f} dB)...")
                    recording_active = True
                    recording_start_time = current_time
                    silence_counter = silence_duration
                    
                    # Ajouter le buffer précédent pour capturer le début
                    recording_data = list(recording_buffer)
                    
                    # Debug info
                    buffer_duration = len(recording_buffer) * blocksize / sample_rate
                    print(f"   📦 Buffer pré-enregistrement: {len(recording_buffer)} chunks ({format_duration(buffer_duration)})")
            
            # Mise à jour du statut
            if current_time - last_status_update >= status_update_interval:
                update_status_display(current_time)
                last_status_update = current_time
                
        except queue.Empty:
            # Pas de nouvelles données, continuer
            continue
        except Exception as e:
            print(f"\n❌ Erreur traitement audio: {e}")

def update_status_display(current_time):
    """Met à jour l'affichage du statut"""
    global recording_active, recording_data, recording_start_time, silence_counter
    
    if recording_active and recording_data:
        duration = current_time - recording_start_time
        samples_count = sum(len(chunk) for chunk in recording_data)
        file_size = estimate_file_size(samples_count, sample_rate, channels)
        
        status = f"\r🔴 ENREGISTREMENT - Durée: {format_duration(duration)} | Taille: {format_size(file_size)} | Silence: {silence_counter:.1f}s"
    else:
        status = f"\r⚪ EN ATTENTE - Écoute continue active..."
    
    print(status, end='', flush=True)

def schedule_daily_merge():
    """Programme la fusion quotidienne"""
    def run_daily_merge():
        while not stop_recording:
            now = datetime.datetime.now()
            merge_hour, merge_minute = map(int, daily_merge_time.split(':'))
            next_merge = now.replace(hour=merge_hour, minute=merge_minute, second=0, microsecond=0)
            
            if next_merge <= now:
                next_merge += datetime.timedelta(days=1)
            
            wait_seconds = (next_merge - now).total_seconds()
            
            # Attendre par petits intervalles pour pouvoir s'arrêter proprement
            while wait_seconds > 0 and not stop_recording:
                sleep_time = min(60, wait_seconds)  # Dormir max 1 minute à la fois
                time.sleep(sleep_time)
                wait_seconds -= sleep_time
            
            if not stop_recording:
                print(f"\n🕐 {daily_merge_time} - Fusion quotidienne...")
                merge_audio_files()
    
    merge_thread = threading.Thread(target=run_daily_merge, daemon=True)
    merge_thread.start()
    return merge_thread

def main():
    """Fonction principale"""
    global stop_recording
    
    print("🚀 Démarrage de l'enregistreur audio optimisé...")
    print("   Configuration:")
    print(f"   - Fréquence: {sample_rate} Hz")
    print(f"   - Canaux: {channels}")
    print(f"   - Taille de bloc: {blocksize}")
    print(f"   - Seuil: {threshold_db} dB")
    print(f"   - Silence max: {silence_duration}s")
    print(f"   - Qualité: {audio_quality}")
    print(f"   - Suppression silences: {'✓' if remove_silence else '✗'}")
    print(f"   - Format de sortie: WAV")
    
    create_to_send_folder()
    merge_thread = schedule_daily_merge()
    
    # Démarrer le thread de traitement audio
    process_thread = threading.Thread(target=process_audio, daemon=True)
    process_thread.start()
    
    try:
        # Démarrer l'enregistrement continu avec callback
        with sd.InputStream(samplerate=sample_rate, channels=channels, dtype='int16',
                           blocksize=blocksize, callback=audio_callback):
            
            print("\n🎧 Enregistrement continu actif...")
            print("Appuyez sur Ctrl+C pour arrêter\n")
            
            # Boucle principale
            while True:
                time.sleep(1)
                
    except KeyboardInterrupt:
        print("\n\n🛑 Arrêt en cours...")
        stop_recording = True
        
        # Sauvegarder l'enregistrement en cours si nécessaire
        if recording_active and recording_data:
            full_recording = np.concatenate(recording_data, axis=0)
            timestamp = int(time.time())
            file_base = f"{output_prefix}_{timestamp}"
            
            print("💾 Sauvegarde finale optimisée...")
            final_path, actual_size = save_optimized_audio(
                full_recording, sample_rate, file_base, audio_quality
            )
            
            duration = time.time() - recording_start_time
            
            print(f"💾 Enregistrement final sauvegardé : {os.path.basename(final_path)}")
            print(f"   Durée: {format_duration(duration)} | Taille: {format_size(actual_size)}")
        
        # Proposer une fusion
        response = input("\nVoulez-vous fusionner les enregistrements ? (o/N): ")
        if response.lower() in ['o', 'oui', 'y', 'yes']:
            print("🔄 Fusion en cours...")
            merge_audio_files()
            print("✅ Fusion terminée !")
            
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        stop_recording = True
    
    print("🏁 Script terminé.")

if __name__ == "__main__":
    main()