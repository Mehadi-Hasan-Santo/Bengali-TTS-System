import numpy as np
import librosa

def mel_mse(reference_audio, synthesized_audio, sr=22050, n_mels=80, n_fft=1024, hop_length=256):
    """
    Calculate Mel-Spectrogram Mean Squared Error between reference and synthesized audio.
    
    Args:
        reference_audio (np.ndarray): Reference audio waveform
        synthesized_audio (np.ndarray): Synthesized audio waveform
        sr (int): Sampling rate (default: 22050)
        n_mels (int): Number of mel bands (default: 80)
        n_fft (int): FFT window size (default: 1024)
        hop_length (int): Number of samples between successive frames (default: 256)
    
    Returns:
        float: Mel-MSE score
    """
    # Compute mel spectrograms for both audio signals
    mel_ref = librosa.feature.melspectrogram(
        y=reference_audio,
        sr=sr,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length
    )
    
    mel_syn = librosa.feature.melspectrogram(
        y=synthesized_audio,
        sr=sr,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length
    )
    
    # Convert to log scale
    mel_ref_db = librosa.power_to_db(mel_ref, ref=np.max)
    mel_syn_db = librosa.power_to_db(mel_syn, ref=np.max)
    
    # Ensure both spectrograms have the same length
    min_length = min(mel_ref_db.shape[1], mel_syn_db.shape[1])
    mel_ref_db = mel_ref_db[:, :min_length]
    mel_syn_db = mel_syn_db[:, :min_length]
    
    # Calculate MSE
    mse = np.mean((mel_ref_db - mel_syn_db) ** 2)
    
    return mse

def main():
    # Example usage
    # Load your audio files here
    reference_audio = "../SelectedData/2308268001.wav"
    synthesized_audio = "../SelectedData/synthesized_2308268001.wav"
    ref_audio, sr = librosa.load(reference_audio, sr=None)
    syn_audio, _ = librosa.load(synthesized_audio, sr=sr)
    
    # Calculate Mel-MSE
    score = mel_mse(ref_audio, syn_audio, sr=sr)
    print(f"Mel-MSE score: {score}")

if __name__ == "__main__":
    main()
