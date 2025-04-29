import numpy as np
from typing import Union, Sequence
import librosa
import os

def extract_mceps(audio_path: str, n_mels: int = 13, sr: int = 22050) -> np.ndarray:
    """
    Extract mel-cepstral coefficients from an audio file.
    
    Args:
        audio_path: Path to the audio file
        n_mels: Number of mel coefficients to extract
        sr: Sample rate for audio processing
    
    Returns:
        np.ndarray: Mel-cepstral coefficients
    """
    # Load audio file
    y, _ = librosa.load(audio_path, sr=sr)
    
    # Extract mel spectrogram
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    
    # Convert to log scale
    log_mel_spec = librosa.power_to_db(mel_spec)
    
    # Extract MFCCs (mel-cepstral coefficients)
    mceps = librosa.feature.mfcc(S=log_mel_spec, n_mfcc=n_mels)
    
    return mceps.T  # Return transposed to get [frames, coeffs] shape

def calculate_mcd(reference_mceps: Union[np.ndarray, Sequence[float]],
                 synthesized_mceps: Union[np.ndarray, Sequence[float]]) -> float:
    """
    Calculate Mel Cepstral Distortion (MCD) between reference and synthesized mel-cepstral coefficients.
    
    MCD measures the spectral distance between two speech signals in the mel-cepstral domain.
    Lower values indicate better match between reference and synthesized speech.
    
    Args:
        reference_mceps: Array of mel-cepstral coefficients from reference speech
        synthesized_mceps: Array of mel-cepstral coefficients from synthesized speech
    
    Returns:
        float: Mel Cepstral Distortion value in dB
    
    Raises:
        ValueError: If input arrays have different lengths or are empty
    """
    # Convert inputs to numpy arrays if they aren't already
    ref_mceps = np.array(reference_mceps, dtype=np.float64)
    syn_mceps = np.array(synthesized_mceps, dtype=np.float64)
    
    # Validate inputs
    if ref_mceps.shape != syn_mceps.shape:
        raise ValueError("Reference and synthesized mel-cepstral coefficients must have the same shape")
    
    if ref_mceps.size == 0:
        raise ValueError("Input arrays cannot be empty")
    
    # Calculate the squared difference between coefficients
    squared_diff = (ref_mceps - syn_mceps) ** 2
    
    # Sum the differences and apply the MCD formula
    # MCD = (10 / ln(10)) * sqrt(2 * sum(squared_differences))
    mcd = (10.0 / np.log(10.0)) * np.sqrt(2.0 * np.sum(squared_diff))
    
    return mcd

def batch_calculate_mcd(reference_batch: np.ndarray,
                       synthesized_batch: np.ndarray,
                       return_mean: bool = True) -> Union[float, np.ndarray]:
    """
    Calculate MCD for batches of mel-cepstral coefficients.
    
    Args:
        reference_batch: Batch of reference mel-cepstral coefficients [batch_size, n_coeffs]
        synthesized_batch: Batch of synthesized mel-cepstral coefficients [batch_size, n_coeffs]
        return_mean: If True, returns mean MCD across batch. If False, returns array of MCDs
    
    Returns:
        Union[float, np.ndarray]: Mean MCD value or array of MCD values for each pair
    """
    if reference_batch.shape != synthesized_batch.shape:
        raise ValueError("Reference and synthesized batches must have the same shape")
    
    batch_size = reference_batch.shape[0]
    mcd_values = np.zeros(batch_size)
    
    for i in range(batch_size):
        mcd_values[i] = calculate_mcd(reference_batch[i], synthesized_batch[i])
    
    return np.mean(mcd_values) if return_mean else mcd_values

if __name__ == "__main__":
    # Example usage with actual audio files
    reference_audio = "SelectedData/2308268001.wav"
    synthesized_audio = "SelectedData/synthesized_2308268001.wav"
    
    # Check if files exist
    if not os.path.exists(reference_audio) or not os.path.exists(synthesized_audio):
        raise FileNotFoundError("Audio files not found")
    
    # Extract mel-cepstral coefficients
    ref_mceps = extract_mceps(reference_audio)
    syn_mceps = extract_mceps(synthesized_audio)
    
    # Ensure both sequences have the same length (take shorter one)
    min_length = min(len(ref_mceps), len(syn_mceps))
    ref_mceps = ref_mceps[:min_length]
    syn_mceps = syn_mceps[:min_length]
    
    # Calculate MCD
    mcd_value = calculate_mcd(ref_mceps, syn_mceps)
    print(f"MCD between reference and synthesized audio: {mcd_value:.2f} dB")
    
    # # If you have multiple files, you can process them in batch
    # reference_files = ["path/to/ref1.wav", "path/to/ref2.wav"]
    # synthesized_files = ["path/to/syn1.wav", "path/to/syn2.wav"]
    
    # # Process batch of files
    # ref_batch = np.array([extract_mceps(f) for f in reference_files])
    # syn_batch = np.array([extract_mceps(f) for f in synthesized_files])
    
    # # Calculate mean MCD for the batch
    # mean_mcd = batch_calculate_mcd(ref_batch, syn_batch, return_mean=True)
    # print(f"Mean batch MCD: {mean_mcd:.2f} dB")
