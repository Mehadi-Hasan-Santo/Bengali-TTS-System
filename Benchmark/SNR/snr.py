import numpy as np
import soundfile as sf

class SignalNoiseRatio:
    def __init__(self):
        pass

    def calculate_snr(self, reference_path: str, synthesized_path: str) -> float:
        """
        Calculate Signal-to-Noise Ratio between reference and synthesized audio.
        
        Args:
            reference_path (str): Path to the reference audio file
            synthesized_path (str): Path to the synthesized audio file
            
        Returns:
            float: SNR value in decibels (dB)
        """
        # Load audio files
        reference_signal, _ = sf.read(reference_path)
        synthesized_signal, _ = sf.read(synthesized_path)

        # Ensure both signals have the same length
        min_length = min(len(reference_signal), len(synthesized_signal))
        reference_signal = reference_signal[:min_length]
        synthesized_signal = synthesized_signal[:min_length]

        # Calculate signal power (numerator)
        signal_power = np.sum(reference_signal ** 2)

        # Calculate noise power (denominator)
        noise = reference_signal - synthesized_signal
        noise_power = np.sum(noise ** 2)

        # Avoid division by zero
        if noise_power == 0:
            return float('inf')

        # Calculate SNR in dB
        snr = 10 * np.log10(signal_power / noise_power)

        return snr

    def __call__(self, reference_path: str, synthesized_path: str) -> dict:
        """
        Calculate SNR and return as a dictionary.
        
        Args:
            reference_path (str): Path to the reference audio file
            synthesized_path (str): Path to the synthesized audio file
            
        Returns:
            dict: Dictionary containing SNR score
        """
        snr_value = self.calculate_snr(reference_path, synthesized_path)
        return {"snr": snr_value}

def main():
    # Example usage
    snr_calculator = SignalNoiseRatio()
    reference_path = "../SelectedData/2308268001.wav"
    synthesized_path = "../SelectedData/synthesized_2308268001.wav"
    
    result = snr_calculator(reference_path, synthesized_path)
    print(f"Signal-to-Noise Ratio: {result['snr']:.2f} dB")

if __name__ == "__main__":
    main()
