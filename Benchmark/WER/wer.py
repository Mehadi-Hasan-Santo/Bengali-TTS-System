import torch
import torchaudio
from typing import List, Tuple
import numpy as np
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

class WordErrorRate:
    def __init__(self):
        # Use the same Bengali ASR model as in PER
        model_name = "arijitx/wav2vec2-xls-r-300m-bengali"
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name)

    def audio_to_text(self, audio_path: str) -> str:
        """Convert audio file to text using Wav2Vec 2.0"""
        # Load and preprocess audio
        waveform, sample_rate = torchaudio.load(audio_path)
        if sample_rate != 16000:
            waveform = torchaudio.transforms.Resample(sample_rate, 16000)(waveform)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Process through Wav2Vec 2.0
        inputs = self.processor(waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt")
        
        with torch.no_grad():
            logits = self.model(inputs.input_values).logits
        
        # Get predicted text
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.batch_decode(predicted_ids)[0]
        
        return transcription.strip()

    def calculate_wer(self, reference_path: str, synthesized_path: str) -> Tuple[float, dict]:
        """Calculate Word Error Rate (WER) between reference and synthesized audio"""
        # Get transcriptions
        syn_text = self.audio_to_text(synthesized_path)
        
        # Split into words
        ref_words = reference_path.split()  # Using reference text directly
        syn_words = syn_text.split()

        # Initialize counters
        substitutions = 0
        deletions = 0
        insertions = 0
        
        # Dynamic programming matrix
        dp = [[0] * (len(syn_words) + 1) for _ in range(len(ref_words) + 1)]
        
        # Initialize first row and column
        for i in range(len(ref_words) + 1):
            dp[i][0] = i
        for j in range(len(syn_words) + 1):
            dp[0][j] = j
            
        # Fill the matrix
        for i in range(1, len(ref_words) + 1):
            for j in range(1, len(syn_words) + 1):
                if ref_words[i-1] == syn_words[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = min(
                        dp[i-1][j] + 1,    # deletion
                        dp[i][j-1] + 1,    # insertion
                        dp[i-1][j-1] + 1   # substitution
                    )
        
        # Backtrace to count operations
        i, j = len(ref_words), len(syn_words)
        while i > 0 and j > 0:
            if ref_words[i-1] == syn_words[j-1]:
                i, j = i-1, j-1
            else:
                if dp[i][j] == dp[i-1][j-1] + 1:
                    substitutions += 1
                    i, j = i-1, j-1
                elif dp[i][j] == dp[i-1][j] + 1:
                    deletions += 1
                    i -= 1
                else:
                    insertions += 1
                    j -= 1
        
        # Handle remaining indels
        deletions += i
        insertions += j
        
        # Calculate WER
        total_words = len(ref_words)
        wer = (substitutions + deletions + insertions) / max(total_words, 1)  # Avoid division by zero

        stats = {
            'reference_words': total_words,
            'synthesized_words': len(syn_words),
            'substitutions': substitutions,
            'deletions': deletions,
            'insertions': insertions,
            'wer': wer,
            'reference_text': ' '.join(ref_words),
            'synthesized_text': ' '.join(syn_words)
        }

        return wer, stats

def main():
    wer_calculator = WordErrorRate()

    # Example paths and reference text
    reference_path = "../SelectedData/2308268001.wav"
    synthesized_path = "../SelectedData/synthesized_2308268001.wav"
    reference_text = "সেইজন্যই সংস্কৃতির নামে বলা অনেক কথায় ও রচনায় সত্যমিথ্যা দুইই একত্রে জড়িয়ে যাওয়া দুটো লতার মতো মাথা তুলে আছে" 

    wer, stats = wer_calculator.calculate_wer(reference_text, synthesized_path)

    print(f"Word Error Rate: {wer:.7f}")
    print("\nDetailed Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    main()
