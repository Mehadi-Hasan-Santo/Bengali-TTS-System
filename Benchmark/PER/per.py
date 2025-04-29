import torch
import torchaudio
from typing import List, Tuple
import numpy as np
import editdistance
import epitran
import os
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

class PhonemeErrorRate:
    def __init__(self):
        try:
            # Initialize Bangla phonemizer
            self.bn_phonemizer = epitran.Epitran('ben-Beng-east')
            
            # Use a public Bengali ASR model
            model_name = "arijitx/wav2vec2-xls-r-300m-bengali"
            self.processor = Wav2Vec2Processor.from_pretrained(model_name)
            self.model = Wav2Vec2ForCTC.from_pretrained(model_name)
        except Exception as e:
            print(f"Error initializing PhonemeErrorRate: {str(e)}")
            raise

    def audio_to_text(self, audio_path: str) -> str:
        """Convert audio file to text using Wav2Vec 2.0"""
        try:
            # Load and preprocess audio
            waveform, sample_rate = torchaudio.load(audio_path)
            if sample_rate != 16000:
                waveform = torchaudio.transforms.Resample(sample_rate, 16000)(waveform)
            
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            # Process through Wav2Vec 2.0
            inputs = self.processor(waveform.squeeze().numpy(), 
                                  sampling_rate=16000, 
                                  return_tensors="pt", 
                                  padding=True)
            
            with torch.no_grad():
                logits = self.model(inputs.input_values).logits
            
            # Get predicted text
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = self.processor.batch_decode(predicted_ids)[0]
            
            return transcription.strip()
        except Exception as e:
            print(f"Error in audio_to_text for {audio_path}: {str(e)}")
            return ""

    def text_to_phonemes(self, text: str) -> List[str]:
        """Convert Bangla text to phoneme sequence"""
        try:
            if not text:
                return []
            # Clean the text before processing
            text = text.strip()
            return list(self.bn_phonemizer.transliterate(text))
        except Exception as e:
            print(f"Error in text_to_phonemes: {str(e)}")
            print(f"Problematic text: {text}")
            return []

    def audio_to_phonemes(self, audio_path: str) -> List[str]:
        """Convert audio to phoneme sequence"""
        try:
            text = self.audio_to_text(audio_path)
            return self.text_to_phonemes(text)
        except Exception as e:
            print(f"Error in audio_to_phonemes for {audio_path}: {str(e)}")
            return []

    def calculate_per(self, reference_text: str, synthesized_path: str) -> Tuple[float, dict]:
        """Calculate Phoneme Error Rate (PER) between reference text and synthesized audio"""
        try:
            # Convert reference text directly to phonemes
            ref_phonemes = self.text_to_phonemes(reference_text)
            
            # Convert synthesized audio to phonemes
            syn_phonemes = self.audio_to_phonemes(synthesized_path)

            if not ref_phonemes or not syn_phonemes:
                print("Warning: Empty phoneme sequence detected")
                return 1.0, {
                    'reference_phonemes': len(ref_phonemes),
                    'synthesized_phonemes': len(syn_phonemes),
                    'levenshtein_distance': 0,
                    'per': 1.0,
                    'ref_phoneme_sequence': ' '.join(ref_phonemes),
                    'syn_phoneme_sequence': ' '.join(syn_phonemes)
                }

            # Compute Levenshtein distance
            distance = editdistance.eval(ref_phonemes, syn_phonemes)
            per = distance / max(len(ref_phonemes), 1)  # Avoid division by zero

            stats = {
                'reference_phonemes': len(ref_phonemes),
                'synthesized_phonemes': len(syn_phonemes),
                'levenshtein_distance': distance,
                'per': per,
                'ref_phoneme_sequence': ' '.join(ref_phonemes),
                'syn_phoneme_sequence': ' '.join(syn_phonemes)
            }

            return per, stats
        except Exception as e:
            print(f"Error in calculate_per: {str(e)}")
            print(f"Reference text: {reference_text}")
            print(f"Synthesized path: {synthesized_path}")
            return 1.0, {
                'error': str(e),
                'per': 1.0
            }

def main():
    """Example usage"""
    per_calculator = PhonemeErrorRate()

    # Example paths
    reference_path = "../SelectedData/2308268001.wav"
    synthesized_path = "../SelectedData/synthesized_2308268001.wav"

    per, stats = per_calculator.calculate_per(reference_path, synthesized_path)

    print(f"Phoneme Error Rate: {per:.7f}")
    print("\nDetailed Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    main()
