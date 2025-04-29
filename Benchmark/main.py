import os
import time
import json
import numpy as np
import pandas as pd
from typing import Dict, List
from SNR.snr import SignalNoiseRatio
from WER.wer import WordErrorRate
from PER.per import PhonemeErrorRate
from MSE.melmse import mel_mse
import librosa

class BenchmarkEvaluator:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.snr_calculator = SignalNoiseRatio()
        self.wer_calculator = WordErrorRate()
        self.per_calculator = PhonemeErrorRate()
        self.synthesis_times = self.load_synthesis_times()
        
    def load_synthesis_times(self) -> Dict[str, float]:
        """Load synthesis times from synthesis_time.txt"""
        synthesis_times = {}
        try:
            with open(os.path.join(self.data_dir, 'synthesis_time.txt'), 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 2:
                        filename, time_taken = parts
                        # Store with .wav extension to match the audio filenames
                        synthesis_times[f"{filename}.wav"] = float(time_taken)
        
            if not synthesis_times:
                print("Warning: No synthesis times loaded from synthesis_time.txt")
            else:
                print(f"Loaded {len(synthesis_times)} synthesis time entries")
            
        except FileNotFoundError:
            print("Warning: synthesis_time.txt not found. RTF calculations will be inaccurate.")
        except Exception as e:
            print(f"Error loading synthesis times: {str(e)}")
        
        return synthesis_times
        
    def get_file_paths(self) -> List[Dict[str, str]]:
        """Get paths for all reference, synthesized audio and text files"""
        file_paths = []
        
        for file in os.listdir(self.data_dir):
            if file.endswith('.wav') and not file.startswith('synthesized_audio'):
                base_name = file[:-4]  # Remove .wav
                file_paths.append({
                    'reference_audio': os.path.join(self.data_dir, file),
                    'synthesized_audio': os.path.join(self.data_dir, f'synthesized_audio{file}'),
                    'text_file': os.path.join(self.data_dir, f'{base_name}.txt')
                })
        
        return file_paths

    def calculate_rtf(self, synthesis_time: float, audio_duration: float) -> float:
        """Calculate Real-Time Factor"""
        return synthesis_time / audio_duration if audio_duration > 0 else float('inf')

    def evaluate_file(self, file_paths: Dict[str, str]) -> Dict[str, float]:
        """Evaluate a single set of files"""
        try:
            # First verify all files exist
            for key, path in file_paths.items():
                if not os.path.exists(path):
                    raise FileNotFoundError(f"File not found: {path}")

            # Load reference text with explicit UTF-8 encoding
            with open(file_paths['text_file'], 'r', encoding='utf-8') as f:
                reference_text = f.read().strip()

            # Load audio files for duration calculation
            try:
                ref_audio, sr = librosa.load(file_paths['reference_audio'], sr=None)
                syn_audio, _ = librosa.load(file_paths['synthesized_audio'], sr=sr)
            except Exception as e:
                print(f"Error loading audio files: {str(e)}")
                return None

            audio_duration = librosa.get_duration(y=ref_audio, sr=sr)

            # Get synthesis time from stored data
            file_name = os.path.basename(file_paths['reference_audio'])
            synthesis_time = self.synthesis_times.get(file_name, 0.0)
            
            if synthesis_time == 0.0:
                print(f"Warning: No synthesis time found for {file_name}")

            # Calculate metrics one by one with error handling
            results = {'file_name': file_name}
            
            try:
                results['snr'] = self.snr_calculator(file_paths['reference_audio'], 
                                                   file_paths['synthesized_audio'])['snr']
            except Exception as e:
                print(f"Error calculating SNR: {str(e)}")
                results['snr'] = None

            try:
                results['wer'] = self.wer_calculator.calculate_wer(reference_text, 
                                                                 file_paths['synthesized_audio'])[0]
            except Exception as e:
                print(f"Error calculating WER: {str(e)}")
                results['wer'] = None

            try:
                results['per'] = self.per_calculator.calculate_per(reference_text, 
                                                                 file_paths['synthesized_audio'])[0]
            except Exception as e:
                print(f"Error calculating PER: {str(e)}")
                results['per'] = None

            try:
                results['mel_mse'] = mel_mse(ref_audio, syn_audio, sr=sr)
            except Exception as e:
                print(f"Error calculating MEL MSE: {str(e)}")
                results['mel_mse'] = None

            results['rtf'] = self.calculate_rtf(synthesis_time, audio_duration)
            results['audio_duration'] = audio_duration
            results['synthesis_time'] = synthesis_time
            
            # Check if we have at least some valid metrics
            if all(v is None for v in [results['snr'], results['wer'], 
                                      results['per'], results['mel_mse']]):
                print(f"All metrics failed for file {file_name}")
                return None
            
            return results

        except UnicodeDecodeError as e:
            print(f"Unicode decode error for {file_paths['text_file']}: {str(e)}")
            return None
        except FileNotFoundError as e:
            print(f"File not found: {str(e)}")
            return None
        except Exception as e:
            print(f"Unexpected error processing {file_paths['reference_audio']}: {str(e)}")
            print(f"File paths being processed: {file_paths}")
            return None

    def run_benchmark(self) -> None:
        """Run benchmark on all files and save results"""
        all_results = []
        file_paths = self.get_file_paths()
        
        print(f"Starting evaluation of {len(file_paths)} files...")
        
        for i, paths in enumerate(file_paths, 1):
            print(f"Processing file {i}/{len(file_paths)}: {paths['reference_audio']}")
            results = self.evaluate_file(paths)
            if results is not None:
                # Convert any NumPy types to native Python types
                converted_results = {}
                for key, value in results.items():
                    if isinstance(value, (np.float32, np.float64)):
                        converted_results[key] = float(value)
                    elif isinstance(value, (np.int32, np.int64)):
                        converted_results[key] = int(value)
                    else:
                        converted_results[key] = value
                all_results.append(converted_results)
        
        if not all_results:
            print("No files were successfully processed. Check the file encodings and paths.")
            return

        # Create DataFrame and exclude non-numeric columns for mean calculation
        df = pd.DataFrame(all_results)
        numeric_columns = ['snr', 'wer', 'per', 'mel_mse', 'rtf', 'audio_duration', 'synthesis_time']
        
        # Calculate average metrics only for numeric columns
        average_metrics = df[numeric_columns].mean()
        std_metrics = df[numeric_columns].std()
        
        # Convert all numpy values to Python native types
        def convert_to_native(obj):
            if isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_native(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native(item) for item in obj]
            return obj

        # Save detailed results
        results_dict = {
            'individual_results': convert_to_native(all_results),
            'average_metrics': {
                'snr_mean': float(average_metrics['snr']),
                'snr_std': float(std_metrics['snr']),
                'wer_mean': float(average_metrics['wer']),
                'wer_std': float(std_metrics['wer']),
                'per_mean': float(average_metrics['per']),
                'per_std': float(std_metrics['per']),
                'mel_mse_mean': float(average_metrics['mel_mse']),
                'mel_mse_std': float(std_metrics['mel_mse']),
                'rtf_mean': float(average_metrics['rtf']),
                'rtf_std': float(std_metrics['rtf'])
            }
        }
        
        # Save results to JSON file
        with open('benchmark_results.json', 'w', encoding='utf-8') as f:
            json.dump(results_dict, f, indent=2, ensure_ascii=False)
        
        # Save results to CSV for easier analysis
        df.to_csv('benchmark_results.csv', index=False)
        
        # Print summary
        print("\nBenchmark Results Summary:")
        print(f"Number of files evaluated: {len(all_results)}")
        print("\nAverage Metrics:")
        for metric, value in results_dict['average_metrics'].items():
            print(f"{metric}: {value:.4f}")

def main():
    # Initialize and run benchmark
    evaluator = BenchmarkEvaluator("./SelectedDataSet")
    evaluator.run_benchmark()

if __name__ == "__main__":
    main()
