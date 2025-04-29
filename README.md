
# Grad-TTS

Official implementation of the Grad-TTS model based on Diffusion Probabilistic Modelling. For all details check out our paper accepted to ICML 2021 via [this](https://arxiv.org/abs/2105.06337) link.

**Authors**: Vadim Popov\*, Ivan Vovk\*, Vladimir Gogoryan, Tasnima Sadekova, Mikhail Kudinov.

<sup>\*Equal contribution.</sup>

## Abstract

**Demo page** with voiced abstract: [link](https://grad-tts.github.io/).

Recently, denoising diffusion probabilistic models and generative score matching have shown high potential in modelling complex data distributions while stochastic calculus has provided a unified point of view on these techniques allowing for flexible inference schemes. In this paper we introduce Grad-TTS, a novel text-to-speech model with score-based decoder producing mel-spectrograms by gradually transforming noise predicted by encoder and aligned with text input by means of Monotonic Alignment Search. The framework of stochastic differential equations helps us to generalize conventional diffusion probabilistic models to the case of reconstructing data from noise with different parameters and allows to make this reconstruction flexible by explicitly controlling trade-off between sound quality and inference speed. Subjective human evaluation shows that Grad-TTS is competitive with state-of-the-art text-to-speech approaches in terms of Mean Opinion Score.

## Installation

Firstly, install all Python package requirements:

```bash
pip install -r requirements.txt
```

Secondly, build `monotonic_align` code (Cython):

```bash
cd model/monotonic_align; python setup.py build_ext --inplace; cd ../..
```

**Note**: code is tested on Python==3.6.9.

## Training

1. Make filelists of your audio data like ones included into `resources/filelists` folder. For single speaker training refer to `jspeech` filelists and to `libri-tts` filelists for multispeaker.

2. Set experiment configuration in `params.py` file. Key parameters include:
   - `n_spks`: Set to 1 for single speaker or appropriate number for multispeaker
   - `pe_scale`: Set to 1 or 1000 depending on positional encoding scale
   - `log_dir`: Directory for storing checkpoints and logs

3. Start training with the appropriate script:
   ```bash
   # For single speaker training
   export CUDA_VISIBLE_DEVICES=YOUR_GPU_ID
   python train.py
   
   # For multispeaker training
   export CUDA_VISIBLE_DEVICES=YOUR_GPU_ID
   python train_multi_speaker.py
   ```

4. Monitor training progress with tensorboard:
   ```bash
   tensorboard --logdir=YOUR_LOG_DIR --port=8888
   ```

## Inference

The system provides multiple ways to generate speech from text:

### Command-line Inference

1. Download pre-trained models or use your own:
   - Pre-trained models available [here](https://drive.google.com/drive/folders/1grsfccJbmEuSBGQExQKr3cVxNV0xEOZ7?usp=sharing)
   - Place Grad-TTS and HiFi-GAN checkpoints in the `checkpts` folder

2. Create a text file with sentences to synthesize (similar to `resources/filelists/synthesis.txt`).

3. Configure model parameters in `params.py`:
   - For single speaker: set `params.n_spks=1`
   - For multispeaker: set `params.n_spks=247` (or appropriate number)
   - Set `params.pe_scale=1` for older models or `params.pe_scale=1000` for newer models

4. Run the inference script:
   ```bash
   python inference.py -f <your-text-file> -c <grad-tts-checkpoint> -t <number-of-timesteps> -s <speaker-id-if-multispeaker>
   ```
   
   Parameters:
   - `-f`: Path to text file with sentences
   - `-c`: Path to Grad-TTS checkpoint
   - `-t`: Number of iterations for reverse diffusion (default: 10)
   - `-s`: Speaker ID (only for multispeaker models)

5. Generated audio files will be saved in the `out` folder.

### Programmatic Inference

For integration into applications, use the `TTSGenerator` class:

```python
from inference import TTSGenerator

# Initialize the TTS system
tts_generator = TTSGenerator()

# Generate audio for a given text
audio_data = tts_generator.generate_audio("Your text to synthesize")

# The audio_data can be saved or played directly
```

The `TTSGenerator` class handles:
- Model initialization and loading
- Text preprocessing and phonemization
- Mel-spectrogram generation
- Audio synthesis with HiFi-GAN vocoder

### Interactive Inference

For interactive experimentation:

1. Use the provided Jupyter Notebook `inference.ipynb`
2. Or try our [Google Colab Demo](https://colab.research.google.com/drive/1YNrXtkJQKcYDmIYJeyX8s5eXxB4zgpZI?usp=sharing)

These interactive options allow for real-time parameter adjustment and immediate audio feedback.

## Benchmarking

The system includes a comprehensive benchmarking framework to evaluate TTS performance:

1. Prepare your evaluation dataset in the `SelectedDataSet` directory with:
   - Reference audio files (`.wav`)
   - Synthesized audio files (named `synthesized_audio*.wav`)
   - Text files (`.txt`) containing the reference text
   - A `synthesis_time.txt` file recording generation times

2. Run the benchmark evaluation:
   ```bash
   cd Benchmark
   python main.py
   ```

3. The benchmark evaluates multiple metrics:
   - Signal-to-Noise Ratio (SNR)
   - Word Error Rate (WER)
   - Phoneme Error Rate (PER)
   - Mel-spectrogram Mean Squared Error
   - Real-Time Factor (RTF)

4. Results are saved in two formats:
   - `benchmark_results.json`: Detailed metrics with individual and average results
   - `benchmark_results.csv`: Tabular format for easier analysis

5. The console output provides a summary of key performance metrics.

## References

* HiFi-GAN model is used as vocoder, official github repository: [link](https://github.com/jik876/hifi-gan).
* Monotonic Alignment Search algorithm is used for unsupervised duration modelling, official github repository: [link](https://github.com/jaywalnut310/glow-tts).
* Phonemization utilizes CMUdict, official github repository: [link](https://github.com/cmusphinx/cmudict).
