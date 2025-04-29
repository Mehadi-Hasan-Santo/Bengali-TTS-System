

# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

import argparse
import json
import datetime as dt
import numpy as np
from scipy.io.wavfile import write
import tempfile
import os
from starlette.responses import FileResponse
import torch

import params
from model import GradTTS, GradTTSSDP, GradTTSSDPContext


from text import text_to_sequence, cmudict
from text.symbols import symbols
from utils import intersperse
from librosa import pyin

import sys
sys.path.append('./hifi-gan/')
from env import AttrDict
from models import Generator as HiFiGAN
# from model.stft_loss import MultiResolutionSTFTLoss
from meldataset import mel_spectrogram
import epitran
from text.bn_phonemiser import bangla_text_normalize, replace_number_with_text

# import IPython.display as ipd

HIFIGAN_CONFIG = './checkpts/hifigan-config.json'
HIFIGAN_CHECKPT = './checkpts/hifigan.pt'



# Assuming all necessary imports for GradTTSSDP and HiFiGAN are already made
def get_ipa_tokens_from_text(tokens):
    ipa_dict = {
        'ɖ̤': 0, 'kʰ': 1, 'n': 2, 'm': 3, 'ɡ̤': 4, 'ʃ': 5, 'b̤': 6, 'd̪̤': 7,
        'ŋ': 8, 'pʰ': 9, 'ɽ̤': 10, 'k': 11, 'a': 12, 'b': 13, 'r': 14, 'ʈʰ': 15,
        'V': 16, 'ɖ': 17, 't̪ʰ': 18, 'p': 19, 'z': 20, 'e': 21, 't̪': 22, 'u': 23,
        'j': 24, 'd̪': 25, 'o': 26, 'i': 27, 'd͡z': 28, 's': 29, 'd͡z̤': 30, 'ঃ': 31,
        'h': 32, '্': 33, 'ɽ': 34, '̃': 35, 'l': 36, 'ʈ': 37, 'ɡ': 38, 'ɔ': 39, ' ': 40
    }
    ipa_tokens = [ipa_dict[tok] for tok in tokens if tok in ipa_dict]
    return ipa_tokens

def get_text(text):
    bn_phonemizer = epitran.Epitran('ben-Beng-east')
    text_norm = replace_number_with_text(text)
    text_norm = bangla_text_normalize(text_norm)
    text_tokens = bn_phonemizer.trans_list(text_norm)
    ipa_tokens = get_ipa_tokens_from_text(text_tokens)
    return ipa_tokens

# Load the model and vocoder
print('Initializing model...')
generator = GradTTSSDP(len(symbols) + 1, params.n_spks, params.spk_emb_dim,
                    params.n_enc_channels, params.filter_channels,
                    params.filter_channels_dp, params.n_heads, params.n_enc_layers,
                    params.enc_kernel, params.enc_dropout, params.window_size,
                    params.n_feats, params.dec_dim, params.beta_min, params.beta_max,
                    pe_scale=1000)  # pe_scale=1 for `grad-tts-old.pt`
generator.load_state_dict(torch.load("logs/new_exp_mahedi_tomal/grad_100.pt", map_location=lambda loc, storage: loc))
_ = generator.eval()
print(f'Number of parameters: {generator.nparams}')

print('Initializing HiFi-GAN...')
with open(HIFIGAN_CONFIG) as f:
    h = AttrDict(json.load(f))
vocoder = HiFiGAN(h)
vocoder.load_state_dict(torch.load(HIFIGAN_CHECKPT, map_location=lambda loc, storage: loc)['generator'])
_ = vocoder.eval()
vocoder.remove_weight_norm()



text = "সেইজন্যই সংস্কৃতির নামে বলা অনেক কথায় ও রচনায় সত্যমিথ্যা দুইই একত্রে জড়িয়ে যাওয়া দুটো লতার মতো মাথা তুলে আছে"
x = torch.LongTensor(get_text(text))[None]
x_lengths = torch.LongTensor([x.shape[-1]])

# Generate audio
y_enc, y_dec, attn = generator.forward(x, x_lengths, n_timesteps=50, temperature=1.3,
                                        stoc=False, spk=None if params.n_spks == 1 else torch.LongTensor([15]),
                                        length_scale=0.91)
audio = (vocoder.forward(y_dec).cpu().squeeze().clamp(-1, 1).detach().numpy() * 32768).astype(np.int16)
filename = f"synthesized_audio2308268001.wav"
write(filename, 22050, audio)
print('Inference done! Please Check out the `out` folder for the samples...')
