# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

import json
import numpy as np
from scipy.io.wavfile import write
import io
import torch

import params
from model import GradTTSSDP
import sys
sys.path.append('./hifi-gan/')
from env import AttrDict
from models import Generator as HiFiGAN
import epitran
from text.bn_phonemiser import bangla_text_normalize, replace_number_with_text
from text.symbols import symbols

HIFIGAN_CONFIG = './checkpts/hifigan-config.json'
HIFIGAN_CHECKPT = './checkpts/hifigan.pt'

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

class TTSGenerator:
    def __init__(self):
        print('Initializing model...')
        self.generator = GradTTSSDP(len(symbols) + 1, params.n_spks, params.spk_emb_dim,
                        params.n_enc_channels, params.filter_channels,
                        params.filter_channels_dp, params.n_heads, params.n_enc_layers,
                        params.enc_kernel, params.enc_dropout, params.window_size,
                        params.n_feats, params.dec_dim, params.beta_min, params.beta_max,
                        pe_scale=1000)
        self.generator.load_state_dict(torch.load("logs/new_exp_mahedi_tomal/grad_100.pt", map_location=lambda loc, storage: loc))
        self.generator.eval()

        print('Initializing HiFi-GAN...')
        with open(HIFIGAN_CONFIG) as f:
            h = AttrDict(json.load(f))
        self.vocoder = HiFiGAN(h)
        self.vocoder.load_state_dict(torch.load(HIFIGAN_CHECKPT, map_location=lambda loc, storage: loc)['generator'])
        self.vocoder.eval()
        self.vocoder.remove_weight_norm()

    def generate_audio(self, text):
        x = torch.LongTensor(get_text(text))[None]
        x_lengths = torch.LongTensor([x.shape[-1]])

        y_enc, y_dec, attn = self.generator.forward(x, x_lengths, n_timesteps=50, temperature=1.3,
                                            stoc=False, spk=None if params.n_spks == 1 else torch.LongTensor([15]),
                                            length_scale=0.91)
        audio = (self.vocoder.forward(y_dec).cpu().squeeze().clamp(-1, 1).detach().numpy() * 32768).astype(np.int16)
        
        virtual_file = io.BytesIO()
        write(virtual_file, 22050, audio)
        virtual_file.seek(0)
        return virtual_file.read()

# Initialize the TTS generator
tts_generator = TTSGenerator()