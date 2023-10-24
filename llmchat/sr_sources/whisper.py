from . import SRSource
from discord import User, Client
from llmchat.config import Config
from llmchat.persistence import PersistentData
from speech_recognition import AudioData
from whispercpp import Whisper as WhisperCPP
import torch
from llmchat.logger import logger
import numpy as np

class Whisper(SRSource):
    def __init__(self, client: Client, config: Config, db: PersistentData):
        super(Whisper, self).__init__(client, config, db)
        self.model = WhisperCPP.from_pretrained("base") #note that you have to download the model yourself and move it to the cache if you're using the pypi wheels https://github.com/aarnphm/whispercpp/issues/126

    def recognize_speech(self, data: AudioData):
        resampled = data.get_raw_data(convert_rate=16_000)
        resampled = np.frombuffer(resampled, dtype=np.int16).flatten().astype(np.float32) / 32768.0
        decoded = self.model.transcribe(resampled)
        return decoded