import os.path

import discord

from . import TTSSource
from discord import User, Client
from llmchat.config import Config
from llmchat.persistence import PersistentData
from llmchat.logger import logger
import torch
import torchaudio
import io


from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

class XTTS(TTSSource):
    def __init__(self, client: Client, config: Config, db: PersistentData):
        super(XTTS, self).__init__(client, config, db)
        config = XttsConfig()
        config.load_json("models/XTTS-v2/config.json")
        self.model = Xtts.init_from_config(config)
        self.model.load_checkpoint(config, checkpoint_dir="models/XTTS-v2/")
        if torch.cuda.is_available():
            self.model.cuda()
        logger.info("Loaded XTTS model.")

    async def generate_speech(self, content: str) -> io.BufferedIOBase:
        gpt_cond_latent, speaker_embedding = self.model.get_conditioning_latents(audio_path=[self.config.xtts_voice])
        out = self.model.inference(
            content,
            "en",
            gpt_cond_latent,
            speaker_embedding,
            temperature=0.7, # Add custom parameters here
        )
        buf = io.BytesIO()
        torchaudio.save(buf, torch.tensor(out["wav"]).unsqueeze(0), 24000, format='wav')
        buf.seek(0)
        return buf

    @property
    def current_voice_name(self) -> str:
        return self.config.xtts_speaker

    def list_voices(self) -> list[discord.SelectOption]:
        # return [discord.SelectOption(label=v, value=v) for v in [f"en_{n}" for n in range(0, 118)]] # 117 voices
        return [discord.SelectOption(label=v, value=v) for v in glob.glob("voices/*.wav")]

    def set_voice(self, voice_id: str) -> None:
        self.config.xtts_speaker = voice_id
