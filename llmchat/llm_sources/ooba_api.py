# -*- coding: utf-8 -*-
"""
Client for the Ooba API.
"""

from . import LLMSource
from llmchat.config import Config
from llmchat.persistence import PersistentData
from llmchat.logger import logger
import discord
import requests

# For local streaming, the websockets are hosted without ssl - http://
HOST = llmchat.config.oobabooga_listen_port
URI = f'http://{HOST}/api/v1/generate'

class OobaClient(LLMSource):
    """
    Client for the Ooba API.  
    """

    def __init__(self, client: discord.Client, config: Config, db: PersistentData):
        super(OobaClient, self).__init__(client, config, db)

    def similar_messages(self, last_message, messages_pool):
        similar_matches = []
        similarity_threshold = self.config.openai_similarity_threshold  # messages with a similarity rating equal to or above this number will be included in the reminder.
        # get embedding for last message
        last_message_embedding = self.db.query_embedding(last_message[2])
        if last_message_embedding:
            similar_matches = self.db.get_most_similar(last_message_embedding, threshold=similarity_threshold, messages_pool=messages_pool)[:self.config.openai_max_similar_messages]
        else:
            logger.warn("Unable to find embedding for message " + last_message[2])
        return similar_matches

    def get_prompt(self, invoker: discord.User = None) -> str:
        initial = self.get_initial(invoker)
        all_messages = self.db.get_recent_messages()
        recent_messages = all_messages[-self.config.llm_context_messages_count:]
        ooc_messages = all_messages[:-self.config.llm_context_messages_count]  # everything but the messages in the context limit
        similar_messages = []
        if self.config.openai_use_embeddings and ooc_messages:
            similar_matches = self.similar_messages(recent_messages[-1], ooc_messages)
            if similar_matches:
                logger.debug("Bot will be reminded of:\n\t"+'\n\t'.join([f"{message[1]} ({round(similarity * 100)}% similar)" for message, similarity in similar_matches]))
                # sort by message_id
                messages, similarities = list(zip(*similar_matches))
                similar_messages = list(messages)
                similar_messages.sort(key=lambda m: m[2])
        context = [initial] 
        context.append("###RELEVANT MEMORIES:")
        for memory in similar_messages:
             context.append(memory)
        context.append("###CURRENT CONVERSATION:")
        for message in recent_messages:
             context.append(message)
        return "\n".join(context)

    async def generate_response(
        self, invoker: discord.User = None, _retry_count=0
    ) -> str:
            # completion_tokens = self.config.llm_max_tokens
            prompt = self.get_prompt(invoker)
            logger.debug(prompt)
            request = {
                'prompt': prompt,
                'max_new_tokens': 250,
                'auto_max_new_tokens': False,
                'max_tokens_second': 0,

                # Generation params. If 'preset' is set to different than 'None', the values
                # in presets/preset-name.yaml are used instead of the individual numbers.
                'preset': 'None',
                'do_sample': True,
                'temperature': 0.7,
                'top_p': 0.1,
                'typical_p': 1,
                'epsilon_cutoff': 0,  # In units of 1e-4
                'eta_cutoff': 0,  # In units of 1e-4
                'tfs': 1,
                'top_a': 0,
                'repetition_penalty': 1.18,
                'repetition_penalty_range': 0,
                'top_k': 40,
                'min_length': 0,
                'no_repeat_ngram_size': 0,
                'num_beams': 1,
                'penalty_alpha': 0,
                'length_penalty': 1,
                'early_stopping': False,
                'mirostat_mode': 0,
                'mirostat_tau': 5,
                'mirostat_eta': 0.1,
                'grammar_string': '',
                'guidance_scale': 1,
                'negative_prompt': '',

                'seed': -1,
                'add_bos_token': True,
                'truncation_length': 2048,
                'ban_eos_token': False,
                'custom_token_bans': '',
                'skip_special_tokens': True,
                'stopping_strings': []
            }

            response = requests.post(URI, json=request)

            if response.status_code == 200:
                result = response.json()['results'][0]['text']
                return result
