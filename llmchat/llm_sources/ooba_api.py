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
import json
import datetime
import pytz

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
            similar_matches = self.db.get_most_similar(last_message_embedding, threshold=similarity_threshold, messages_pool=messages_pool, include_previous_message=self.config.memory_include_previous_message)[:self.config.openai_max_similar_messages]
        else:
            logger.warn("Unable to find embedding for message " + last_message[2])
        return similar_matches

    async def format_with_author(self, message):
        fmt_message = ""
        author_id, content, message_id = message
        if author_id == -1:
            #do nothing to the message content if the author is not available
            return content
        elif author_id == self.client.user.id:
            fmt_message = f"{self.config.bot_name}: {content}"
        else:
            name = (await self.client.fetch_user(author_id)).display_name
            identity = self.db.get_identity(author_id)
            if identity is not None:
                name_, identity = identity
                name = name_
            fmt_message = f"{name}: {content}"
        return fmt_message

    async def add_author_to_messages(self, messages):
        final_messages = []
        for i in messages:
            fmt_message = await self.format_with_author(i)
            final_messages.append(fmt_message)
        return final_messages

    async def get_recent_discord_messages(self, channel, N, T):
        messages = [m async for m in channel.history(limit=N)]
        current_time = datetime.datetime.now(pytz.utc)
        current_time = current_time.astimezone(pytz.timezone('US/Pacific'))
        recent_messages = []
        #skip the most recent message; already included as the initial message content.
        for message in messages[::-1]:
            time_diff = (current_time - message.created_at).total_seconds()
            if time_diff <= T:
                recent_messages.append(message.author.name + ":" + message.clean_content)
            else:
                break
        return recent_messages

    @property
    def current_model_name(self) -> str:
        host = str(self.config.oobabooga_listen_port)
        uri = f'http://{host}/api/v1/model'
        return requests.get(uri).json()['result']

    async def get_prompt(self, invoker: discord.User = None, channel = None) -> str:
        initial = self.get_initial(invoker)
        all_messages = self.db.get_recent_messages()
        recent_messages = all_messages[-self.config.llm_context_messages_count:]
        ooc_messages = all_messages[:-self.config.llm_context_messages_count]  # everything but the messages in the context limit
        similar_messages = []
        if (self.config.openai_use_embeddings or self.config.use_local_embeddings) and ooc_messages:
            similar_matches = self.similar_messages(recent_messages[-1], ooc_messages)
            if similar_matches:
                logger.debug("Bot will be reminded of:\n\t"+'\n\t'.join([f"{message[1]} ({round(similarity * 100)}% similar)" for message, similarity in similar_matches]))
                # sort by message_id
                messages, similarities = list(zip(*similar_matches))
                similar_messages = list(messages)
                similar_messages.sort(key=lambda m: m[2])
        context = [initial] 
        context.append("###RELEVANT MEMORIES:")
        memories = await self.add_author_to_messages(similar_messages)
        for memory in memories:
             context.append(memory)
        context.append("###CURRENT CONVERSATION:")
        current_conversation = []
        if channel != None:
            logger.debug(f"Trying to fetch history from channel ID: {channel.id}")
            #TODO: allow N and T to be set as parameters in the config.
            #no technical reason this needs to be the same value as the N most recent messages excluded from memory/N most recent used for voice context but probably has reasonable behavior
            N = self.config.llm_context_messages_count 
            T = 24 * 60 * 60 #number of seconds in a day- thus excluding messages from discord history that are more than a day old.
            current_conversation = await self.get_recent_discord_messages(channel, N, T)
            for message in current_conversation[:-1]:
                context.append(message)
            #add the most recent message as-recorded. This allows for compatibility with BLIP and similar implementations.
            #this means that we need to prepend the author, though. 
            current_message = await self.format_with_author(recent_messages[-1])
            context.append(current_message)
        else:
            #otherwise, fetch recent messages directly from the relevant channel.
            current_conversation = await self.add_author_to_messages(recent_messages)
            for message in current_conversation:
                context.append(message)
        return "\n".join(context)

    async def generate_response(
        self, invoker: discord.User = None, channel = None
    ) -> str:
            # completion_tokens = self.config.llm_max_tokens
            prompt = await self.get_prompt(invoker, channel)
            logger.debug(prompt)
            request = {
                'user_input': prompt,
                'max_new_tokens': int(self.config.llm_max_tokens),
                'auto_max_new_tokens': False,
                'max_tokens_second': 0,
                'history': {'internal':[],'visible':[]},
                'mode': 'chat',  # Valid options: 'chat', 'chat-instruct', 'instruct'
                'character': self.config.character,
                # 'instruction_template': 'Vicuna-v1.1',  # Will get autodetected if unset
                # 'your_name': 'You',
                'regenerate': False,
                '_continue': False,
                'chat_instruct_command': 'Continue the chat dialogue below. Write a single reply for the character "<|character|>".\n\n<|prompt|>',

                # Generation params. If 'preset' is set to different than 'None', the values
                # in presets/preset-name.yaml are used instead of the individual numbers.
                'preset': 'Midnight Enigma',
                'do_sample': True,
                'temperature': float(self.config.llm_temperature),
                'top_p': 0.1,
                'typical_p': 1,
                'epsilon_cutoff': 0,  # In units of 1e-4
                'eta_cutoff': 0,  # In units of 1e-4
                'tfs': 1,
                'top_a': 0,
                'repetition_penalty': 1.18,
                'repetition_penalty_range': 0,
                'presence_penalty': float(self.config.llm_presence_penalty),
                'frequency_penalty': float(self.config.llm_frequency_penalty),
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
            host = str(self.config.oobabooga_listen_port)
            uri = f'http://{host}/api/v1/chat'
            response = requests.post(uri, json=request)

            if response.status_code == 200:
                result = response.json()
                return result['results'][0]['history']['internal'][0][-1]
