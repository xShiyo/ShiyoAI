import os
os.environ['HF_HOME'] = "/Volumes/cool sd card thing/huggingface_cache"

import json
import time
import asyncio
import signal
import re
from datetime import datetime, timedelta
from dotenv import load_dotenv
from memory import store_in_memory
import aiohttp
import discord
import discord.opus
from discord import FFmpegPCMAudio
from personality import Personality
# from faster_whisper import WhisperModel
import wave
from rapidfuzz import process, fuzz
import logging
import boto3
import uuid
from collections import deque

# New imports for semantic memory
from sentence_transformers import SentenceTransformer
import numpy as np


load_dotenv()

TOKEN = os.getenv("DISCORD_TOKEN")
BACKEND_URL = os.getenv("BACKEND_URL")
MODEL_API_URL = BACKEND_URL + "/generate" if BACKEND_URL else None

SHORT_TERM_MEMORY_FILE = "short_term_memory.json"
LAST_SHUTDOWN_FILE = "last_shutdown.txt"
WHITELIST_FILE = "whitelist.json"
OFFLINE_WIPE_SECONDS = 10 * 60

OWNER_ID = 490436968543551493
LOG_CHANNEL_ID = 1383158697861058650
MEMORY_CHANNEL_ID = 1383129869730447600

logging.basicConfig(
    filename='bot.log',
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


def log_dm(direction, author, target, content):
    logging.info(f"DM {direction} | From: {author} | To: {target} | Content: {content}")


intents = discord.Intents.default()
intents.message_content = True
intents.messages = True
intents.dm_messages = True
intents.guilds = True
intents.members = True

client = discord.Client(intents=intents)

short_term_memory = []
whitelist = set()
temp_whitelist = {}
personality_state = {"mood": "neutral", "interests": [], "last_update": time.time()}
startup_time = time.time()

# Semantic memory globals
embedding_model = None
short_term_embeddings = deque(maxlen=500)

# Chat pacing globals
recent_message_times = deque(maxlen=10)
recent_message_buffer = deque(maxlen=10)
last_response_time = 0


def init_embedding_model():
    global embedding_model
    if embedding_model is not None:
        return
    try:
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    except Exception as e:
        logging.error(f"Failed to load embedding model: {e}")
        embedding_model = None


def embed_text(text: str) -> np.ndarray:
    if embedding_model is None:
        init_embedding_model()
    if embedding_model is None:
        return np.zeros(384)
    try:
        return embedding_model.encode(text, convert_to_numpy=True)
    except Exception as e:
        logging.error(f"Embedding failed: {e}")
        return np.zeros(384)


def build_memory_embeddings():
    """Embed all short-term memory entries and refresh the embeddings deque."""
    global short_term_embeddings
    short_term_embeddings.clear()

    if not short_term_memory:
        return

    init_embedding_model()
    if embedding_model is None:
        return

    try:
        texts = [entry["content"] for entry in short_term_memory]
        embeddings_list = embedding_model.encode(texts, convert_to_numpy=True)
        for emb in embeddings_list:
            short_term_embeddings.append(emb)
    except Exception as e:
        logging.error(f"Failed to build memory embeddings: {e}")


def ask_bedrock(prompt: str) -> str:
    """Sends a prompt to Bedrock (Mistral 7B) and gets a direct response."""
    try:
        bedrock_runtime = boto3.client(service_name='bedrock-runtime', region_name='us-east-1')
        formatted_prompt = f"<s>[INST] {prompt} [/INST]"

        body = json.dumps({
            "prompt": formatted_prompt,
            "max_tokens": 1024,
            "temperature": 0.7
        })

        model_id = 'mistral.mistral-7b-instruct-v0:2'

        response = bedrock_runtime.invoke_model(
            body=body,
            modelId=model_id,
            contentType='application/json',
            accept='application/json'
        )

        response_body = json.loads(response.get('body').read())
        return response_body.get('outputs')[0].get('text')

    except Exception as e:
        logging.error(f"Failed to call Bedrock: {e}")
        return f"Sorry, I encountered an error with Bedrock: {e}"


async def run_bedrock(prompt: str) -> str:
    """Run the blocking Bedrock call in a thread to avoid blocking the event loop."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, ask_bedrock, prompt)


def load_last_shutdown_time():
    if os.path.exists(LAST_SHUTDOWN_FILE):
        with open(LAST_SHUTDOWN_FILE, "r") as f:
            try:
                return float(f.read().strip())
            except:
                return 0
    return 0


def save_last_shutdown_time():
    with open(LAST_SHUTDOWN_FILE, "w") as f:
        f.write(str(time.time()))


def load_short_term_memory():
    if os.path.exists(SHORT_TERM_MEMORY_FILE):
        with open(SHORT_TERM_MEMORY_FILE, "r") as f:
            try:
                return json.load(f)
            except:
                return []
    return []


def save_short_term_memory():
    with open(SHORT_TERM_MEMORY_FILE, "w") as f:
        json.dump(short_term_memory, f, indent=2)


def load_whitelist():
    global whitelist, temp_whitelist
    if os.path.exists(WHITELIST_FILE):
        with open(WHITELIST_FILE, "r") as f:
            try:
                data = json.load(f)
                if isinstance(data, dict):
                    whitelist = set(data.get("whitelist", []))
                    temp_raw = data.get("temp_whitelist", {})
                    temp_whitelist.clear()
                    for uid_str, expiry_str in temp_raw.items():
                        expiry = datetime.strptime(expiry_str, "%Y-%m-%dT%H:%M:%S")
                        if expiry > datetime.utcnow():
                            temp_whitelist[int(uid_str)] = expiry
                else:
                    raise ValueError("Whitelist file must be a JSON object.")
            except Exception as e:
                print(f"Error loading whitelist: {e}")
                asyncio.create_task(send_error_to_discord(f"Error loading whitelist: {e}"))


def save_whitelist():
    temp_serializable = {str(uid): expiry.strftime("%Y-%m-%dT%H:%M:%S") for uid, expiry in temp_whitelist.items()}
    data = {
        "whitelist": list(whitelist),
        "temp_whitelist": temp_serializable
    }
    with open(WHITELIST_FILE, "w") as f:
        json.dump(data, f, indent=2)


def parse_duration(duration_str):
    unit_map = {
        'm': 'minutes',
        'h': 'hours',
        'd': 'days',
        'w': 'weeks',
        'mo': 'days',
        'y': 'days'
    }
    match = re.match(r"(\d+)([a-z]+)", duration_str)
    if not match:
        return None
    value, unit = match.groups()
    value = int(value)

    if unit == 'mo':
        return timedelta(days=value * 30)
    elif unit == 'y':
        return timedelta(days=value * 365)
    elif unit in unit_map:
        return timedelta(**{unit_map[unit]: value})
    return None


def check_temp_whitelist():
    now = datetime.utcnow()
    expired = [uid for uid, expiry in temp_whitelist.items() if expiry < now]
    for uid in expired:
        del temp_whitelist[uid]
    if expired:
        save_whitelist()


async def send_error_to_discord(error_text):
    log_channel = client.get_channel(LOG_CHANNEL_ID)
    if log_channel:
        await log_channel.send(f"❌ Error: {error_text}")


if not discord.opus.is_loaded():
    try:
        discord.opus.load_opus('libopus.so')
    except OSError:
        try:
            discord.opus.load_opus('./libopus.dylib')
        except OSError:
            try:
                discord.opus.load_opus('')
            except OSError:
                print("❌ Failed to load Opus library. Voice features will not work.")

try:
    base_sink = discord.AudioSink
except AttributeError:
    class DummySink:
        def __init__(self):
            self.audio_data = bytearray()

        def write(self, data):
            self.audio_data.extend(data)

        def clear(self):
            self.audio_data = bytearray()

    base_sink = DummySink


class VoiceReceiver(base_sink):
    def __init__(self):
        super().__init__()

    def write(self, data):
        if hasattr(super(), 'write'):
            super().write(data)
        else:
            self.audio_data.extend(data)

    def clear(self):
        if hasattr(super(), 'clear'):
            super().clear()
        else:
            self.audio_data = bytearray()


async def join_and_listen(vc):
    await vc.connect()
    receiver = VoiceReceiver()
    vc.start_recording(receiver, lambda e: print(f"Recording finished: {e}"))

    await asyncio.sleep(30)

    vc.stop_recording()
    print(f"Recorded {len(receiver.audio_data)} bytes of audio")

    with open("output.raw", "wb") as f:
        f.write(receiver.audio_data)

    await vc.disconnect()


def update_personality_state(memory, state):
    positive_words = ["good", "great", "happy", "fun", "love"]
    negative_words = ["bad", "sad", "angry", "hate"]
    pos_count = 0
    neg_count = 0

    for entry in memory[-20:]:
        content = entry.get("content", "").lower()
        pos_count += sum(content.count(w) for w in positive_words)
        neg_count += sum(content.count(w) for w in negative_words)

    if pos_count > neg_count:
        state["mood"] = "happy"
    elif neg_count > pos_count:
        state["mood"] = "grumpy"
    else:
        state["mood"] = "neutral"

    state["last_update"] = time.time()
    return state


def get_relevant_memory_semantic(user_message, memory, max_entries=5):
    if not memory:
        return []
    if len(short_term_embeddings) == 0:
        build_memory_embeddings()
    if len(short_term_embeddings) == 0:
        return memory[-max_entries:]

    user_emb = embed_text(user_message)
    embedding_array = np.array(short_term_embeddings)
    if embedding_array.ndim == 1:
        embedding_array = embedding_array.reshape(1, -1)

    eps = 1e-10
    emb_norms = (np.linalg.norm(embedding_array, axis=1) * np.linalg.norm(user_emb)) + eps
    similarities = np.dot(embedding_array, user_emb) / emb_norms
    similarities = np.nan_to_num(similarities)
    top_indices = np.argsort(similarities)[-max_entries:][::-1]
    return [memory[i] for i in top_indices]


def build_prompt(user_name: str, user_message: str, personality_state: dict, memory: list):
    """Builds a direct prompt that includes mood-based instructions."""

    mood = personality_state.get("mood", "neutral")
    relevant_memory = get_relevant_memory_semantic(user_message, memory)
    memory_text = "\n".join(
        f"{m['user']}: {m['content']}" for m in relevant_memory
    ) if relevant_memory else "No relevant conversation history."

    personality_guide = f"You are a Discord user named Allay-sama. Your creator is Shiyo. Your current mood is {mood}. If you don't feel like replying, return an empty response."

    if mood == "grumpy":
        personality_guide += "Respond in a blunt or sarcastic way. "
    elif mood == "happy":
        personality_guide += "Respond in a playful or cheerful way. "

    prompt = (
        f"{personality_guide}\n\n"
        f"### CONVERSATION CONTEXT\n{memory_text}\n\n"
        f"### YOUR TASK\n"
        f"A user named '{user_name}' just sent this message: \"{user_message}\"\n"
        f"Write a short, direct reply to '{user_name}'.\n"
        f"**To send a private message, start your response with 'DM:'. Only send if and when necessary.**\n"
        f"Do not give too many expressions about inside thoughts"
        f"Do not need use punctuation or capital letters except if the response won't make any sense without it."
        f"DO NOT include context that you are not saying to the user directly"
        f"DO NOT write a script. DO NOT write dialogue for the user. DO NOT list multiple possible responses.\n"
        f"Just write your single, natural response.\n\n"
        f"Allay's Reply:"
        f"DO NOT include '[Allay-sama]:' in your response"
    )

    print(f"--- PROMPT SENT TO BEDROCK ---\n{prompt}\n-----------------------------")
    return prompt


@client.event
async def on_ready():
    global short_term_memory

    print("Loading sentence embedding model...")
    init_embedding_model()
    print("Embedding model loaded.")

    load_whitelist()
    check_temp_whitelist()

    last_shutdown = load_last_shutdown_time()
    offline_duration = startup_time - last_shutdown if last_shutdown else 0
    print(f"Offline duration: {offline_duration} seconds")

    if offline_duration > OFFLINE_WIPE_SECONDS:
        print("Bot was offline for more than 10 minutes — wiping memory.")
        short_term_memory = []
        short_term_embeddings.clear()
        save_short_term_memory()
    else:
        print("Bot was offline less than 10 minutes — loading and processing memory.")
        short_term_memory = load_short_term_memory()
        build_memory_embeddings()

    client.loop.create_task(chat_response_loop())
    client.loop.create_task(autonomous_messaging_loop())
    print(f"{client.user} is now running!")


async def chat_response_loop():
    """Autonomous chat response loop for general conversation pacing."""
    global last_response_time
    while True:
        await asyncio.sleep(1)
        if not recent_message_buffer:
            continue

        time_since_last_msg = time.time() - recent_message_times[-1]
        time_since_last_reply = time.time() - last_response_time
        # Placeholder: add pacing logic when ready

# whisper_model = WhisperModel("small")  # can also use "medium" or "large" if you want better accuracy

# class VoiceReceiver(discord.AudioSink):
#    def __init__(self):
#        self.buffers = {}  # user_id -> bytearray
#
#    def write(self, data):
#        if data.user_id not in self.buffers:
#            self.buffers[data.user_id] = bytearray()
#        self.buffers[data.user_id].extend(data.audio)

#    def pop_user_audio(self, user_id):
#        if user_id in self.buffers:
#            audio = self.buffers[user_id]
#            self.buffers[user_id] = bytearray()
#            return audio
#        return None

#voice_receiver = VoiceReceiver()


def save_pcm_to_wav(pcm_bytes, filename="temp.wav"):
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit PCM
        wf.setframerate(48000)
        wf.writeframes(pcm_bytes)
    return filename

#async def transcribe_audio(user_id, pcm_bytes, text_channel):
#    if not pcm_bytes:
#        return
#    wav_file = save_pcm_to_wav(pcm_bytes)
#    segments, info = whisper_model.transcribe(wav_file, beam_size=5)
#    transcript = " ".join([segment.text for segment in segments]).strip()
#    if transcript:
#        user = client.get_user(user_id)
#        await text_channel.send(f"{user.name}: {transcript}")
#    os.remove(wav_file)


@client.event
async def on_message(message):
    global personality_state

    if message.author == client.user:
        return

    user_message = message.content
    entry = {"user": str(message.author), "content": user_message, "timestamp": time.time()}
    short_term_memory.append(entry)
    new_embedding = embed_text(user_message)
    short_term_embeddings.append(new_embedding)
    save_short_term_memory()

    if client.user.mentioned_in(message):
        personality_state = update_personality_state(short_term_memory, personality_state)

        prompt = build_prompt(
            message.author.display_name,
            user_message,
            personality_state,
            short_term_memory
        )

        try:
            await asyncio.sleep(0.5)
            await message.channel.trigger_typing()
            ai_response = await run_bedrock(prompt)

            if not ai_response or not ai_response.strip():
                return

            await message.channel.send(ai_response)

        except Exception as e:
            logging.error(f"Error in on_message: {e}")
            await message.channel.send("Someone tell Shiyo there is a problem with my AI.")


@client.event
async def on_message_delete(message):
    pass


@client.event
async def on_guild_join(guild):
    pass


@client.event
async def on_guild_remove(guild):
    pass


async def autonomous_messaging_loop():
    while True:
        await asyncio.sleep(60)
        # Autonomous messaging code, e.g., post in a channel
        pass


async def shutdown():
    print("Shutting down...")
    save_last_shutdown_time()
    save_short_term_memory()
    save_whitelist()
    await client.close()


def exit_gracefully(signum, frame):
    asyncio.get_event_loop().create_task(shutdown())


signal.signal(signal.SIGINT, exit_gracefully)
signal.signal(signal.SIGTERM, exit_gracefully)


if __name__ == "__main__":
    try:
        client.run(TOKEN)
    except Exception as e:
        print(f"Fatal error: {e}")
