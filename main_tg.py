import asyncio
import torch
from aiogram import Bot, Dispatcher, types
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.client.telegram import TelegramAPIServer
from aiogram.filters import Command
from transformers import AutoTokenizer, AutoModelForCausalLM
from ddgs import DDGS
import requests
from bs4 import BeautifulSoup
import re

# ============================================================
#                   НАСТРОЙКИ
# ============================================================

BOT_TOKEN = "8927830103:AAHO9tua-qvKfOTEdAp7wAYPcbBAbmdifkA"
PROXY_URL = "https://kirill-ai.limon-123321.workers.dev"

# ============================================================
#                   ИНФОРМАЦИЯ О БОТЕ
# ============================================================
# ✏️ ЗДЕСЬ ВЫ МОЖЕТЕ НАПИСАТЬ ЛЮБОЙ ТЕКСТ

INFO_TEXT = """
Крутая нейросеть версия 0.3

0.2 добавлено - поиск информации в интернете(/search)
0.3 бот переведен с tkinter на телеграм бота, был улучшен поиск, исправление ошибок

Создан @cs_go_kirill (Кирилл) и @SlideLowMurk
"""


# ============================================================
#                   МОДЕЛЬ
# ============================================================

print("Загрузка модели Qwen2.5-0.5B...")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B-Instruct",
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print("Модель загружена.")

# Быстрые ответы (включая английские варианты)
SELF_ANSWERS = {
    "кто ты": "Я — Kirill AI, ваш виртуальный ассистент.",
    "ты кто": "Я — Kirill AI, ваш виртуальный ассистент.",
    "кто тебя создал": "Меня создал Кирилл.",
    "как тебя зовут": "Меня зовут Kirill AI.",
    "твоё имя": "Моё имя — Kirill AI.",
    "представься": "Я — Kirill AI, ваш виртуальный ассистент.",
    "что ты умеешь": "Я умею отвечать на вопросы и искать информацию в интернете.",
    "расскажи о себе": "Я — Kirill AI, ассистент",
    "who you": "Я — Kirill AI, ваш виртуальный ассистент.",
    "who are you": "Я — Kirill AI, ваш виртуальный ассистент.",
    "what are you": "Я — Kirill AI, ваш виртуальный ассистент.",
    "your name": "Меня зовут Kirill AI.",
}


def get_fast_answer(text):
    low = text.lower().strip()
    for key, answer in SELF_ANSWERS.items():
        if key in low:
            return answer
    return None


# ============================================================
#                   ПОИСК В ИНТЕРНЕТЕ (быстрый)
# ============================================================

def search_and_analyze_fast(query):
    """Быстрый поиск: только русская Википедия, один источник"""
    try:
        print(f"Поиск: {query}")

        with DDGS() as ddgs:
            results = list(ddgs.text(f"{query} site:ru.wikipedia.org", max_results=5))

            if not results:
                results = list(ddgs.text(query, max_results=5))

            if not results:
                return None

            for result in results:
                url = result.get('href', '')
                if 'wikipedia.org' not in url.lower():
                    continue

                try:
                    headers = {'User-Agent': 'Mozilla/5.0'}
                    resp = requests.get(url, headers=headers, timeout=10)
                    soup = BeautifulSoup(resp.text, 'html.parser')

                    for el in soup(['script', 'style', 'nav', 'footer', 'header']):
                        el.decompose()

                    content = soup.find('div', {'id': 'mw-content-text'})
                    if content:
                        paragraphs = content.find_all('p')[:8]
                        texts = []
                        for p in paragraphs:
                            t = p.get_text(strip=True)
                            if len(t) > 40:
                                texts.append(t)
                        text = ' '.join(texts)
                        text = re.sub(r'\s+', ' ', text)
                        text = re.sub(r'\[\d+\]', '', text)
                        print(f"✓ {result.get('title', '')[:50]}")
                        return [text.strip()[:2000]]
                except Exception as e:
                    print(f"Ошибка загрузки {url}: {e}")
                    continue

        return None
    except Exception as e:
        print(f"Ошибка поиска: {e}")
        return None


# ============================================================
#                   ГЕНЕРАЦИЯ ОТВЕТА
# ============================================================

SYSTEM_PROMPT = (
    "Ты — Kirill AI, русскоязычный ассистент, созданный Кириллом. "
    "Ты НЕ Claude, НЕ Anthropic, НЕ OpenAI, НЕ Qwen. "
    "Ты отвечаешь ТОЛЬКО на русском языке. "
    "Никогда не представляйся как Claude, Anthropic или любая другая компания. "
    "Если тебя спрашивают 'кто ты' — отвечай: 'Я — Kirill AI, ваш виртуальный ассистент.' "
    "Если не знаешь ответа — скажи 'Я не знаю'. Кратко, по делу."
)


def generate_answer(user_input, search_mode=False):
    # Быстрые ответы
    fast = get_fast_answer(user_input)
    if fast:
        return fast

    # Поиск в интернете
    if search_mode:
        analyzed = search_and_analyze_fast(user_input)
        if not analyzed:
            return "Не удалось найти информацию в интернете."

        context = analyzed[0][:1500]

        messages = [
            {
                "role": "system",
                "content": (
                    "Ты — Kirill AI. Отвечай ТОЛЬКО на русском языке. "
                    "Перескажи текст из Википедии кратко, 2-4 предложениями. Без ссылок."
                )
            },
            {
                "role": "user",
                "content": f"Вопрос: {user_input}\n\nТекст на русском:\n{context}\n\nОтветь на русском:"
            }
        ]
    else:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Ответь на русском языке: {user_input}"}
        ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=250,
            temperature=0.6,
            top_p=0.85,
            do_sample=True,
            repetition_penalty=1.15,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    answer = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()

    for bad in ["Anthropic", "Claude", "OpenAI", "Qwen", "Alibaba"]:
        answer = answer.replace(bad, "Kirill AI")

    return answer if answer else "Я не знаю."


# ============================================================
#                   БОТ
# ============================================================

api_server = TelegramAPIServer.from_base(PROXY_URL)
session = AiohttpSession(api=api_server)
bot = Bot(token=BOT_TOKEN, session=session)
dp = Dispatcher()

search_modes = {}


@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    await message.answer(
        "Kirill AI 0.3 готов к работе.\n\n"
        "Команды:\n"
        "/info — информация о боте\n"
        "/search — включить/выключить поиск в интернете"
    )


# ============================================================
#                   КОМАНДА /info
# ============================================================

@dp.message(Command("info"))
async def cmd_info(message: types.Message):
    await message.answer(INFO_TEXT.strip())


@dp.message(Command("search"))
async def cmd_search(message: types.Message):
    uid = message.from_user.id
    search_modes[uid] = not search_modes.get(uid, False)
    status = "включён" if search_modes[uid] else "выключен"
    await message.answer(f"🔍 Поиск в интернете {status}.")


@dp.message()
async def handle_message(message: types.Message):
    uid = message.from_user.id
    search_mode = search_modes.get(uid, False)

    await message.answer("⏳ Думаю...")
    answer = generate_answer(message.text, search_mode=search_mode)
    await message.answer(answer)


async def main():
    print("Бот запущен.")
    await dp.start_polling(bot)
