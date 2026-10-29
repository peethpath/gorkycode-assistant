import json
import hashlib
import os
from datetime import datetime, timedelta

import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from supabase import create_client
if not hasattr(st, "cache"):
    st.cache = st.cache_data
from streamlit_cookies_manager import EncryptedCookieManager

load_dotenv()

def get_setting(name):
    """Read configuration from local .env or Streamlit Cloud secrets."""
    value = os.getenv(name)
    if value:
        return value
    try:
        return st.secrets[name]
    except Exception:
        return None

GROQ_API_KEY = get_setting("GROQ_API_KEY")
SUPABASE_URL = get_setting("SUPABASE_URL")
SUPABASE_KEY = get_setting("SUPABASE_KEY")
COOKIES_PASSWORD = get_setting("COOKIES_PASSWORD")
MODEL = "openai/gpt-oss-120b"

# These are initialized for each Streamlit browser session below.
supabase = None
CURRENT_USER_ID = None


def parse_request(user_text):
    if not GROQ_API_KEY:
        raise ValueError("Не найден API-ключ в файле .env")

    client = Groq(api_key=GROQ_API_KEY)
    now = datetime.now()

    prompt = f"""
Ты помощник, который разбирает запросы пользователя
на задачи, напоминания и встречи.

Текущая дата: {now.strftime("%Y-%m-%d")}
Текущее время: {now.strftime("%H:%M")}

Запрос пользователя: {user_text}

Определи:
- title — краткое название дела;
- kind — тип: Задача, Напоминание или Встреча;
- event_date — дата в формате YYYY-MM-DD;
- event_time — время в формате HH:MM.

Правила определения даты:

1. «Сегодня» — текущая дата.
2. «Завтра» — следующий календарный день.
3. «Послезавтра» — дата через два дня.
4. «Через N дней» — прибавь N дней к текущей дате.
5. Если пользователь указывает день недели, например
«в понедельник», «во вторник», «в пятницу»,
определи ближайшую будущую дату этого дня недели.
Если этот день уже прошёл на текущей неделе,
выбери его на следующей неделе.
Если сегодня именно этот день недели, используй сегодняшнюю дату,
если пользователь явно не указал следующую неделю.
6. «На следующей неделе» без конкретного дня —
это недостаточно для определения точной даты, верни null.
7. Не придумывай время, если пользователь его не указал.
8. Если пользователь пишет «надо сделать», «нужно подготовить»,
«сделать», «купить» или другую задачу без просьбы напомнить,
обычно выбирай kind «Задача».
Если пользователь просит напомнить — выбирай «Напоминание».
Если речь о запланированной встрече с человеком —
выбирай «Встреча».

Примеры:
- «В понедельник надо сделать доклад» —
  title: «Сделать доклад», kind: «Задача»,
  event_date: ближайший будущий понедельник, event_time: null.
- «Завтра в 15:00 позвонить тренеру» —
  title: «Позвонить тренеру», kind: «Задача»,
  event_date: завтрашняя дата, event_time: «15:00».
- «Напомни завтра купить продукты» —
  title: «Купить продукты», kind: «Напоминание»,
  event_date: завтрашняя дата, event_time: null.

Все даты рассчитывай относительно указанной выше текущей даты.
Формат даты — YYYY-MM-DD, времени — HH:MM.

Если дата или время не указаны и их нельзя определить,
верни null для соответствующего поля.

Верни только JSON без пояснений и без Markdown.
Пример:
{{"title": "Позвонить тренеру",
  "kind": "Напоминание",
  "event_date": "2026-10-29",
  "event_time": "15:00"}}
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_completion_tokens=1500,
        temperature=0.1,
    )

    content = response.choices[0].message.content

    if not content:
        raise ValueError("ИИ вернул пустой ответ. Попробуй ещё раз.")

    content = content.strip()

    if content.startswith("```"):
        content = content.removeprefix("```json")
        content = content.removeprefix("```")
        content = content.removesuffix("```").strip()

    return json.loads(content)

def detect_intent(user_text):
    if not GROQ_API_KEY:
        raise ValueError("Не найден API-ключ в файле .env")

    client = Groq(api_key=GROQ_API_KEY)
    now = datetime.now()

    prompt = f"""
Ты определяешь намерение пользователя в голосовом
ассистенте для личных задач и встреч.

Текущая дата: {now.strftime("%Y-%m-%d")}
Текущее время: {now.strftime("%H:%M")}

Запрос пользователя: {user_text}

Выбери действие:
- create — создать задачу, напоминание или встречу;
- list — показать записи за определённый период;
- search — найти ранее созданную запись;
- clarify — если намерение невозможно определить.

Правила:
1. «В пятницу нужно отправить отчёт» — create.
2. «Что у меня сегодня?» — list.
3. «Какие встречи на следующей неделе?» — list.
4. «Найди задачу про презентацию» — search.
5. «Когда была встреча с Иваном?» — search.
6. Если спрашивают о записях за период, используй list.
7. Для list определи date_from и date_to включительно.
8. Для сегодня и завтра используй текущую дату как основу.
Если пользователь спрашивает «что у меня в понедельник»,
«какие дела во вторник» или называет другой день недели,
определи дату этого дня: если он ещё впереди на текущей неделе,
используй текущую неделю; если уже прошёл — следующую.
Если сегодня именно этот день недели, используй сегодняшнюю дату.
Для действия list установи date_from и date_to равными найденной дате.
9. Если пользователь спрашивает «что у меня в понедельник»,
«какие дела во вторник» или называет другой день недели,
определи ближайшую будущую дату этого дня недели.
Если этот день на текущей неделе уже прошёл,
используй дату на следующей неделе.
Для такого запроса установи date_from и date_to
равными этой дате.
9. Если период не указан, для list используй сегодняшний день.
10. Для search укажи search_query — ключевые слова
    для поиска названия, человека или темы.
11. kind_filter может быть «Задача», «Напоминание»,
    «Встреча» или «Все».
12. Если тип не указан, используй «Все».

Верни только JSON без Markdown:
{{
  "action": "list",
  "kind_filter": "Все",
  "date_from": "{now.strftime("%Y-%m-%d")}",
  "date_to": "{now.strftime("%Y-%m-%d")}",
  "search_query": null
}}
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_completion_tokens=1500,
        temperature=0.1,
    )

    content = response.choices[0].message.content

    if not content:
        raise ValueError("ИИ вернул пустой ответ.")

    content = content.strip()

    if content.startswith("```"):
        content = content.removeprefix("```json")
        content = content.removeprefix("```")
        content = content.removesuffix("```").strip()

    result = json.loads(content)

    if result.get("action") not in [
        "create", "list", "search", "clarify"
    ]:
        raise ValueError("ИИ вернул неизвестное действие.")

    return result
   
def transcribe_audio(audio_file):
    if not GROQ_API_KEY:
        raise ValueError("Не найден API-ключ в файле .env")

    client = Groq(api_key=GROQ_API_KEY)

    transcription = client.audio.transcriptions.create(
        file=(
            audio_file.name,
            audio_file.getvalue(),
            audio_file.type or "audio/wav",
        ),
        model="whisper-large-v3-turbo",
        language="ru",
        response_format="json",
    )

    return transcription.text.strip()


def add_item(title, kind, date, time):
    """Save a record to the current anonymous user's cloud account."""
    if supabase is None or CURRENT_USER_ID is None:
        raise RuntimeError("Облачная база ещё не подключена.")

    supabase.table("items").insert({
        "title": title,
        "kind": kind,
        "event_date": date.isoformat(),
        "event_time": time.strftime("%H:%M") if time else "",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "completed": False,
        # The database's RLS policy checks this against auth.uid().
        "user_id": CURRENT_USER_ID,
    }).execute()


def get_items():
    """Read only records allowed by the current user's RLS policies."""
    if supabase is None or CURRENT_USER_ID is None:
        return []
    response = (
        supabase.table("items")
        .select("*")
        .order("id", desc=True)
        .execute()
    )
    return response.data or []


st.set_page_config(
    page_title="Голосовой помощник",
     page_icon="🎙️",
    layout="wide",
)


# Persistent, encrypted browser cookie keeps the anonymous Supabase session
# between visits. The original app.py remains untouched.
if not SUPABASE_URL or not SUPABASE_KEY:
    st.error("Не заданы SUPABASE_URL и SUPABASE_KEY в файле .env.")
    st.stop()

if not COOKIES_PASSWORD:
    st.error(
        "Не задан COOKIES_PASSWORD. Добавь отдельный случайный секрет "
        "в .env перед запуском облачной версии."
    )
    st.stop()

cookies = EncryptedCookieManager(
    prefix="gorkycode-assistant/",
    password=COOKIES_PASSWORD,
)

if not cookies.ready():
    st.stop()


def initialize_cloud_session():
    """Restore the browser's anonymous session, or create one on first visit."""
    client = create_client(SUPABASE_URL, SUPABASE_KEY)
    saved_session = cookies.get("supabase_session")
    st.caption(
        f"Диагностика сессии: "
        f"{'найдена' if saved_session else 'не найдена'}"
    )

    if saved_session:
        try:
            saved = json.loads(saved_session)
            auth_response = client.auth.set_session(
                saved["access_token"],
                saved["refresh_token"],
            )
            session = auth_response.session
        except Exception as exc:
            st.error(
                "Не удалось восстановить анонимный профиль из этого браузера. "
                "Мы не будем автоматически создавать новый профиль, чтобы "
                "не скрыть доступ к старой истории."
            )
            st.caption(f"Техническая причина: {exc}")
            if st.button("Создать новый профиль", type="primary"):
                cookies.delete("supabase_session")
                cookies.save()
                st.rerun()
            st.stop()
    else:
        try:
            auth_response = client.auth.sign_in_anonymously()
            session = auth_response.session
        except Exception as exc:
            st.error(f"Не удалось создать анонимный профиль: {exc}")
            st.stop()

    if not session or not session.user:
        st.error("Supabase не вернул действующую пользовательскую сессию.")
        st.stop()

    updated_session = json.dumps({
        "access_token": session.access_token,
        "refresh_token": session.refresh_token,
    })

    if updated_session != saved_session:
        cookies["supabase_session"] = updated_session
        cookies.save()

    return client, str(session.user.id)

if not cookies.ready():
    st.stop()
if (
    "cloud_supabase" not in st.session_state
    or "cloud_user_id" not in st.session_state
):
    try:
        cloud_client, cloud_user_id = initialize_cloud_session()
        st.session_state["cloud_supabase"] = cloud_client
        st.session_state["cloud_user_id"] = cloud_user_id
    except Exception as exc:
        st.error(f"Не удалось подключиться к облачной базе: {exc}")
        st.stop()

supabase = st.session_state["cloud_supabase"]
CURRENT_USER_ID = st.session_state["cloud_user_id"]

st.markdown("""

<style>

/* Цвет кнопок отправки форм */
div[data-testid="stFormSubmitButton"] button {
    background: #52765A !important;
    background-color: #52765A !important;
    color: white !important;
    border: 1px solid #52765A !important;
    border-radius: 10px !important;
}

div[data-testid="stFormSubmitButton"] button:hover {
    background: #3F6047 !important;
    background-color: #3F6047 !important;
    border-color: #3F6047 !important;
    color: white !important;
}


/* Исправляем цвет главной кнопки */
button[data-testid="stBaseButton-primary"],
button[kind="primary"] {
    background: #52765A !important;
    background-color: #52765A !important;
    color: #FFFFFF !important;
    border: 1px solid #52765A !important;
    box-shadow: none !important;
}

button[data-testid="stBaseButton-primary"]:hover,
button[kind="primary"]:hover {
    background: #3F6047 !important;
    background-color: #3F6047 !important;
    color: #FFFFFF !important;
    border-color: #3F6047 !important;
}

/* Главные кнопки */
button[kind="primary"] {
    background-color: #52765A !important;
    border: 1px solid #52765A !important;
    color: #FFFFFF !important;
    border-radius: 10px !important;
    min-height: 44px;
    font-weight: 600;
    transition: background-color 0.2s ease;
}

button[kind="primary"]:hover {
    background-color: #3F6047 !important;
    border-color: #3F6047 !important;
    color: #FFFFFF !important;
}

/* Общий фон и ширина страницы */
.stApp {
    background-color: #F7F8F6;
    color: #252B27;
}

.block-container {
    max-width: 1000px;
    padding-top: 2.5rem;
    padding-bottom: 3rem;
}

/* Заголовки */
h1, h2, h3 {
    color: #252B27;
    letter-spacing: -0.025em;
}

/* Карточки и формы */
div[data-testid="stForm"] {
    background-color: #FFFFFF;
    border: 1px solid #E2E7E1;
    border-radius: 16px;
    padding: 1.2rem;
}

/* Поля ввода */
div[data-testid="stTextInput"] input,
div[data-testid="stDateInput"] input,
div[data-testid="stTimeInput"] input,
div[data-testid="stSelectbox"] div[role="combobox"] {
    border-radius: 10px;
}

/* Кнопки */
.stButton > button {
    border-radius: 10px;
    border: 1px solid #D8DED8;
    background-color: #FFFFFF;
    color: #29372D;
    min-height: 42px;
    transition: all 0.2s ease;
}

.stButton > button:hover {
    border-color: #52765A;
    color: #31583A;
    background-color: #F0F5EF;
}

/* Уведомления */
div[data-testid="stAlert"] {
    border-radius: 12px;
}

/* Разделители */
hr {
    border-color: #E4E8E3;
}

/* Шапка приложения */
.hero {
    background: linear-gradient(135deg, #FFFFFF 0%, #EDF4ED 100%);
    border: 1px solid #E2E7E1;
    border-radius: 20px;
    padding: 28px 32px;
    margin-bottom: 28px;
}

.hero-eyebrow {
    color: #52765A;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.12em;
    margin-bottom: 12px;
}

.hero h1 {
    margin: 0;
    font-size: 2.4rem;
    line-height: 1.2;
}

.hero p {
    margin: 12px 0 0;
    color: #66716A;
    font-size: 1rem;
}

/* Карточки задач */
div[data-testid="stVerticalBlockBorderWrapper"] {
    border: 1px solid #E2E7E1 !important;
    border-radius: 14px !important;
    background-color: #FFFFFF;
}

/* Компактнее расположение элементов */
div[data-testid="stVerticalBlockBorderWrapper"] [data-testid="stVerticalBlock"] {
    gap: 0.6rem;
}
/* Эффект при наведении на карточку */
div[data-testid="stVerticalBlockBorderWrapper"] {
    box-shadow: 0 2px 8px rgba(39, 65, 45, 0.04);
    transition: box-shadow 0.2s ease, border-color 0.2s ease;
}

div[data-testid="stVerticalBlockBorderWrapper"]:hover {
    border-color: #C6D5C7 !important;
    box-shadow: 0 5px 14px rgba(39, 65, 45, 0.08);
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">ВАШ ЛИЧНЫЙ AI-ПЛАНИРОВЩИК</div>
    <h1>🎙️ Голосовой помощник</h1>
    <p>Задачи, напоминания и встречи — в одном месте.</p>
</div>
""", unsafe_allow_html=True)
st.subheader("✨ Создать запись с помощью ИИ")


audio_file = st.audio_input("🎙️ Или запиши запрос голосом")

if "last_audio_hash" not in st.session_state:
    st.session_state["last_audio_hash"] = None

if audio_file is not None:
    audio_hash = hashlib.sha256(
        audio_file.getvalue()
    ).hexdigest()

    if audio_hash != st.session_state["last_audio_hash"]:
        st.session_state["last_audio_hash"] = audio_hash

        try:
            recognized_text = transcribe_audio(audio_file)

            if recognized_text:
                st.session_state["user_request"] = recognized_text
                st.session_state["voice_notice"] = (
                    "Голос распознан автоматически. "
                    "Проверь текст и нажми «Разобрать запрос»."
                )
            else:
                st.warning(
                    "Не удалось распознать речь. "
                    "Попробуй записать ещё раз."
                )

        except Exception as e:
            st.error(f"Не удалось распознать голос: {e}")


if st.session_state.get("voice_notice"):
    st.info(st.session_state["voice_notice"])
    st.session_state["voice_notice"] = None

user_request = st.text_input(
    "Напиши запрос обычными словами",
    placeholder="Например, завтра в 15:00 позвонить тренеру",
    key="user_request"
)

ai_submitted = st.button(
    "Разобрать запрос",
    use_container_width=True,
    type="primary",
)


if "ai_parsed" not in st.session_state:
    st.session_state.ai_parsed = None

if "flash_message" not in st.session_state:
    st.session_state.flash_message = None

if st.session_state.flash_message:
    st.success(st.session_state.flash_message)
    st.session_state.flash_message = None

if "ai_response" not in st.session_state:
    st.session_state.ai_response = None

if "ai_result_items" not in st.session_state:
    st.session_state.ai_result_items = []

if ai_submitted:
    st.session_state.ai_parsed = None
    st.session_state.ai_response = None
    st.session_state.ai_result_items = []

    if not user_request.strip():
        st.warning("Сначала напиши запрос.")
    else:
        try:
            intent = detect_intent(user_request)
            action = intent["action"]

            
            
            if action == "create":
                parsed = parse_request(user_request)
                event_date_text = parsed.get("event_date")
                event_time_text = parsed.get("event_time")

                if event_date_text and event_time_text:
                    event_datetime = datetime.strptime(
                        f"{event_date_text} {event_time_text}",
                        "%Y-%m-%d %H:%M"
                    )

                    if event_datetime < datetime.now():
                        st.session_state.ai_parsed = None
                        st.session_state.ai_response = (
                            "Указанные дата и время уже прошли "
                            f"({event_datetime.strftime('%d.%m.%Y в %H:%M')}). "
                            "Проверь распознанный текст и укажи "
                            "будущую дату и время."
                        )
                    else:
                        st.session_state.ai_parsed = parsed
                else:
                    st.session_state.ai_parsed = parsed


            elif action == "list":
                date_from = intent.get("date_from")
                date_to = intent.get("date_to")

                if not date_from:
                    date_from = datetime.now().date().isoformat()

                if not date_to:
                    date_to = date_from

                start = datetime.strptime(
                    date_from, "%Y-%m-%d"
                ).date()

                end = datetime.strptime(
                    date_to, "%Y-%m-%d"
                ).date()

                if end < start:
                    raise ValueError(
                        "Дата окончания раньше даты начала."
                    )

                kind_filter = intent.get(
                    "kind_filter", "Все"
                )

                if kind_filter not in [
                    "Задача", "Напоминание", "Встреча", "Все"
                ]:
                    kind_filter = "Все"

                found = []

                for item in get_items():
                    event_date = datetime.strptime(
                        item["event_date"], "%Y-%m-%d"
                    ).date()

                    if (
                        start <= event_date <= end
                        and (
                            kind_filter == "Все"
                            or item["kind"] == kind_filter
                        )
                    ):
                        found.append(dict(item))

                st.session_state.ai_result_items = found

                if found:
                    st.session_state.ai_response = (
                        f"Нашёл записей: {len(found)}."
                    )
                else:
                    st.session_state.ai_response = (
                        "На этот период записей нет."
                    )

            
            elif action == "search":
                all_items = [
                    dict(item) for item in get_items()
                ]

                if not all_items:
                    st.session_state.ai_result_items = []
                    st.session_state.ai_response = (
                        "Пока нет сохранённых записей для поиска."
                    )
                else:
                    records = [
                        {
                            "id": item["id"],
                            "title": item["title"],
                            "kind": item["kind"],
                            "date": item["event_date"],
                            "time": item["event_time"],
                        }
                        for item in all_items
                    ]

                    client = Groq(api_key=GROQ_API_KEY)

                    search_prompt = f"""
Ты помощник для поиска задач и встреч.

Запрос пользователя: {user_request}

Список записей из базы:
{json.dumps(records, ensure_ascii=False)}

Найди записи, подходящие по смыслу запроса.
Формулировки могут отличаться.

Не придумывай записи и идентификаторы.
Выбирай только существующие id.
Если совпадений нет, верни пустой список.

Верни только JSON:
{{"ids": [1, 3]}}
"""

                    response = client.chat.completions.create(
                        model=MODEL,
                        messages=[
                            {
                                "role": "user",
                                "content": search_prompt
                            }
                        ],
                        max_completion_tokens=500,
                        temperature=0.1,
                    )

                    content = response.choices[0].message.content

                    if not content:
                        raise ValueError(
                            "ИИ не вернул результат поиска."
                        )

                    content = content.strip()

                    if content.startswith("```"):
                        content = content.removeprefix("```json")
                        content = content.removeprefix("```")
                        content = content.removesuffix("```").strip()

                    result = json.loads(content)

                    valid_ids = {
                        item["id"] for item in all_items
                    }

                    matched_ids = {
                        record_id
                        for record_id in result.get("ids", [])
                        if isinstance(record_id, int)
                        and record_id in valid_ids
                    }

                    found = [
                        item for item in all_items
                        if item["id"] in matched_ids
                    ]

                    st.session_state.ai_result_items = found

                    if found:
                        st.session_state.ai_response = (
                            f"Нашёл подходящих записей: {len(found)}."
                        )
                    else:
                        st.session_state.ai_response = (
                            "Не нашёл подходящих записей. "
                            "Попробуй описать задачу другими словами."
                        )


            else:
                st.session_state.ai_response = (
                    "Уточни, пожалуйста: создать запись, "
                    "посмотреть планы или найти старую задачу?"
                )

        except Exception as e:
            st.error(f"Не удалось обработать запрос: {e}")
if st.session_state.ai_response:
    st.markdown("### Ответ помощника")
    st.write(st.session_state.ai_response)

    for item in st.session_state.ai_result_items:
        with st.container(border=True):
            st.markdown(f"**{item['title']}**")

            event_date = datetime.strptime(
                item["event_date"], "%Y-%m-%d"
            ).date()

            st.caption(
                f"{item['kind']} · "
                f"{event_date.strftime('%d.%m.%Y')} · "
                f"{item['event_time']}"
            )           

if st.session_state.ai_parsed:
    parsed = st.session_state.ai_parsed

    if (
        not parsed.get("title")
        or parsed.get("kind") not in ["Задача", "Напоминание", "Встреча"]
        or not parsed.get("event_date")
        
    ):
        st.warning("Не хватает данных для записи.")
        st.info(
            "Укажи название дела, дату и время "
            "в запросе и попробуй ещё раз."
        )
    else:
        st.markdown("### Проверь запись")
        st.write(f"**Название:** {parsed['title']}")
        st.write(f"**Тип:** {parsed['kind']}")

        try:
            event_date = datetime.strptime(
                parsed["event_date"], "%Y-%m-%d"
            ).date()

            event_time = (
                datetime.strptime(parsed["event_time"], "%H:%M").time()
                 if parsed.get("event_time")
                 else None
            )
              
            st.write(
                f"**Дата:** {event_date.strftime('%d.%m.%Y')}"
            )
            st.write(
                f"**Время:** {event_time.strftime('%H:%M') if event_time else 'Не указано'}"
            )

            col_save, col_cancel = st.columns(2)

            with col_save:
                if st.button(
                    "Сохранить",
                    key="save_ai",
                    use_container_width=True
                ):
                    add_item(
                        parsed["title"],
                        parsed["kind"],
                        event_date,
                        event_time
                    )

                    st.session_state.ai_parsed = None
                    st.session_state.flash_message = (
                        "Запись успешно сохранена!"
                    )
                    st.rerun()

            with col_cancel:
                if st.button(
                    "Отменить",
                    key="cancel_ai",
                    use_container_width=True
                ):
                    st.session_state.ai_parsed = None
                    st.rerun()

        except ValueError:
            st.error(
                "ИИ вернул дату или время в неправильном формате. "
                "Попробуй сформулировать запрос точнее."
            )

st.subheader("Добавить запись")

with st.form("new_item"):
    title = st.text_input(
        "Что нужно сделать?",
        placeholder="Например, подготовить презентацию",
    )

    kind = st.selectbox(
        "Тип записи",
        ["Задача", "Напоминание", "Встреча"],
    )

    date = st.date_input("Дата", value=datetime.now().date())
    time = st.time_input("Время", value=datetime.now().time())

    submitted = st.form_submit_button(
        "Добавить",
        use_container_width=True,
    )

    if submitted:
        if not title.strip():
            st.warning("Сначала введи название записи.")
        elif datetime.combine(date, time) < datetime.now():
            st.warning(
                "Указанные дата и время уже прошли. "
                "Выбери будущее время."
            )
        else:
            add_item(title.strip(), kind, date, time)
            st.success("Запись сохранена в базе данных!")


st.divider()
st.subheader("📋 Мои записи")

today = datetime.now().date()

col1, col2 = st.columns(2)


with col1:
    start_date = st.date_input(
        "С даты",
        value=today
    )

with col2:
    end_date = st.date_input(
        "По дату",
        value=today + timedelta(days=30)
    )


type_filter = st.selectbox(
    "Тип записи",
    ["Все", "Задача", "Напоминание", "Встреча"]
)

search_text = st.text_input(
    "Поиск по названию",
    placeholder="Например, презентация"
)


items = get_items()

upcoming_items = []
overdue_items = []
history_items = []

now = datetime.now().replace(second=0, microsecond=0)

for item in items:
    item_time = item["event_time"] or "23:59"
    event_datetime = datetime.strptime(
        f"{item['event_date']} {item_time}",
        "%Y-%m-%d %H:%M"
        )

    event_date = event_datetime.date()

    if not (start_date <= event_date <= end_date):
        continue

    if (
        type_filter != "Все"
        and item["kind"] != type_filter
    ):
        continue

    if search_text.lower() not in item["title"].lower():
        continue

    if item["completed"]:
        history_items.append(item)
    elif event_datetime >= now:
        upcoming_items.append(item)
    elif item["kind"] in ["Задача", "Напоминание"]:
        overdue_items.append(item)
    else:
        history_items.append(item)

upcoming_items.sort(
    key=lambda item: (item["event_date"], item["event_time"])
)

overdue_items.sort(
    key=lambda item: (item["event_date"], item["event_time"])
)

history_items.sort(
    key=lambda item: (item["event_date"], item["event_time"]),
    reverse=True
)



def show_item(item):
    st.markdown(f"**{item['title']}**")

    event_date = datetime.strptime(
        item["event_date"], "%Y-%m-%d"
    ).date()

    st.caption(
        f"{item['kind']} · "
        f"{event_date.strftime('%d.%m.%Y')} · "
        f"{item['event_time']}"
    )

    



def complete_item(item_id):
    if supabase is None or CURRENT_USER_ID is None:
        raise RuntimeError("Облачная база ещё не подключена.")
    (
        supabase.table("items")
        .update({"completed": True})
        .eq("id", item_id)
        .eq("user_id", CURRENT_USER_ID)
        .execute()
    )


def uncomplete_item(item_id):
    if supabase is None or CURRENT_USER_ID is None:
        raise RuntimeError("Облачная база ещё не подключена.")
    (
        supabase.table("items")
        .update({"completed": False})
        .eq("id", item_id)
        .eq("user_id", CURRENT_USER_ID)
        .execute()
    )


def delete_item(item_id):
    if supabase is None or CURRENT_USER_ID is None:
        raise RuntimeError("Облачная база ещё не подключена.")
    (
        supabase.table("items")
        .delete()
        .eq("id", item_id)
        .eq("user_id", CURRENT_USER_ID)
        .execute()
    )

st.subheader("📅 Ближайшие дела")

if "delete_confirm_id" not in st.session_state:
    st.session_state.delete_confirm_id = None

if st.session_state.delete_confirm_id is not None:
    delete_id = st.session_state.delete_confirm_id

    items_for_delete = [
        dict(item) for item in get_items()
    ]

    item_to_delete = next(
        (
            item for item in items_for_delete
            if item["id"] == delete_id
        ),
        None
    )

    if item_to_delete:
        st.warning(
            f"Удалить запись «{item_to_delete['title']}»? "
            "Это действие нельзя отменить."
        )

        col_yes, col_no = st.columns(2)

        with col_yes:
            if st.button(
                "Да, удалить",
                key="confirm_delete"
            ):
                delete_item(delete_id)
                st.session_state.delete_confirm_id = None
                st.rerun()

        with col_no:
            if st.button(
                "Отмена",
                key="cancel_delete"
            ):
                st.session_state.delete_confirm_id = None
                st.rerun()
    else:
        st.session_state.delete_confirm_id = None

if not upcoming_items:
    st.info("Предстоящих дел нет.")

for item in upcoming_items:
    with st.container(border=True):
        show_item(item)

        
        col_delete, col_complete = st.columns(2)

        with col_delete:
            if st.button(
                "🗑️ Удалить",
                key=f"delete_{item['id']}"
            ):
                st.session_state.delete_confirm_id = item["id"]
                st.rerun()

        with col_complete:
            if st.button(
                "✓ Выполнено",
                key=f"complete_upcoming_{item['id']}"
            ):
                complete_item(item["id"])
                st.rerun()



if overdue_items:
    st.subheader("⏰ Просроченные дела")

    for item in overdue_items:
        with st.container(border=True):
            show_item(item)

            if st.button(
                "✓ Выполнено",
                key=f"complete_overdue_{item['id']}"
            ):
                complete_item(item["id"])
                st.rerun()


with st.expander(
    f"🗂️ История — {len(history_items)} записей"
):
    if not history_items:
        st.info("История пока пуста.")
    else:
        for item in history_items:
            with st.container(border=True):
                show_item(item)

                if item["completed"]:
                    st.caption("Статус: выполнено")
                    if st.button(
                        "↩ Вернуть в активные",
                        key=f"undo_complete_{item['id']}"
                    ):
                        uncomplete_item(item["id"])
                        st.rerun()