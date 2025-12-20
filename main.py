import json
import os
import time
from typing import TypedDict

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from langgraph.graph import StateGraph
from google.generativeai import configure, GenerativeModel

# --- Настройки ---

configure(api_key=GEMINI_API_KEY)
model = GenerativeModel("gemini-2.5-flash")

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}

# --- State ---
class AgentState(TypedDict):
    inn: str
    egrul_data: dict
    raw_financial_data: dict
    downloaded_xml_paths: list
    plaintiff_cases: list
    respondent_cases: list
    markdown_report: str

# --- Функция 1: Получение данных ЕГРЮЛ ---
def fetch_egrul_data(state: AgentState) -> AgentState:
    inn = state["inn"]
    url = EGRUL_API_URL_TEMPLATE.format(inn=inn)
    
    print(f"📋 Загружаю данные ЕГРЮЛ для ИНН {inn}...")
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()
        print("✅ Данные ЕГРЮЛ загружены")
    except Exception as e:
        data = {"error": f"Failed to fetch EGRUL: {str(e)}"}
        print(f"❌ Ошибка ЕГРЮЛ: {str(e)}")

    return {"egrul_data": data}

# --- Функция 2: Получение финансовых данных ---
def fetch_financial_data(state: AgentState) -> AgentState:
    inn = state["inn"]
    url = FIN_API_URL_TEMPLATE.format(inn=inn)
    
    print(f"💰 Загружаю финансовые данные для ИНН {inn}...")

    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        print("✅ Финансовые данные загружены")
    except Exception as e:
        data = {"error": f"Failed to fetch financial: {str(e)}"}
        print(f"❌ Ошибка финансов: {str(e)}")

    return {"raw_financial_data": data}

# --- Функция 3: Скачивание бухгалтерской отчетности ---
def download_xml_reports(state: AgentState) -> AgentState:
    inn = state["inn"]
    out_dir = f"reports_{inn}"
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"📊 Загружаю бухгалтерскую отчетность для ИНН {inn}...")
    print("⏳ Ожидание 5 секунд перед запросом...")
    time.sleep(5)
    
    for attempt in range(3):
        try:
            resp = requests.get(
                f"{BASE_BOH_URL}ls.php",
                params={"inn": inn},
                headers=HEADERS,
                timeout=15
            )
            resp.raise_for_status()
            break
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                wait = 20 * (attempt + 1)
                print(f"⚠️ Ошибка 429. Ждем {wait} секунд...")
                time.sleep(wait)
                if attempt == 2:
                    return {"downloaded_xml_paths": [{"error": "429 после 3 попыток"}]}
            else:
                return {"downloaded_xml_paths": [{"error": f"HTTP Error: {str(e)}"}]}
        except Exception as e:
            return {"downloaded_xml_paths": [{"error": f"Request failed: {str(e)}"}]}

    soup = BeautifulSoup(resp.text, "html.parser")
    links = soup.find_all('a', string='Скачать')
    
    if not links:
        print("⚠️ Бухгалтерские файлы не найдены")
        return {"downloaded_xml_paths": []}
    
    print(f"📁 Найдено файлов: {len(links)}")
    downloaded = []

    for idx, link in enumerate(links, 1):
        href = link.get("href")
        if not href:
            continue

        if idx > 1:
            print(f"⏳ Ждем 10 секунд...")
            time.sleep(10)
        
        url = urljoin(BASE_BOH_URL, href)
        filename = f"report_{inn}_{idx}.pdf"
        path = os.path.join(out_dir, filename)
        
        print(f"📥 [{idx}/{len(links)}]: {filename}")
        
        for file_attempt in range(3):
            try:
                r = requests.get(url, headers=HEADERS, timeout=15)
                r.raise_for_status()
                with open(path, "wb") as f:
                    f.write(r.content)
                downloaded.append(path)
                print(f"✅ OK")
                break
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:
                    wait = 20 * (file_attempt + 1)
                    print(f"⚠️ 429. Ждем {wait} сек...")
                    time.sleep(wait)
                else:
                    break

    print(f"✅ Скачано {len(downloaded)} файлов")
    return {"downloaded_xml_paths": downloaded}

# --- Функция 4: Получение арбитражных дел ---
def get_arbitration_cases_by_inn(inn: str, api_key: str) -> dict:
    url = ARBITRATION_API_URL
    params = {
        'key': api_key,
        'Inn': inn,
        'InnType': 'Any'
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        if data.get('Success') != 1:
            error_msg = data.get('error', 'API returned Success != 1')
            print(f"⚠️ {error_msg}")
        return data

    except requests.exceptions.HTTPError as e:
        return {"error": f"HTTP Error: {e}", "details": response.text if response else None}
    except requests.exceptions.RequestException as e:
        return {"error": f"Request Error: {str(e)}"}
    except ValueError:
        return {"error": "Response is not valid JSON", "content": response.text}

def parse_arbitration_cases(arb_data: dict, company_inn: str):
    cases = arb_data.get("Cases", [])
    plaintiff_cases = []
    respondent_cases = []

    for case in cases:
        is_plaintiff = False
        is_respondent = False

        for plaintiff in case.get("Plaintiffs") or []:
            if plaintiff and plaintiff.get("Inn") == company_inn:
                is_plaintiff = True
                break

        for respondent in case.get("Respondents") or []:
            if respondent and respondent.get("Inn") == company_inn:
                is_respondent = True
                break

        if is_plaintiff:
            plaintiff_cases.append(case)
        if is_respondent:
            respondent_cases.append(case)

    return plaintiff_cases, respondent_cases

def fetch_arbitration_data(state: AgentState) -> AgentState:
    inn = state["inn"]
    
    print(f"⚖️ Загружаю судебные дела для ИНН {inn}...")
    
    raw_arb_data = get_arbitration_cases_by_inn(inn, ARBITRATION_API_KEY)
    plt_cases, resp_cases = parse_arbitration_cases(raw_arb_data, inn)
    
    print(f"✅ Найдено дел: истец={len(plt_cases)}, ответчик={len(resp_cases)}")

    return {
        "plaintiff_cases": plt_cases,
        "respondent_cases": resp_cases
    }

# --- Функция 5: Чтение файлов ---
def read_xml_files_as_text(paths):
    texts = []
    for path in paths:
        if isinstance(path, dict) and "error" in path:
            texts.append(path["error"])
            continue
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            texts.append(content)
        except:
            texts.append(f"Не удалось прочитать {os.path.basename(path)}")
    return "\n\n".join(texts)

# --- Функция 6: Финальный анализ и создание отчета ---
def merge_and_analyze(state: AgentState) -> AgentState:
    egrul_data = state["egrul_data"]
    fin_data = state["raw_financial_data"]
    xml_texts = read_xml_files_as_text(state["downloaded_xml_paths"])
    plt_cases = state["plaintiff_cases"]
    resp_cases = state["respondent_cases"]

    # Ограничение размера
    if len(xml_texts) > 50000:
        xml_texts = xml_texts[:50000] + "\n\n[ОБРЕЗАНО]"

    prompt = f"""
Ты — эксперт по комплексному корпоративному анализу компаний.

Создай подробный структурированный markdown-отчет на основе всех доступных данных.

# СТРУКТУРА ОТЧЕТА:

## 1. 🏢 КОРПОРАТИВНАЯ ИНФОРМАЦИЯ (ЕГРЮЛ)

### Основные сведения:
- ИНН, ОГРН
- Полное и краткое наименование
- Организационно-правовая форма
- Дата регистрации
- Статус компании
- Уставный капитал

### Юридический адрес:
Полный адрес

### 👥 Руководство:
Таблица с ФИО, должностями, датами назначения

### 💼 Структура владения:

**Учредители:**
Таблица с именами/наименованиями, долями, ИНН (для юрлиц)

**Mermaid диаграмма структуры:**
\`\`\`mermaid
graph TD
    A[Компания<br/>ИНН: XXX]
    B[Учредитель 1<br/>Доля: XX%]
    C[Учредитель 2<br/>Доля: XX%]
    
    B --> A
    C --> A
\`\`\`

**Анализ структуры:**
- Кто контролирует компанию (конечные бенефициары)
- Есть ли связанные юридические лица
- Прозрачность структуры владения

### 🏭 Виды деятельности (ОКВЭД):
Основной и дополнительные

### ⚠️ Корпоративные риски:
- Массовый адрес
- Номинальный директор
- Недавние изменения
- Другие красные флаги

---

## 2. 💰 ФИНАНСОВЫЙ АНАЛИЗ

### Ключевые показатели:
Из финансовых данных - выручка, прибыль, активы, обязательства

### Динамика:
Изменения по годам/периодам

### Финансовое состояние:
Платежеспособность, ликвидность, рентабельность

---

## 3. 📊 БУХГАЛТЕРСКАЯ ОТЧЕТНОСТЬ

### Анализ отчетов:
- Какие формы представлены
- Периоды отчетности
- Основные статьи баланса
- Финансовый результат

### Выявленные особенности:
Аномалии, пустые значения, изменения

---

## 4. ⚖️ СУДЕБНАЯ АКТИВНОСТЬ

### Статистика:
- Всего дел: XX
- Компания как истец: XX дел
- Компания как ответчик: XX дел

### Дела в качестве истца:
Топ-5 наиболее значимых дел (номер, суть, сумма, статус)

### Дела в качестве ответчика:
Топ-5 наиболее значимых дел

### Судебные риски:
- Типичные споры
- Крупные претензии
- Проигранные дела
- Общая оценка судебной нагрузки

---

## 5. 🎯 КОМПЛЕКСНАЯ ОЦЕНКА И ВЫВОДЫ

### Сильные стороны:
### Слабые стороны:
### Ключевые риски:
### Инвестиционная привлекательность:

**Итоговая оценка:** (по шкале 1-10)

---

# ИСХОДНЫЕ ДАННЫЕ:

## Данные ЕГРЮЛ:
{json.dumps(egrul_data, ensure_ascii=False, indent=2)}

## Финансовые данные:
{json.dumps(fin_data, ensure_ascii=False, indent=2)}

## Бухгалтерская отчетность:
{xml_texts}

## Судебные дела (истец):
{json.dumps(plt_cases, ensure_ascii=False, indent=2)}

## Судебные дела (ответчик):
{json.dumps(resp_cases, ensure_ascii=False, indent=2)}

---

ТРЕБОВАНИЯ:
1. Все таблицы в markdown
2. Обязательно построй Mermaid диаграмму структуры владения
3. Будь конкретен, используй цифры и факты
4. Выдели риски жирным шрифтом
5. Дай четкие выводы и рекомендации
"""

    print("\n🤖 Генерирую финальный отчет через Gemini...")
    
    try:
        time.sleep(3)
        response = model.generate_content(prompt)
        report = response.text or "Отчет не был сгенерирован."
        print("✅ Отчет готов")
    except Exception as e:
        report = f"Ошибка генерации отчета: {str(e)}"
        print(f"❌ Ошибка: {str(e)}")

    return {"markdown_report": report}

# --- Построение графа ---
workflow = StateGraph(AgentState)

workflow.add_node("fetch_egrul", fetch_egrul_data)
workflow.add_node("fetch_financial", fetch_financial_data)
workflow.add_node("download_xml", download_xml_reports)
workflow.add_node("fetch_arbitration", fetch_arbitration_data)
workflow.add_node("merge_and_analyze", merge_and_analyze)

workflow.set_entry_point("fetch_egrul")
workflow.add_edge("fetch_egrul", "fetch_financial")
workflow.add_edge("fetch_financial", "download_xml")
workflow.add_edge("download_xml", "fetch_arbitration")
workflow.add_edge("fetch_arbitration", "merge_and_analyze")
workflow.add_edge("merge_and_analyze", "__end__")

app = workflow.compile()

# --- Запуск ---
if __name__ == "__main__":
    print("="*80)
    print("📊 КОМПЛЕКСНЫЙ АНАЛИЗ КОМПАНИИ")
    print("="*80)
    
    user_input_inn = input("\n🔢 Введите ИНН компании: ").strip()
    
    initial_state = {
        "inn": user_input_inn,
        "egrul_data": {},
        "raw_financial_data": {},
        "downloaded_xml_paths": [],
        "plaintiff_cases": [],
        "respondent_cases": [],
        "markdown_report": ""
    }

    print("\n" + "="*80)
    print("🚀 ЗАПУСК АНАЛИЗА")
    print("="*80 + "\n")

    result = app.invoke(initial_state)

    print("\n" + "="*80)
    print("📋 ИТОГОВЫЙ ОТЧЕТ")
    print("="*80 + "\n")
    print(result["markdown_report"])

    # Сохранение отчета
    out_dir = f"reports_{user_input_inn}"
    os.makedirs(out_dir, exist_ok=True)
    report_file = os.path.join(out_dir, "final_corporate_report.md")
    
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(result["markdown_report"])
    
    print("\n" + "="*80)
    print(f"💾 Отчет сохранен: {report_file}")
    print("="*80)
