import streamlit as st
import json
import os
import time
import gzip
import re
import io
from typing import TypedDict, Dict, List, Any
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from langgraph.graph import StateGraph
from google.generativeai import configure, GenerativeModel
import xml.etree.ElementTree as ET
from dotenv import load_dotenv

# Загрузка переменных окружения из .env файла
load_dotenv()

# PDF библиотеки
try:
    import pdfplumber
    PDF_LIB = "pdfplumber"
except ImportError:
    try:
        import fitz
        PDF_LIB = "pymupdf"
    except ImportError:
        PDF_LIB = None

# OCR библиотеки (опционально)
try:
    from pdf2image import convert_from_path
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

# Exa API для поиска в СМИ
try:
    from exa_py import Exa
    EXA_AVAILABLE = True
except ImportError:
    EXA_AVAILABLE = False
    st.warning("⚠️ exa_py не установлен. Поиск в СМИ недоступен. Установите: pip install exa_py")

# PIL для работы с изображениями
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# --- Конфигурация ---
st.set_page_config(page_title="Комплексный анализ компании", page_icon="📊", layout="wide")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
EGRUL_API_URL_TEMPLATE = "https://egrul.itsoft.ru/{inn}.json.gz"
FIN_API_URL_TEMPLATE = "https://egrul.itsoft.ru/fin/?{inn}"
ARBITRATION_API_URL = "https://parser-api.com/parser/arbitr_api/search"
BASE_BOH_URL = "https://egrul.itsoft.ru/bo/"
ARBITRATION_API_KEY = os.getenv("ARBITRATION_API_KEY")
EXA_API_KEY = os.getenv("EXA_API_KEY")

configure(api_key=GEMINI_API_KEY)
model = GenerativeModel("gemini-2.5-flash")

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Accept': 'application/json, text/html, */*',
    'Accept-Language': 'ru-RU,ru;q=0.9,en;q=0.8',
    'Accept-Encoding': 'gzip, deflate'
}

class AgentState(TypedDict):
    inn: str
    egrul_parsed: dict
    fin_parsed: dict
    boh_parsed: list
    courts_parsed: dict
    related_companies: list
    media_mentions: list
    markdown_report: str

def normalize_inn(inn: str) -> str:
    inn = inn.strip()
    if len(inn) < 10:
        return inn.zfill(10)
    elif len(inn) < 12 and len(inn) > 10:
        return inn.zfill(12)
    return inn

def safe_get(data, *keys):
    """Безопасное извлечение вложенных значений"""
    for key in keys:
        if isinstance(data, dict):
            data = data.get(key)
        else:
            return None
        if data is None:
            return None
    return data

# ==================== УЛУЧШЕННЫЙ ПАРСИНГ PDF ====================

def _clean_number(num_str: str):
    """Преобразует русские/европейские форматы чисел в float/int, учитывает отрицательные в скобках."""
    if not num_str:
        return None
    s = str(num_str)
    # Удаляем вспомогательные знаки, оставляем цифры, пробелы, запятые, точки, минусы, скобки
    s = s.replace('\xa0', ' ').replace('\u2009', ' ')
    s = s.strip()
    # если в скобках — отрицательное
    neg = False
    if re.match(r'^\(.*\)$', s):
        neg = True
        s = s.strip('()')
    # убрать все символы кроме цифр, запятой, точки, минуса и пробела
    s = re.sub(r'[^0-9\-,\. ]+', '', s)
    # заменим пробелы в тысячных на пусто
    s = s.replace(' ', '')
    # заменим запятую на точку если есть
    if ',' in s and '.' not in s:
        s = s.replace(',', '.')
    try:
        if s == '':
            return None
        if '.' in s:
            val = float(s)
        else:
            val = int(s)
        return -val if neg else val
    except:
        return None

def parse_accounting_from_text(text: str) -> dict:
    """
    Попытка извлечь структурированные поля бухотчетности из плоского текста.
    Ищем блоки 'БАЛАНС' (Актив/Пассив) и 'Финансовые результаты' (Выручка, Чистая прибыль и т.д.)
    Возвращаем структуру похожую на parse_accounting_xml -> structured_data
    """
    res = {
        "report_type": None,
        "report_year": None,
        "company_info": {},
        "balance": {},
        "financial_results": {},
        "structured_data": {}
    }

    if not text or not text.strip():
        return res

    txt = text

    # 1) Попытка найти год: "за 2020 г.", "за 2020 год", "за 2020"
    year_match = re.search(r'за\s+(\d{4})\s*(год|г\.)?', txt, flags=re.IGNORECASE)
    if not year_match:
        # альтернативно - поиск 4-значного года рядом с 'баланс' или 'отчет'
        year_match = re.search(r'(баланс|отчет).{0,30}?(\d{4})', txt, flags=re.IGNORECASE)
    if year_match:
        res["report_year"] = year_match.group(1) if len(year_match.groups()) == 1 else year_match.group(2)

    # 2) Попытка найти ИНН
    inn_match = re.search(r'ИНН[:\s]*([0-9]{10,12})', txt, flags=re.IGNORECASE)
    if inn_match:
        res["company_info"]["inn"] = inn_match.group(1)

    # 3) Попытка найти наименование компании (строка перед ИНН или заголовок документа)
    if inn_match:
        start = max(0, inn_match.start() - 200)
        snippet = txt[start:inn_match.start()]
        # возьмём последнюю немногосимвольную строку
        lines = [l.strip() for l in snippet.splitlines() if l.strip()]
        if lines:
            res["company_info"]["name"] = lines[-1][:200]

    # 4) Блоки Баланс / Актив / Пассив
    # Находим начало блока 'БАЛАНС' и 'ПАССИВ'/'АКТИВ'
    bal_match = re.search(r'(баланс|БАЛАНС)', txt, flags=re.IGNORECASE)
    if bal_match:
        res["report_type"] = "Бухгалтерский баланс"
        # возьмём окно текста вокруг блока (случайно 1000 символов вперед)
        start = bal_match.start()
        window = txt[start:start+5000]
        # Популярные позиции — попробуем найти строки с этими ключами и парсить числа справа
        items = {
            "Внеоборотные активы": r'Внеоборотн\w*\s+актив\w*\D*([0-9\-\(\)\s\.,]+)',
            "Запасы": r'Запас\w*\D*([0-9\-\(\)\s\.,]+)',
            "Денежные средства": r'Денежн\w*\s*ср\w*\D*([0-9\-\(\)\s\.,]+)',
            "Финансовые вложения": r'Финан\w*\s*влож\w*\D*([0-9\-\(\)\s\.,]+)',
            "Нематериальные активы": r'Нематер\w*\s*актив\w*\D*([0-9\-\(\)\s\.,]+)',
            "Капитал и резервы": r'Капитал\w*.*резерв\w*\D*([0-9\-\(\)\s\.,]+)',
            "Долгосрочные заемные средства": r'Долгосрочн\w*\s*заем\w*\s*средств\D*([0-9\-\(\)\s\.,]+)',
            "Кредиторская задолженность": r'Кредитор\w*\s*задолж\w*\D*([0-9\-\(\)\s\.,]+)'
        }
        active_details = {}
        passive_details = {}
        # Сначала ищем явные пары "label ... number" в окне
        for label, pattern in items.items():
            m = re.search(pattern, window, flags=re.IGNORECASE)
            if m:
                val = _clean_number(m.group(1))
                # решаем куда класть - простая эвристика по слову "актив"/"пассив" в лейбле
                if re.search(r'актив', label, flags=re.IGNORECASE) or label in ["Внеоборотные активы","Денежные средства","Финансовые вложения","Нематериальные активы","Запасы"]:
                    active_details[label] = val
                else:
                    passive_details[label] = val

        # Также пытаемся найти "ИТОГО АКТИВ" и "ИТОГО ПАССИВ"
        total_active = re.search(r'(итог\w*.*актив\w*|всего.*актив\w*).{0,40}?([0-9\-\(\)\s\.,]+)', window, flags=re.IGNORECASE)
        total_passive = re.search(r'(итог\w*.*пассив\w*|всего.*пассив\w*).{0,40}?([0-9\-\(\)\s\.,]+)', window, flags=re.IGNORECASE)
        if total_active:
            res["balance"].setdefault("active", {})["total_current"] = _clean_number(total_active.group(2))
        if total_passive:
            res["balance"].setdefault("passive", {})["total_current"] = _clean_number(total_passive.group(2))

        if active_details:
            res["balance"].setdefault("active", {})["details"] = active_details
        if passive_details:
            res["balance"].setdefault("passive", {})["details"] = passive_details

    # 5) Финрез — Выручка / Чистая прибыль / Налог на прибыль
    fin_patterns = {
        "revenue": r'Выруч\w*\D*([0-9\-\(\)\s\.,]+)',
        "net_profit": r'Чист\w*\s*приб\w*\D*([0-9\-\(\)\s\.,]+)',
        "income_tax": r'Нал(ог)?\s*на\s*прибыль\D*([0-9\-\(\)\s\.,]+)',
        "expenses": r'Расх\w*\D*([0-9\-\(\)\s\.,]+)'
    }
    for key, patt in fin_patterns.items():
        m = re.search(patt, txt, flags=re.IGNORECASE)
        if m:
            # иногда группа 1 - нужная, иногда 2 (после скобок). Берём последнюю ненулевую
            grp = None
            if m.groups():
                # берём последнюю группу содержащую цифры
                for g in reversed(m.groups()):
                    if g and re.search(r'\d', str(g)):
                        grp = g
                        break
            if not grp:
                grp = m.group(1)
            res["financial_results"][key] = {"current": _clean_number(grp)}

    # 6) Если ничего не нашлось — отметим отсутствие
    res["structured_data"] = {
        "balance": res.get("balance", {}),
        "financial_results": res.get("financial_results", {})
    }

    return res

def extract_from_pdf(pdf_path: str) -> dict:
    """
    Многоступенчатая функция извлечения: pdfplumber -> pymupdf -> таблицы -> OCR.
    Возвращает dict со 'text', 'success', и, по возможности, 'structured_data', 'report_type', 'report_year', 'company_info'
    """
    result = {
        "filename": os.path.basename(pdf_path),
        "success": False,
        "text": "",
        "structured_data": {},
        "report_type": None,
        "report_year": None,
        "company_info": {}
    }

    # 0) Быстрая проверка заголовка
    try:
        with open(pdf_path, 'rb') as f:
            header = f.read(10)
            if not header.startswith(b'%PDF'):
                # попробуем как XML/текст
                try:
                    with open(pdf_path, 'r', encoding='utf-8') as xf:
                        content = xf.read()
                        if '<?xml' in content or content.strip().startswith('<'):
                            return parse_accounting_xml(pdf_path)
                except:
                    pass
                result["error"] = f"Не PDF файл. Заголовок: {header[:20]}"
                return result
    except Exception as e:
        result["error"] = f"Ошибка чтения файла: {e}"
        return result
    
    # 0.5) Диагностика PDF (количество страниц, защита)
    pdf_info = {}
    try:
        if PDF_LIB == "pdfplumber":
            import pdfplumber
            with pdfplumber.open(pdf_path) as pdf:
                pdf_info["pages"] = len(pdf.pages)
                pdf_info["encrypted"] = pdf.metadata.get('Encrypted', False) if pdf.metadata else False
        elif PDF_LIB == "pymupdf":
            import fitz
            doc = fitz.open(pdf_path)
            pdf_info["pages"] = len(doc)
            pdf_info["encrypted"] = doc.is_encrypted
            doc.close()
        
        result["pdf_info"] = pdf_info
        
        if pdf_info.get("encrypted"):
            result["error"] = "PDF защищен паролем или зашифрован"
            return result
            
        if pdf_info.get("pages", 0) == 0:
            result["error"] = "PDF не содержит страниц"
            return result
            
    except Exception as e:
        result["pdf_diagnostic_error"] = str(e)

    extracted_text = []
    try:
        # 1) pdfplumber
        if PDF_LIB == "pdfplumber":
            try:
                import pdfplumber
                with pdfplumber.open(pdf_path) as pdf:
                    for page in pdf.pages:
                        # текст
                        page_text = page.extract_text(x_tolerance=2, y_tolerance=2) or ""
                        # таблицы: попробуем извлечь таблицы как fallback
                        try:
                            tables = page.extract_tables()
                            if tables:
                                # переведём таблицы в строки
                                for t in tables:
                                    for row in t:
                                        row_text = " | ".join([c if c else "" for c in row])
                                        page_text += "\n" + row_text
                        except Exception:
                            pass
                        extracted_text.append(page_text)
            except Exception as e:
                # pdfplumber отсутствует или упал — перейдём к pymupdf
                extracted_text = []

        # 2) pymupdf (fitz)
        if (not extracted_text or all(not p.strip() for p in extracted_text)) and PDF_LIB == "pymupdf":
            try:
                import fitz
                doc = fitz.open(pdf_path)
                for i in range(len(doc)):
                    page = doc.load_page(i)
                    # get_text("blocks") или "text" — пробуем "blocks" чтобы получить более сырые блоки
                    try:
                        text = page.get_text("text") or ""
                        if not text.strip():
                            # попробуем blocks
                            blocks = page.get_text("blocks") or []
                            # blocks -> sort by y -> join
                            blocks_sorted = sorted(blocks, key=lambda b: b[1])
                            page_text = "\n".join([str(b[4]) for b in blocks_sorted if b and b[4]])
                        else:
                            page_text = text
                    except Exception:
                        page_text = page.get_text() or ""
                    extracted_text.append(page_text)
                doc.close()
            except Exception:
                pass

        # 3) Если нет текста — попробуем OCR через pytesseract/pdf2image
        joined = "\n\n=== НОВАЯ СТРАНИЦА ===\n\n".join([p for p in extracted_text if p and p.strip()])
        
        if not joined.strip():
            result["extraction_method"] = "none"
            result["pages_processed"] = len(extracted_text)
            
            if OCR_AVAILABLE:
                # OCR fallback
                try:
                    from pdf2image import convert_from_path
                    import pytesseract
                    pages = convert_from_path(pdf_path, dpi=200)
                    ocr_texts = []
                    for page_im in pages:
                        txt = pytesseract.image_to_string(page_im, lang='rus+eng')
                        ocr_texts.append(txt)
                    joined = "\n\n=== НОВАЯ СТРАНИЦА ===\n\n".join(ocr_texts)
                    result["ocr_used"] = True
                    result["extraction_method"] = "ocr"
                except Exception as e:
                    # OCR упал
                    result["error"] = f"OCR доступен, но произошла ошибка: {str(e)}"
                    return result
            else:
                # OCR не установлен
                result["error"] = ("Пустой текст. Вероятно PDF — скан без текстового слоя. "
                                   "Установите OCR: pip install pdf2image pytesseract")
                return result
        else:
            # Определяем какой метод сработал
            if PDF_LIB == "pdfplumber":
                result["extraction_method"] = "pdfplumber"
            elif PDF_LIB == "pymupdf":
                result["extraction_method"] = "pymupdf"
            else:
                result["extraction_method"] = "unknown"

        # Сохраняем текст
        result["text"] = joined
        result["success"] = True

        # 4) Специализированный текст-парсер бухгалтерии
        try:
            parsed = parse_accounting_from_text(joined)
            if parsed:
                # Мержим возвращаемые поля
                result["structured_data"] = parsed.get("structured_data", {})
                result["report_type"] = parsed.get("report_type") or result.get("report_type")
                result["report_year"] = parsed.get("report_year") or result.get("report_year")
                result["company_info"] = parsed.get("company_info") or result.get("company_info")
                
                # Формируем multi_year_data если есть структурированные данные
                if result["structured_data"].get("balance") or result["structured_data"].get("financial_results"):
                    result["multi_year_data"] = {
                        "years": [result.get("report_year", "N/A")],
                        "balance": {},
                        "financial_results": {}
                    }
                    
                    # Переносим данные баланса
                    if result["structured_data"].get("balance"):
                        balance = result["structured_data"]["balance"]
                        if balance.get("active"):
                            active = balance["active"]
                            if active.get("total_current"):
                                result["multi_year_data"]["balance"]["АКТИВ (всего)"] = [active.get("total_current")]
                            if active.get("details"):
                                for key, val in active["details"].items():
                                    result["multi_year_data"]["balance"][key] = [val]
                        
                        if balance.get("passive"):
                            passive = balance["passive"]
                            if passive.get("total_current"):
                                result["multi_year_data"]["balance"]["ПАССИВ (всего)"] = [passive.get("total_current")]
                            if passive.get("details"):
                                for key, val in passive["details"].items():
                                    result["multi_year_data"]["balance"][key] = [val]
                    
                    # Переносим финрезультаты
                    if result["structured_data"].get("financial_results"):
                        fin_res = result["structured_data"]["financial_results"]
                        if fin_res.get("revenue"):
                            result["multi_year_data"]["financial_results"]["Выручка"] = [fin_res["revenue"].get("current")]
                        if fin_res.get("net_profit"):
                            result["multi_year_data"]["financial_results"]["Чистая прибыль (убыток)"] = [fin_res["net_profit"].get("current")]
                        if fin_res.get("expenses"):
                            result["multi_year_data"]["financial_results"]["Расходы"] = [fin_res["expenses"].get("current")]
                        if fin_res.get("income_tax"):
                            result["multi_year_data"]["financial_results"]["Налог на прибыль"] = [fin_res["income_tax"].get("current")]
        
        except Exception as e:
            # не критично — оставим просто текст
            result["parsing_error"] = str(e)

    except Exception as e:
        result["error"] = str(e)
        result["success"] = False

    return result

# ==================== ПАРСИНГ XML БУХОТЧЕТНОСТИ ====================

def parse_accounting_xml(xml_path: str) -> dict:
    """
    Специализированный парсер для XML бухгалтерской отчетности
    Поддерживает формы 0710001 (Баланс) и 0710002 (Финрезультаты)
    """
    result = {
        "filename": os.path.basename(xml_path),
        "success": False,
        "report_type": None,
        "report_year": None,
        "company_info": {},
        "balance": {},
        "financial_results": {},
        "text": "",
        "structured_data": {}
    }
    
    try:
        # Читаем с правильной кодировкой
        with open(xml_path, 'r', encoding='windows-1251') as f:
            content = f.read()
        
        tree = ET.ElementTree(ET.fromstring(content))
        root = tree.getroot()
        
        # Извлекаем основные атрибуты документа
        doc = root.find('.//Документ')
        if doc is not None:
            result["report_year"] = doc.get('ОтчетГод')
            result["report_date"] = doc.get('ДатаДок')
            okud = doc.get('КНД')
            
            # Определяем тип отчета
            if okud == "0710096":  # Упрощенная форма
                result["report_type"] = "Упрощенная форма"
            
            # Информация о компании
            svnp = doc.find('.//СвНП')
            if svnp is not None:
                # Данные о компании в НПЮЛ (налогоплательщик юр. лицо)
                npul = svnp.find('.//НПЮЛ')
                if npul is not None:
                    company_attrs = npul.attrib
                    result["company_info"]["inn"] = company_attrs.get('ИННЮЛ')
                    result["company_info"]["kpp"] = company_attrs.get('КПП')
                    result["company_info"]["name"] = company_attrs.get('НаимОрг')
                    result["company_info"]["address"] = company_attrs.get('АдрМН')
                
                # Классификаторы в СвНП
                svnp_attrs = svnp.attrib
                result["company_info"]["okved"] = svnp_attrs.get('ОКВЭД2')
                result["company_info"]["okopf"] = svnp_attrs.get('ОКОПФ')
                result["company_info"]["okpo"] = svnp_attrs.get('ОКПО')
                result["company_info"]["okfs"] = svnp_attrs.get('ОКФС')
            
            # Подписант
            podp = doc.find('.//Подписант')
            if podp is not None:
                fio = podp.find('.//ФИО')
                if fio is not None:
                    result["company_info"]["signatory"] = {
                        "surname": fio.get('Фамилия'),
                        "name": fio.get('Имя'),
                        "patronymic": fio.get('Отчество')
                    }
            
            # БАЛАНС (форма 0710001)
            balance = doc.find('.//Баланс')
            if balance is not None:
                result["report_type"] = "Бухгалтерский баланс"
                
                # АКТИВ
                active = balance.find('.//Актив')
                if active is not None:
                    result["balance"]["active"] = {
                        "total_current": active.get('СумОтч'),
                        "total_prev_year": active.get('СумПрдШв'),
                        "total_prev_2years": active.get('СумПрдЩ'),
                        "details": {}
                    }
                    
                    # Внеоборотные активы
                    mat_vne = active.find('.//МатВнеАкт')
                    if mat_vne is not None:
                        result["balance"]["active"]["details"]["non_current_assets"] = {
                            "current": mat_vne.get('СумОтч'),
                            "prev_year": mat_vne.get('СумПрдШв'),
                            "prev_2years": mat_vne.get('СумПрдЩ')
                        }
                    
                    # Финансовые вложения
                    fin_vlozh = active.find('.//ФинВлож')
                    if fin_vlozh is not None:
                        result["balance"]["active"]["details"]["financial_investments"] = {
                            "current": fin_vlozh.get('СумОтч'),
                            "prev_year": fin_vlozh.get('СумПрдШв'),
                            "prev_2years": fin_vlozh.get('СумПрдЩ')
                        }
                    
                    # Запасы
                    zapasy = active.find('.//Запасы')
                    if zapasy is not None:
                        result["balance"]["active"]["details"]["inventory"] = {
                            "current": zapasy.get('СумОтч'),
                            "prev_year": zapasy.get('СумПрдШв'),
                            "prev_2years": zapasy.get('СумПрдЩ')
                        }
                    
                    # Денежные средства
                    denezhn = active.find('.//ДенежнСр')
                    if denezhn is not None:
                        result["balance"]["active"]["details"]["cash"] = {
                            "current": denezhn.get('СумОтч'),
                            "prev_year": denezhn.get('СумПрдШв'),
                            "prev_2years": denezhn.get('СумПрдЩ')
                        }
                    
                    # Нематериальные финансовые активы
                    ne_mat = active.find('.//НеМатФинАкт')
                    if ne_mat is not None:
                        result["balance"]["active"]["details"]["intangible_assets"] = {
                            "current": ne_mat.get('СумОтч'),
                            "prev_year": ne_mat.get('СумПрдШв'),
                            "prev_2years": ne_mat.get('СумПрдЩ')
                        }
                
                # ПАССИВ
                passive = balance.find('.//Пассив')
                if passive is not None:
                    result["balance"]["passive"] = {
                        "total_current": passive.get('СумОтч'),
                        "total_prev_year": passive.get('СумПрдШв'),
                        "total_prev_2years": passive.get('СумПрдЩ'),
                        "details": {}
                    }
                    
                    # Капитал и резервы
                    kap_rez = passive.find('.//КапРез')
                    if kap_rez is not None:
                        result["balance"]["passive"]["details"]["equity"] = {
                            "current": kap_rez.get('СумОтч'),
                            "prev_year": kap_rez.get('СумПрдШв'),
                            "prev_2years": kap_rez.get('СумПрдЩ')
                        }
                    
                    # Долгосрочные заемные средства
                    dlg_zaem = passive.find('.//ДлгЗаемСредств')
                    if dlg_zaem is not None:
                        result["balance"]["passive"]["details"]["long_term_debt"] = {
                            "current": dlg_zaem.get('СумОтч'),
                            "prev_year": dlg_zaem.get('СумПрдШв'),
                            "prev_2years": dlg_zaem.get('СумПрдЩ')
                        }
                    
                    # Другие долгосрочные обязательства
                    dr_dolg = passive.find('.//ДрДолгосрОбяз')
                    if dr_dolg is not None:
                        result["balance"]["passive"]["details"]["other_long_term_liab"] = {
                            "current": dr_dolg.get('СумОтч'),
                            "prev_year": dr_dolg.get('СумПрдШв'),
                            "prev_2years": dr_dolg.get('СумПрдЩ')
                        }
                    
                    # Кредиторская задолженность
                    kredit = passive.find('.//КредитЗадолж')
                    if kredit is not None:
                        result["balance"]["passive"]["details"]["accounts_payable"] = {
                            "current": kredit.get('СумОтч'),
                            "prev_year": kredit.get('СумПрдШв'),
                            "prev_2years": kredit.get('СумПрдЩ')
                        }
            
            # ОТЧЕТ О ФИНАНСОВЫХ РЕЗУЛЬТАТАХ (форма 0710002)
            fin_rez = doc.find('.//ФинРез')
            if fin_rez is not None:
                if not result["report_type"]:
                    result["report_type"] = "Отчет о финансовых результатах"
                else:
                    result["report_type"] += " + Отчет о финансовых результатах"
                
                # Выручка
                viruch = fin_rez.find('.//Выруч')
                if viruch is not None:
                    result["financial_results"]["revenue"] = {
                        "current": viruch.get('СумОтч'),
                        "previous": viruch.get('СумПред')
                    }
                
                # Расходы по обычной деятельности
                rashod = fin_rez.find('.//РасхОбДеят')
                if rashod is not None:
                    result["financial_results"]["expenses"] = {
                        "current": rashod.get('СумОтч'),
                        "previous": rashod.get('СумПред')
                    }
                
                # Прочие доходы
                proch_doh = fin_rez.find('.//ПрочДоход')
                if proch_doh is not None:
                    result["financial_results"]["other_income"] = {
                        "current": proch_doh.get('СумОтч'),
                        "previous": proch_doh.get('СумПред')
                    }
                
                # Прочие расходы
                proch_rash = fin_rez.find('.//ПрочРасход')
                if proch_rash is not None:
                    result["financial_results"]["other_expenses"] = {
                        "current": proch_rash.get('СумОтч'),
                        "previous": proch_rash.get('СумПред')
                    }
                
                # Налог на прибыль
                nal_prib = fin_rez.find('.//НалПрибДох')
                if nal_prib is not None:
                    result["financial_results"]["income_tax"] = {
                        "current": nal_prib.get('СумОтч'),
                        "previous": nal_prib.get('СумПред')
                    }
                
                # Чистая прибыль/убыток
                chist_prib = fin_rez.find('.//ЧистПрибУб')
                if chist_prib is not None:
                    result["financial_results"]["net_profit"] = {
                        "current": chist_prib.get('СумОтч'),
                        "previous": chist_prib.get('СумПред')
                    }
        
        # Формируем текстовое представление
        text_parts = []
        text_parts.append(f"=== {result['report_type']} за {result['report_year']} год ===\n")
        text_parts.append(f"Компания: {result['company_info'].get('name', 'N/A')}")
        text_parts.append(f"ИНН: {result['company_info'].get('inn', 'N/A')}")
        text_parts.append(f"КПП: {result['company_info'].get('kpp', 'N/A')}\n")
        
        if result["balance"]:
            text_parts.append("\n--- БАЛАНС ---")
            if "active" in result["balance"]:
                text_parts.append(f"АКТИВ (всего): {result['balance']['active'].get('total_current', 'N/A')} тыс. руб.")
            if "passive" in result["balance"]:
                text_parts.append(f"ПАССИВ (всего): {result['balance']['passive'].get('total_current', 'N/A')} тыс. руб.")
        
        if result["financial_results"]:
            text_parts.append("\n--- ФИНАНСОВЫЕ РЕЗУЛЬТАТЫ ---")
            if "revenue" in result["financial_results"]:
                text_parts.append(f"Выручка: {result['financial_results']['revenue'].get('current', 'N/A')} тыс. руб.")
            if "net_profit" in result["financial_results"]:
                text_parts.append(f"Чистая прибыль: {result['financial_results']['net_profit'].get('current', 'N/A')} тыс. руб.")
        
        result["text"] = "\n".join(text_parts)
        result["structured_data"] = {
            "balance": result["balance"],
            "financial_results": result["financial_results"]
        }
        
        # Формируем данные для СВОДНОЙ ТАБЛИЦЫ по всем годам
        multi_year = {"years": [], "balance": {}, "financial_results": {}}
        
        # Определяем годы
        if result["report_year"]:
            try:
                year_current = int(result["report_year"])
                year_prev = year_current - 1
                year_prev2 = year_current - 2
                multi_year["years"] = [str(year_current), str(year_prev), str(year_prev2)]
            except:
                multi_year["years"] = [result["report_year"], "N/A", "N/A"]
        
        # БАЛАНС - собираем все показатели
        if result["balance"]:
            if "active" in result["balance"]:
                active = result["balance"]["active"]
                
                # Всего активов
                multi_year["balance"]["АКТИВ (всего)"] = [
                    active.get("total_current"),
                    active.get("total_prev_year"),
                    active.get("total_prev_2years")
                ]
                
                # Детализация активов
                if "details" in active:
                    details = active["details"]
                    
                    if "non_current_assets" in details:
                        nca = details["non_current_assets"]
                        multi_year["balance"]["Внеоборотные активы"] = [
                            nca.get("current"),
                            nca.get("prev_year"),
                            nca.get("prev_2years")
                        ]
                    
                    if "financial_investments" in details:
                        fi = details["financial_investments"]
                        multi_year["balance"]["Финансовые вложения"] = [
                            fi.get("current"),
                            fi.get("prev_year"),
                            fi.get("prev_2years")
                        ]
                    
                    if "inventory" in details:
                        inv = details["inventory"]
                        multi_year["balance"]["Запасы"] = [
                            inv.get("current"),
                            inv.get("prev_year"),
                            inv.get("prev_2years")
                        ]
                    
                    if "cash" in details:
                        cash = details["cash"]
                        multi_year["balance"]["Денежные средства"] = [
                            cash.get("current"),
                            cash.get("prev_year"),
                            cash.get("prev_2years")
                        ]
                    
                    if "intangible_assets" in details:
                        ia = details["intangible_assets"]
                        multi_year["balance"]["Нематериальные активы"] = [
                            ia.get("current"),
                            ia.get("prev_year"),
                            ia.get("prev_2years")
                        ]
            
            if "passive" in result["balance"]:
                passive = result["balance"]["passive"]
                
                # Всего пассивов
                multi_year["balance"]["ПАССИВ (всего)"] = [
                    passive.get("total_current"),
                    passive.get("total_prev_year"),
                    passive.get("total_prev_2years")
                ]
                
                # Детализация пассивов
                if "details" in passive:
                    details = passive["details"]
                    
                    if "equity" in details:
                        eq = details["equity"]
                        multi_year["balance"]["Капитал и резервы"] = [
                            eq.get("current"),
                            eq.get("prev_year"),
                            eq.get("prev_2years")
                        ]
                    
                    if "long_term_debt" in details:
                        ltd = details["long_term_debt"]
                        multi_year["balance"]["Долгосрочные заемные средства"] = [
                            ltd.get("current"),
                            ltd.get("prev_year"),
                            ltd.get("prev_2years")
                        ]
                    
                    if "other_long_term_liab" in details:
                        oltl = details["other_long_term_liab"]
                        multi_year["balance"]["Другие долгосрочные обязательства"] = [
                            oltl.get("current"),
                            oltl.get("prev_year"),
                            oltl.get("prev_2years")
                        ]
                    
                    if "accounts_payable" in details:
                        ap = details["accounts_payable"]
                        multi_year["balance"]["Кредиторская задолженность"] = [
                            ap.get("current"),
                            ap.get("prev_year"),
                            ap.get("prev_2years")
                        ]
        
        # ФИНАНСОВЫЕ РЕЗУЛЬТАТЫ (только 2 года: текущий и предыдущий)
        if result["financial_results"]:
            fin_years = multi_year["years"][:2] if len(multi_year["years"]) >= 2 else multi_year["years"]
            
            if "revenue" in result["financial_results"]:
                rev = result["financial_results"]["revenue"]
                multi_year["financial_results"]["Выручка"] = [
                    rev.get("current"),
                    rev.get("previous")
                ]
            
            if "expenses" in result["financial_results"]:
                exp = result["financial_results"]["expenses"]
                multi_year["financial_results"]["Расходы по обычной деятельности"] = [
                    exp.get("current"),
                    exp.get("previous")
                ]
            
            if "other_income" in result["financial_results"]:
                oi = result["financial_results"]["other_income"]
                multi_year["financial_results"]["Прочие доходы"] = [
                    oi.get("current"),
                    oi.get("previous")
                ]
            
            if "other_expenses" in result["financial_results"]:
                oe = result["financial_results"]["other_expenses"]
                multi_year["financial_results"]["Прочие расходы"] = [
                    oe.get("current"),
                    oe.get("previous")
                ]
            
            if "income_tax" in result["financial_results"]:
                it = result["financial_results"]["income_tax"]
                multi_year["financial_results"]["Налог на прибыль"] = [
                    it.get("current"),
                    it.get("previous")
                ]
            
            if "net_profit" in result["financial_results"]:
                np_val = result["financial_results"]["net_profit"]
                multi_year["financial_results"]["Чистая прибыль (убыток)"] = [
                    np_val.get("current"),
                    np_val.get("previous")
                ]
        
        result["multi_year_data"] = multi_year
        result["success"] = True
        
    except Exception as e:
        result["error"] = str(e)
        result["text"] = f"Ошибка парсинга XML: {str(e)}"
    
    return result

def extract_from_xml(xml_path: str) -> dict:
    """Извлекает ВСЕ данные из XML - сначала пробует специализированный парсер"""
    
    # Пробуем специализированный парсер для бухотчетности
    accounting_result = parse_accounting_xml(xml_path)
    if accounting_result.get("success"):
        return accounting_result
    
    # Fallback: общий парсер
    result = {
        "filename": os.path.basename(xml_path),
        "success": False,
        "data": {},
        "text": ""
    }
    
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        elements = []
        for elem in root.iter():
            if elem.text and elem.text.strip():
                tag = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag
                elements.append(f"{tag}: {elem.text.strip()}")
        
        result["text"] = "\n".join(elements)
        result["data"]["elements_count"] = len(elements)
        result["data"]["root_tag"] = root.tag
        result["success"] = True
        
    except Exception as e:
        result["error"] = str(e)
    
    return result

def extract_from_pdf(pdf_path: str) -> dict:
    """Извлекает текст из PDF"""
    result = {
        "filename": os.path.basename(pdf_path),
        "success": False,
        "text": ""
    }
    
    try:
        # Проверка что это PDF
        with open(pdf_path, 'rb') as f:
            header = f.read(10)
            if not header.startswith(b'%PDF'):
                # Пробуем как XML
                try:
                    with open(pdf_path, 'r', encoding='utf-8') as xf:
                        content = xf.read()
                        if '<?xml' in content or content.strip().startswith('<'):
                            return extract_from_xml(pdf_path)
                except:
                    pass
                result["error"] = f"Не PDF. Заголовок: {header}"
                return result
        
        # Извлекаем текст
        if PDF_LIB == "pdfplumber":
            import pdfplumber
            with pdfplumber.open(pdf_path) as pdf:
                texts = []
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        texts.append(text)
                result["text"] = "\n\n=== НОВАЯ СТРАНИЦА ===\n\n".join(texts)
                result["success"] = True
                
        elif PDF_LIB == "pymupdf":
            import fitz
            doc = fitz.open(pdf_path)
            texts = []
            for i in range(len(doc)):
                texts.append(doc[i].get_text())
            doc.close()
            result["text"] = "\n\n=== НОВАЯ СТРАНИЦА ===\n\n".join(texts)
            result["success"] = True
        else:
            result["error"] = "Нет библиотеки для PDF"
        
    except Exception as e:
        result["error"] = str(e)
    
    return result

# ==================== ПАРСИНГ ЕГРЮЛ ====================
def parse_egrul_data(egrul_json: dict) -> dict:
    """ПОЛНЫЙ парсинг ЕГРЮЛ JSON - РЕАЛЬНАЯ СТРУКТУРА API"""
    parsed = {
        "raw_json_structure": list(egrul_json.keys()),
        "basic_info": {},
        "address": {},
        "directors": [],
        "founders": [],
        "okved": {},
        "capital": None,
        "all_fields": {},
        "related_companies": []
    }
    
    try:
        # Основной контейнер - СвЮЛ (Сведения о юридическом лице)
        svul = egrul_json.get("СвЮЛ", {})
        if not svul:
            # Fallback на старую структуру
            svul = egrul_json
        
        # Основные атрибуты (ИНН, ОГРН и т.д.)
        attrs = svul.get("@attributes", {})
        parsed["basic_info"]["inn"] = attrs.get("ИНН")
        parsed["basic_info"]["ogrn"] = attrs.get("ОГРН")
        parsed["basic_info"]["kpp"] = attrs.get("КПП")
        parsed["basic_info"]["reg_date"] = attrs.get("ДатаОГРН")
        parsed["basic_info"]["data_extract_date"] = attrs.get("ДатаВып")
        
        # ОПФ (организационно-правовая форма)
        parsed["basic_info"]["opf_code"] = attrs.get("КодОПФ")
        parsed["basic_info"]["opf_full"] = attrs.get("ПолнНаимОПФ")
        
        # Наименование
        sv_naim = svul.get("СвНаимЮЛ", {})
        naim_attrs = sv_naim.get("@attributes", {})
        parsed["basic_info"]["full_name"] = naim_attrs.get("НаимЮЛПолн")
        
        # Короткое наименование
        sv_naim_sokr = sv_naim.get("СвНаимЮЛСокр", {})
        if isinstance(sv_naim_sokr, dict):
            sokr_attrs = sv_naim_sokr.get("@attributes", {})
            parsed["basic_info"]["short_name"] = sokr_attrs.get("НаимСокр")
        
        parsed["all_fields"] = parsed["basic_info"].copy()
        
        # Адрес
        sv_addr = svul.get("СвАдресЮЛ", {})
        addr_rf = sv_addr.get("АдресРФ", {})
        if addr_rf:
            addr_attrs = addr_rf.get("@attributes", {})
            region_info = addr_rf.get("Регион", {}).get("@attributes", {})
            street_info = addr_rf.get("Улица", {}).get("@attributes", {})
            
            # Собираем полный адрес
            address_parts = []
            if addr_attrs.get("Индекс"):
                address_parts.append(addr_attrs["Индекс"])
            if region_info.get("НаимРегион"):
                address_parts.append(region_info["НаимРегион"])
            if street_info.get("ТипУлица") and street_info.get("НаимУлица"):
                address_parts.append(f"{street_info['ТипУлица']} {street_info['НаимУлица']}")
            if addr_attrs.get("Дом"):
                address_parts.append(f"д. {addr_attrs['Дом']}")
            if addr_attrs.get("Корпус"):
                address_parts.append(f"корп. {addr_attrs['Корпус']}")
            if addr_attrs.get("Кварт"):
                address_parts.append(f"кв. {addr_attrs['Кварт']}")
            
            parsed["address"] = {
                "index": addr_attrs.get("Индекс"),
                "region": region_info.get("НаимРегион"),
                "street_type": street_info.get("ТипУлица"),
                "street": street_info.get("НаимУлица"),
                "house": addr_attrs.get("Дом"),
                "corpus": addr_attrs.get("Корпус"),
                "apartment": addr_attrs.get("Кварт"),
                "full": ", ".join(address_parts) if address_parts else None
            }
        
        # Руководители (СведДолжнФЛ)
        directors_data = svul.get("СведДолжнФЛ", [])
        if not isinstance(directors_data, list):
            directors_data = [directors_data] if directors_data else []
        
        for director in directors_data:
            sv_fl = director.get("СвФЛ", {})
            fl_attrs = sv_fl.get("@attributes", {})
            
            sv_dolzhn = director.get("СвДолжн", {})
            dolzhn_attrs = sv_dolzhn.get("@attributes", {})
            
            full_name = f"{fl_attrs.get('Фамилия', '')} {fl_attrs.get('Имя', '')} {fl_attrs.get('Отчество', '')}".strip()
            
            parsed["directors"].append({
                "surname": fl_attrs.get("Фамилия"),
                "name": fl_attrs.get("Имя"),
                "patronymic": fl_attrs.get("Отчество"),
                "full_name": full_name,
                "position": dolzhn_attrs.get("НаимДолжн"),
                "inn": fl_attrs.get("ИННФЛ")
            })
        
        # Учредители
        sv_uchredit = svul.get("СвУчредит", {})
        
        # Физлица-учредители
        uchr_fl = sv_uchredit.get("УчрФЛ", [])
        if not isinstance(uchr_fl, list):
            uchr_fl = [uchr_fl] if uchr_fl else []
        
        for founder in uchr_fl:
            sv_fl = founder.get("СвФЛ", {})
            fl_attrs = sv_fl.get("@attributes", {})
            
            dolya = founder.get("ДоляУстКап", {})
            razmer_doli = dolya.get("РазмерДоли", {})
            
            full_name = f"{fl_attrs.get('Фамилия', '')} {fl_attrs.get('Имя', '')} {fl_attrs.get('Отчество', '')}".strip()
            
            parsed["founders"].append({
                "type": "ФЛ",
                "surname": fl_attrs.get("Фамилия"),
                "name": fl_attrs.get("Имя"),
                "patronymic": fl_attrs.get("Отчество"),
                "full_name": full_name,
                "inn": fl_attrs.get("ИННФЛ"),
                "share": razmer_doli.get("Процент"),
                "nominal_value": dolya.get("@attributes", {}).get("НоминСтоим")
            })
        
        # Юрлица-учредители (для связанных компаний)
        uchr_ul = sv_uchredit.get("УчрЮЛ", [])
        if not isinstance(uchr_ul, list):
            uchr_ul = [uchr_ul] if uchr_ul else []
        
        for founder in uchr_ul:
            sv_ul = founder.get("СвЮЛ", {})
            ul_attrs = sv_ul.get("@attributes", {})
            
            dolya = founder.get("ДоляУстКап", {})
            razmer_doli = dolya.get("РазмерДоли", {})
            
            founder_inn = ul_attrs.get("ИННЮЛ")
            founder_name = ul_attrs.get("НаимЮЛ")
            share = razmer_doli.get("Процент")
            
            parsed["founders"].append({
                "type": "ЮЛ",
                "name": founder_name,
                "inn": founder_inn,
                "ogrn": ul_attrs.get("ОГРН"),
                "share": share,
                "nominal_value": dolya.get("@attributes", {}).get("НоминСтоим")
            })
            
            # Добавляем в связанные компании
            if founder_inn:
                parsed["related_companies"].append({
                    "inn": founder_inn,
                    "name": founder_name,
                    "ogrn": ul_attrs.get("ОГРН"),
                    "relation": "Учредитель",
                    "share": share
                })
        
        # ОКВЭД
        sv_okved = svul.get("СвОКВЭД", {})
        
        # Основной ОКВЭД
        okved_osn = sv_okved.get("СвОКВЭДОсн", {})
        if okved_osn:
            osn_attrs = okved_osn.get("@attributes", {})
            parsed["okved"]["main_code"] = osn_attrs.get("КодОКВЭД")
            parsed["okved"]["main_name"] = osn_attrs.get("НаимОКВЭД")
        
        # Дополнительные ОКВЭД
        okved_dop = sv_okved.get("СвОКВЭДДоп", [])
        if not isinstance(okved_dop, list):
            okved_dop = [okved_dop] if okved_dop else []
        
        parsed["okved"]["additional"] = []
        for okved in okved_dop:
            okved_attrs = okved.get("@attributes", {})
            parsed["okved"]["additional"].append({
                "code": okved_attrs.get("КодОКВЭД"),
                "name": okved_attrs.get("НаимОКВЭД")
            })
        
        # Уставный капитал
        sv_ust_kap = svul.get("СвУстКап", {})
        if sv_ust_kap:
            kap_attrs = sv_ust_kap.get("@attributes", {})
            parsed["capital"] = kap_attrs.get("СумКап")
        
    except Exception as e:
        parsed["parsing_error"] = str(e)
        import traceback
        parsed["parsing_traceback"] = traceback.format_exc()
    
    return parsed

# ==================== ПОСТРОЕНИЕ ГРАФА СВЯЗЕЙ ====================
def build_company_network_diagram(company_info: dict, related_companies: list) -> str:
    """Строит Mermaid диаграмму сети связанных компаний"""
    company_name = company_info.get("short_name") or company_info.get("full_name") or "Компания"
    company_inn = company_info.get("inn") or "N/A"
    
    def clean_name(name):
        if not name:
            return "N/A"
        return name.replace('"', '').replace('[', '').replace(']', '').replace('(', '').replace(')', '')[:50]
    
    diagram = "```mermaid\ngraph TD\n"
    
    # Центральная компания с дополнительной информацией
    main_label = f"{clean_name(company_name)}<br/>ИНН: {company_inn}"
    
    diagram += f'    MAIN["{main_label}"]\n'
    diagram += "    style MAIN fill:#4CAF50,stroke:#2E7D32,stroke-width:3px,color:#fff\n\n"
    
    # Добавляем связанные компании
    if related_companies:
        for idx, company in enumerate(related_companies, 1):
            node_id = f"REL{idx}"
            rel_name = clean_name(company.get("name"))
            rel_inn = company.get("inn", "N/A")
            relation = company.get("relation", "Связь")
            share = company.get("share", "")
            
            # Определяем цвет в зависимости от типа связи
            if "Учредитель" in relation:
                color = "#2196F3"
                stroke = "#1565C0"
                arrow_label = "Владеет"
            elif "Дочерн" in relation:
                color = "#FF9800"
                stroke = "#E65100"
                arrow_label = "Контролирует"
            else:
                color = "#9E9E9E"
                stroke = "#616161"
                arrow_label = "Связь"
            
            # Формируем метку узла
            node_label = f"{rel_name}<br/>ИНН: {rel_inn}<br/>{relation}"
            if share:
                node_label += f'<br/>📊 Доля: {share}%'
            
            diagram += f'    {node_id}["{node_label}"]\n'
            diagram += f'    style {node_id} fill:{color},stroke:{stroke},stroke-width:2px,color:#fff\n'
            
            # Связь со стрелкой
            if "Учредитель" in relation:
                if share:
                    diagram += f'    {node_id} -->|{arrow_label} {share}%| MAIN\n'
                else:
                    diagram += f'    {node_id} -->|{arrow_label}| MAIN\n'
            else:
                diagram += f'    MAIN -->|{arrow_label}| {node_id}\n'
    else:
        diagram += '    NOTE["❌ Связанные компании не найдены<br/>Данные отсутствуют в ЕГРЮЛ"]\n'
        diagram += '    style NOTE fill:#FFC107,stroke:#FF6F00,stroke-width:2px,color:#333\n'
        diagram += '    NOTE -.->|нет данных| MAIN\n'
    
    diagram += "```\n"
    return diagram

# ==================== ПАРСИНГ ФИНАНСОВ ====================
def parse_financial_data(fin_json: dict) -> dict:
    """ПОЛНЫЙ парсинг финансов - РЕАЛЬНАЯ СТРУКТУРА API"""
    parsed = {
        "raw_json_structure": list(fin_json.keys()) if isinstance(fin_json, dict) else [],
        "income_expenses": [],
        "taxes": [],
        "employees": [],
        "tax_systems": [],
        "company_size": None,
        "support": [],
        "msp_status": [],
        "all_data": {}
    }
    
    try:
        # Коды налогов (справочник)
        TAX_CODES = {
            "1": "Налог на прибыль организаций / УСН",
            "2": "Транспортный налог",
            "3": "Земельный налог",
            "5": "Страховые взносы"
        }
        
        # Обрабатываем данные по годам
        years_data = {}
        for key, value in fin_json.items():
            # Проверяем что это год (цифры)
            if key.isdigit() and isinstance(value, dict):
                year = key
                year_data = value
                
                # Доходы/расходы
                income = year_data.get("income")
                outcome = year_data.get("outcome")
                
                if income or outcome:
                    parsed["income_expenses"].append({
                        "year": year,
                        "income": int(income) if income else None,
                        "outcome": int(outcome) if outcome else None,
                        "profit": (int(income) - int(outcome)) if (income and outcome) else None
                    })
                
                # Численность
                n = year_data.get("n")
                if n:
                    parsed["employees"].append({
                        "year": year,
                        "count": int(n) if n else None
                    })
                
                # Система налогообложения
                tax_system = year_data.get("tax_system")
                if tax_system:
                    tax_system_name = "УСН" if tax_system == "2" else f"Код {tax_system}"
                    if tax_system_name not in parsed["tax_systems"]:
                        parsed["tax_systems"].append({
                            "code": tax_system,
                            "name": tax_system_name
                        })
                
                # Налоги
                taxes = year_data.get("tax", {})
                if isinstance(taxes, dict):
                    for tax_code, tax_amount in taxes.items():
                        parsed["taxes"].append({
                            "year": year,
                            "code": tax_code,
                            "name": TAX_CODES.get(tax_code, f"Налог код {tax_code}"),
                            "amount": int(tax_amount) if tax_amount else None
                        })
                
                years_data[year] = year_data
        
        # МСП статус
        msp = fin_json.get("msp", [])
        if isinstance(msp, list):
            for item in msp:
                cat_name = {
                    "1": "Микропредприятие",
                    "2": "Малое предприятие", 
                    "3": "Среднее предприятие"
                }.get(str(item.get("cat")), f"Категория {item.get('cat')}")
                
                parsed["msp_status"].append({
                    "date": item.get("inc_date"),
                    "category": cat_name,
                    "category_code": item.get("cat")
                })
                
                # Определяем размер компании (берем последний)
                parsed["company_size"] = cat_name
        
        # Господдержка
        support = fin_json.get("support", [])
        if isinstance(support, list):
            for item in support:
                # Формы поддержки
                form_names = {
                    "100": "Субсидия",
                    "200": "Грант",
                    "400": "Льгота"
                }
                
                # Типы поддержки
                type_names = {
                    "103": "Финансовая поддержка",
                    "204": "Грант на развитие",
                    "401": "Налоговые льготы",
                    "406": "Льготные кредиты"
                }
                
                form_id = item.get("form_id")
                type_id = item.get("type_id")
                
                parsed["support"].append({
                    "year": item.get("accept_date", "")[:4] if item.get("accept_date") else None,
                    "from_inn": item.get("from_inn"),
                    "form": form_names.get(form_id, f"Форма {form_id}"),
                    "form_code": form_id,
                    "type": type_names.get(type_id, f"Тип {type_id}"),
                    "type_code": type_id,
                    "amount": float(item.get("s", 0)) if item.get("s") else None,
                    "amount_type": item.get("s_type"),
                    "accept_date": item.get("accept_date"),
                    "term": item.get("term"),
                    "end_date": item.get("end_date")
                })
        
        # Сортируем данные по годам (новые первые)
        parsed["income_expenses"].sort(key=lambda x: x["year"], reverse=True)
        parsed["taxes"].sort(key=lambda x: x["year"], reverse=True)
        parsed["employees"].sort(key=lambda x: x["year"], reverse=True)
        
        parsed["all_data"] = {
            "years_count": len(years_data),
            "years": sorted(years_data.keys(), reverse=True)
        }
        
    except Exception as e:
        parsed["parsing_error"] = str(e)
        import traceback
        parsed["parsing_traceback"] = traceback.format_exc()
    
    return parsed

# ==================== ФУНКЦИИ СБОРА ДАННЫХ ====================
def fetch_egrul_data(state: AgentState) -> AgentState:
    inn = normalize_inn(state["inn"])
    url = EGRUL_API_URL_TEMPLATE.format(inn=inn)
    
    st.info(f"📋 Загружаю ЕГРЮЛ для {inn}...")
    time.sleep(10)
    
    for attempt in range(3):
        try:
            response = requests.get(url, headers=HEADERS, timeout=30)
            response.raise_for_status()
            
            try:
                content = gzip.decompress(response.content)
                raw_data = json.loads(content.decode('utf-8'))
            except:
                raw_data = response.json()
            
            # ПАРСИМ
            parsed_data = parse_egrul_data(raw_data)
            st.success(f"✅ ЕГРЮЛ: найдено {len(parsed_data['raw_json_structure'])} полей")
            
            return {
                "egrul_parsed": parsed_data,
                "related_companies": parsed_data.get("related_companies", [])
            }
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                wait = 60 * (attempt + 1)
                st.warning(f"⚠️ 429. Ждем {wait} сек...")
                time.sleep(wait)
                if attempt == 2:
                    return {"egrul_parsed": {"error": "429"}, "related_companies": []}
            else:
                return {"egrul_parsed": {"error": f"HTTP {e.response.status_code}"}, "related_companies": []}
        except Exception as e:
            return {"egrul_parsed": {"error": str(e)}, "related_companies": []}

def fetch_financial_data(state: AgentState) -> AgentState:
    inn = normalize_inn(state["inn"])
    url = FIN_API_URL_TEMPLATE.format(inn=inn)
    
    st.info(f"💰 Загружаю финансы для {inn}...")
    time.sleep(15)
    
    for attempt in range(3):
        try:
            response = requests.get(url, headers=HEADERS, timeout=30)
            response.raise_for_status()
            
            try:
                raw_data = response.json()
            except:
                content = gzip.decompress(response.content)
                raw_data = json.loads(content.decode('utf-8'))
            
            parsed_data = parse_financial_data(raw_data)
            st.success(f"✅ Финансы: найдено {len(parsed_data['raw_json_structure'])} полей")
            return {"fin_parsed": parsed_data}
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                wait = 60 * (attempt + 1)
                st.warning(f"⚠️ 429. Ждем {wait} сек...")
                time.sleep(wait)
                if attempt == 2:
                    return {"fin_parsed": {"error": "429"}}
            else:
                return {"fin_parsed": {"error": f"HTTP {e.response.status_code}"}}
        except Exception as e:
            return {"fin_parsed": {"error": str(e)}}

def analyze_pdf_with_gemini(pdf_path: str, filename: str) -> dict:
    """
    Отправляет PDF напрямую в Gemini для анализа как инвест-аналитик
    """
    result = {
        "filename": filename,
        "success": False,
        "report_type": None,
        "report_year": None,
        "company_info": {},
        "multi_year_data": None,
        "analysis": None
    }
    
    try:
        # Читаем PDF как бинарные данные
        with open(pdf_path, 'rb') as f:
            pdf_data = f.read()
        
        # Конвертируем в base64 для отправки в API
        import base64
        pdf_base64 = base64.b64encode(pdf_data).decode('utf-8')
        
        # Промпт для Gemini как инвест-аналитика
        prompt = """Ты — профессиональный инвест-аналитик. Проанализируй эту бухгалтерскую отчетность.

**Твоя задача:**
1. Извлечь ВСЕ финансовые показатели из документа
2. Создать структурированные данные для анализа
3. Дать профессиональную оценку финансового состояния

**ВЕРНИ ТОЛЬКО ВАЛИДНЫЙ JSON в следующем формате:**

```json
{
  "report_type": "Бухгалтерский баланс" или "Отчет о финансовых результатах" или "Отчет о финансовом положении" или "Отчет о движении денежных средств" или "Отчет об изменениях в капитале" или "Полная отчетность",
  "report_year": "2023",
  "company_info": {
    "name": "ООО КОМПАНИЯ",
    "inn": "1234567890"
  },
  "multi_year_data": {
    "years": ["2021", "2022", "2023"],
    "balance": {
      "АКТИВ (всего)": [4500, 5200, 6047],
      "Основные средства": [2800, 3200, 3800],
      "Запасы": [400, 500, 597],
      "Дебиторская задолженность": [600, 700, 850],
      "Денежные средства": [150, 180, 192],
      "Краткосрочные финансовые вложения": [100, 120, 128],
      "ПАССИВ (всего)": [4500, 5200, 6047],
      "Капитал и резервы": [3800, 4500, 5145],
      "Долгосрочные заемные средства": [200, 120, 250],
      "Краткосрочные заемные средства": [50, 30, 50],
      "Кредиторская задолженность": [450, 550, 602],
      "Оборотный капитал": [750, 900, 1080],
      "Оборотный капитал / Выручка, %": [1.67, 1.88, 2.06],
      "Оборачиваемость запасов, раз": [112.5, 96.0, 87.9],
      "Оборачиваемость дебиторской задолженности, раз": [75.0, 68.6, 61.8],
      "Оборачиваемость кредиторской задолженности, раз": [100.0, 87.3, 87.2],
      "Чистый долг": [100, -30, 108],
      "Чистый долг / Операционная прибыль, x": [0.06, -0.02, 0.05]
    },
    "financial_results": {
      "Выручка": [45000, 48000, 52489],
      "Темп роста выручки, %": [null, 6.67, 9.35],
      "Валовая прибыль": [5000, 5500, 6100],
      "Рентабельность по валовой прибыли, %": [11.11, 11.46, 11.63],
      "Операционная прибыль": [1800, 2100, 2353],
      "Рентабельность по операционной прибыли, %": [4.00, 4.38, 4.48],
      "Чистая прибыль (убыток)": [1800, 2100, 2353],
      "Рентабельность по чистой прибыли, %": [4.00, 4.38, 4.48],
      "Налог на прибыль": [600, 750, 850]
    },
    "cash_flows": {
      "Сальдо потоков от операционной деятельности": [2500, 2800, 3100],
      "Сальдо потоков от инвестиционной деятельности": [-800, -900, -1000],
      "Сальдо потоков от финансовой деятельности": [-500, -600, -700],
      "Итоговый денежный поток за период": [1200, 1300, 1400],
      "Приобретение основных средств": [600, 700, 800],
      "Разница: Операционный поток - Приобретение ОС": [1900, 2100, 2300]
    }
  },
  "analysis": {
    "financial_health": "Отличное/Хорошее/Удовлетворительное/Плохое",
    "key_metrics": {
      "revenue_growth": "+9.3%",
      "profit_margin": "4.5%",
      "roa": "8.2%",
      "roe": "15.6%",
      "debt_to_equity": "0.18",
      "current_ratio": "2.8",
      "quick_ratio": "2.1"
    },
    "strengths": [
      "Стабильный рост выручки",
      "Низкая долговая нагрузка",
      "Положительная динамика прибыли",
      "Высокая ликвидность"
    ],
    "weaknesses": [
      "Рост кредиторской задолженности",
      "Снижение оборачиваемости активов"
    ],
    "risks": [
      "Возможная зависимость от кредитования",
      "Снижение маржинальности"
    ],
    "investment_rating": "A/A-/B+/B/B-/C",
    "investment_recommendation": "Рекомендуется для инвестиций / Требует осторожности / Не рекомендуется",
    "summary": "Краткое резюме финансового состояния (2-3 предложения с конкретными цифрами)"
  }
}
```

**ВАЖНЫЕ ПРАВИЛА:**
- Извлекай ТОЛЬКО данные из документа, НЕ придумывай
- Если данных нет — оставляй поле пустым или null
- Все суммы указывай в тысячах рублей (как в документе)
- Рассчитай ВСЕ финансовые показатели и коэффициенты:

**Из отчета о финансовых результатах:**
  * Выручка - извлекай из документа
  * Темп роста выручки, % = ((Выручка текущего года - Выручка предыдущего года) / Выручка предыдущего года) * 100. Для первого года = null
  * Валовая прибыль = Выручка - Себестоимость продаж
  * Рентабельность по валовой прибыли, % = (Валовая прибыль / Выручка) * 100
  * Операционная прибыль = Валовая прибыль - Коммерческие расходы - Управленческие расходы
  * Рентабельность по операционной прибыли, % = (Операционная прибыль / Выручка) * 100
  * Чистая прибыль - извлекай из документа
  * Рентабельность по чистой прибыли, % = (Чистая прибыль / Выручка) * 100

**Из отчета о финансовом положении (баланс):**
  * Активы итого - извлекай из документа
  * Основные средства - извлекай из документа
  * Запасы - извлекай из документа
  * Оборачиваемость запасов, раз = Выручка / Средние запасы (средние = (Запасы на начало + Запасы на конец) / 2)
  * Дебиторская задолженность - извлекай из документа
  * Оборачиваемость дебиторской задолженности, раз = Выручка / Средняя дебиторская задолженность
  * Кредиторская задолженность - извлекай из документа
  * Оборачиваемость кредиторской задолженности, раз = Выручка / Средняя кредиторская задолженность
  * Оборотный капитал = Оборотные активы - Краткосрочные обязательства
  * Оборотный капитал / Выручка, % = (Оборотный капитал / Выручка) * 100
  * Капитал и резервы - извлекай из документа
  * Долгосрочные заемные средства - извлекай из документа
  * Краткосрочные заемные средства - извлекай из документа
  * Чистый долг = (Долгосрочные заемные средства + Краткосрочные заемные средства) - (Денежные средства + Краткосрочные финансовые вложения)
  * Чистый долг / Операционная прибыль, x = Чистый долг / Операционная прибыль

**Из отчета о движении денежных средств:**
  * Сальдо потоков от операционной деятельности - извлекай из документа
  * Сальдо потоков от инвестиционной деятельности - извлекай из документа
  * Сальдо потоков от финансовой деятельности - извлекай из документа
  * Итоговый денежный поток за период = Сальдо операционной + Сальдо инвестиционной + Сальдо финансовой
  * Приобретение основных средств - извлекай из раздела инвестиционной деятельности
  * Разница: Операционный поток - Приобретение ОС = Сальдо потоков от операционной деятельности - Приобретение основных средств

**Дополнительные коэффициенты для analysis.key_metrics:**
  * ROE (рентабельность капитала) = Чистая прибыль / Капитал и резервы
  * ROA (рентабельность активов) = Чистая прибыль / Активы итого
  * Current Ratio (текущая ликвидность) = Оборотные активы / Краткосрочные обязательства
  * Quick Ratio (быстрая ликвидность) = (Оборотные активы - Запасы) / Краткосрочные обязательства
  * Debt/Equity (долг/капитал) = (Долгосрочные + Краткосрочные обязательства) / Капитал и резервы

- Используй массив years для всех найденных годов в документе
- Дай профессиональную оценку инвестиционной привлекательности
- ВЕРНИ ТОЛЬКО JSON, без markdown блоков и текста вокруг

Анализируй документ как профессиональный финансовый аналитик крупного инвестиционного фонда."""

        # Отправляем в Gemini
        import google.generativeai as genai
        
        # Создаем части запроса
        parts = [
            prompt,
            {
                "mime_type": "application/pdf",
                "data": pdf_base64
            }
        ]
        
        # Генерируем ответ
        response = model.generate_content(parts)
        
        # Парсим JSON из ответа
        response_text = response.text.strip()
        
        # Убираем markdown блоки если есть
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()
        
        # Парсим JSON
        data = json.loads(response_text)
        
        # Заполняем результат
        result["success"] = True
        result["report_type"] = data.get("report_type")
        result["report_year"] = data.get("report_year")
        result["company_info"] = data.get("company_info", {})
        result["multi_year_data"] = data.get("multi_year_data")
        result["analysis"] = data.get("analysis")
        result["text"] = ""  # Для совместимости
        
    except json.JSONDecodeError as e:
        result["error"] = f"Ошибка парсинга JSON от AI: {str(e)}"
        result["raw_response"] = response_text if 'response_text' in locals() else None
    except Exception as e:
        result["error"] = f"Ошибка AI-анализа: {str(e)}"
    
    return result

def download_and_parse_boh(state: AgentState) -> AgentState:
    inn = normalize_inn(state["inn"])
    out_dir = f"reports_{inn}"
    os.makedirs(out_dir, exist_ok=True)
    
    st.info("📊 Загружаю бухотчетность...")
    time.sleep(20)
    
    for attempt in range(3):
        try:
            resp = requests.get(f"{BASE_BOH_URL}ls.php", params={"inn": inn}, headers=HEADERS, timeout=15)
            resp.raise_for_status()
            break
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                wait = 60 * (attempt + 1)
                st.warning(f"⚠️ 429. Ждем {wait} сек...")
                time.sleep(wait)
                if attempt == 2:
                    return {"boh_parsed": [{"error": "429"}]}
            else:
                return {"boh_parsed": [{"error": f"HTTP {e.response.status_code}"}]}

    soup = BeautifulSoup(resp.text, "html.parser")
    links = soup.find_all('a', string='Скачать')
    
    if not links:
        st.warning("⚠️ Файлов не найдено")
        return {"boh_parsed": []}
    
    # АНАЛИЗИРУЕМ ВСЕ ФАЙЛЫ БЕЗ ОГРАНИЧЕНИЙ!
    total_files = len(links)
    
    # Подсчитываем типы файлов
    xml_count = sum(1 for link in links if '.xml' in link.get('href', '').lower())
    pdf_count = total_files - xml_count
    
    st.info(f"📊 Найдено файлов отчетности: {total_files} (XML: {xml_count}, PDF: {pdf_count})")
    st.info(f"📥 Скачиваю и анализирую ВСЕ файлы...")
    
    parsed_files = []
    
    progress = st.progress(0)
    
    for idx, link in enumerate(links, 1):
        href = link.get("href")
        if not href:
            continue

        if idx > 1:
            time.sleep(20)
        
        url = urljoin(BASE_BOH_URL, href)
        is_xml = '.xml' in href.lower()
        ext = '.xml' if is_xml else '.pdf'
        filename = f"report_{inn}_{idx}{ext}"
        path = os.path.join(out_dir, filename)
        
        st.text(f"📥 {filename}")
        
        try:
            r = requests.get(url, headers=HEADERS, timeout=30, stream=True)
            r.raise_for_status()
            
            with open(path, "wb") as f:
                for chunk in r.iter_content(8192):
                    if chunk:
                        f.write(chunk)
            
            if is_xml or path.endswith('.xml'):
                # XML - используем старый парсер
                parsed = extract_from_xml(path)
            else:
                # PDF - отправляем напрямую в Gemini как инвест-аналитик
                st.info(f"🤖 Анализирую {filename} с помощью AI...")
                parsed = analyze_pdf_with_gemini(path, filename)
            
            parsed_files.append(parsed)
            
            if parsed.get("success"):
                # Показываем результат AI-анализа
                if parsed.get("analysis"):
                    analysis = parsed["analysis"]
                    rating = analysis.get("investment_rating", "N/A")
                    health = analysis.get("financial_health", "N/A")
                    recommendation = analysis.get("investment_recommendation", "")
                    
                    # Эмодзи для рейтингов
                    rating_emoji = {
                        "A": "🟢", "A-": "🟢", 
                        "B+": "🟡", "B": "🟡", "B-": "🟡",
                        "C": "🔴", "C-": "🔴"
                    }.get(rating, "⚪")
                    
                    st.success(f"✅ {filename} {rating_emoji} Рейтинг: {rating} | Состояние: {health}")
                    if recommendation:
                        st.info(f"   💡 {recommendation}")
                else:
                    # Для XML файлов - старый формат
                    char_count = len(parsed.get("text", ""))
                    info_parts = [f"{char_count} символов"]
                    
                    if parsed.get("report_year"):
                        info_parts.append(f"Год: {parsed['report_year']}")
                    
                    if parsed.get("structured_data"):
                        sd = parsed["structured_data"]
                        if sd.get("balance"):
                            info_parts.append("Баланс ✓")
                        if sd.get("financial_results"):
                            info_parts.append("FinРез ✓")
                    
                    st.success(f"✅ {filename} " + " | ".join(info_parts))
            else:
                error_msg = parsed.get('error', 'Неизвестная ошибка')
                st.warning(f"⚠️ {filename}: {error_msg}")
                
        except Exception as e:
            parsed_files.append({"filename": filename, "error": str(e)})
            st.error(f"❌ {filename}: {str(e)}")
        
        progress.progress(idx / len(links))
    
    progress.empty()
    
    # Подробная статистика
    successful = [f for f in parsed_files if f.get("success")]
    
    # Разбивка по типам
    xml_files = [f for f in parsed_files if f.get("filename", "").endswith('.xml')]
    pdf_files = [f for f in parsed_files if f.get("filename", "").endswith('.pdf')]
    
    xml_success = [f for f in xml_files if f.get("success")]
    pdf_success = [f for f in pdf_files if f.get("success")]
    
    st.success(f"✅ Обработано: {len(successful)}/{len(parsed_files)} файлов")
    st.info(f"   📄 XML: {len(xml_success)}/{len(xml_files)} | 📋 PDF: {len(pdf_success)}/{len(pdf_files)}")
    
    return {"boh_parsed": parsed_files}

def get_case_pdf_url(case_number: str) -> str:
    """
    Получает URL PDF файла решения по делу через API
    """
    if not case_number:
        return None
    
    try:
        # Получаем детали дела
        details_url = f"https://parser-api.com/parser/arbitr_api/details_by_number?key={ARBITRATION_API_KEY}&CaseNumber={case_number}"
        response = requests.get(details_url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if data.get("Success") != 1:
            return None
        
        cases = data.get("Cases", [])
        if not cases:
            return None
        
        case_data = cases[0]
        
        # Ищем PDF файл в инстанциях
        instances = case_data.get("CaseInstances", [])
        for instance in instances:
            # Проверяем основной файл инстанции
            file_info = instance.get("File")
            if file_info and file_info.get("URL"):
                return file_info["URL"]
            
            # Проверяем события (ищем финальное решение)
            events = instance.get("InstanceEvents", [])
            for event in events:
                if event.get("FinishEvent") == 1:  # Финальное решение
                    pdf_url = event.get("File")
                    if pdf_url:
                        return pdf_url
        
        # Если не нашли финальное, берем любой PDF из событий
        for instance in instances:
            events = instance.get("InstanceEvents", [])
            for event in events:
                pdf_url = event.get("File")
                if pdf_url:
                    return pdf_url
        
        return None
        
    except Exception as e:
        return None

def get_case_summary_from_api(case_number: str) -> str:
    """
    Получает детальную информацию о деле и формирует понятное описание СУТИ КОНФЛИКТА
    На основе РЕАЛЬНЫХ полей API (не выдуманных!)
    """
    if not case_number:
        return "Номер дела отсутствует"
    
    try:
        # Получаем ДЕТАЛЬНУЮ информацию о деле
        details_url = f"https://parser-api.com/parser/arbitr_api/details_by_number?key={ARBITRATION_API_KEY}&CaseNumber={case_number}"
        response = requests.get(details_url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if data.get("Success") != 1:
            return "Детали дела недоступны"
        
        # В ответе может быть массив Cases
        cases = data.get("Cases", [])
        if not cases:
            return "Детали дела недоступны"
        
        case_data = cases[0]  # Берем первое дело
        
        # Собираем информацию из РЕАЛЬНО существующих полей
        info_parts = []
        
        # 1. Состояние дела (State)
        state = case_data.get("State", "")
        if state:
            info_parts.append(f"Состояние: {state}")
        
        # 2. Сумма иска - ищем в событиях (ClaimSum)
        claim_sum = None
        instances = case_data.get("CaseInstances", [])
        for instance in instances:
            events = instance.get("InstanceEvents", [])
            for event in events:
                if event.get("ClaimSum"):
                    claim_sum = event["ClaimSum"]
                    break
            if claim_sum:
                break
        
        if claim_sum:
            info_parts.append(f"Сумма иска: {claim_sum:,.0f} руб.")
        
        # 3. Тип решения - из последнего события
        decision_info = None
        for instance in instances:
            events = instance.get("InstanceEvents", [])
            if events:
                # Ищем финальное событие (решение)
                for event in events:
                    if event.get("FinishEvent") == 1:
                        event_type = event.get("EventTypeName", "")
                        content_type = event.get("EventContentTypeName", "")
                        if content_type:
                            decision_info = content_type
                        elif event_type:
                            decision_info = event_type
                        break
        
        if decision_info:
            info_parts.append(f"Решение: {decision_info}")
        
        # 4. Тип дела для контекста
        case_type = case_data.get("CaseType", "")
        type_desc = {
            "А": "административный спор",
            "Б": "банкротство",
            "Г": "гражданский спор"
        }.get(case_type, "спор")
        
        # Если ничего не собрали - используем минимальную информацию
        if not info_parts:
            # Пытаемся хотя бы указать участников
            plaintiffs = case_data.get("Plaintiffs", [])
            respondents = case_data.get("Respondents", [])
            
            if plaintiffs and respondents:
                p_name = plaintiffs[0].get("Name", "")[:30] if plaintiffs else ""
                r_name = respondents[0].get("Name", "")[:30] if respondents else ""
                
                if p_name and r_name:
                    return f"{type_desc.capitalize()}: {p_name} vs {r_name}"
            
            return "Детали дела недоступны в API"
        
        # Формируем текст для AI
        full_context = "\n".join(info_parts)
        
        # Отправляем в AI для создания краткого резюме
        prompt = f"""Ты - юридический аналитик. Создай КРАТКОЕ описание (до 15 слов) сути судебного дела на основе ДОСТУПНОЙ информации:

{full_context}

Тип дела: {type_desc}

ВАЖНО:
- Будь конкретным
- Укажи сумму если есть
- Укажи суть решения если есть
- Максимум 15 слов!

ПРИМЕРЫ ХОРОШИХ ОПИСАНИЙ:
❌ ПЛОХО: "Административный спор"
✅ ХОРОШО: "Административный спор, иск 500 тыс. руб., иск удовлетворен"

❌ ПЛОХО: "Банкротство"
✅ ХОРОШО: "Банкротство, требование кредиторов 2 млн руб."

Твое краткое описание (до 15 слов):"""

        response_ai = model.generate_content(prompt)
        summary = response_ai.text.strip()
        
        # Убираем лишние символы
        summary = summary.replace('"', '').replace("'", "").replace('✅', '').replace('❌', '').strip()
        
        # Ограничиваем длину
        if len(summary) > 150:
            summary = summary[:147] + "..."
        
        return summary
        
    except Exception as e:
        return f"Ошибка получения деталей: {str(e)[:30]}"

def fetch_arbitration_data(state: AgentState) -> AgentState:
    inn = normalize_inn(state["inn"])
    
    st.info("⚖️ Загружаю судебные дела...")
    
    result = {
        "plaintiff_cases": [],
        "respondent_cases": [],
        "third_party_cases": [],
        "total": 0,
        "plaintiff_total": 0,
        "respondent_total": 0,
        "third_party_total": 0
    }
    
    try:
        # Типы дел для расшифровки
        CASE_TYPES = {
            "А": "Административное",
            "Б": "Банкротное",
            "Г": "Гражданское"
        }
        
        # 1. ЗАГРУЖАЕМ ДЕЛА ГДЕ КОМПАНИЯ - ИСТЕЦ
        st.info("📋 Загружаю дела (истец)...")
        try:
            response_plaintiff = requests.get(
                ARBITRATION_API_URL, 
                params={'key': ARBITRATION_API_KEY, 'Inn': inn, 'InnType': 'Plaintiff'}
            )
            response_plaintiff.raise_for_status()
            data_plaintiff = response_plaintiff.json()
            
            if data_plaintiff.get("Success") == 1:
                cases_plaintiff = data_plaintiff.get("Cases", [])
                
                for idx, case in enumerate(cases_plaintiff):
                    case_number = safe_get(case, "CaseNumber")
                    case_type = safe_get(case, "CaseType")
                    
                    # Собираем всех участников с детальной информацией
                    plaintiffs_list = []
                    for p in (case.get("Plaintiffs") or []):
                        if isinstance(p, dict):
                            plaintiffs_list.append({
                                "name": p.get("Name"),
                                "inn": p.get("Inn"),
                                "address": p.get("Address")
                            })
                    
                    respondents_list = []
                    for r in (case.get("Respondents") or []):
                        if isinstance(r, dict):
                            respondents_list.append({
                                "name": r.get("Name"),
                                "inn": r.get("Inn"),
                                "address": r.get("Address")
                            })
                    
                    # Получаем краткое содержание дела (для ВСЕХ дел!)
                    case_summary = "..."
                    pdf_url = None
                    if case_number:
                        st.text(f"   🔍 Анализирую дело {case_number}...")
                        case_summary = get_case_summary_from_api(case_number)
                        # Получаем PDF URL
                        pdf_url = get_case_pdf_url(case_number)
                    
                    parsed_case = {
                        "number": case_number,
                        "case_id": safe_get(case, "CaseId"),
                        "case_type": CASE_TYPES.get(case_type, case_type),
                        "case_type_code": case_type,
                        "court": safe_get(case, "Court", "Name") if isinstance(safe_get(case, "Court"), dict) else safe_get(case, "Court"),
                        "start_date": safe_get(case, "StartDate"),
                        "plaintiffs": plaintiffs_list,
                        "respondents": respondents_list,
                        "summary": case_summary,
                        "pdf_url": pdf_url,
                        "details_url": f"https://parser-api.com/parser/arbitr_api/details_by_number?key={ARBITRATION_API_KEY}&CaseNumber={case_number}" if case_number else None
                    }
                    
                    result["plaintiff_cases"].append(parsed_case)
                
                result["plaintiff_total"] = len(cases_plaintiff)
                st.success(f"✅ Истец: {len(cases_plaintiff)} дел")
        
        except Exception as e:
            st.warning(f"⚠️ Ошибка загрузки дел (истец): {str(e)}")
        
        # 2. ЗАГРУЖАЕМ ДЕЛА ГДЕ КОМПАНИЯ - ОТВЕТЧИК
        st.info("📋 Загружаю дела (ответчик)...")
        try:
            response_respondent = requests.get(
                ARBITRATION_API_URL, 
                params={'key': ARBITRATION_API_KEY, 'Inn': inn, 'InnType': 'Respondent'}
            )
            response_respondent.raise_for_status()
            data_respondent = response_respondent.json()
            
            if data_respondent.get("Success") == 1:
                cases_respondent = data_respondent.get("Cases", [])
                
                for idx, case in enumerate(cases_respondent):
                    case_number = safe_get(case, "CaseNumber")
                    case_type = safe_get(case, "CaseType")
                    
                    # Собираем всех участников с детальной информацией
                    plaintiffs_list = []
                    for p in (case.get("Plaintiffs") or []):
                        if isinstance(p, dict):
                            plaintiffs_list.append({
                                "name": p.get("Name"),
                                "inn": p.get("Inn"),
                                "address": p.get("Address")
                            })
                    
                    respondents_list = []
                    for r in (case.get("Respondents") or []):
                        if isinstance(r, dict):
                            respondents_list.append({
                                "name": r.get("Name"),
                                "inn": r.get("Inn"),
                                "address": r.get("Address")
                            })
                    
                    # Получаем краткое содержание дела (для ВСЕХ дел!)
                    case_summary = "..."
                    pdf_url = None
                    if case_number:
                        st.text(f"   🔍 Анализирую дело {case_number}...")
                        case_summary = get_case_summary_from_api(case_number)
                        # Получаем PDF URL
                        pdf_url = get_case_pdf_url(case_number)
                    
                    parsed_case = {
                        "number": case_number,
                        "case_id": safe_get(case, "CaseId"),
                        "case_type": CASE_TYPES.get(case_type, case_type),
                        "case_type_code": case_type,
                        "court": safe_get(case, "Court", "Name") if isinstance(safe_get(case, "Court"), dict) else safe_get(case, "Court"),
                        "start_date": safe_get(case, "StartDate"),
                        "plaintiffs": plaintiffs_list,
                        "respondents": respondents_list,
                        "summary": case_summary,
                        "pdf_url": pdf_url,
                        "details_url": f"https://parser-api.com/parser/arbitr_api/details_by_number?key={ARBITRATION_API_KEY}&CaseNumber={case_number}" if case_number else None
                    }
                    
                    result["respondent_cases"].append(parsed_case)
                
                result["respondent_total"] = len(cases_respondent)
                st.success(f"✅ Ответчик: {len(cases_respondent)} дел")
        
        except Exception as e:
            st.warning(f"⚠️ Ошибка загрузки дел (ответчик): {str(e)}")
        
        # 3. ЗАГРУЖАЕМ ДЕЛА ГДЕ КОМПАНИЯ - ТРЕТЬЕ ЛИЦО
        st.info("📋 Загружаю дела (третье лицо)...")
        try:
            response_third = requests.get(
                ARBITRATION_API_URL, 
                params={'key': ARBITRATION_API_KEY, 'Inn': inn, 'InnType': 'Third'}
            )
            response_third.raise_for_status()
            data_third = response_third.json()
            
            if data_third.get("Success") == 1:
                cases_third = data_third.get("Cases", [])
                
                for idx, case in enumerate(cases_third):
                    case_number = safe_get(case, "CaseNumber")
                    case_type = safe_get(case, "CaseType")
                    
                    # Собираем всех участников
                    plaintiffs_list = []
                    for p in (case.get("Plaintiffs") or []):
                        if isinstance(p, dict):
                            plaintiffs_list.append({
                                "name": p.get("Name"),
                                "inn": p.get("Inn")
                            })
                    
                    respondents_list = []
                    for r in (case.get("Respondents") or []):
                        if isinstance(r, dict):
                            respondents_list.append({
                                "name": r.get("Name"),
                                "inn": r.get("Inn")
                            })
                    
                    # Получаем краткое содержание дела (для ВСЕХ дел!)
                    case_summary = "..."
                    pdf_url = None
                    if case_number:
                        st.text(f"   🔍 Анализирую дело {case_number}...")
                        case_summary = get_case_summary_from_api(case_number)
                        # Получаем PDF URL
                        pdf_url = get_case_pdf_url(case_number)
                    
                    parsed_case = {
                        "number": case_number,
                        "case_id": safe_get(case, "CaseId"),
                        "case_type": CASE_TYPES.get(case_type, case_type),
                        "case_type_code": case_type,
                        "court": safe_get(case, "Court", "Name") if isinstance(safe_get(case, "Court"), dict) else safe_get(case, "Court"),
                        "start_date": safe_get(case, "StartDate"),
                        "plaintiffs": plaintiffs_list,
                        "respondents": respondents_list,
                        "summary": case_summary,
                        "pdf_url": pdf_url,
                        "details_url": f"https://parser-api.com/parser/arbitr_api/details_by_number?key={ARBITRATION_API_KEY}&CaseNumber={case_number}" if case_number else None
                    }
                    
                    result["third_party_cases"].append(parsed_case)
                
                result["third_party_total"] = len(cases_third)
                if len(cases_third) > 0:
                    st.success(f"✅ Третье лицо: {len(cases_third)} дел")
        
        except Exception as e:
            st.warning(f"⚠️ Ошибка загрузки дел (третье лицо): {str(e)}")
        
        # ИТОГО
        result["total"] = result["plaintiff_total"] + result["respondent_total"] + result["third_party_total"]
        
        st.success(f"✅ ИТОГО: {result['total']} дел (истец: {result['plaintiff_total']}, ответчик: {result['respondent_total']}, третье лицо: {result['third_party_total']})")
        
        return {"courts_parsed": result}
        
    except Exception as e:
        st.error(f"❌ Суды: {str(e)}")
        return {"courts_parsed": {"error": str(e), "total": 0}}

def fetch_media_mentions(state: AgentState) -> AgentState:
    """Ищет упоминания компании в СМИ через Exa API"""
    
    if not EXA_AVAILABLE:
        st.warning("⚠️ Exa API недоступен. Установите: pip install exa_py")
        return {"media_mentions": []}
    
    egrul = state.get("egrul_parsed", {})
    company_name = egrul.get("short_name") or egrul.get("full_name", "")
    
    if not company_name:
        return {"media_mentions": []}
    
    st.info(f"📰 Ищу упоминания в СМИ...")
    
    try:
        # Инициализируем Exa клиент
        exa = Exa(api_key=EXA_API_KEY)
        
        # Формируем поисковый запрос
        query = f"статья новость упоминание {company_name}"
        
        # Поиск с получением контента
        response = exa.search_and_contents(
            query,
            num_results=10,
            use_autoprompt=True,
            text={"max_characters": 500},  # Краткий контент
            highlights={"num_sentences": 3,  # Ключевые предложения
                       "highlights_per_url": 3}
        )
        
        mentions = []
        for result in response.results:
            mention = {
                "title": result.title or "Без названия",
                "url": result.url,
                "published_date": result.published_date if hasattr(result, 'published_date') else None,
                "author": result.author if hasattr(result, 'author') else None,
                "text": result.text if hasattr(result, 'text') else None,
                "highlights": result.highlights if hasattr(result, 'highlights') else []
            }
            mentions.append(mention)
        
        st.success(f"✅ СМИ: найдено {len(mentions)} упоминаний")
        return {"media_mentions": mentions}
        
    except Exception as e:
        st.warning(f"⚠️ Ошибка поиска в СМИ: {str(e)}")
        return {"media_mentions": []}

# ==================== ГЕНЕРАЦИЯ ОТЧЕТА ====================
def merge_and_analyze(state: AgentState) -> AgentState:
    egrul = state["egrul_parsed"]
    fin = state["fin_parsed"]
    boh = state["boh_parsed"]
    courts = state["courts_parsed"]
    related = state["related_companies"]
    media = state.get("media_mentions", [])
    
    # Строим диаграмму сети компаний
    network_diagram = build_company_network_diagram(egrul.get("basic_info", {}), related)
    
    # Формируем текст с файлами бухотчетности
    boh_section = []
    for i, f in enumerate(boh):
        status = '✅ Успешно' if f.get('success') else f'❌ {f.get("error", "Ошибка")}'
        
        # Если есть AI-анализ - показываем полную информацию
        if f.get('analysis'):
            multi_data = f.get('multi_year_data', {})
            analysis = f['analysis']
            
            # Увеличены лимиты в 10 раз чтобы Gemini получал ВСЕ данные!
            data_preview = json.dumps(multi_data, ensure_ascii=False, indent=2)[:20000]
            analysis_preview = json.dumps(analysis, ensure_ascii=False, indent=2)[:15000]
            
            boh_section.append(f"""### {i+1}. {f['filename']} - {status} 🤖 AI-АНАЛИЗ
**Тип:** {f.get('report_type', 'N/A')}
**Год отчета:** {f.get('report_year', 'N/A')}
**Компания:** {f.get('company_info', {}).get('name', 'N/A')} (ИНН: {f.get('company_info', {}).get('inn', 'N/A')})
**Годы в данных:** {', '.join(multi_data.get('years', []))}

**СВОДНЫЕ ДАННЫЕ (для построения таблицы по всем годам):**
```json
{data_preview}
```

**AI-АНАЛИЗ ИНВЕСТ-АНАЛИТИКА:**
```json
{analysis_preview}
```

**Краткие выводы:**
- **Финансовое состояние:** {analysis.get('financial_health', 'N/A')}
- **Инвестиционный рейтинг:** {analysis.get('investment_rating', 'N/A')}
- **Рекомендация:** {analysis.get('investment_recommendation', 'N/A')}
- **Резюме:** {analysis.get('summary', 'N/A')}
""")
        # Показываем СВОДНЫЕ данные по всем годам если есть (для XML)
        elif f.get('multi_year_data'):
            multi_data = f['multi_year_data']
            # Увеличен лимит в 10 раз
            data_preview = json.dumps(multi_data, ensure_ascii=False, indent=2)[:20000]
            boh_section.append(f"""### {i+1}. {f['filename']} - {status}
**Тип:** {f.get('report_type', 'N/A')}
**Год отчета:** {f.get('report_year', 'N/A')}
**Годы в таблице:** {', '.join(multi_data.get('years', []))}

**СВОДНЫЕ ДАННЫЕ (для построения таблицы по всем годам):**
```json
{data_preview}
```""")
        elif f.get('structured_data'):
            # Увеличен лимит в 10 раз
            data_preview = json.dumps(f['structured_data'], ensure_ascii=False, indent=2)[:15000]
            boh_section.append(f"### {i+1}. {f['filename']} - {status}\n**Тип:** {f.get('report_type', 'N/A')}\n**Год:** {f.get('report_year', 'N/A')}\n```json\n{data_preview}\n```")
        else:
            # Увеличен лимит в 10 раз
            text_preview = f.get('text', '')[:8000]
            boh_section.append(f"### {i+1}. {f['filename']} - {status}\n```\n{text_preview}\n```")
    
    boh_text = "\n\n".join(boh_section)
    
    # Формируем данные БЕЗ ЗАГЛУШЕК
    data_summary = f"""
# ИЗВЛЕЧЕННЫЕ ДАННЫЕ (БЕЗ ЗАГЛУШЕК)

## ЕГРЮЛ - Структура JSON:
Найдено полей: {egrul.get('raw_json_structure', [])}

## ЕГРЮЛ - Основная информация:
```json
{json.dumps(egrul.get('all_fields', {}), ensure_ascii=False, indent=2)}
```

## ЕГРЮЛ - Адрес:
```json
{json.dumps(egrul.get('address', {}), ensure_ascii=False, indent=2)}
```

## ЕГРЮЛ - Руководители ({len(egrul.get('directors', []))} чел.):
```json
{json.dumps(egrul.get('directors', []), ensure_ascii=False, indent=2)}
```

## ЕГРЮЛ - Учредители ({len(egrul.get('founders', []))} чел.):
```json
{json.dumps(egrul.get('founders', []), ensure_ascii=False, indent=2)}
```

## ЕГРЮЛ - Связанные компании ({len(related)} шт.):
```json
{json.dumps(related, ensure_ascii=False, indent=2)}
```

## ЕГРЮЛ - ОКВЭД:
```json
{json.dumps(egrul.get('okved', {}), ensure_ascii=False, indent=2)}
```

## ЕГРЮЛ - Уставный капитал:
{egrul.get('capital') if egrul.get('capital') is not None else 'Отсутствует в JSON'}

---

## ФИНАНСЫ - Структура JSON:
Найдено полей: {fin.get('raw_json_structure', [])}

## ФИНАНСЫ - Доходы/расходы ({len(fin.get('income_expenses', []))} записей):
```json
{json.dumps(fin.get('income_expenses', []), ensure_ascii=False, indent=2)}
```

## ФИНАНСЫ - Налоги ({len(fin.get('taxes', []))} записей):
```json
{json.dumps(fin.get('taxes', []), ensure_ascii=False, indent=2)}
```

## ФИНАНСЫ - Численность ({len(fin.get('employees', []))} записей):
```json
{json.dumps(fin.get('employees', []), ensure_ascii=False, indent=2)}
```

## ФИНАНСЫ - Налоговые системы:
{fin.get('tax_systems') if fin.get('tax_systems') else 'Отсутствует'}

## ФИНАНСЫ - Размер компании:
{fin.get('company_size') if fin.get('company_size') else 'Отсутствует'}

## ФИНАНСЫ - Господдержка ({len(fin.get('support', []))} записей):
```json
{json.dumps(fin.get('support', []), ensure_ascii=False, indent=2)}
```

---

## БУХОТЧЕТНОСТЬ - Файлов: {len(boh)}

{boh_text}

---

## СУДЫ - Всего дел: {courts.get('total', 0)}

**Статистика:**
- Истец: {courts.get('plaintiff_total', 0)} дел
- Ответчик: {courts.get('respondent_total', 0)} дел
- Третье лицо: {courts.get('third_party_total', 0)} дел

### Компания как ИСТЕЦ ({courts.get('plaintiff_total', 0)} дел):
```json
{json.dumps(courts.get('plaintiff_cases', []), ensure_ascii=False, indent=2)}
```

### Компания как ОТВЕТЧИК ({courts.get('respondent_total', 0)} дел):
```json
{json.dumps(courts.get('respondent_cases', []), ensure_ascii=False, indent=2)}
```

### Компания как ТРЕТЬЕ ЛИЦО ({courts.get('third_party_total', 0)} дел):
```json
{json.dumps(courts.get('third_party_cases', []), ensure_ascii=False, indent=2)}
```

---

## 📰 УПОМИНАНИЯ В СМИ

**Всего упоминаний:** {len(media)}

{json.dumps(media, ensure_ascii=False, indent=2) if media else "[]"}
"""

    prompt = f"""
Ты — форматировщик данных. Твоя задача: переформатировать JSON в читаемый markdown отчет.

**ЗАПРЕЩЕНО:**
- Придумывать данные
- Добавлять информацию от себя
- Писать "Нет данных" если в JSON есть null или пусто - пиши "Отсутствует в API"

**РАЗРЕШЕНО:**
- Переформатировать JSON в таблицы
- Делать выводы НА ОСНОВЕ ЦИФР из данных
- Строить диаграммы если есть учредители

# Создай отчет по структуре:

## 📊 КОМПЛЕКСНЫЙ АНАЛИЗ КОМПАНИИ

### 1. 🏢 ОСНОВНЫЕ СВЕДЕНИЯ
Преобразуй `all_fields` в таблицу:
| Параметр | Значение |
|----------|----------|

Если поле = null → "Отсутствует в API"

### 2. 📍 ЮРИДИЧЕСКИЙ АДРЕС
Если есть `full` - покажи полностью. Если нет - собери из частей (регион, город, улица, дом)

### 3. 👥 РУКОВОДСТВО
Таблица ВСЕХ руководителей:
| ФИО | Должность | Дата назначения | ИНН |
|-----|-----------|-----------------|-----|

### 4. 💼 УЧРЕДИТЕЛИ И СТРУКТУРА ВЛАДЕНИЯ

#### Таблица учредителей:
| Наименование | Тип | ИНН | Доля % |
|--------------|-----|-----|--------|

#### Сеть связанных компаний:
{network_diagram}

**ВАЖНО: Диаграмму выше вставь В ТОЧНОСТИ как есть, НЕ изменяя синтаксис Mermaid!**

**Анализ корпоративной структуры:**
- Количество связанных юридических лиц: {len(related)}
- Прозрачность структуры владения
- Потенциальные бенефициары

### 5. 🏭 ВИДЫ ДЕЯТЕЛЬНОСТИ (ОКВЭД)

**Основной:**
Код и наименование

**Дополнительные:**
Список (если есть)

### 6. 💰 ФИНАНСОВЫЕ ПОКАЗАТЕЛИ

#### Доходы и расходы (по годам):
Таблица. Если массив пустой → "Данные отсутствуют в API"

#### Налоги (по годам):
Таблица

#### Среднесписочная численность:
Таблица по годам

#### Другие показатели:
- Налоговые системы
- Размер организации
- Господдержка

### 7. 📊 БУХГАЛТЕРСКАЯ ОТЧЕТНОСТЬ

**ВАЖНО: Используй multi_year_data для создания СВОДНОЙ ТАБЛИЦЫ!**

**Для файлов с AI-АНАЛИЗОМ (у которых есть поле `analysis`):**

Используй данные из `multi_year_data` и `analysis` для создания полноценного раздела:

#### Финансовые показатели (сводная таблица):

Построй таблицу по ВСЕМ годам из массива `years`:

| Показатель | {{Год 1}} | {{Год 2}} | {{Год 3}} | Изменение (%) |
|------------|---------|---------|---------|---------------|
| **ОТЧЕТ О ФИНАНСОВЫХ РЕЗУЛЬТАТАХ** | | | | |
| Выручка | ... | ... | ... | ... |
| Темп роста выручки, % | - | ... | ... | ... |
| Валовая прибыль | ... | ... | ... | ... |
| Рентабельность по валовой прибыли, % | ... | ... | ... | ... |
| Операционная прибыль | ... | ... | ... | ... |
| Рентабельность по операционной прибыли, % | ... | ... | ... | ... |
| Чистая прибыль (убыток) | ... | ... | ... | ... |
| Рентабельность по чистой прибыли, % | ... | ... | ... | ... |
| **ОТЧЕТ О ФИНАНСОВОМ ПОЛОЖЕНИИ (БАЛАНС)** | | | | |
| АКТИВ (всего) | ... | ... | ... | ... |
| Основные средства | ... | ... | ... | ... |
| Запасы | ... | ... | ... | ... |
| Оборачиваемость запасов, раз | ... | ... | ... | ... |
| Дебиторская задолженность | ... | ... | ... | ... |
| Оборачиваемость дебиторской задолженности, раз | ... | ... | ... | ... |
| Денежные средства | ... | ... | ... | ... |
| Краткосрочные финансовые вложения | ... | ... | ... | ... |
| ПАССИВ (всего) | ... | ... | ... | ... |
| Капитал и резервы | ... | ... | ... | ... |
| Долгосрочные заемные средства | ... | ... | ... | ... |
| Краткосрочные заемные средства | ... | ... | ... | ... |
| Кредиторская задолженность | ... | ... | ... | ... |
| Оборачиваемость кредиторской задолженности, раз | ... | ... | ... | ... |
| Оборотный капитал | ... | ... | ... | ... |
| Оборотный капитал / Выручка, % | ... | ... | ... | ... |
| Чистый долг | ... | ... | ... | ... |
| Чистый долг / Операционная прибыль, x | ... | ... | ... | ... |
| **ОТЧЕТ О ДВИЖЕНИИ ДЕНЕЖНЫХ СРЕДСТВ** | | | | |
| Сальдо потоков от операционной деятельности | ... | ... | ... | ... |
| Сальдо потоков от инвестиционной деятельности | ... | ... | ... | ... |
| Сальдо потоков от финансовой деятельности | ... | ... | ... | ... |
| Итоговый денежный поток за период | ... | ... | ... | ... |
| Приобретение основных средств | ... | ... | ... | ... |
| Разница: Операционный поток - Приобретение ОС | ... | ... | ... | ... |

#### 🤖 ПРОФЕССИОНАЛЬНЫЙ АНАЛИЗ ИНВЕСТ-АНАЛИТИКА:

**Финансовое состояние:** {{analysis.financial_health}}

**Ключевые финансовые коэффициенты:**
Создай таблицу из `analysis.key_metrics` и рассчитанных показателей из `multi_year_data`:

| Показатель | Значение | Норма |
|------------|----------|-------|
| **Рентабельность** | | |
| Рентабельность по валовой прибыли, % | Из multi_year_data.financial_results | >20% |
| Рентабельность по операционной прибыли, % | Из multi_year_data.financial_results | >10% |
| Рентабельность по чистой прибыли, % | Из multi_year_data.financial_results | >5% |
| ROE (рентабельность капитала) | {{roe}} | >15% |
| ROA (рентабельность активов) | {{roa}} | >10% |
| **Рост** | | |
| Темп роста выручки, % | Из multi_year_data.financial_results | >5% |
| **Ликвидность** | | |
| Коэфф. текущей ликвидности | {{current_ratio}} | >2.0 |
| Коэфф. быстрой ликвидности | {{quick_ratio}} | >1.0 |
| Оборотный капитал / Выручка, % | Из multi_year_data.balance | 5-15% |
| **Оборачиваемость** | | |
| Оборачиваемость запасов, раз | Из multi_year_data.balance | >5 |
| Оборачиваемость дебиторской задолженности, раз | Из multi_year_data.balance | >10 |
| Оборачиваемость кредиторской задолженности, раз | Из multi_year_data.balance | >8 |
| **Долговая нагрузка** | | |
| Долг/Капитал | {{debt_to_equity}} | <0.5 |
| Чистый долг / Операционная прибыль, x | Из multi_year_data.balance | <3.0 |

**Сильные стороны:**
Список из `analysis.strengths`

**Слабые стороны:**
Список из `analysis.weaknesses`

**Риски:**
Список из `analysis.risks` (если есть)

**Инвестиционная оценка:**
- **Рейтинг:** {{analysis.investment_rating}}
- **Рекомендация:** {{analysis.investment_recommendation}}
- **Резюме:** {{analysis.summary}}

---

**Для файлов БЕЗ AI-анализа (старые XML файлы):**

Для каждого файла с multi_year_data:

#### Отчет о финансовом положении (Баланс) - сводная таблица по всем годам:

Создай ОДНУ таблицу в формате:

| Показатель | Год 1 | Год 2 | Год 3 | Изменение (%) |
|------------|-------|-------|-------|---------------|
| АКТИВ (всего) | ... | ... | ... | ... |
| Основные средства | ... | ... | ... | ... |
| Запасы | ... | ... | ... | ... |
| Оборачиваемость запасов, раз | ... | ... | ... | ... |
| Дебиторская задолженность | ... | ... | ... | ... |
| Оборачиваемость дебиторской задолженности, раз | ... | ... | ... | ... |
| Денежные средства | ... | ... | ... | ... |
| Краткосрочные финансовые вложения | ... | ... | ... | ... |
| ПАССИВ (всего) | ... | ... | ... | ... |
| Капитал и резервы | ... | ... | ... | ... |
| Долгосрочные заемные средства | ... | ... | ... | ... |
| Краткосрочные заемные средства | ... | ... | ... | ... |
| Кредиторская задолженность | ... | ... | ... | ... |
| Оборачиваемость кредиторской задолженности, раз | ... | ... | ... | ... |
| Оборотный капитал | ... | ... | ... | ... |
| Оборотный капитал / Выручка, % | ... | ... | ... | ... |
| Чистый долг | ... | ... | ... | ... |
| Чистый долг / Операционная прибыль, x | ... | ... | ... | ... |

#### Отчет о финансовых результатах (сводная таблица):

| Показатель | Год 1 | Год 2 | Год 3 | Изменение (%) |
|------------|-------|-------|-------|---------------|
| Выручка | ... | ... | ... | ... |
| Темп роста выручки, % | - | ... | ... | ... |
| Валовая прибыль | ... | ... | ... | ... |
| Рентабельность по валовой прибыли, % | ... | ... | ... | ... |
| Операционная прибыль | ... | ... | ... | ... |
| Рентабельность по операционной прибыли, % | ... | ... | ... | ... |
| Чистая прибыль (убыток) | ... | ... | ... | ... |
| Рентабельность по чистой прибыли, % | ... | ... | ... | ... |

#### Отчет о движении денежных средств (сводная таблица):

| Показатель | Год 1 | Год 2 | Год 3 | Изменение (%) |
|------------|-------|-------|-------|---------------|
| Сальдо потоков от операционной деятельности | ... | ... | ... | ... |
| Сальдо потоков от инвестиционной деятельности | ... | ... | ... | ... |
| Сальдо потоков от финансовой деятельности | ... | ... | ... | ... |
| Итоговый денежный поток за период | ... | ... | ... | ... |
| Приобретение основных средств | ... | ... | ... | ... |
| Разница: Операционный поток - Приобретение ОС | ... | ... | ... | ... |

**Анализ:**
- Динамика показателей (рост/падение выручки, прибыли)
- Рентабельность (по валовой, операционной и чистой прибыли)
- Оборачиваемость (запасов, дебиторской и кредиторской задолженности)
- Ликвидность (оборотный капитал, коэффициенты ликвидности)
- Долговая нагрузка (чистый долг, соотношение долг/капитал)
- Денежные потоки (операционная, инвестиционная, финансовая деятельность)
- Ключевые выводы

### 8. ⚖️ СУДЕБНАЯ АКТИВНОСТЬ

**Всего дел: {courts.get('total', 0)}** (истец: {courts.get('plaintiff_total', 0)}, ответчик: {courts.get('respondent_total', 0)}, третье лицо: {courts.get('third_party_total', 0)})

#### 📋 Компания как ИСТЕЦ ({{courts.get('plaintiff_total', 0)}} дел):

Для каждого дела создай строку таблицы с КРАТКОЙ информацией:

| № Дела | Тип | Суд | Дата | Ответчики | Суть дела | Детали |
|--------|-----|-----|------|-----------|-----------|--------|
| А40-123456/2024 | Административное | АС г. Москвы | 2024-03-15 | ООО Контрагент (ИНН: 123...) | Взыскание задолженности 500 тыс. руб. | [📄 PDF](URL) |

**Требования к таблице:**
- Номер дела из поля `number`
- Тип дела из поля `case_type` (Административное/Банкротное/Гражданское)
- Суд из поля `court`
- Дата из поля `start_date`
- Ответчики: выпиши ВСЕХ из массива `respondents` с их ИНН (формат: "Название (ИНН: ...)")
- **Суть дела: ОБЯЗАТЕЛЬНО используй поле `summary` для каждого дела**
- **Детали: Используй поле `pdf_url` для ссылки на PDF решения. Если `pdf_url` есть → [📄 PDF](pdf_url), если нет → "PDF недоступен"**

**После таблицы краткий анализ:**
- Основные контрагенты (кто чаще ответчик)
- Типы дел (какие преобладают)

#### 📋 Компания как ОТВЕТЧИК ({{courts.get('respondent_total', 0)}} дел):

Аналогичная таблица, но колонка "Ответчики" заменяется на "Истцы":

| № Дела | Тип | Суд | Дата | Истцы | Суть дела | Детали |
|--------|-----|-----|------|-------|-----------|--------|
| А41-789012/2023 | Гражданское | АС МО | 2023-11-10 | ООО Поставщик (ИНН: 456...) | Возмещение убытков 1.2 млн руб. | [📄 PDF](URL) |

**Требования:**
- **Суть дела: ОБЯЗАТЕЛЬНО используй поле `summary` для каждого дела**
- **Детали: Используй поле `pdf_url` для ссылки на PDF решения. Если `pdf_url` есть → [📄 PDF](pdf_url), если нет → "PDF недоступен"**
- Остальные поля аналогично таблице истца

**После таблицы краткий анализ:**
- Основные истцы (кто чаще подает иски)
- Типы дел (какие риски преобладают)
- Судебные риски

#### 📋 Компания как ТРЕТЬЕ ЛИЦО ({{courts.get('third_party_total', 0)}} дел):

Если есть дела (third_party_total > 0), создай аналогичную таблицу с обеими колонками:

| № Дела | Тип | Суд | Дата | Истцы | Ответчики | Суть дела | Детали |
|--------|-----|-----|------|-------|-----------|-----------|--------|

**Требования:**
- **Суть дела: ОБЯЗАТЕЛЬНО используй поле `summary` для каждого дела**
- **Детали: Используй поле `pdf_url` для ссылки на PDF решения. Если `pdf_url` есть → [📄 PDF](pdf_url), если нет → "PDF недоступен"**

**Общий анализ судебной активности:**
- Всего дел и динамика
- Наиболее активные контрагенты
- Судебные риски (если компания часто ответчик)
- Рекомендации

### 9. 📰 УПОМИНАНИЯ В СМИ

**Если есть упоминания (media не пуст), создай раздел:**

Для каждого упоминания создай короткую карточку:

**Заголовок:** [Название статьи](URL)  
**Дата публикации:** Дата (если есть)  
**Автор:** Имя автора (если есть)  
**Краткое содержание:** Используй поле `highlights` (ключевые фрагменты) или `text` (первые 300 символов)

**Требования:**
- Название статьи из поля `title` - сделай кликабельной ссылкой на `url`
- Дата из поля `published_date` 
- Автор из поля `author`
- Для краткого содержания используй:
  1. Если есть `highlights` - выведи их списком (по одному на строку)
  2. Иначе используй первые 300 символов из `text`
- Если нет данных - пропусти это поле

**Пример карточки:**

**Заголовок:** [Компания запустила новый проект](https://example.com/article)  
**Дата публикации:** 2024-11-15  
**Краткое содержание:**  
- Компания объявила о запуске нового проекта стоимостью 50 млн руб.
- Проект направлен на развитие цифровых технологий
- Планируется создание 100 новых рабочих мест

---

**После всех упоминаний добавь краткий анализ:**
- Общая тональность (позитивная/нейтральная/негативная)
- Основные темы упоминаний
- Динамика медиа-активности (если видны даты)

### 10. 🎯 КОМПЛЕКСНАЯ ОЦЕНКА

**На основе ТОЛЬКО реальных данных:**

#### Сильные стороны:
- Что хорошего видно в цифрах

#### Слабые стороны:
- Что настораживает

#### Корпоративные риски:
- Концентрация владения
- Связанные стороны
- Судебные риски

#### Рекомендации:
Конкретные, основанные на данных

**Итоговая оценка:** X/10 (обоснуй цифрами)

---

# ДАННЫЕ ДЛЯ АНАЛИЗА:
{data_summary}

---

**ВАЖНО:**
- Если видишь `null`, `None`, `[]` - пиши "Отсутствует в API"
- НЕ придумывай данные
- Диаграмма сети уже готова - вставь её как есть
- Все выводы ТОЛЬКО на основе реальных цифр
- **ОБЯЗАТЕЛЬНО используй multi_year_data из бухотчетности для построения СВОДНОЙ таблицы по всем годам**
- Значения null в таблице отображай как "-" (прочерк)
"""

    st.info("🤖 Форматирую отчет...")
    
    try:
        time.sleep(3)
        response = model.generate_content(prompt)
        report = response.text or "Ошибка генерации"
        st.success("✅ Готово")
    except Exception as e:
        report = f"Ошибка: {str(e)}"
        st.error(f"❌ {report}")
    
    return {"markdown_report": report}

# ==================== ГРАФ ====================
@st.cache_resource
def build_workflow():
    workflow = StateGraph(AgentState)
    workflow.add_node("fetch_egrul", fetch_egrul_data)
    workflow.add_node("fetch_financial", fetch_financial_data)
    workflow.add_node("download_and_parse_boh", download_and_parse_boh)
    workflow.add_node("fetch_arbitration", fetch_arbitration_data)
    workflow.add_node("fetch_media", fetch_media_mentions)
    workflow.add_node("merge_and_analyze", merge_and_analyze)

    workflow.set_entry_point("fetch_egrul")
    workflow.add_edge("fetch_egrul", "fetch_financial")
    workflow.add_edge("fetch_financial", "download_and_parse_boh")
    workflow.add_edge("download_and_parse_boh", "fetch_arbitration")
    workflow.add_edge("fetch_arbitration", "fetch_media")
    workflow.add_edge("fetch_media", "merge_and_analyze")
    workflow.add_edge("merge_and_analyze", "__end__")
    
    return workflow.compile()

# ==================== UI ====================
def main():
    st.title("📊 Комплексный анализ компании")
    st.caption("🚀 Версия 3.14 - Добавлены PDF файлы решений по делам")
    st.markdown("---")
    
    with st.sidebar:
        st.header("ℹ️ Информация о системе")
        
        # Версия системы
        st.success("**🚀 Версия:** 3.14")
        st.caption("📄 Добавлены PDF решений по делам")
        
        st.markdown("---")
        
        if PDF_LIB:
            st.success(f"✅ PDF библиотека: **{PDF_LIB}**")
        else:
            st.error("❌ Нет библиотеки для PDF")
            st.code("pip install pdfplumber")
        
        st.markdown("---")
        
        st.info("""
        **📋 Что анализируем:**
        - Корпоративные данные (ЕГРЮЛ)
        - Сеть связанных компаний
        - Финансовые показатели
        - Бухгалтерская отчетность (структурированная)
        - Судебная активность
        """)
        
        st.markdown("---")
        
        st.success("""
        **✅ Новое в v3.14:**
        - 📄 PDF файлы решений по делам
        - 🔗 Прямые ссылки на документы
        - ✅ Извлечение из API kad.arbitr
        - 🔧 v3.13: Реальные поля API
        - ✅ v3.11: Убраны все лимиты
        """)
        
        st.markdown("---")
        
        st.info("""
        **🎯 Особенности:**
        - БЕЗ заглушек данных
        - Визуализация связей компаний
        - Парсинг XML (Windows-1251) и PDF
        - Структурированные данные баланса
        - AI-аналитик для отчетности
        """)
        
        st.markdown("---")
        
        st.warning("""
        **⚠️ Лимиты API:**
        - 100 запросов/сутки
        - 1 анализ ≈ 8-10 запросов
        - ~10-12 анализов в день
        """)
        
        st.markdown("---")
        st.caption("Powered by Gemini AI")
    
    # Основной интерфейс
    col1, col2 = st.columns([3, 1])
    
    with col1:
        inn = st.text_input(
            "🔢 Введите ИНН компании:",
            placeholder="Например: 7730588444",
            help="10-значный ИНН для организаций или 12-значный для ИП"
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        analyze_button = st.button("🚀 Анализ", type="primary", use_container_width=True)
    
    if analyze_button:
        if not inn or len(inn.strip()) < 10:
            st.error("❌ Пожалуйста, введите корректный ИНН (минимум 10 цифр)")
        else:
            inn_norm = normalize_inn(inn)
            st.info(f"📌 Нормализованный ИНН: **{inn_norm}**")
            
            st.markdown("---")
            st.subheader("🔄 Процесс анализа")
            
            initial_state = {
                "inn": inn,
                "egrul_parsed": {},
                "fin_parsed": {},
                "boh_parsed": [],
                "courts_parsed": {},
                "related_companies": [],
                "markdown_report": ""
            }
            
            with st.spinner("⏳ Полный анализ займет 3-5 минут..."):
                try:
                    app = build_workflow()
                    result = app.invoke(initial_state)
                    
                    st.markdown("---")
                    st.subheader("📋 Итоговый отчет")
                    
                    # Рендерим отчет с поддержкой Mermaid диаграмм
                    report_parts = result["markdown_report"].split("```mermaid")
                    
                    st.markdown(report_parts[0])  # Часть до первой диаграммы
                    
                    # Обрабатываем каждую mermaid диаграмму
                    for i in range(1, len(report_parts)):
                        part = report_parts[i]
                        diagram_end = part.find("```")
                        
                        if diagram_end != -1:
                            # Извлекаем код диаграммы
                            diagram_code = part[:diagram_end].strip()
                            remaining_text = part[diagram_end + 3:]
                            
                            # Рендерим Mermaid через mermaid.ink API
                            try:
                                import base64
                                graphbytes = diagram_code.encode("utf8")
                                base64_bytes = base64.b64encode(graphbytes)
                                base64_string = base64_bytes.decode("ascii")
                                img_url = f"https://mermaid.ink/img/{base64_string}"
                                
                                st.image(img_url, caption="Сеть связанных компаний", use_container_width=True)
                            except Exception as e:
                                # Fallback: показываем код
                                st.code(diagram_code, language="mermaid")
                                st.warning(f"Не удалось отрендерить диаграмму: {e}")
                            
                            # Оставшийся текст после диаграммы
                            st.markdown(remaining_text)
                        else:
                            st.markdown(part)
                    
                    # Дополнительная информация
                    st.markdown("---")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.download_button(
                            "💾 Скачать отчет (.md)",
                            result["markdown_report"].encode('utf-8'),
                            f"report_{inn_norm}.md",
                            "text/markdown",
                            use_container_width=True
                        )
                    
                    with col2:
                        st.link_button(
                            "📊 Бухотчетность на сайте",
                            f"{BASE_BOH_URL}ls.php?inn={inn_norm}",
                            use_container_width=True
                        )
                    
                    with col3:
                        # Скачать сырые данные
                        raw_data = {
                            "egrul": result.get("egrul_parsed", {}),
                            "fin": result.get("fin_parsed", {}),
                            "boh": result.get("boh_parsed", []),
                            "courts": result.get("courts_parsed", {}),
                            "related": result.get("related_companies", [])
                        }
                        st.download_button(
                            "📄 Скачать JSON",
                            json.dumps(raw_data, ensure_ascii=False, indent=2).encode('utf-8'),
                            f"data_{inn_norm}.json",
                            "application/json",
                            use_container_width=True
                        )
                    
                    # Статистика
                    st.markdown("---")
                    st.subheader("📈 Статистика анализа")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        related_count = len(result.get("related_companies", []))
                        st.metric("Связанные компании", related_count)
                    
                    with col2:
                        boh_success = len([f for f in result.get("boh_parsed", []) if f.get("success")])
                        boh_total = len(result.get("boh_parsed", []))
                        st.metric("Файлы бухотчетности", f"{boh_success}/{boh_total}")
                    
                    with col3:
                        courts_total = result.get("courts_parsed", {}).get("total", 0)
                        st.metric("Судебных дел", courts_total)
                    
                    with col4:
                        fin_records = len(result.get("fin_parsed", {}).get("income_expenses", []))
                        st.metric("Финансовых записей", fin_records)
                    
                except Exception as e:
                    st.error(f"❌ Ошибка выполнения анализа: {str(e)}")
                    st.exception(e)
    
    # Футер
    st.markdown("---")
    st.caption("⚠️ Источники данных: egrul.itsoft.ru, parser-api.com | Версия 3.0 с правильным парсингом бухотчетности")

if __name__ == "__main__":
    main()