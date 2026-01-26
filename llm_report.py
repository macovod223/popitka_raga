# Модуль для генерации структурированных отчётов через локальную LLM по HTTP API
# Модель/рантайм могут быть любыми, если сервер поддерживает совместимый endpoint.

import json
import logging
import os
import re
import time
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from rag_core.constants import PROJECT_ROOT

logger = logging.getLogger("llm_report")

# Параметры по умолчанию для локального HTTP API
DEFAULT_LLM_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "qwen"
DEFAULT_TIMEOUT = 120
DEFAULT_GEN_PARAMS = {
    "temperature": 0.0,
    "top_p": 1.0,
    "num_predict": 100,
    "num_ctx": 2048,
}

_TRANSFORMERS_CACHE: Dict[str, Any] = {}
_MODEL_ALIASES: Dict[str, str] = {
    "qwen": "Qwen/Qwen2.5-0.5B-Instruct",
    "falcon": "tiiuae/Falcon3-1B-Instruct",
}

def _resolve_local_snapshot(model_name: str) -> Optional[str]:
    if not model_name: return None
    candidate = Path(model_name)
    if candidate.exists(): return str(candidate)
    
    hub_root = os.getenv("HF_HOME")
    if hub_root: hub_root = Path(hub_root) / "hub"
    else: hub_root = Path.home() / ".cache" / "huggingface" / "hub"
    
    model_dir_name = f"models--{model_name.replace('/', '--')}"
    model_dir = hub_root / model_dir_name
    
    snapshots_dir = model_dir / "snapshots"
    if not snapshots_dir.exists():
        model_dir = hub_root / model_name.replace('/', '--')
        snapshots_dir = model_dir / "snapshots"
        if not snapshots_dir.exists(): return None
        
    try:
        snapshots = sorted([p for p in snapshots_dir.iterdir() if p.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True)
        for snap in snapshots:
            if (snap / "config.json").exists() or any(snap.glob("*.safetensors")) or any(snap.glob("*.bin")):
                return str(snap)
    except Exception:
        pass
    return None

def strip_emojis(text: str) -> str:
    if not text: return text
    out_chars = []
    for ch in text:
        cat = unicodedata.category(ch)
        code = ord(ch)
        if cat in ("So", "Sk") or (0x1F300 <= code <= 0x1FAFF) or (0x2600 <= code <= 0x27BF): continue
        out_chars.append(ch)
    return "".join(out_chars)

def clean_llm_analysis(text: str) -> str:
    if not text: return text
    
    text = strip_emojis(text).strip()
    text = re.sub(r"(?im)^\s*<\|[^>]+\|>\s*$", "", text).strip()

    # Исправление лексики и галлюцинаций
    corrections = {
        r"(?i)аномалия": "аналоги",
        r"(?i)аналоги[яие]": "аналоги",
        r"(?i)видеопраммата": "видеопамяти",
        r"(?i)оператiva": "оперативной",
        r"(?i)оператива": "оперативная",
        r"(?i)охлатительного": "охлаждения",
        r"(?i)охлатывания": "охлаждения",
        r"(?i)навигатора": "накопителя",
        r"(?i)быстрохолдементный": "быстрый",
        r"(?i)памятного места": "памяти",
        r"(?i)ограниченными возможностями": "высокими требованиями",
        r"(?i)unlocked": "разблокирован",
        r"(?i)унилоке": "разблокирован",
    }
    
    for pattern, replacement in corrections.items():
        text = re.sub(pattern, replacement, text)
        
    # Удаление блоков на восточных языках (китайский, японский, корейский)
    text = re.sub(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]+", "", text)
    
    # Принудительное форматирование заголовков
    required_headers = [
        ("КРАТКИЙ ВЫВОД:", ["КРАТКИЙ ВЫВОД", "ВЫВОД"]),
        ("ПОЧЕМУ ЭТО АНАЛОГИ:", ["ПОЧЕМУ ЭТО АНАЛОГИ", "ПОЧЕМУ"]),
        ("КЛЮЧЕВЫЕ ОТЛИЧИЯ:", ["КЛЮЧЕВЫЕ ОТЛИЧИЯ", "ОТЛИЧИЯ"]),
        ("РИСКИ / НЕУВЕРЕННОСТЬ:", ["РИСКИ", "НЕУВЕРЕННОСТЬ"]),
    ]
    
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    final_lines = []
    
    # Если в тексте вообще нет заголовков, но есть строки
    has_any_header = any(h[0].upper() in text.upper() for h in required_headers)
    
    if not has_any_header and len(lines) > 0:
        for i, line in enumerate(lines[:len(required_headers)]):
            final_lines.append(f"{required_headers[i][0]} {line}")
        text = "\n".join(final_lines)
    else:
        # Пытаемся нормализовать существующие заголовки
        current_header_idx = -1
        for line in lines:
            found_header = False
            for i, (header, aliases) in enumerate(required_headers):
                for alias in aliases:
                    if line.upper().startswith(alias.upper()):
                        # Чистим строку от старого заголовка
                        content = re.sub(rf"^{alias}.*?:\s*", "", line, flags=re.I).strip()
                        final_lines.append(f"{header} {content}")
                        current_header_idx = i
                        found_header = True
                        break
                if found_header: break
            
            if not found_header and current_header_idx != -1:
                # Добавляем строку к текущему разделу
                final_lines[-1] += " " + line

        text = "\n".join(final_lines)

    # Очистка от мусора в конце
    junk_markers = ["ДАННЫЕ:", "ОТЧЕТ:", "###"]
    for marker in junk_markers:
        idx = text.find(marker)
        if idx != -1: text = text[:idx].strip()

    return text.strip()

def local_llm_generate(prompt: str, model: str = DEFAULT_MODEL, llm_url: str = DEFAULT_LLM_URL, timeout: int = DEFAULT_TIMEOUT, run_id: str = "llm", **gen_params: Any) -> Optional[str]:
    params = {**DEFAULT_GEN_PARAMS, **gen_params}
    params["model"], params["prompt"], params["stream"] = model, prompt, False
    try:
        response = requests.post(llm_url, json=params, timeout=timeout)
        response.raise_for_status()
        text = response.json().get("response", "").strip()
        return text if text else None
    except Exception as e:
        logger.error(f"Ошибка LLM API: {e}")
        return None

def transformers_generate(prompt: str, model: str, *, temperature: float = 0.0, top_p: float = 1.0, num_predict: int = 100, num_ctx: int = 2048, allow_download: bool = False, run_id: str = "hf") -> Optional[str]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except: return None
    
    resolved_model = _MODEL_ALIASES.get(model, model)
    if not allow_download:
        snap = _resolve_local_snapshot(resolved_model)
        if snap: resolved_model = snap
    
    try:
        if resolved_model not in _TRANSFORMERS_CACHE:
            dtype = torch.float32
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                dtype = torch.float16
            
            tok = AutoTokenizer.from_pretrained(resolved_model, use_fast=True, local_files_only=not allow_download)
            mdl = AutoModelForCausalLM.from_pretrained(resolved_model, dtype=dtype, low_cpu_mem_usage=True, local_files_only=not allow_download)
            mdl.to("cpu").eval()
            _TRANSFORMERS_CACHE[resolved_model] = (tok, mdl)
        else: tok, mdl = _TRANSFORMERS_CACHE[resolved_model]
        
        max_input = max(32, int(num_ctx - num_predict - 32))
        try: tok.truncation_side = "left"
        except: pass
        
        enc = tok(prompt, return_tensors="pt", truncation=True, max_length=max_input)
        
        do_sample = temperature > 0.0
        gen_kwargs = {
            "max_new_tokens": num_predict, 
            "do_sample": do_sample, 
            "temperature": temperature if do_sample else None,
            "top_p": top_p if do_sample else None,
            "repetition_penalty": 1.3,
            "no_repeat_ngram_size": 3,
            "pad_token_id": tok.eos_token_id,
            "eos_token_id": tok.eos_token_id
        }
        gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}

        with torch.no_grad(): out = mdl.generate(**enc, **gen_kwargs)
        text = tok.decode(out[0][enc["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
        return text if text else None
    except Exception as e:
        logger.error(f"Ошибка transformers ({model}): {e}")
        return None

def _truncate_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars: return text
    return text[:max_chars-3] + "..." if max_chars > 3 else text[:max_chars]

def build_report_prompt(query_item: Dict[str, Any], analog_items: List[tuple], run_meta: Optional[Dict[str, Any]] = None, max_chars: int = 800, adaptive_limit: bool = True, num_ctx: int = 2048, num_predict: int = 100, language: str = "ru") -> str:
    query_title = _truncate_text(query_item.get("title", "N/A"), 80)
    
    prompt = f"""ЗАДАЧА: Напиши отчет о товаре и аналогах на РУССКОМ ЯЗЫКЕ.
Используй ровно 4 раздела.

ТОВАР: {query_title}
АНАЛОГИ:
"""
    for i, a in enumerate(analog_items[:2], 1):
        cand = a[2]
        title = _truncate_text(cand.get("title", ""), 60)
        prompt += f"- {title}\n"
    
    prompt += """
ОТЧЕТ:
КРАТКИЙ ВЫВОД:
"""
    return prompt

def save_report(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        text = strip_emojis(text)
        with open(path, "w", encoding="utf-8") as f: f.write(text)
        logger.info(f"Отчёт сохранён: {path}")
    except Exception as e: logger.error(f"Ошибка сохранения: {e}")
