ПРОЕКТ: ПОИСК АНАЛОГОВ ТОВАРОВ

Цель: Найти похожие товары на основе текстового описания, сохраняя суть (категорию).


1. Установка:
   source .venv/bin/activate
   python3 -m pip install -r requirements_comparison.txt rank-bm25 faiss-cpu

2. Запуск (минимальная команда):
   python3 rag_prob.py --disable-generation
   # Использует значения по умолчанию: индексы 0,10,20, top-k=5, модель по умолчанию

БЕЗ LLM (быстрее, fallback отчёт, больше данных):
   python3 rag_prob.py --indices 0,60,121,182,243,304,365,426,487,548,609,670,731,792,853 --top-k 5 --disable-generation

С LLM (анализ через google/flan-t5-base):
   python3 rag_prob.py --indices 0,60,121,182,243,304,365,426,487,548,609,670,731,792,853 --top-k 5
   # Примечание: LLM генерация может занять время (несколько секунд на товар)

С ДРУГОЙ LLM МОДЕЛЬЮ:
   python3 rag_prob.py --indices 0,60,121,182,243,304,365,426,487,548,609,670,731,792,853 --top-k 5 --gen-model google/flan-t5-large

РЕАЛИЗОВАНО

- RAG-подобный подход (rag_prob.py) - основной, использует эмбеддинги
  примечание: используется RAG-подобная архитектура, где Generation - это генерация
  аналитического отчёта о результатах поиска, а не генерация ответа пользователю
- BM25 baseline (baseline_bm25.py) - для сравнения
- TF-IDF baseline (baseline_tfidf.py) - для сравнения
- FAISS векторная БД (vector_db_faiss.py) - для больших датасетов
- Гибридный подход BM25+RAG (hybrid_bm25_rag.py) - флагманский, комбинирует лучшие стороны
  sparse retrieval (BM25) и dense retrieval (RAG) через Reciprocal Rank Fusion (RRF)
- Сравнение всех подходов (compare_approaches.py)

ПРИМЕРЫ ЗАПУСКА

1. RAG-подобный подход (rag_prob.py):
   # БЕЗ LLM (быстро, fallback отчёт):
   python3 rag_prob.py --indices 0,60,121,182,243,304,365,426,487,548,609,670,731,792,853 --top-k 5 --disable-generation
   
   # С LLM (медленнее, но с анализом через google/flan-t5-base):
   python3 rag_prob.py --indices 0,60,121,182,243,304,365,426,487,548,609,670,731,792,853 --top-k 5
   
   # С другой моделью эмбеддингов (по умолчанию: sentence-transformers/paraphrase-multilingual-mpnet-base-v2):
   python3 rag_prob.py --indices 0,60,121,182,243,304,365,426,487,548,609,670,731,792,853 --top-k 5 --embedding-model intfloat/multilingual-e5-base --disable-generation
   python3 rag_prob.py --indices 0,60,121,182,243,304,365,426,487,548,609,670,731,792,853 --top-k 5 --embedding-model intfloat/multilingual-e5-base
   # Другие флагманские модели:
   python3 rag_prob.py --indices 0,60,121,182,243,304,365,426,487,548,609,670,731,792,853 --top-k 5 --embedding-model intfloat/multilingual-e5-large-instruct --disable-generation

2. BM25 baseline (baseline_bm25.py):
   python3 baseline_bm25.py --indices 0,60,121,182,243,304,365,426,487,548,609,670,731,792,853 --top-k 5
   python3 baseline_bm25.py --indices 0,60,121,182,243,304,365,426,487,548,609,670,731,792,853 --top-k 5 --k1 2.0 --b 0.75

3. TF-IDF baseline (baseline_tfidf.py):
   python3 baseline_tfidf.py --indices 0,60,121,182,243,304,365,426,487,548,609,670,731,792,853 --top-k 5
   python3 baseline_tfidf.py --indices 0,60,121,182,243,304,365,426,487,548,609,670,731,792,853 --top-k 5 --max-features 5000 --ngram-range 1,2

4. FAISS векторная БД (vector_db_faiss.py):
   python3 vector_db_faiss.py --indices 0,60,121,182,243,304,365,426,487,548,609,670,731,792,853 --top-k 5 --index-type Flat
   python3 vector_db_faiss.py --indices 0,60,121,182,243,304,365,426,487,548,609,670,731,792,853 --top-k 5 --index-type Flat --embedding-model intfloat/multilingual-e5-base

5. Гибридный подход BM25+RAG (hybrid_bm25_rag.py):
   # С RRF (Reciprocal Rank Fusion) - рекомендуется:
   python3 hybrid_bm25_rag.py --indices 0,60,121,182,243,304,365,426,487,548,609,670,731,792,853 --top-k 5
   
   # С взвешенной суммой вместо RRF:
   python3 hybrid_bm25_rag.py --indices 0,60,121,182,243,304,365,426,487,548,609,670,731,792,853 --top-k 5 --no-rrf --bm25-weight 0.3 --rag-weight 0.7
   
   # С другой моделью эмбеддингов:
   python3 hybrid_bm25_rag.py --indices 0,60,121,182,243,304,365,426,487,548,609,670,731,792,853 --top-k 5 --embedding-model intfloat/multilingual-e5-base

6. Сравнение всех подходов (compare_approaches.py):
   # БЕЗ кросс-категорий (по умолчанию, сохранение сути):
   python3 compare_approaches.py --indices 0,60,121,182,243,304,365,426,487,548,609,670,731,792,853 --top-k 5
   
   # С кросс-категориями (для экспериментов):
   python3 compare_approaches.py --indices 0,60,121,182,243,304,365,426,487,548,609,670,731,792,853 --top-k 5 --allow-cross-category
   
   # Дополнительные опции:
   python3 compare_approaches.py --indices 0,60,121,182,243,304,365,426,487,548,609,670,731,792,853 --top-k 5 --skip-rag
   python3 compare_approaches.py --indices 0,60,121,182,243,304,365,426,487,548,609,670,731,792,853 --top-k 5 --embedding-model intfloat/multilingual-e5-base

ПАРАМЕТРЫ

--indices 0,10,20          Индексы товаров для поиска (15 индексов равномерно распределены)
                           По умолчанию: 0,10,20 (для быстрого теста)
--top-k 5                  Количество аналогов (по умолчанию: 5)
--disable-generation       Отключить LLM генерацию (быстрее, использует fallback отчёт)
--gen-model NAME           LLM модель для генерации отчёта (по умолчанию: google/flan-t5-base)
--allow-cross-category     Разрешить аналоги из других категорий (для экспериментов)
                           По умолчанию: фильтр по категории включен (сохранение сути)
--embedding-model NAME     Модель эмбеддингов (по умолчанию: sentence-transformers/paraphrase-multilingual-mpnet-base-v2)
                           Рекомендуемые флагманские модели:
                           - sentence-transformers/paraphrase-multilingual-mpnet-base-v2 (по умолчанию, популярная)
                           - intfloat/multilingual-e5-base (хорошая производительность)
                           - intfloat/multilingual-e5-large-instruct (лучшая по MMTEB, но медленнее)

СОХРАНЕНИЕ СУТИ (категория)

Реализация использует фильтр по категории - все найденные аналоги гарантированно из той же
категории. Категория является первым и основным критерием "сути" товара.

Для экспериментов можно использовать --allow-cross-category, чтобы разрешить аналоги из других категорий.

ГЕНЕРАЦИЯ ОТЧЁТА

Без LLM (--disable-generation):
- Быстрый запуск, не требует загрузки LLM
- Fallback отчёт с анализом метрик и выводами
- Показывает: количество аналогов, средний скор, совпадение категорий/брендов, оценку качества

С LLM (без --disable-generation):
- Использует google/flan-t5-base для генерации аналитического отчёта
- Генерирует текстовое заключение на русском языке
- Анализирует сохранение сути, сходства/различия, числовые характеристики
- примечание: generation используется для генерации отчёта о результатах поиска,
  а не для генерации ответа пользователю или нормализации результата

РЕЗУЛЬТАТЫ

После запуска создаются файлы *_results.txt и *_comparison.txt в директории comparison_results/
Результаты сравнения всех подходов сохраняются в разных файлах в зависимости от режима:
- comparison_results/comparison_results.txt - результаты БЕЗ кросс-категорий (allow_cross_category=False)
- comparison_results/comparison_results_cross_category.txt - результаты С кросс-категориями (allow_cross_category=True)
Графики производительности сохраняются в comparison_results/performance_comparison.png

РЕЗУЛЬТАТЫ СРАВНЕНИЯ (без кросс-категорий, allow_cross_category=False):
- Hybrid BM25+RAG: средний скор 0.815, совпадение категорий 100.0% (ФЛАГМАНСКИЙ)
- RAG (Dense Retrieval): средний скор 0.788, совпадение категорий 100.0%
- FAISS: средний скор 0.635, совпадение категорий 100.0%
- BM25: средний скор 0.618, совпадение категорий 100.0%
- TF-IDF: средний скор 0.540, совпадение категорий 100.0%

РЕЗУЛЬТАТЫ СРАВНЕНИЯ (с кросс-категориями, allow_cross_category=True):
- Hybrid BM25+RAG: средний скор 0.820, совпадение категорий 92.0% (ФЛАГМАНСКИЙ)
- RAG (Dense Retrieval): средний скор 0.794, совпадение категорий 86.7%
- FAISS: средний скор 0.662, совпадение категорий 86.7%
- BM25: средний скор 0.625, совпадение категорий 94.7%
- TF-IDF: средний скор 0.547, совпадение категорий 90.7%
Примечание: при разрешении кросс-категорий средний скор может быть выше, но сохранение сути снижается

Лучший подход: Hybrid BM25+RAG - лучший средний скор (0.815 без кросс-категорий, 0.820 с кросс-категориями)
и 100% сохранение сути (без кросс-категорий). Флагманский гибридный подход объединяет лучшие стороны
sparse retrieval (BM25) и dense retrieval (RAG) через комбинацию Reciprocal Rank Fusion (RRF) и
взвешенной суммы нормализованных скоров, обеспечивая максимальное качество при сохранении
преимуществ обоих методов

