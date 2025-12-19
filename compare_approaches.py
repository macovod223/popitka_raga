# сравнительный скрипт для оценки различных подходов к поиску аналогов
# этот скрипт запускает все доступные методы поиска (RAG, BM25, TF-IDF, FAISS, BM25 + RAG)
# на одних и тех же данных и сравнивает результаты по метрикам
# 
# Использование:
#     python compare_approaches.py --indices 0,10,20 --top-k 5
# 
# Результаты сохраняются в:
#     - comparison_results.txt - детальное сравнение
#     - comparison_summary.txt - краткая сводка

import argparse
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None

from rag_prob import DEFAULT_DATASET, DEFAULT_INDICES, DEFAULT_TOP_K, parse_example_indices

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("comparison")


@dataclass
class ComparisonConfig:
    dataset: Path = DEFAULT_DATASET
    indices: tuple = DEFAULT_INDICES
    top_k: int = DEFAULT_TOP_K
    allow_cross_category: bool = False
    embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2"
    output_dir: Path = Path("comparison_results")
    run_rag: bool = True
    run_bm25: bool = True
    run_tfidf: bool = True
    run_faiss: bool = True
    run_hybrid: bool = True


class ApproachRunner:
    # класс для запуска различных подходов и сбора результатов

    def __init__(self, config: ComparisonConfig):
        self.config = config
        self.output_dir = config.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run_rag(self) -> bool:
        # запуск RAG подхода (dense retrieval)
        if not self.config.run_rag:
            logger.info("Пропуск RAG (отключён)")
            return False

        logger.info("")
        logger.info("[1/5] запуск rag подхода (dense retrieval)")
        start_time = time.time()

        try:
            indices_str = ",".join(str(i) for i in self.config.indices)
            output_file = self.output_dir / "rag_comparison.txt"

            cmd = [
                sys.executable,
                "rag_prob.py",
                "--dataset",
                str(self.config.dataset),
                "--embedding-model",
                self.config.embedding_model,
                "--indices",
                indices_str,
                "--top-k",
                str(self.config.top_k),
                "--disable-generation",  # Отключаем генерацию для скорости
                "--output-file",
                str(output_file),
            ]

            if self.config.allow_cross_category:
                cmd.append("--allow-cross-category")

            env = os.environ.copy()
            result = subprocess.run(cmd, capture_output=True, text=True, check=True, env=env)
            elapsed = time.time() - start_time
            logger.info("")
            logger.info(f"RAG выполнен успешно за {elapsed:.1f} сек")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Ошибка при выполнении RAG: {e}")
            logger.error(f"Stderr: {e.stderr}")
            return False
        except FileNotFoundError:
            logger.error("Файл rag_prob.py не найден")
            return False

    def run_bm25(self) -> bool:
        # запуск BM25 baseline
        if not self.config.run_bm25:
            logger.info("Пропуск BM25 (отключён)")
            return False

        logger.info("")
        logger.info("[2/5] запуск bm25 baseline")
        start_time = time.time()

        try:
            indices_str = ",".join(str(i) for i in self.config.indices)
            output_file = self.output_dir / "bm25_comparison.txt"

            cmd = [
                sys.executable,
                "baseline_bm25.py",
                "--dataset",
                str(self.config.dataset),
                "--indices",
                indices_str,
                "--top-k",
                str(self.config.top_k),
                "--output-file",
                str(output_file),
            ]

            if self.config.allow_cross_category:
                cmd.append("--allow-cross-category")

            env = os.environ.copy()
            result = subprocess.run(cmd, capture_output=True, text=True, check=True, env=env)
            elapsed = time.time() - start_time
            logger.info("")
            logger.info(f"BM25 выполнен успешно за {elapsed:.1f} сек")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Ошибка при выполнении BM25: {e}")
            logger.error(f"Stderr: {e.stderr}")
            return False
        except FileNotFoundError:
            logger.error("Файл baseline_bm25.py не найден")
            return False

    def run_tfidf(self) -> bool:
        # запуск TF-IDF baseline
        if not self.config.run_tfidf:
            logger.info("Пропуск TF-IDF (отключён)")
            return False

        logger.info("")
        logger.info("[3/5] запуск tf-idf baseline")
        start_time = time.time()

        try:
            indices_str = ",".join(str(i) for i in self.config.indices)
            output_file = self.output_dir / "tfidf_comparison.txt"

            cmd = [
                sys.executable,
                "baseline_tfidf.py",
                "--dataset",
                str(self.config.dataset),
                "--indices",
                indices_str,
                "--top-k",
                str(self.config.top_k),
                "--output-file",
                str(output_file),
            ]

            if self.config.allow_cross_category:
                cmd.append("--allow-cross-category")

            env = os.environ.copy()
            result = subprocess.run(cmd, capture_output=True, text=True, check=True, env=env)
            elapsed = time.time() - start_time
            logger.info("")
            logger.info(f"TF-IDF выполнен успешно за {elapsed:.1f} сек")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Ошибка при выполнении TF-IDF: {e}")
            logger.error(f"Stderr: {e.stderr}")
            return False
        except FileNotFoundError:
            logger.error("Файл baseline_tfidf.py не найден")
            return False

    def run_faiss(self) -> bool:
        # запуск FAISS векторной БД
        if not self.config.run_faiss:
            logger.info("Пропуск FAISS (отключён)")
            return False

        logger.info("")
        logger.info("[4/5] запуск faiss vector db")
        start_time = time.time()

        try:
            indices_str = ",".join(str(i) for i in self.config.indices)
            output_file = self.output_dir / "faiss_comparison.txt"

            cmd = [
                sys.executable,
                "vector_db_faiss.py",
                "--dataset",
                str(self.config.dataset),
                "--embedding-model",
                self.config.embedding_model,
                "--indices",
                indices_str,
                "--top-k",
                str(self.config.top_k),
                "--output-file",
                str(output_file),
                "--index-type",
                "Flat",  # Используем Flat для стабильности (HNSW может падать на macOS)
            ]

            if self.config.allow_cross_category:
                cmd.append("--allow-cross-category")

            # отключаем mps для faiss (избегаем sigsegv на macos)
            env = os.environ.copy()
            env['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True, env=env)
            elapsed = time.time() - start_time
            logger.info("")
            logger.info(f"FAISS выполнен успешно за {elapsed:.1f} сек")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Ошибка при выполнении FAISS: {e}")
            logger.error(f"Stderr: {e.stderr}")
            return False
        except FileNotFoundError:
            logger.error("Файл vector_db_faiss.py не найден")
            return False

    def run_hybrid(self) -> bool:
        # запуск гибридного подхода BM25+RAG
        if not self.config.run_hybrid:
            logger.info("Пропуск Hybrid BM25+RAG (отключён)")
            return False

        logger.info("")
        logger.info("[5/5] запуск гибридного подхода BM25+RAG")
        start_time = time.time()

        try:
            indices_str = ",".join(str(i) for i in self.config.indices)
            output_file = self.output_dir / "hybrid_bm25_rag_comparison.txt"

            cmd = [
                sys.executable,
                "hybrid_bm25_rag.py",
                "--dataset",
                str(self.config.dataset),
                "--embedding-model",
                self.config.embedding_model,
                "--indices",
                indices_str,
                "--top-k",
                str(self.config.top_k),
                "--output-file",
                str(output_file),
            ]

            if self.config.allow_cross_category:
                cmd.append("--allow-cross-category")

            env = os.environ.copy()
            result = subprocess.run(cmd, capture_output=True, text=True, check=True, env=env)
            elapsed = time.time() - start_time
            logger.info("")
            logger.info(f"Hybrid BM25+RAG выполнен успешно за {elapsed:.1f} сек")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Ошибка при выполнении Hybrid BM25+RAG: {e}")
            logger.error(f"Stderr: {e.stderr}")
            return False
        except FileNotFoundError:
            logger.error("Файл hybrid_bm25_rag.py не найден")
            return False

    def parse_results_file(self, file_path: Path) -> Optional[dict]:
        # парсинг файла результатов для извлечения метрик
        # формат файла:
        # method run @ timestamp
        # Запросов: X | Всего аналогов: Y | Средний скор: Z
        # Совпадение категорий: A/B (C%)
        if not file_path.exists():
            return None

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # ищем общую статистику
            lines = content.split("\n")
            stats = {}
            for line in lines:
                if "Запросов:" in line:
                    # извлекаем количество запросов и средний скор
                    parts = line.split("|")
                    for part in parts:
                        if "Запросов:" in part:
                            stats["total_queries"] = int(
                                part.split(":")[1].strip().split()[0]
                            )
                        elif "Средний скор:" in part:
                            stats["avg_score"] = float(
                                part.split(":")[1].strip().split()[0]
                            )
                elif "Совпадение категорий:" in line:
                    # извлекаем процент совпадения категорий
                    if "/" in line and "(" in line:
                        parts = line.split("(")
                        if len(parts) > 1:
                            percent_str = parts[1].split("%")[0]
                            stats["category_match_percent"] = float(percent_str)

            return stats if stats else None
        except Exception as e:
            logger.warning(f"Не удалось распарсить {file_path}: {e}")
            return None

    def compare_results(self) -> None:
        # сравнение результатов всех подходов
        logger.info("\nсравнение результатов")

        approaches = {
            "RAG (Dense Retrieval)": self.output_dir / "rag_comparison.txt",
            "BM25": self.output_dir / "bm25_comparison.txt",
            "TF-IDF": self.output_dir / "tfidf_comparison.txt",
            "FAISS": self.output_dir / "faiss_comparison.txt",
            "Hybrid BM25+RAG": self.output_dir / "hybrid_bm25_rag_comparison.txt",
        }

        results = {}
        for approach_name, file_path in approaches.items():
            stats = self.parse_results_file(file_path)
            if stats:
                results[approach_name] = stats
                logger.info(f"\n{approach_name}:")
                logger.info(f"  Средний скор: {stats.get('avg_score', 0):.3f}")
                logger.info(
                    f"  Совпадение категорий: {stats.get('category_match_percent', 0):.1f}%"
                )
            else:
                logger.warning(f"Не удалось загрузить результаты для {approach_name}")

        # сохраняем сравнение в файл
        self._write_comparison(results)

    def _write_comparison(self, results: Dict[str, dict]) -> None:
        # запись результатов сравнения в файлы
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # детальное сравнение
        # сохраняем в разные файлы в зависимости от режима кросс-категорий
        if self.config.allow_cross_category:
            comparison_file = self.output_dir / "comparison_results_cross_category.txt"
        else:
            comparison_file = self.output_dir / "comparison_results.txt"
        lines = []
        lines.append(f"сравнение подходов @ {timestamp}")
        lines.append(f"Датасет: {self.config.dataset}")
        lines.append(f"Индексы запросов: {self.config.indices}")
        lines.append(f"Top-K: {self.config.top_k}")
        lines.append(f"Разрешить кросс-категории: {self.config.allow_cross_category}")
        lines.append("")

        # таблица сравнения
        lines.append("метрики:")
        lines.append(
            f"{'Подход':<25} {'Средний скор':<15} {'Категории %':<15}"
        )

        for approach_name, stats in results.items():
            avg_score = stats.get("avg_score", 0)
            cat_match = stats.get("category_match_percent", 0)
            lines.append(
                f"{approach_name:<25} {avg_score:<15.3f} {cat_match:<15.1f}"
            )

        lines.append("")

        with comparison_file.open("w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        logger.info(f"\nРезультаты сравнения сохранены в {comparison_file}")
        
        # создаём графики производительности
        self._create_performance_plots(results)
    
    def _create_performance_plots(self, results: Dict[str, dict]) -> None:
        # создание графиков производительности всех моделей
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("matplotlib не установлен, графики не будут созданы")
            logger.info("установите: pip install matplotlib")
            return
        
        if not results:
            logger.warning("нет данных для графиков")
            return
        
        try:
            # подготавливаем данные
            approaches = list(results.keys())
            avg_scores = [results[a].get("avg_score", 0) for a in approaches]
            cat_matches = [results[a].get("category_match_percent", 0) for a in approaches]
            
            # создаём фигуру с несколькими графиками
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            fig.suptitle("сравнение производительности моделей", fontsize=16, fontweight='bold')
            
            # график 1: средние скоры
            ax1 = axes[0]
            bars1 = ax1.bar(approaches, avg_scores, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
            ax1.set_title("средний скор по моделям", fontsize=12)
            ax1.set_ylabel("скор", fontsize=10)
            ax1.set_xticks(range(len(approaches)))
            ax1.set_xticklabels(approaches, rotation=15, ha='right', fontsize=9)
            ax1.grid(axis='y', alpha=0.3)
            for i, (bar, score) in enumerate(zip(bars1, avg_scores)):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{score:.3f}', ha='center', va='bottom', fontsize=8)
            
            # график 2: совпадение категорий
            ax2 = axes[1]
            bars2 = ax2.bar(approaches, cat_matches, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
            ax2.set_title("совпадение категорий (%)", fontsize=12)
            ax2.set_ylabel("процент", fontsize=10)
            ax2.set_xticks(range(len(approaches)))
            ax2.set_xticklabels(approaches, rotation=15, ha='right', fontsize=9)
            ax2.set_ylim([0, 105])
            ax2.grid(axis='y', alpha=0.3)
            for i, (bar, match) in enumerate(zip(bars2, cat_matches)):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{match:.1f}%', ha='center', va='bottom', fontsize=8)
            
            plt.tight_layout()
            
            # сохраняем график
            plot_file = self.output_dir / "performance_comparison.png"
            plt.savefig(plot_file, dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"графики производительности сохранены в {plot_file}")
        except Exception as e:
            logger.warning(f"не удалось создать графики: {e}")

    def run_all(self) -> None:
        # запуск всех подходов и сравнение результатов
        total_start = time.time()
        logger.info("сравнительный анализ подходов к поиску аналогов")
        logger.info(f"Датасет: {self.config.dataset}")
        logger.info(f"Индексы: {self.config.indices}")
        logger.info(f"Top-K: {self.config.top_k}")
        logger.info("")

        # запускаем все подходы
        self.run_rag()
        self.run_bm25()
        self.run_tfidf()
        self.run_faiss()
        self.run_hybrid()

        # сравниваем результаты
        logger.info("")
        logger.info("сравнение результатов")
        t_compare = time.time()
        self.compare_results()
        compare_time = time.time() - t_compare
        
        total_elapsed = time.time() - total_start
        logger.info("")
        logger.info(f"сравнение завершено")
        logger.info(f"  время сравнения: {compare_time:.1f} сек")
        logger.info(f"  общее время: {total_elapsed:.1f} сек")


def parse_args() -> argparse.Namespace:
    # парсинг аргументов командной строки
    parser = argparse.ArgumentParser(
        description="Сравнение различных подходов к поиску аналогов товаров."
    )
    parser.add_argument(
        "--dataset", default=str(DEFAULT_DATASET), help="Путь к CSV датасету"
    )
    parser.add_argument(
        "--indices",
        default=",".join(str(i) for i in DEFAULT_INDICES),
        help="Список индексов через запятую",
    )
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Количество аналогов")
    parser.add_argument(
        "--allow-cross-category",
        action="store_true",
        help="Разрешить аналоги из других категорий",
    )
    parser.add_argument(
        "--embedding-model",
        default="paraphrase-multilingual-MiniLM-L12-v2",
        help="Модель эмбеддингов для RAG и FAISS",
    )
    parser.add_argument(
        "--output-dir",
        default="comparison_results",
        help="Директория для сохранения результатов",
    )
    parser.add_argument(
        "--skip-rag", action="store_true", help="Пропустить RAG подход"
    )
    parser.add_argument(
        "--skip-bm25", action="store_true", help="Пропустить BM25 baseline"
    )
    parser.add_argument(
        "--skip-tfidf", action="store_true", help="Пропустить TF-IDF baseline"
    )
    parser.add_argument(
        "--skip-faiss", action="store_true", help="Пропустить FAISS"
    )
    parser.add_argument(
        "--skip-hybrid", action="store_true", help="Пропустить Hybrid BM25+RAG"
    )
    return parser.parse_args()


def build_config_from_cli(args: argparse.Namespace) -> ComparisonConfig:
    # создание конфигурации из аргументов CLI
    indices = parse_example_indices(args.indices)
    if args.top_k <= 0:
        raise ValueError("top_k должен быть больше нуля")

    return ComparisonConfig(
        dataset=Path(args.dataset),
        indices=tuple(indices),
        top_k=args.top_k,
        allow_cross_category=args.allow_cross_category,
        embedding_model=args.embedding_model,
        output_dir=Path(args.output_dir),
        run_rag=not args.skip_rag,
        run_bm25=not args.skip_bm25,
        run_tfidf=not args.skip_tfidf,
        run_faiss=not args.skip_faiss,
        run_hybrid=not args.skip_hybrid,
    )


def main() -> None:
    # главная функция
    args = parse_args()
    config = build_config_from_cli(args)
    runner = ApproachRunner(config)
    runner.run_all()


if __name__ == "__main__":
    main()
