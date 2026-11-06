"""Лабораторная работа: от Bag of Words к смысловым пространствам.

Задание: docs/origin_task.ipynb. Все разделы выполняются последовательно.

Установка зависимостей:
    python -m pip install numpy matplotlib scikit-learn gensim

Запуск всей работы (при первом запуске нужна сеть для загрузки Word2Vec):
    python Task_02.py

Запуск без окон с графиками:
    python Task_02.py --no-show

Запуск только разделов BoW без загрузки модели:
    python Task_02.py --skip-word2vec --no-show

Графики сохраняются в results/task_02 рядом со скриптом. Каталог можно
изменить через --output-dir. Модель кешируется в .cache/gensim-data рядом
со скриптом; переменная GENSIM_DATA_DIR позволяет выбрать другой кеш.
NLTK не требуется: стоп-слова в этой работе сохраняются, лемматизация
не выполняется, чтобы исследовать именно исходное представление BoW.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


TEST_TEXTS = [
    "собака укусила человека",
    "человек укусил собаку",
    "стреляю из лука",
    "плачу от лука",
    "люблю смотреть фильмы",
    "фильмы смотреть люблю",
]

MODEL_NAME = "word2vec-ruscorpora-300"

WORD_GROUPS = {
    "Люди": ["человек_NOUN", "люди_NOUN", "личность_NOUN", "персона_NOUN"],
    "Время": ["время_NOUN", "год_NOUN", "день_NOUN", "месяц_NOUN"],
    "Речь": ["говорить_VERB", "сказать_VERB", "молчать_VERB", "кричать_VERB"],
    "Размер": ["большой_ADJ", "маленький_ADJ", "огромный_ADJ", "крошечный_ADJ"],
    "Качество": ["хороший_ADJ", "плохой_ADJ", "отличный_ADJ", "ужасный_ADJ"],
    # Дополнительная тематическая группа по заданию.
    "Семья": ["отец_NOUN", "мать_NOUN", "сын_NOUN", "дочь_NOUN", "семья_NOUN"],
}


# Раздел 1. Реализация Bag of Words.
class SimpleBagOfWords:
    """Частотное представление текста с воспроизводимым порядком признаков."""

    def __init__(self):
        self.vocabulary = {}
        self.word_index = {}
        self.vocab_size = 0

    def preprocess_text(self, text):
        """Переводит текст в нижний регистр и заменяет пунктуацию пробелами."""
        # Unicode-буквы (в том числе ё) и цифры сохраняются; '_' — разделитель.
        return " ".join(re.findall(r"[^\W_]+", text.lower()))

    def fit(self, texts):
        """Строит словарь уникальных слов в алфавитном порядке."""
        unique_words = sorted(
            {word for text in texts for word in self.preprocess_text(text).split()}
        )
        if not unique_words:
            raise ValueError("После предобработки в обучающих текстах нет слов.")

        self.vocab_size = len(unique_words)
        self.vocabulary = {word: idx for idx, word in enumerate(unique_words)}
        self.word_index = {idx: word for word, idx in self.vocabulary.items()}
        return self

    def transform(self, texts):
        """Считает частоты слов; слова вне обученного словаря игнорирует."""
        if not self.vocabulary:
            raise ValueError("Перед transform необходимо вызвать fit.")

        texts = list(texts)
        vectors = np.zeros((len(texts), self.vocab_size), dtype=np.int64)
        for row, text in enumerate(texts):
            for word in self.preprocess_text(text).split():
                index = self.vocabulary.get(word)
                if index is not None:
                    vectors[row, index] += 1
        return vectors

    def fit_transform(self, texts):
        """Обучает модель и преобразует те же тексты, включая генераторы."""
        texts = list(texts)
        return self.fit(texts).transform(texts)

    def get_feature_names(self):
        """Возвращает слова в том же порядке, что и столбцы матрицы."""
        return list(self.vocabulary)


# Разделы 2–3. Векторы и ограничения BoW.
def demonstrate_bow_limitations(bow_model, texts, vectors):
    """Показывает потерю порядка слов, неоднозначность и разные словоформы."""
    print("\nПотеря порядка слов:")
    for index in (4, 5):
        print(f"  '{texts[index]}': {vectors[index]}")
    print("Векторы равны:", np.array_equal(vectors[4], vectors[5]))

    # У 'мать' и 'дочь' совпадают формы именительного и винительного падежей:
    # в этих фразах порядок меняет естественное прочтение ролей действующих лиц.
    reordered = ["мать любит дочь", "дочь любит мать"]
    reordered_model = SimpleBagOfWords()
    reordered_vectors = reordered_model.fit_transform(reordered)
    print("\nПерестановка слов с изменением смысла:")
    print("  Словарь дополнительного примера:", reordered_model.get_feature_names())
    for text, vector in zip(reordered, reordered_vectors):
        print(f"  '{text}': {vector}")
    print("Векторы равны:", np.array_equal(*reordered_vectors))
    print("BoW не хранит порядок слов и не определяет, кто кого любит.")

    print("\nИгнорирование контекста: оружие и овощ обозначены словом 'лука'.")
    onion_index = bow_model.vocabulary["лука"]
    for index in (2, 3):
        print(f"  '{texts[index]}': {vectors[index]}")
        print(
            f"  Частота 'лука' (столбец {onion_index}): {vectors[index, onion_index]}"
        )
    in_both = bool(vectors[2, onion_index] > 0 and vectors[3, onion_index] > 0)
    print("Слово 'лука' присутствует в обеих фразах:", in_both)
    print("Полные векторы равны:", np.array_equal(vectors[2], vectors[3]))
    print(
        "Один признак 'лука' не различает значения слова, хотя другие слова различаются."
    )

    print("\nЗависимость от словоформ:")
    print(
        "Векторы первых двух исходных фраз равны:",
        np.array_equal(vectors[0], vectors[1]),
    )
    print("'собака' и 'собаку', 'человек' и 'человека', 'укусила' и 'укусил'")
    print("попадают в разные столбцы: здесь нет лемматизации.")


# Раздел 4. Визуализация частотных векторов.
def visualize_bow_vectors(texts, vectors, title="BoW-векторы (PCA)", output_path=None):
    """Строит PCA-проекцию шести исходных примеров и возвращает фигуру."""
    pca = PCA(n_components=2)
    vectors_2d = pca.fit_transform(vectors)
    fig, ax = plt.subplots(figsize=(12, 8))
    colors = ["red", "red", "blue", "blue", "green", "green"]

    for index, (text, color) in enumerate(zip(texts, colors)):
        x, y = vectors_2d[index]
        ax.scatter(x, y, c=color, s=90, alpha=0.7, edgecolors="black")
        short_text = text if len(text) <= 32 else text[:29] + "..."
        # Последние два вектора совпадают; подписи разводим по вертикали.
        align_right = x > vectors_2d[:, 0].mean()
        offset = (-8 if align_right else 8, 12 if index % 2 == 0 else -18)
        ax.annotate(
            short_text,
            (x, y),
            xytext=offset,
            textcoords="offset points",
            fontsize=9,
            ha="right" if align_right else "left",
        )

    ax.set(xlabel="Компонента PCA 1", ylabel="Компонента PCA 2", title=title)
    ax.margins(0.25)
    ax.grid(True, alpha=0.3)
    ax.legend(
        handles=[
            Patch(facecolor="red", label="Укусы"),
            Patch(facecolor="blue", label="Лук"),
            Patch(facecolor="green", label="Фильмы"),
        ]
    )
    fig.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, dpi=160, bbox_inches="tight")

    print("\nОбъясненная дисперсия PCA:", pca.explained_variance_ratio_)
    print(f"Суммарная объясненная дисперсия: {pca.explained_variance_ratio_.sum():.3f}")
    print("Совпавшие точки последних двух фраз отражают совпадение их BoW-векторов.")
    print(
        "Разные исходные векторы тоже могут совпасть после PCA: часть информации потеряна."
    )
    return fig


# Раздел 5. Предобученная модель Word2Vec с тегами частей речи.
def load_word2vec_model():
    """Загружает русскую модель из кеша или скачивает ее при первом запуске."""
    os.environ.setdefault(
        "GENSIM_DATA_DIR",
        str(Path(__file__).resolve().parent / ".cache" / "gensim-data"),
    )
    # Отложенный импорт позволяет выполнить BoW без установленного gensim.
    import gensim.downloader as api

    print(f"\nЗагружаем {MODEL_NAME}. При первом запуске потребуется скачать модель.")
    print("Каталог кеша:", os.environ["GENSIM_DATA_DIR"])
    model = api.load(MODEL_NAME)
    print(f"Модель загружена: {len(model):,} слов, размерность {model.vector_size}.")
    return model


def demonstrate_vector_arithmetic_with_tags(model):
    """Проверяет гипотезы аналогий и выводит пять ближайших слов для каждой."""
    examples = [
        # Существительные из задания и дополнительная аналогия стран и столиц.
        (["король_NOUN", "женщина_NOUN"], ["мужчина_NOUN"], "королева_NOUN"),
        (["отец_NOUN", "дочь_NOUN"], ["сын_NOUN"], "мать_NOUN"),
        (["париж_NOUN", "россия_NOUN"], ["франция_NOUN"], "москва_NOUN"),
        # В исходном примере нет устойчивого основания ожидать 'плохой'.
        # Оставляем его как проверяемую гипотезу, а не гарантированный ответ.
        (["большой_ADJ", "хороший_ADJ"], ["маленький_ADJ"], "плохой_ADJ"),
        (["хороший_ADJ", "ужасный_ADJ"], ["плохой_ADJ"], "отличный_ADJ"),
        # Дополнительная глагольная аналогия движения в разных средах.
        (["идти_VERB", "вода_NOUN"], ["земля_NOUN"], "плыть_VERB"),
        # Контрольные операции из ноутбука: слагаемые взаимно сокращаются.
        (["бежать_VERB", "стоять_VERB"], ["стоять_VERB"], "бежать_VERB"),
        (["день_NOUN", "ночь_NOUN"], ["ночь_NOUN"], "день_NOUN"),
    ]
    results = []
    print("\nВекторная арифметика: ожидаемое слово — гипотеза для проверки.")
    for positive, negative, expected in examples:
        expression = " + ".join(positive) + " - " + " - ".join(negative)
        print(f"\n{expression}; гипотеза: {expected}")
        missing = sorted({word for word in positive + negative if word not in model})
        if missing:
            print("  Пропуск: в словаре нет", ", ".join(missing))
            continue

        neighbors = model.most_similar(positive=positive, negative=negative, topn=5)
        for rank, (word, similarity) in enumerate(neighbors, 1):
            print(f"  {rank}. {word:<25} {similarity:.4f}")

        # Gensim складывает нормированные векторы исходных слов. Строим тот же
        # запрос для оценки гипотезы, в том числе когда она исключена из top-5.
        query = sum((model.get_vector(word, norm=True) for word in positive))
        query = query - sum((model.get_vector(word, norm=True) for word in negative))
        norm = np.linalg.norm(query)
        expected_similarity = None
        if expected in model and norm > 0:
            expected_similarity = float(
                np.dot(query / norm, model.get_vector(expected, norm=True))
            )
            print(f"  Косинусное сходство с гипотезой: {expected_similarity:.4f}")
        elif expected not in model:
            print("  Слово-гипотеза отсутствует в словаре модели.")

        if expected in positive + negative:
            print("  Контроль с сокращением: результат равен исходному вектору.")
            print("  most_similar исключает слова запроса, поэтому его нет в top-5.")
        else:
            print(
                "  Гипотеза входит в top-5:",
                any(word == expected for word, _ in neighbors),
            )
        results.append((expected, neighbors, expected_similarity))
    analogies = [
        result for result in results if result[0] not in {"бежать_VERB", "день_NOUN"}
    ]
    matches = sum(
        any(word == expected for word, _ in neighbors)
        for expected, neighbors, _ in analogies
    )
    print(f"\nИтог: гипотеза попала в top-5 в {matches} из {len(analogies)} аналогий.")
    print("Контрольные операции с сокращением слагаемых в этот итог не включены.")
    return results


def find_similar_words_with_tags(model, word, topn=8):
    """Ищет ближайших соседей с учетом точного ключа 'лемма_ЧАСТЬРЕЧИ'."""
    print(f"\nСлова, близкие к '{word}':")
    if word not in model:
        print("  Слово отсутствует в словаре. Проверьте лемму и тег части речи.")
        return []
    neighbors = model.most_similar(positive=[word], topn=topn)
    for rank, (neighbor, similarity) in enumerate(neighbors, 1):
        print(f"  {rank}. {neighbor:<25} {similarity:.4f}")
    return neighbors


def visualize_word2vec_with_tags(model, output_path=None):
    """Собирает доступные слова и строит t-SNE-проекцию семантических групп."""
    all_words, all_vectors, categories = [], [], []
    for category, words in WORD_GROUPS.items():
        for word in words:
            if word in model:
                all_words.append(word)
                all_vectors.append(model[word])
                categories.append(category)
            else:
                print(f"  Пропущено отсутствующее слово: {word}")

    if len(all_words) < 2:
        print("Недостаточно известных слов для t-SNE: требуется минимум два.")
        return None

    all_vectors_array = np.asarray(all_vectors, dtype=np.float32)
    # Нормировка согласует евклидовы расстояния с косинусной близостью.
    norms = np.linalg.norm(all_vectors_array, axis=1, keepdims=True)
    all_vectors_array /= np.maximum(norms, np.finfo(np.float32).eps)
    tsne = TSNE(
        n_components=2,
        random_state=42,
        perplexity=min(5, len(all_words) - 1),
        init="random",
        learning_rate="auto",
    )
    vectors_2d = tsne.fit_transform(all_vectors_array)
    fig, ax = plt.subplots(figsize=(14, 10))
    colors = ["red", "blue", "green", "orange", "purple", "brown"]
    categories = np.asarray(categories)

    for color, category in zip(colors, WORD_GROUPS):
        mask = categories == category
        if not mask.any():
            continue
        ax.scatter(
            vectors_2d[mask, 0],
            vectors_2d[mask, 1],
            c=color,
            label=category,
            s=75,
            alpha=0.8,
            edgecolors="black",
        )
    for word, (x, y) in zip(all_words, vectors_2d):
        ax.annotate(word, (x, y), xytext=(5, 6), textcoords="offset points", fontsize=9)

    ax.set(
        title="Семантическое пространство Word2Vec (t-SNE)",
        xlabel="Координата t-SNE 1",
        ylabel="Координата t-SNE 2",
    )
    ax.margins(0.18)
    ax.legend(title="Тематические группы")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, dpi=160, bbox_inches="tight")
    print(f"\nt-SNE: {len(all_words)} слов, perplexity={tsne.perplexity}.")
    print("Цвета заданы вручную; t-SNE обучается без этих меток.")
    print("Оценивайте локальные соседства: расстояния между группами и оси")
    print("проекции не являются точной мерой семантической близости.")
    return fig


def print_conclusions(word2vec_completed):
    """Формулирует выводы, отделяя наблюдения от общих свойств методов."""
    print("""
ВЫВОДЫ

Bag of Words - преимущества:
1. Простая и быстрая модель; каждый столбец имеет понятный смысл: частоту слова.
2. Не требует предобученного корпуса и подходит как базовое представление текста.

Bag of Words - недостатки и наблюдения:
1. Фразы о фильмах имеют одинаковые векторы. Перестановка слов в примере с
   матерью и дочерью также не меняет вектор: порядок не учитывается.
2. 'Лук' занимает один столбец и для оружия, и для овоща: значения не разделяются.
3. Разные словоформы создают разные признаки. Синонимы тоже не сближаются
   автоматически; при росте словаря векторы становятся большими и разреженными.
4. Близость на рисунке отражает частотные признаки, а не доказанное сходство смысла.

Word2Vec - преимущества:
1. Плотные векторы фиксированной размерности передают распределительное
   сходство: слова с похожими контекстами получают близкие представления.
2. Позволяет искать соседей по косинусному сходству и проверять аналогии
   векторной арифметикой; словарь данной модели содержит теги частей речи.

Word2Vec - недостатки:
1. Статическая модель дает один вектор на ключ 'лемма_тег', поэтому разные
   значения 'лук_NOUN' по-прежнему смешиваются. Контекст фразы она не кодирует.
2. Слова вне словаря недоступны; результат зависит от корпуса и предобработки.
   Антонимы могут быть близки, потому что употребляются в похожих контекстах.
3. Аналогии - эмпирические гипотезы, а не правила: top-5 и сходство с ожидаемым
   словом позволяют проверить результат. Сокращение слагаемых - лишь контроль.
4. t-SNE помогает изучать локальные соседства, но не доказывает наличие
   кластеров или точность аналогий в исходном 300-мерном пространстве.

Переход от BoW к Word2Vec добавляет распределительную семантику, но понимание
контекста конкретной фразы требует контекстных моделей представления текста.
""")
    if word2vec_completed:
        print("Раздел Word2Vec выполнен: выше приведены фактические соседи и аналогии.")
    else:
        print("Раздел Word2Vec не выполнен; его свойства выше — теоретические выводы.")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--no-show", action="store_true", help="Сохранить графики без показа окон"
    )
    parser.add_argument(
        "--skip-word2vec", action="store_true", help="Выполнить только разделы BoW"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results" / "task_02",
        help="Каталог для PNG-графиков",
    )
    args = parser.parse_args(argv)
    if args.no_show:
        plt.switch_backend("Agg")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("РАЗДЕЛЫ 1–3. BAG OF WORDS")
    bow_model = SimpleBagOfWords()
    vectors = bow_model.fit_transform(TEST_TEXTS)
    print("Словарь:", bow_model.get_feature_names())
    print("Размер словаря:", bow_model.vocab_size)
    print("Векторы BoW:")
    for text, vector in zip(TEST_TEXTS, vectors):
        print(f"  '{text}': {vector}")
    demonstrate_bow_limitations(bow_model, TEST_TEXTS, vectors)

    print("\nРАЗДЕЛ 4. ВИЗУАЛИЗАЦИЯ BOW")
    visualize_bow_vectors(
        TEST_TEXTS, vectors, output_path=args.output_dir / "bow_pca.png"
    )

    word2vec_completed = False
    load_failed = False
    if not args.skip_word2vec:
        print("\nРАЗДЕЛ 5. WORD2VEC")
        try:
            word2vec_model = load_word2vec_model()
        except (ImportError, OSError, ValueError, RuntimeError, EOFError) as error:
            # BoW и выводы остаются доступны; обращений к несуществующей модели нет.
            load_failed = True
            print(f"Не удалось загрузить Word2Vec: {error}", file=sys.stderr)
            print(
                "Проверьте зависимости, сеть и кеш модели; повторите запуск.",
                file=sys.stderr,
            )
            print(
                "Для запуска только BoW используйте --skip-word2vec.", file=sys.stderr
            )
        else:
            demonstrate_vector_arithmetic_with_tags(word2vec_model)
            for word in ("человек_NOUN", "говорить_VERB", "большой_ADJ", "хороший_ADJ"):
                find_similar_words_with_tags(word2vec_model, word)
            figure = visualize_word2vec_with_tags(
                word2vec_model, output_path=args.output_dir / "word2vec_tsne.png"
            )
            word2vec_completed = figure is not None

    print_conclusions(word2vec_completed)
    print("\nКаталог графиков:", args.output_dir.resolve())
    if not args.no_show:
        plt.show()
    plt.close("all")
    return 1 if load_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
