# Molecular Transformer

Генерация молекулярных структур (SMILES) с помощью decoder-only трансформера на датасете MOSES.

## Результаты

Сгенерированные молекулы (Top 20):

![Top 20 Molecules](assets/visualization_results/molecules_top20.png)

Сгенерированные молекулы (Random 20):

![Random 20 Molecules](assets/visualization_results/molecules_random20.png)

Распределение молекулярных свойств (MW, LogP, HBA, HBD, TPSA, RotBonds, Rings, Aromatic):

![Molecular Properties](assets/visualization_results/molecular_properties.png)

## Установка

```bash
pip install -r requirements.txt
```

## Запуск

Обучение модели:
```bash
python src/train_standalone.py
```

Генерация молекул:
```bash
python src/generate_logged.py --num 50 --validate
```

Построение графиков и структур молекул:
```bash
python src/visualize_results.py
```

## Структура

- `src/model_decoder.py` — архитектура трансформера
- `src/Tokenizer.py` — токенизатор SMILES
- `src/dataset.py` — загрузка датасета MOSES
- `src/train_standalone.py` — обучение модели
- `src/generate_logged.py` — генерация молекул
- `src/visualize_results.py` — визуализация молекул и свойств
- `src/validate.py` — валидация через RDKit
