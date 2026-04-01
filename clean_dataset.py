#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт для очистки датасета от сгенерированных (исторических) данных.
Оставляет только реальные записи, полученные парсингом hh.ru.
Использование:
    python clean_dataset.py [--input INPUT.csv] [--output OUTPUT.csv] [--year YYYY] [--no-backup]
"""

import argparse
import pandas as pd
import os
from datetime import date

def clean_dataset(input_file, output_file, backup=True, threshold_year=None):
    """
    Удаляет строки с датой 1 января или годом меньше threshold_year.
    Остальные строки сохраняет в output_file.
    """
    # Читаем CSV
    df = pd.read_csv(input_file)
    print(f"Загружено {len(df)} строк из {input_file}")

    # Преобразуем дату с автоматическим определением формата
    df['date'] = pd.to_datetime(df['date'], format='mixed', errors='coerce')
    # Отбрасываем строки с некорректной датой
    df = df.dropna(subset=['date'])
    print(f"После удаления некорректных дат: {len(df)} строк")

    # Определяем пороговый год (если не задан – текущий)
    if threshold_year is None:
        threshold_year = date.today().year

    # Фильтруем: оставляем только строки с годом >= threshold_year и датой не 1 января
    # Также можно явно удалить строки, у которых день и месяц 1 января
    mask = (df['date'].dt.year >= threshold_year) & (df['date'].dt.strftime('%m-%d') != '01-01')
    cleaned_df = df[mask].copy()
    print(f"Оставлено реальных данных: {len(cleaned_df)} строк")

    # Если нужно, делаем бэкап оригинального файла
    if backup and os.path.exists(input_file):
        backup_file = input_file + '.backup'
        os.rename(input_file, backup_file)
        print(f"Создана резервная копия: {backup_file}")

    # Сохраняем результат
    cleaned_df.to_csv(output_file, index=False)
    print(f"Очищенный датасет сохранён в {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Удаляет сгенерированные данные из датасета, оставляя только реальные парсинговые.')
    parser.add_argument('--input', '-i', default='professions_daily.csv',
                        help='Исходный файл с данными (по умолчанию professions_daily.csv)')
    parser.add_argument('--output', '-o', default='professions_real.csv',
                        help='Выходной файл для очищенных данных (по умолчанию professions_real.csv)')
    parser.add_argument('--year', '-y', type=int, default=None,
                        help='Пороговый год (включительно). Данные за более ранние годы будут удалены. По умолчанию текущий год.')
    parser.add_argument('--no-backup', action='store_true',
                        help='Не создавать резервную копию исходного файла')
    args = parser.parse_args()

    clean_dataset(args.input, args.output, backup=not args.no_backup, threshold_year=args.year)