import pandas as pd
import os
from datetime import date
from data_loader import parse_hh_salaries, get_expert_data, get_regional_data, get_backup_salaries

DATA_FILE = "professions_daily.csv"
CURRENT_DATE = date.today()
AVG_RUSSIAN_SALARY = 70000

def load_existing_data():
    if os.path.exists(DATA_FILE):
        df = pd.read_csv(DATA_FILE)
        # Преобразуем колонку date в datetime, автоматически определяя формат
        df['date'] = pd.to_datetime(df['date'], format='mixed', errors='coerce')
        # Если всё ещё есть ошибки, попробуем без формата
        if df['date'].isna().any():
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        return df
    return None

def fetch_today_data():
    expert_df = get_expert_data()
    professions_list = expert_df['profession'].tolist()

    try:
        salary_df = parse_hh_salaries(professions_list)
        print("✅ Данные hh.ru обновлены")
    except Exception as e:
        print(f"❌ Ошибка парсинга hh.ru: {e}")
        print("🔄 Использую резервные данные")
        salary_df = get_backup_salaries()

    merged_df = expert_df.merge(salary_df, on='profession', how='left')
    if merged_df.empty:
        return None

    regions = get_regional_data()
    new_rows = []

    for _, row in merged_df.iterrows():
        prof = row['profession']
        category = row['category']
        ai_impact = row['ai_impact']
        trend = row['trend']
        verdict = row['verdict']
        base_salary = row.get('salary_mean', AVG_RUSSIAN_SALARY)

        automation = ai_impact * (0.5 + 0.04 * (CURRENT_DATE.year - 2015))
        automation = min(automation, 0.95)

        for region, reg_data in regions.items():
            region_factor = reg_data['salary'] / AVG_RUSSIAN_SALARY
            auto_factor = reg_data.get('auto_factor', 0.9)
            demand_factor = 1.5 if region in ['Москва', 'Санкт-Петербург'] else 1.0
            salary_region = base_salary * region_factor
            demand = 1000 * demand_factor

            new_rows.append({
                'date': CURRENT_DATE,
                'year': CURRENT_DATE.year,
                'profession': prof,
                'category': category,
                'region': region,
                'latitude': reg_data['lat'],
                'longitude': reg_data['lon'],
                'salary': round(salary_region),
                'demand': round(demand),
                'automation_level': round(automation * auto_factor, 3),
                'verdict': verdict,
                'trend': trend
            })

    return pd.DataFrame(new_rows)

def main():
    existing = load_existing_data()
    if existing is not None and not existing.empty:
        # Преобразование уже сделано в load_existing_data
        if CURRENT_DATE in existing['date'].dt.date.unique():
            print(f"Данные за {CURRENT_DATE} уже есть. Выход.")
            return

    new_data = fetch_today_data()
    if new_data is None or new_data.empty:
        print("Нет новых данных для добавления.")
        return

    if existing is None:
        new_data.to_csv(DATA_FILE, index=False)
        print(f"Создан новый датасет: {len(new_data)} записей.")
    else:
        combined = pd.concat([existing, new_data], ignore_index=True)
        combined.to_csv(DATA_FILE, index=False)
        print(f"Добавлено {len(new_data)} записей за {CURRENT_DATE}")

if __name__ == "__main__":
    main()
