# update_data.py
import pandas as pd
import os
from datetime import date
from data_loader import parse_hh_salaries, get_expert_data, get_regional_data, get_backup_salaries

DATA_FILE = "professions_daily.csv"
CURRENT_DATE = date.today()
AVG_RUSSIAN_SALARY = 70000

def load_existing_data():
    if os.path.exists(DATA_FILE):
        return pd.read_csv(DATA_FILE, parse_dates=['date'])
    return None

def fetch_and_append():
    expert_df = get_expert_data()
    professions_list = expert_df['profession'].tolist()

    try:
        salary_df = parse_hh_salaries(professions_list)
    except Exception as e:
        print(f"Ошибка парсинга: {e}")
        salary_df = get_backup_salaries()

    merged_df = expert_df.merge(salary_df, on='profession', how='left')
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

    new_df = pd.DataFrame(new_rows)
    return new_df

def main():
    existing = load_existing_data()
    if existing is not None and not existing.empty:
        if CURRENT_DATE in existing['date'].dt.date.unique():
            print(f"Данные за {CURRENT_DATE} уже есть. Выход.")
            return
    new_data = fetch_and_append()
    if new_data is not None and not new_data.empty:
        if existing is None:
            new_data.to_csv(DATA_FILE, index=False)
        else:
            combined = pd.concat([existing, new_data], ignore_index=True)
            combined.to_csv(DATA_FILE, index=False)
        print(f"Добавлено {len(new_data)} записей за {CURRENT_DATE}")

if __name__ == "__main__":
    main()
