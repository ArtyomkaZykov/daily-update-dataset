import requests
import time
import pandas as pd
import numpy as np
from geopy.geocoders import Nominatim
import streamlit as st


# ========== ПАРСЕР HH.RU ==========
@st.cache_data(ttl=86400)  # Кэшируем на сутки
def parse_hh_salaries(professions_list):
    """
    Парсит реальные зарплаты с hh.ru для списка профессий
    """
    base_url = "https://api.hh.ru/vacancies"
    results = []

    progress_bar = st.progress(0)
    status_text = st.empty()

    for idx, profession in enumerate(professions_list):
        status_text.text(f"🔄 Парсим {profession}...")
        salaries = []

        # Собираем данные по 3 страницам
        for page in range(3):
            params = {
                "text": profession,
                "area": 113,  # Россия
                "per_page": 50,
                "page": page,
                "only_with_salary": True,
                "currency": "RUR"
            }

            try:
                response = requests.get(base_url, params=params)
                if response.status_code == 200:
                    data = response.json()
                    for item in data.get('items', []):
                        salary = item.get('salary')
                        if salary and salary.get('from'):
                            # Берем нижнюю границу зарплаты
                            salaries.append(salary.get('from'))
                time.sleep(0.2)  # Не долбим API
            except Exception as e:
                st.warning(f"Ошибка при парсинге {profession}: {e}")

        if salaries:
            results.append({
                'profession': profession,
                'salary_mean': int(np.mean(salaries)),
                'salary_median': int(np.median(salaries)),
                'salary_min': min(salaries),
                'salary_max': max(salaries),
                'vacancies_count': len(salaries)
            })
        else:
            # Если не нашли, ставим заглушку
            results.append({
                'profession': profession,
                'salary_mean': 70000,
                'salary_median': 70000,
                'salary_min': 50000,
                'salary_max': 90000,
                'vacancies_count': 0
            })

        progress_bar.progress((idx + 1) / len(professions_list))
        time.sleep(0.5)

    status_text.text("✅ Парсинг hh.ru завершен!")
    return pd.DataFrame(results)


# ========== ЭКСПЕРТНЫЕ ДАННЫЕ (ИСПРАВЛЕНО) ==========
def get_expert_data():
    """
    Возвращает экспертные оценки (категории, влияние ИИ, вердикты)
    Теперь включают ВСЕ профессии, упоминаемые в дашборде
    """
    return pd.DataFrame([
        # Профессии под угрозой
        {"profession": "Копирайтер", "category": "рутинная", "ai_impact": 0.85, "trend": "declining",
         "verdict": "вымрет"},
        {"profession": "Корректор", "category": "рутинная", "ai_impact": 0.80, "trend": "declining",
         "verdict": "вымрет"},
        {"profession": "Бухгалтер", "category": "рутинная", "ai_impact": 0.75, "trend": "declining",
         "verdict": "вымрет"},
        {"profession": "Телемаркетолог", "category": "рутинная", "ai_impact": 0.80, "trend": "declining",
         "verdict": "вымрет"},

        # IT-профессии (ПОЛНЫЙ СПИСОК)
        {"profession": "Junior-разработчик", "category": "it", "ai_impact": 0.70, "trend": "declining",
         "verdict": "под угрозой"},
        {"profession": "Middle-разработчик", "category": "it", "ai_impact": 0.50, "trend": "stable",
         "verdict": "трансформируется"},
        {"profession": "Senior-разработчик", "category": "it", "ai_impact": 0.40, "trend": "growing",
         "verdict": "останется"},
        {"profession": "Golang-разработчик", "category": "it", "ai_impact": 0.30, "trend": "growing",
         "verdict": "останется"},
        {"profession": "Python разработчик", "category": "it", "ai_impact": 0.40, "trend": "growing",
         "verdict": "останется"},
        {"profession": "Java разработчик", "category": "it", "ai_impact": 0.40, "trend": "growing",
         "verdict": "останется"},
        {"profession": "ML-инженер", "category": "it", "ai_impact": 0.25, "trend": "growing", "verdict": "останется"},
        {"profession": "Data Scientist", "category": "it", "ai_impact": 0.30, "trend": "growing",
         "verdict": "останется"},
        {"profession": "DevOps-инженер", "category": "it", "ai_impact": 0.30, "trend": "growing",
         "verdict": "останется"},
        {"profession": "Специалист по кибербезопасности", "category": "it", "ai_impact": 0.20, "trend": "growing",
         "verdict": "останется"},
        {"profession": "Product-менеджер", "category": "управленческая", "ai_impact": 0.25, "trend": "growing",
         "verdict": "останется"},

        # Рабочие профессии
        {"profession": "Сварщик", "category": "физическая", "ai_impact": 0.15, "trend": "growing",
         "verdict": "останется"},
        {"profession": "Токарь", "category": "физическая", "ai_impact": 0.15, "trend": "growing",
         "verdict": "останется"},
        {"profession": "Сантехник", "category": "физическая", "ai_impact": 0.10, "trend": "growing",
         "verdict": "останется"},
        {"profession": "Электрик", "category": "физическая", "ai_impact": 0.10, "trend": "growing",
         "verdict": "останется"},

        # Медицина
        {"profession": "Хирург", "category": "творческая", "ai_impact": 0.15, "trend": "growing",
         "verdict": "останется"},
        {"profession": "Врач", "category": "творческая", "ai_impact": 0.20, "trend": "growing", "verdict": "останется"},

        # Остальные
        {"profession": "Учитель", "category": "творческая", "ai_impact": 0.20, "trend": "stable",
         "verdict": "останется"},
        {"profession": "Маркетолог", "category": "творческая", "ai_impact": 0.45, "trend": "growing",
         "verdict": "трансформируется"},
        {"profession": "Финансовый аналитик", "category": "творческая", "ai_impact": 0.40, "trend": "growing",
         "verdict": "трансформируется"},
    ])


# ========== РЕГИОНАЛЬНЫЕ ДАННЫЕ ==========
def get_regional_data():
    """
    Данные по регионам с реалистичными коэффициентами автоматизации
    на основе плотности промышленных роботов и заводов
    """

    return {
            'Чукотский АО': {'salary': 193000, 'lat': 65.0, 'lon': 175.0, 'auto_factor': 0.6},
    'Ямало-Ненецкий АО': {'salary': 173000, 'lat': 66.5, 'lon': 76.5, 'auto_factor': 0.65},
    'Москва': {'salary': 161000, 'lat': 55.8, 'lon': 37.6, 'auto_factor': 1.4},
    'Магаданская область': {'salary': 152500, 'lat': 59.5, 'lon': 150.8, 'auto_factor': 0.6},
    'Ненецкий АО': {'salary': 151000, 'lat': 67.6, 'lon': 52.9, 'auto_factor': 0.6},
    'Сахалинская область': {'salary': 150000, 'lat': 50.0, 'lon': 142.0, 'auto_factor': 0.7},


    'Камчатский край': {'salary': 142800, 'lat': 53.0, 'lon': 158.0, 'auto_factor': 0.65},
    'Ханты-Мансийский АО': {'salary': 138000, 'lat': 61.0, 'lon': 69.0, 'auto_factor': 0.8},
    'Республика Саха (Якутия)': {'salary': 135000, 'lat': 62.0, 'lon': 129.0, 'auto_factor': 0.7},
    'Мурманская область': {'salary': 132000, 'lat': 68.0, 'lon': 33.0, 'auto_factor': 0.9},
    'Санкт-Петербург': {'salary': 130000, 'lat': 59.9, 'lon': 30.3, 'auto_factor': 1.35},
    'Московская область': {'salary': 128000, 'lat': 55.5, 'lon': 37.5, 'auto_factor': 1.3},
    'Тюменская область': {'salary': 125000, 'lat': 57.1, 'lon': 65.5, 'auto_factor': 0.95},
    'Архангельская область': {'salary': 122000, 'lat': 64.5, 'lon': 40.5, 'auto_factor': 0.85},



    'Республика Коми': {'salary': 115000, 'lat': 61.7, 'lon': 50.8, 'auto_factor': 0.8},
    'Красноярский край': {'salary': 112000, 'lat': 56.0, 'lon': 92.9, 'auto_factor': 1.1},
    'Свердловская область': {'salary': 110000, 'lat': 56.8, 'lon': 60.6, 'auto_factor': 1.2},
    'Иркутская область': {'salary': 108000, 'lat': 52.3, 'lon': 104.3, 'auto_factor': 0.9},
    'Томская область': {'salary': 107000, 'lat': 56.5, 'lon': 84.9, 'auto_factor': 0.95},
    'Республика Татарстан': {'salary': 106000, 'lat': 55.8, 'lon': 49.1, 'auto_factor': 1.25},
    'Ленинградская область': {'salary': 105000, 'lat': 60.0, 'lon': 31.0, 'auto_factor': 1.0},
    'Нижегородская область': {'salary': 104000, 'lat': 56.3, 'lon': 44.0, 'auto_factor': 1.2},
    'Пермский край': {'salary': 103000, 'lat': 58.0, 'lon': 56.2, 'auto_factor': 1.05},
    'Хабаровский край': {'salary': 102000, 'lat': 48.5, 'lon': 135.1, 'auto_factor': 0.85},
    'Приморский край': {'salary': 101000, 'lat': 43.1, 'lon': 131.9, 'auto_factor': 0.85},
    'Республика Карелия': {'salary': 100000, 'lat': 61.8, 'lon': 34.3, 'auto_factor': 0.8},



    'Новосибирская область': {'salary': 95000, 'lat': 55.0, 'lon': 82.9, 'auto_factor': 1.05},
    'Самарская область': {'salary': 94000, 'lat': 53.2, 'lon': 50.2, 'auto_factor': 1.15},
    'Кемеровская область': {'salary': 93000, 'lat': 55.4, 'lon': 86.1, 'auto_factor': 0.9},
    'Челябинская область': {'salary': 92000, 'lat': 55.2, 'lon': 61.4, 'auto_factor': 1.15},
    'Омская область': {'salary': 91000, 'lat': 54.9, 'lon': 73.4, 'auto_factor': 0.9},
    'Ростовская область': {'salary': 90000, 'lat': 47.2, 'lon': 39.7, 'auto_factor': 0.9},
    'Краснодарский край': {'salary': 89000, 'lat': 45.0, 'lon': 39.0, 'auto_factor': 0.9},
    'Вологодская область': {'salary': 88000, 'lat': 59.2, 'lon': 39.9, 'auto_factor': 0.85},
    'Ярославская область': {'salary': 87000, 'lat': 57.6, 'lon': 39.9, 'auto_factor': 0.9},
    'Тульская область': {'salary': 86000, 'lat': 54.2, 'lon': 37.6, 'auto_factor': 1.2},
    'Калужская область': {'salary': 85000, 'lat': 54.5, 'lon': 36.0, 'auto_factor': 1.3},
    'Липецкая область': {'salary': 84000, 'lat': 52.6, 'lon': 39.6, 'auto_factor': 0.95},
    'Воронежская область': {'salary': 83000, 'lat': 51.7, 'lon': 39.2, 'auto_factor': 0.9},
    'Белгородская область': {'salary': 82000, 'lat': 50.6, 'lon': 36.6, 'auto_factor': 0.9},
    'Волгоградская область': {'salary': 81000, 'lat': 48.7, 'lon': 44.5, 'auto_factor': 0.85},
    'Саратовская область': {'salary': 80000, 'lat': 51.5, 'lon': 46.0, 'auto_factor': 0.8},
    'Удмуртская Республика': {'salary': 79000, 'lat': 56.9, 'lon': 53.2, 'auto_factor': 0.9},
    'Чувашская Республика': {'salary': 78000, 'lat': 56.1, 'lon': 47.3, 'auto_factor': 0.85},
    'Кировская область': {'salary': 77000, 'lat': 58.6, 'lon': 49.7, 'auto_factor': 0.8},
    'Ульяновская область': {'salary': 76000, 'lat': 54.3, 'lon': 48.4, 'auto_factor': 0.85},
    'Пензенская область': {'salary': 75000, 'lat': 53.2, 'lon': 45.0, 'auto_factor': 0.8},
    'Тамбовская область': {'salary': 74000, 'lat': 52.7, 'lon': 41.5, 'auto_factor': 0.8},
    'Орловская область': {'salary': 73000, 'lat': 52.9, 'lon': 36.1, 'auto_factor': 0.8},
    'Курская область': {'salary': 72000, 'lat': 51.7, 'lon': 36.2, 'auto_factor': 0.85},
    'Брянская область': {'salary': 71000, 'lat': 53.2, 'lon': 34.4, 'auto_factor': 0.8},
    'Смоленская область': {'salary': 70000, 'lat': 54.8, 'lon': 32.0, 'auto_factor': 0.8},



    'Республика Адыгея': {'salary': 68000, 'lat': 44.6, 'lon': 40.1, 'auto_factor': 0.6},
    'Республика Калмыкия': {'salary': 67000, 'lat': 46.3, 'lon': 44.3, 'auto_factor': 0.5},
    'Республика Марий Эл': {'salary': 66000, 'lat': 56.6, 'lon': 47.9, 'auto_factor': 0.6},
    'Республика Мордовия': {'salary': 65000, 'lat': 54.4, 'lon': 45.2, 'auto_factor': 0.6},
    'Республика Северная Осетия': {'salary': 64000, 'lat': 43.0, 'lon': 44.7, 'auto_factor': 0.5},
    'Кабардино-Балкарская Республика': {'salary': 63000, 'lat': 43.5, 'lon': 43.6, 'auto_factor': 0.5},
    'Карачаево-Черкесская Республика': {'salary': 62000, 'lat': 43.8, 'lon': 41.8, 'auto_factor': 0.5},
    'Республика Дагестан': {'salary': 56000, 'lat': 42.5, 'lon': 47.5, 'auto_factor': 0.4},
    'Чеченская Республика': {'salary': 52000, 'lat': 43.3, 'lon': 45.7, 'auto_factor': 0.4},
    'Республика Ингушетия': {'salary': 48000, 'lat': 43.2, 'lon': 44.9, 'auto_factor': 0.4},
    }

# ========== РЕЗЕРВНЫЕ ДАННЫЕ НА СЛУЧАЙ ОШИБКИ ==========
def get_backup_salaries():
    """
    Резервные данные, если hh.ru недоступен
    """
    return pd.DataFrame([
        {'profession': 'Golang-разработчик', 'salary_mean': 550000},
        {'profession': 'Senior-разработчик', 'salary_mean': 284000},
        {'profession': 'ML-инженер', 'salary_mean': 247000},
        {'profession': 'DevOps-инженер', 'salary_mean': 220000},
        {'profession': 'Data Scientist', 'salary_mean': 200000},
        {'profession': 'Хирург', 'salary_mean': 250000},
        {'profession': 'Сварщик', 'salary_mean': 267000},
        {'profession': 'Токарь', 'salary_mean': 185000},
        {'profession': 'Python разработчик', 'salary_mean': 150000},
        {'profession': 'Java разработчик', 'salary_mean': 160000},
        {'profession': 'Junior-разработчик', 'salary_mean': 90000},
        {'profession': 'Middle-разработчик', 'salary_mean': 200000},
        {'profession': 'Копирайтер', 'salary_mean': 70000},
        {'profession': 'Бухгалтер', 'salary_mean': 75000},
        {'profession': 'Сантехник', 'salary_mean': 95000},
        {'profession': 'Электрик', 'salary_mean': 100000},
        {'profession': 'Учитель', 'salary_mean': 85000},
        {'profession': 'Маркетолог', 'salary_mean': 100000},
        {'profession': 'Финансовый аналитик', 'salary_mean': 117000},
    ])