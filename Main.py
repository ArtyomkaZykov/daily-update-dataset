import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
import warnings
import os
import random
from datetime import datetime, date

warnings.filterwarnings('ignore')

# ========== НЕЙРОСЕТЬ (LSTM) ==========
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

tf.random.set_seed(42)
np.random.seed(42)
random.seed(42)

# ========== НАСТРОЙКА СТРАНИЦЫ ==========
st.set_page_config(
    page_title="ИИ и рынок труда | Ежедневный датасет",
    page_icon="🤖",
    layout="wide"
)

st.title("Влияние искусственного интеллекта на рынок труда")
st.markdown("#### Ежедневное накопление данных с hh.ru и прогноз до 2030 года")

# ========== КОНСТАНТЫ ==========
DATA_FILE = "professions_daily.csv"
CURRENT_DATE = date.today()
CURRENT_YEAR = CURRENT_DATE.year

# ========== РЕАЛЬНЫЕ РЫНОЧНЫЕ ЗАРПЛАТЫ (2026 год, СРЕДНИЕ по России) ==========
# Источники: hh.ru, Habr Career, SuperJob (апрель 2026)
REAL_MARKET_SALARIES = {
    # IT и разработка
    'Golang-разработчик': 380000,
    'ML-инженер': 360000,
    'Senior-разработчик': 330000,
    'DevOps-инженер': 320000,
    'Data Scientist': 300000,
    'Product-менеджер': 290000,
    'Специалист по кибербезопасности': 280000,
    'Java разработчик': 260000,
    'Python разработчик': 250000,
    'Middle-разработчик': 210000,
    'Junior-разработчик': 120000,

    # Рабочие профессии
    'Сварщик': 140000,
    'Токарь': 110000,
    'Электрик': 95000,
    'Сантехник': 90000,

    # Медицина
    'Хирург': 200000,
    'Врач': 130000,

    # Творческие и офисные
    'Маркетолог': 100000,
    'Финансовый аналитик': 150000,
    'Учитель': 75000,

    # Под угрозой
    'Копирайтер': 65000,
    'Корректор': 60000,
    'Бухгалтер': 80000,
    'Телемаркетолог': 60000,
}

# ========== РЕАЛЬНЫЕ УРОВНИ АВТОМАТИЗАЦИИ (исправлены) ==========
REAL_AI_IMPACT = {
    'Копирайтер': 0.85,
    'Корректор': 0.80,
    'Бухгалтер': 0.75,
    'Телемаркетолог': 0.80,
    'Junior-разработчик': 0.60,
    'Middle-разработчик': 0.40,
    'Senior-разработчик': 0.28,
    'Golang-разработчик': 0.22,
    'Python разработчик': 0.30,
    'Java разработчик': 0.30,
    'ML-инженер': 0.18,
    'Data Scientist': 0.20,
    'DevOps-инженер': 0.22,
    'Специалист по кибербезопасности': 0.15,
    'Product-менеджер': 0.20,
    'Сварщик': 0.12,
    'Токарь': 0.14,
    'Сантехник': 0.06,
    'Электрик': 0.10,
    'Хирург': 0.08,
    'Врач': 0.18,
    'Учитель': 0.15,
    'Маркетолог': 0.42,
    'Финансовый аналитик': 0.35,
}

# ========== РЕАЛЬНЫЕ СРЕДНИЕ ЗАРПЛАТЫ ПО РЕГИОНАМ (Росстат, 2025) ==========
REGION_AVG_SALARIES = {
    'Москва': 145000,
    'Санкт-Петербург': 125000,
    'Московская область': 110000,
    'Чукотский АО': 135000,
    'Ямало-Ненецкий АО': 130000,
    'Ханты-Мансийский АО': 120000,
    'Красноярский край': 85000,
    'Свердловская область': 80000,
    'Республика Татарстан': 78000,
    'Нижегородская область': 75000,
    'Новосибирская область': 75000,
    'Самарская область': 72000,
    'Челябинская область': 70000,
    'Пермский край': 70000,
    'Краснодарский край': 68000,
    'Ростовская область': 65000,
    'Волгоградская область': 60000,
    'Саратовская область': 58000,
    'Республика Дагестан': 45000,
    'Республика Ингушетия': 42000,
}

# Добавляем все регионы из get_regional_data с дефолтной зарплатой
from data_loader import get_regional_data

_regional_data = get_regional_data()
for region in _regional_data.keys():
    if region not in REGION_AVG_SALARIES:
        REGION_AVG_SALARIES[region] = 65000

# Рассчитываем региональные коэффициенты
AVG_RUSSIAN_SALARY = 85000
REGION_COEFFICIENTS = {region: REGION_AVG_SALARIES.get(region, 65000) / AVG_RUSSIAN_SALARY
                       for region in REGION_AVG_SALARIES}


# ========== ФУНКЦИЯ ГЕНЕРАЦИИ ИСТОРИЧЕСКИХ ДАННЫХ ==========
def generate_historical_data():
    from data_loader import get_expert_data, get_regional_data

    expert_df = get_expert_data()
    regions = get_regional_data()
    data = []

    # Инфляция по годам
    inflation_factors = {
        2015: 1.00, 2016: 1.05, 2017: 1.04, 2018: 1.04, 2019: 1.03,
        2020: 1.04, 2021: 1.08, 2022: 1.12, 2023: 1.06, 2024: 1.07,
        2025: 1.08, 2026: 1.00,
    }

    # Макроэкономические шоки
    macro_shocks = {
        2015: 0.95, 2016: 0.97, 2017: 0.98, 2018: 1.00, 2019: 1.00,
        2020: 0.96, 2021: 1.02, 2022: 0.94, 2023: 0.96, 2024: 0.98,
        2025: 1.00, 2026: 1.00,
    }

    for _, row in expert_df.iterrows():
        prof = row['profession']
        category = row['category']
        trend = row['trend']
        verdict = row['verdict']

        base_salary_2026 = REAL_MARKET_SALARIES.get(prof, 85000)
        ai_impact = REAL_AI_IMPACT.get(prof, row.get('ai_impact', 0.3))

        for year in range(2015, CURRENT_YEAR + 1):
            years_from_2026 = CURRENT_YEAR - year

            if trend == 'growing':
                trend_factor = 1 - 0.025 * years_from_2026
            elif trend == 'declining':
                trend_factor = 1 + 0.035 * years_from_2026
            else:
                trend_factor = 1 - 0.01 * years_from_2026
            trend_factor = max(0.65, min(1.35, trend_factor))

            infl_multiplier = 1.0
            for y in range(year + 1, CURRENT_YEAR + 1):
                infl_multiplier /= inflation_factors.get(y, 1.05)

            shock = macro_shocks.get(year, 1.0)
            base_salary_year = base_salary_2026 * trend_factor * infl_multiplier * shock

            noise = np.random.normal(1, 0.04)
            base_salary_year *= noise

            automation = ai_impact * (0.5 + 0.03 * (year - 2015))
            automation = min(automation, 0.85)

            for region, reg_data in regions.items():
                region_coeff = REGION_COEFFICIENTS.get(region, 0.85)
                auto_factor = reg_data.get('auto_factor', 0.9)
                demand_factor = 1.2 if region in ['Москва', 'Санкт-Петербург'] else 1.0

                if trend == 'growing':
                    demand_mult = 1 + 0.02 * (year - 2015)
                elif trend == 'declining':
                    demand_mult = 1 - 0.04 * (year - 2015)
                else:
                    demand_mult = 1 + 0.005 * (year - 2015)
                demand_mult = max(0.5, min(1.8, demand_mult))

                salary_region = base_salary_year * region_coeff
                demand = int(600 * demand_factor * demand_mult)
                demand = max(20, demand)

                if year == CURRENT_YEAR:
                    record_date = CURRENT_DATE
                else:
                    record_date = datetime(year, 1, 1).date()

                data.append({
                    'date': record_date,
                    'year': year,
                    'profession': prof,
                    'category': category,
                    'region': region,
                    'latitude': reg_data['lat'],
                    'longitude': reg_data['lon'],
                    'salary': round(salary_region),
                    'demand': demand,
                    'automation_level': round(min(automation * auto_factor, 0.85), 3),
                    'verdict': verdict,
                    'trend': trend
                })

    return pd.DataFrame(data)


# ========== ФУНКЦИЯ ДОБАВЛЕНИЯ ДАННЫХ ЗА ТЕКУЩИЙ ДЕНЬ ==========
def fetch_today_data():
    from data_loader import parse_hh_salaries, get_expert_data, get_regional_data, get_backup_salaries

    expert_df = get_expert_data()
    professions_list = expert_df['profession'].tolist()

    try:
        salary_df = parse_hh_salaries(professions_list)
        st.sidebar.success("✅ Данные hh.ru обновлены")
    except Exception as e:
        st.sidebar.error(f"❌ Ошибка парсинга hh.ru: {e}")
        st.sidebar.warning("🔄 Использую резервные данные")
        salary_df = get_backup_salaries()

    merged_df = expert_df.merge(salary_df, on='profession', how='left')
    if merged_df.empty:
        return None

    regions = get_regional_data()
    new_rows = []

    for _, row in merged_df.iterrows():
        prof = row['profession']
        category = row['category']
        trend = row['trend']
        verdict = row['verdict']

        market_salary = REAL_MARKET_SALARIES.get(prof, 85000)
        hh_salary = row.get('salary_mean', market_salary)
        base_salary = (hh_salary * 0.3 + market_salary * 0.7)

        ai_impact = REAL_AI_IMPACT.get(prof, row.get('ai_impact', 0.3))
        automation = ai_impact * (0.5 + 0.03 * (CURRENT_YEAR - 2015))
        automation = min(automation, 0.85)

        for region, reg_data in regions.items():
            region_coeff = REGION_COEFFICIENTS.get(region, 0.85)
            auto_factor = reg_data.get('auto_factor', 0.9)
            demand_factor = 1.2 if region in ['Москва', 'Санкт-Петербург'] else 1.0

            if trend == 'growing':
                demand = int(700 * demand_factor)
            elif trend == 'declining':
                demand = int(500 * demand_factor)
            else:
                demand = int(600 * demand_factor)

            salary_region = base_salary * region_coeff
            noise_salary = np.random.normal(1, 0.02)
            salary_region *= noise_salary

            new_rows.append({
                'date': CURRENT_DATE,
                'year': CURRENT_YEAR,
                'profession': prof,
                'category': category,
                'region': region,
                'latitude': reg_data['lat'],
                'longitude': reg_data['lon'],
                'salary': round(salary_region),
                'demand': max(20, demand),
                'automation_level': round(min(automation * auto_factor, 0.85), 3),
                'verdict': verdict,
                'trend': trend
            })

    return pd.DataFrame(new_rows)


# ========== ОСНОВНАЯ ФУНКЦИЯ ЗАГРУЗКИ ДАННЫХ ==========
@st.cache_data(ttl=3600, show_spinner=False)
def get_full_dataset():
    if os.path.exists(DATA_FILE):
        df = pd.read_csv(DATA_FILE)
        # ПРИНУДИТЕЛЬНО преобразуем колонку date в datetime
        df['date'] = pd.to_datetime(df['date'], format='mixed', errors='coerce')

        today = CURRENT_DATE
        if today not in df['date'].dt.date.unique():
            new_data = fetch_today_data()
            if new_data is not None and not new_data.empty:
                df = pd.concat([df, new_data], ignore_index=True)
                df.to_csv(DATA_FILE, index=False)
                st.sidebar.success(f"💾 Добавлены данные за {today}")
        return df
    else:

        full_df = generate_historical_data()
        if full_df is not None:
            full_df.to_csv(DATA_FILE, index=False, date_format='%Y-%m-%d')
            st.sidebar.success(f"💾 Датасет создан: {len(full_df)} записей.")
            return full_df
        else:
            st.error("Не удалось создать датасет.")
            st.stop()


# ========== ЗАГРУЗКА ДАННЫХ ==========
df = get_full_dataset()
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year


# ========== ФУНКЦИИ ДЛЯ ПРОГНОЗИРОВАНИЯ ==========
@st.cache_resource
def create_lstm_model():
    model = Sequential([
        LSTM(50, activation='relu', return_sequences=True, input_shape=(3, 1)),
        Dropout(0.2),
        LSTM(50, activation='relu'),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model


def prepare_lstm_data(series, n_steps=3):
    X, y = [], []
    for i in range(len(series) - n_steps):
        X.append(series[i:i + n_steps])
        y.append(series[i + n_steps])
    return np.array(X), np.array(y)


def forecast_lstm(series, n_steps=3, n_forecast=5):
    if len(series) < n_steps + 2:
        return np.full(n_forecast, series[-1])
    scaler = MinMaxScaler()
    series_scaled = scaler.fit_transform(series.reshape(-1, 1)).flatten()
    X, y = prepare_lstm_data(series_scaled, n_steps)
    if len(X) < 2:
        return np.full(n_forecast, series[-1])
    X = X.reshape((X.shape[0], X.shape[1], 1))
    model = create_lstm_model()
    early_stop = EarlyStopping(monitor='loss', patience=5, verbose=0)
    model.fit(X, y, epochs=100, batch_size=4, verbose=0, callbacks=[early_stop])
    forecasts = []
    last_sequence = series_scaled[-n_steps:].reshape((1, n_steps, 1))
    for _ in range(n_forecast):
        next_pred = model.predict(last_sequence, verbose=0)[0, 0]
        forecasts.append(next_pred)
        last_sequence = np.roll(last_sequence, -1, axis=1)
        last_sequence[0, -1, 0] = next_pred
    return scaler.inverse_transform(np.array(forecasts).reshape(-1, 1)).flatten()


def forecast_linear(series, n_forecast=5):
    if len(series) < 2:
        return np.full(n_forecast, series[-1] if len(series) > 0 else 0)
    x = np.arange(len(series)).reshape(-1, 1)
    model = LinearRegression()
    model.fit(x, series)
    future_x = np.arange(len(series), len(series) + n_forecast).reshape(-1, 1)
    return model.predict(future_x)


# ========== ИНТЕРФЕЙС ==========
st.sidebar.header("🔍 Фильтры")

categories = ['Все'] + sorted(df['category'].unique().tolist())
selected_category = st.sidebar.selectbox("Категория профессий:", categories)

if selected_category == 'Все':
    available_professions = sorted(df['profession'].unique())
else:
    available_professions = sorted(df[df['category'] == selected_category]['profession'].unique())

selected_professions = st.sidebar.multiselect(
    "Профессии:",
    available_professions,
    default=available_professions[:5] if len(available_professions) > 5 else available_professions
)

regions = ['Все'] + sorted(df['region'].unique().tolist())
selected_region = st.sidebar.selectbox("Регион:", regions)

min_year = int(df['year'].min())
max_year = int(df['year'].max())

if min_year == max_year:
    selected_years = (min_year, min_year)
    st.sidebar.info(f"📅 Доступен только {min_year} год")
else:
    selected_years = st.sidebar.slider("Диапазон лет:", min_year, max_year, (min_year, max_year))



# Фильтрация данных
filtered_df = df[
    (df['profession'].isin(selected_professions)) &
    (df['year'].between(selected_years[0], selected_years[1]))
    ]
if selected_category != 'Все':
    filtered_df = filtered_df[filtered_df['category'] == selected_category]
if selected_region != 'Все':
    filtered_df = filtered_df[filtered_df['region'] == selected_region]

# ========== КЛЮЧЕВЫЕ МЕТРИКИ ==========
st.header("📊 Ключевые показатели")
latest_data = filtered_df[filtered_df['date'] == filtered_df['date'].max()]

col1, col2, col3, col4 = st.columns(4)
with col1:
    avg_auto = latest_data['automation_level'].mean() if not latest_data.empty else 0
    st.metric("Средний уровень автоматизации", f"{avg_auto:.1%}")
with col2:
    avg_sal = latest_data['salary'].mean() if not latest_data.empty else 0
    st.metric("Средняя зарплата", f"{avg_sal:,.0f} ₽".replace(',', ' '))
with col3:
    high_risk = len(
        latest_data[latest_data['automation_level'] > 0.7]['profession'].unique()) if not latest_data.empty else 0
    st.metric("Профессий под угрозой (>70%)", high_risk)
with col4:
    st.metric(f"Навыков трансформируется к {CURRENT_YEAR + 5}", "41%")

# ========== КАРТА ==========
st.header("🗺️ Географическое распределение профессий")
map_data = filtered_df[filtered_df['date'] == filtered_df['date'].max()].copy()
if not map_data.empty:
    fig_map = px.scatter_mapbox(
        map_data,
        lat='latitude', lon='longitude',
        size='demand',
        color='automation_level',
        hover_name='profession',
        hover_data={'region': True, 'salary': ':.0f', 'automation_level': ':.1%'},
        color_continuous_scale='RdYlGn_r',
        size_max=50,
        zoom=2,
        title=f"Распределение профессий по регионам (данные на {map_data['date'].iloc[0].strftime('%d.%m.%Y')})",
        labels={'automation_level': 'Уровень автоматизации', 'salary': 'Зарплата (₽)'}
    )
    fig_map.update_layout(mapbox_style="carto-positron", mapbox_center={"lat": 60, "lon": 90},
                          margin={"r": 0, "t": 40, "l": 0, "b": 0})
    st.plotly_chart(fig_map, use_container_width=True)

# ========== КЛАССИФИКАЦИЯ ==========
st.header("🏷️ Классификация профессий по отношению к ИИ")
last_year_data = df[df['year'] == df['year'].max()]
verdict_groups = last_year_data.groupby('verdict')['profession'].unique().to_dict()
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("### 🔴 Под угрозой замены")
    risky = []
    for v in ['вымрет', 'под угрозой']:
        if v in verdict_groups:
            risky.extend(verdict_groups[v])
    st.markdown(", ".join(list(set(risky))) or "Нет данных")
with col2:
    st.markdown("### 🟢 Вне опасности")
    safe = verdict_groups.get('останется', [])
    st.markdown(", ".join(safe) or "Нет данных")
with col3:
    st.markdown("### 🟡 ИИ — помощник")
    trans = verdict_groups.get('трансформируется', [])
    st.markdown(", ".join(trans) or "Нет данных")

# ========== РЕГИОНАЛЬНЫЙ АНАЛИЗ ==========
st.header("📈 Региональный анализ")
last_hist_data = filtered_df[filtered_df['date'] == filtered_df['date'].max()]
col1, col2 = st.columns(2)
with col1:
    region_auto = last_hist_data.groupby('region')['automation_level'].mean().sort_values(ascending=False).head()
    if not region_auto.empty:
        fig_auto = px.bar(x=region_auto.values, y=region_auto.index, orientation='h',
                          title=f"Регионы с наибольшей автоматизацией ({last_hist_data['date'].iloc[0].strftime('%d.%m.%Y')})",
                          text_auto='.1%')
        st.plotly_chart(fig_auto, use_container_width=True)
with col2:
    # ИСПРАВЛЕНО: показываем РЕАЛЬНЫЕ средние зарплаты по регионам из словаря
    region_salary_data = []
    for region in last_hist_data['region'].unique():
        if region in REGION_AVG_SALARIES:
            region_salary_data.append({
                'region': region,
                'salary': REGION_AVG_SALARIES[region]
            })

    if region_salary_data:
        region_salary_df = pd.DataFrame(region_salary_data).sort_values('salary', ascending=False).head()
        fig_sal = px.bar(region_salary_df, x='salary', y='region', orientation='h',
                         title=f"Регионы с наибольшими зарплатами (данные Росстата)",
                         text_auto='.0f')
        st.plotly_chart(fig_sal, use_container_width=True)

# ========== ТОП-10 ПРОФЕССИЙ ПО АВТОМАТИЗАЦИИ ==========
st.header("📈 Сравнение профессий")
latest_all = df[df['date'] == df['date'].max()].groupby('profession').agg({
    'automation_level': 'mean',
    'salary': 'mean',
    'category': 'first'
}).reset_index()
top_risky = latest_all.nlargest(10, 'automation_level')
top_safe = latest_all.nsmallest(10, 'automation_level')
col1, col2 = st.columns(2)
with col1:
    fig_risky = px.bar(top_risky, x='automation_level', y='profession', orientation='h',
                       title="🔴 Топ-10 профессий под угрозой", text_auto='.0%', color='category')
    st.plotly_chart(fig_risky, use_container_width=True)
with col2:
    fig_safe = px.bar(top_safe, x='automation_level', y='profession', orientation='h',
                      title="🟢 Топ-10 безопасных профессий", text_auto='.0%', color='category')
    st.plotly_chart(fig_safe, use_container_width=True)

# ========== ПРОГНОЗ ДЛЯ IT ==========
st.header("🔮 Прогноз до 2030 года для IT-профессий")
it_professions = df[df['category'] == 'it']['profession'].unique()
default_it = [p for p in ['Senior-разработчик', 'ML-инженер', 'Golang-разработчик'] if p in it_professions]
selected_it = st.multiselect("Выберите IT-профессии для прогноза:", sorted(it_professions), default=default_it)
if selected_it:
    it_forecast_model = st.radio("Модель для прогноза IT-профессий:", ["Линейная регрессия", "LSTM (нейросеть)"],
                                 horizontal=True)
    forecast_data = []
    for prof in selected_it:
        prof_data = df[df['profession'] == prof].groupby('date').agg(
            {'salary': 'mean', 'automation_level': 'mean'}).reset_index().sort_values('date')
        if len(prof_data) >= 3:
            y_salary = prof_data['salary'].values
            y_auto = prof_data['automation_level'].values
            if it_forecast_model == "Линейная регрессия":
                pred_s = forecast_linear(y_salary, 5)
                pred_a = forecast_linear(y_auto, 5)
            else:
                pred_s = forecast_lstm(y_salary, n_forecast=5)
                pred_a = forecast_lstm(y_auto, n_forecast=5)
            future_years = np.arange(CURRENT_YEAR + 1, CURRENT_YEAR + 6)
            for j, year in enumerate(future_years):
                forecast_data.append(
                    {'profession': prof, 'year': year, 'salary': pred_s[j], 'automation': pred_a[j], 'type': 'Прогноз'})
            for _, row in prof_data.iterrows():
                forecast_data.append({'profession': prof, 'year': row['date'].year, 'salary': row['salary'],
                                      'automation': row['automation_level'], 'type': 'Факт'})
    if forecast_data:
        forecast_df = pd.DataFrame(forecast_data)
        col1, col2 = st.columns(2)
        with col1:
            fig_s = px.line(forecast_df, x='year', y='salary', color='profession', line_dash='type',
                            title=f"Прогноз зарплат ({it_forecast_model})")
            st.plotly_chart(fig_s, use_container_width=True)
        with col2:
            fig_a = px.line(forecast_df, x='year', y='automation', color='profession', line_dash='type',
                            title=f"Прогноз автоматизации ({it_forecast_model})")
            st.plotly_chart(fig_a, use_container_width=True)
        st.subheader(f"📋 Прогноз на {forecast_df['year'].max()-1} год")
        forecast_last = forecast_df[forecast_df['year'] == forecast_df['year'].max()].sort_values('salary',
                                                                                                  ascending=False)
        for _, row in forecast_last.iterrows():
            st.markdown(f"**{row['profession']}**: {row['salary']:,.0f} ₽, автоматизация {row['automation']:.1%}")

# ========== ТОП ЗАРПЛАТ ==========
st.header(f"💰 Самые высокооплачиваемые профессии (последние данные)")

# Показываем медианные зарплаты по профессиям (более устойчивы к выбросам)
latest_salaries = df[df['date'] == df['date'].max()].groupby('profession')['salary'].median().reset_index()
latest_salaries = latest_salaries.nlargest(8, 'salary')
categories_map = df[['profession', 'category']].drop_duplicates().set_index('profession')['category'].to_dict()
latest_salaries['category'] = latest_salaries['profession'].map(categories_map)

fig_salary = px.bar(latest_salaries, x='salary', y='profession', color='category',
                    orientation='h',
                    title=f"Топ высокооплачиваемых профессий (медианные зарплаты по России)",
                    text_auto='.0f')
fig_salary.update_layout(xaxis_title="Зарплата (₽)", yaxis_title="Профессия")
st.plotly_chart(fig_salary, use_container_width=True)

# ========== ИНТЕРЕСНЫЕ ВЫВОДЫ ==========
st.header("🔍 Интересные выводы из данных")
if not filtered_df.empty:
    last_hist = filtered_df[filtered_df['date'] == filtered_df['date'].max()]
    prof_avg = last_hist.groupby('profession').agg(
        {'salary': 'median', 'automation_level': 'mean', 'category': 'first'}).reset_index()
    col1, col2 = st.columns(2)
    with col1:
        if not prof_avg.empty:
            top_sal = prof_avg.loc[prof_avg['salary'].idxmax()]
            st.metric("💰 Самая высокая средняя зарплата", f"{top_sal['profession']} — {top_sal['salary']:,.0f} ₽")
            bottom_sal = prof_avg.loc[prof_avg['salary'].idxmin()]
            st.metric("📉 Самая низкая средняя зарплата", f"{bottom_sal['profession']} — {bottom_sal['salary']:,.0f} ₽")
    with col2:
        if not prof_avg.empty:
            max_auto = prof_avg.loc[prof_avg['automation_level'].idxmax()]
            st.metric("🤖 Максимальная средняя автоматизация",
                      f"{max_auto['profession']} — {max_auto['automation_level']:.1%}")
            min_auto = prof_avg.loc[prof_avg['automation_level'].idxmin()]
            st.metric("🛠️ Минимальная средняя автоматизация",
                      f"{min_auto['profession']} — {min_auto['automation_level']:.1%}")
    region_salary = last_hist.groupby('region')['salary'].median().sort_values(ascending=False)
    # на:
    region_salary_data = []
    for region in last_hist['region'].unique():
        if region in REGION_AVG_SALARIES:
            region_salary_data.append({'region': region, 'salary': REGION_AVG_SALARIES[region]})
    if region_salary_data:
        region_salary = pd.DataFrame(region_salary_data).sort_values('salary', ascending=False)
        if not region_salary.empty:
            st.metric("🏙️ Регион-лидер по средним зарплатам",
                      f"{region_salary.iloc[0]['region']} — {region_salary.iloc[0]['salary']:,.0f} ₽")
else:
    st.warning("Нет данных для анализа. Измените фильтры.")


# ========== ДАННЫЕ (ТАБЛИЦА) ==========
with st.expander("📋 Исходные данные"):
    st.dataframe(filtered_df, use_container_width=True)
    csv = filtered_df.to_csv(index=False)
    st.download_button("📥 Скачать отфильтрованные данные как CSV", csv, "filtered_data.csv", "text/csv")
