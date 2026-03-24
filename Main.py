import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
import warnings
import os
from datetime import datetime, date

warnings.filterwarnings('ignore')

# ========== НЕЙРОСЕТЬ (LSTM) ==========
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

tf.random.set_seed(42)
np.random.seed(42)

# ========== НАСТРОЙКА СТРАНИЦЫ ==========
st.set_page_config(
    page_title="ИИ и рынок труда | Ежедневный датасет",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Влияние искусственного интеллекта на рынок труда")
st.markdown("#### Ежедневное накопление данных с hh.ru и прогноз до 2030 года")

# ========== КОНСТАНТЫ ==========
AVG_RUSSIAN_SALARY = 70000          # для региональной нормализации
DATA_FILE = "professions_daily.csv" # единый файл для накопления данных
CURRENT_DATE = date.today()
CURRENT_YEAR = CURRENT_DATE.year

# ========== ФУНКЦИЯ ГЕНЕРАЦИИ ПОЛНОЙ ИСТОРИИ (2015-ТЕКУЩИЙ) ==========
def generate_full_history():
    """
    Генерирует полные данные с 2015 по текущий год (включая сегодня)
    на основе экспертных данных, трендов и региональных коэффициентов.
    """
    from data_loader import parse_hh_salaries, get_expert_data, get_regional_data, get_backup_salaries

    expert_df = get_expert_data()
    professions_list = expert_df['profession'].tolist()

    try:
        salary_df = parse_hh_salaries(professions_list)
    except Exception as e:
        st.warning(f"Ошибка парсинга hh.ru при генерации истории: {e}")
        salary_df = get_backup_salaries()

    merged_df = expert_df.merge(salary_df, on='profession', how='left')
    regions = get_regional_data()
    data = []

    for _, row in merged_df.iterrows():
        prof = row['profession']
        category = row['category']
        ai_impact = row['ai_impact']
        trend = row['trend']
        verdict = row['verdict']
        base_salary = row.get('salary_mean', AVG_RUSSIAN_SALARY)

        for year in range(2015, CURRENT_YEAR + 1):
            # Экстраполяция зарплаты и спроса на основе тренда
            if trend == 'growing':
                salary_year = base_salary / (1 + 0.08 * (CURRENT_YEAR - year))
                demand_year = 1000 * (1 - 0.02 * (CURRENT_YEAR - year))
            elif trend == 'declining':
                salary_year = base_salary * (1 + 0.05 * (CURRENT_YEAR - year))
                demand_year = 1000 * (1 + 0.10 * (CURRENT_YEAR - year))
            else:  # stable
                salary_year = base_salary * (1 + 0.03 * (CURRENT_YEAR - year))
                demand_year = 1000 * (1 + 0.01 * (CURRENT_YEAR - year))

            # Уровень автоматизации (растёт со временем)
            automation = ai_impact * (0.5 + 0.04 * (year - 2015))
            automation = min(automation, 0.95)

            for region, reg_data in regions.items():
                region_factor = reg_data['salary'] / AVG_RUSSIAN_SALARY
                auto_factor = reg_data.get('auto_factor', 0.9)
                demand_factor = 1.5 if region in ['Москва', 'Санкт-Петербург'] else 1.0

                # Дата: для текущего года ставим сегодняшнюю дату, для прошлых — 1 января
                if year == CURRENT_YEAR:
                    record_date = CURRENT_DATE
                else:
                    record_date = datetime(year, 1, 1)

                data.append({
                    'date': record_date,
                    'year': year,
                    'profession': prof,
                    'category': category,
                    'region': region,
                    'latitude': reg_data['lat'],
                    'longitude': reg_data['lon'],
                    'salary': round(salary_year * region_factor),
                    'demand': round(max(demand_year * demand_factor, 10)),
                    'automation_level': round(min(automation * auto_factor, 0.95), 3),
                    'verdict': verdict,
                    'trend': trend
                })

    df = pd.DataFrame(data)
    return df

# ========== ФУНКЦИЯ ДОБАВЛЕНИЯ ДАННЫХ ЗА ТЕКУЩИЙ ДЕНЬ ==========
def fetch_today_data():
    """Парсит текущие данные с hh.ru и возвращает DataFrame только для сегодняшней даты."""
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
        ai_impact = row['ai_impact']
        trend = row['trend']
        verdict = row['verdict']
        base_salary = row.get('salary_mean', AVG_RUSSIAN_SALARY)

        # Расчёт автоматизации на текущий год (используем формулу из старой версии)
        automation = ai_impact * (0.5 + 0.04 * (CURRENT_YEAR - 2015))
        automation = min(automation, 0.95)

        for region, reg_data in regions.items():
            region_factor = reg_data['salary'] / AVG_RUSSIAN_SALARY
            auto_factor = reg_data.get('auto_factor', 0.9)
            demand_factor = 1.5 if region in ['Москва', 'Санкт-Петербург'] else 1.0
            salary_region = base_salary * region_factor
            demand = 1000 * demand_factor

            new_rows.append({
                'date': CURRENT_DATE,
                'year': CURRENT_YEAR,
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

# ========== ОСНОВНАЯ ФУНКЦИЯ ЗАГРУЗКИ ДАННЫХ ==========
@st.cache_data(ttl=3600, show_spinner=False)
def get_full_dataset():
    if os.path.exists(DATA_FILE):
        # Читаем CSV как строки сначала, чтобы избежать проблем с датами
        df = pd.read_csv(DATA_FILE)

        # Принудительно преобразуем колонку date в datetime с обработкой ошибок
        # format='mixed' позволяет обрабатывать разные форматы
        df['date'] = pd.to_datetime(df['date'], format='mixed', errors='coerce')

        # Проверяем, есть ли NaN в датах (это значит, что парсинг не удался)
        if df['date'].isna().any():
            st.warning(f"Обнаружены некорректные даты. Попытка исправить...")
            # Пробуем другой подход: сначала читаем как строки, потом парсим
            df_temp = pd.read_csv(DATA_FILE)
            df_temp['date'] = pd.to_datetime(df_temp['date'], errors='coerce')
            df = df_temp

        today = CURRENT_DATE
        # Преобразуем today в datetime для сравнения
        today_dt = datetime.combine(today, datetime.min.time())

        # Проверяем, есть ли данные за сегодня
        if today_dt not in df['date'].dt.date.unique():
            new_data = fetch_today_data()
            if new_data is not None and not new_data.empty:
                df = pd.concat([df, new_data], ignore_index=True)
                df.to_csv(DATA_FILE, index=False)
                st.sidebar.success(f"💾 Добавлены данные за {today}")
        return df
    else:
        st.sidebar.info("📁 Создаём новый датасет (исторические данные + сегодня)...")
        full_df = generate_historical_data()
        if full_df is not None:
            full_df.to_csv(DATA_FILE, index=False)
            st.sidebar.success(f"💾 Датасет создан: {len(full_df)} записей.")
            return full_df
        else:
            st.error("Не удалось создать датасет.")
            st.stop()
            
    # Файл существует — читаем
    df = pd.read_csv(DATA_FILE, parse_dates=['date'])

    # Проверяем, есть ли данные за все годы от 2015 до текущего
    existing_years = set(df['year'].unique())
    required_years = set(range(2015, CURRENT_YEAR + 1))
    missing_years = required_years - existing_years

    if missing_years:
        st.sidebar.warning(f"Обнаружены отсутствующие годы: {sorted(missing_years)}. Добавляем...")
        # Генерируем полную историю и объединяем с существующими данными
        full_history = generate_full_history()
        # Удаляем из full_history те годы, которые уже есть в df, чтобы не дублировать
        full_history = full_history[~full_history['year'].isin(existing_years)]
        if not full_history.empty:
            df = pd.concat([df, full_history], ignore_index=True)
            df.to_csv(DATA_FILE, index=False)
            st.sidebar.success(f"Добавлены данные за {sorted(missing_years)}")

    # Проверяем, есть ли данные за сегодня (актуальный парсинг)
    today = CURRENT_DATE
    if today not in df['date'].dt.date.unique():
        st.sidebar.info(f"🆕 Добавляем данные за {today.strftime('%d.%m.%Y')}...")
        today_data = fetch_today_data()
        if today_data is not None and not today_data.empty:
            df = pd.concat([df, today_data], ignore_index=True)
            df.to_csv(DATA_FILE, index=False)
            st.sidebar.success(f"💾 Добавлены данные за {today}")

    return df

# ========== ЗАГРУЗКА ДАННЫХ ==========
df = get_full_dataset()
df['date'] = pd.to_datetime(df['date'])

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

# ========== ИНТЕРФЕЙС ==========
st.sidebar.header("🔍 Фильтры")

# Для фильтрации используем год из даты
df['year'] = df['date'].dt.year

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

forecast_model = st.sidebar.radio("Модель прогнозирования:", ["Линейная регрессия", "LSTM (нейросеть)"])

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
    high_risk = len(latest_data[latest_data['automation_level'] > 0.7]['profession'].unique()) if not latest_data.empty else 0
    st.metric("Профессий под угрозой (>70%)", high_risk)
with col4:
    st.metric(f"Навыков трансформируется к {CURRENT_YEAR+5}", "41%")

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
    fig_map.update_layout(mapbox_style="carto-positron", mapbox_center={"lat": 60, "lon": 90}, margin={"r": 0, "t": 40, "l": 0, "b": 0})
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
        fig_auto = px.bar(x=region_auto.values, y=region_auto.index, orientation='h', title=f"Регионы с наибольшей автоматизацией ({last_hist_data['date'].iloc[0].strftime('%d.%m.%Y')})", text_auto='.1%')
        st.plotly_chart(fig_auto, use_container_width=True)
with col2:
    region_salary = last_hist_data.groupby('region')['salary'].mean().sort_values(ascending=False).head()
    if not region_salary.empty:
        fig_sal = px.bar(x=region_salary.values, y=region_salary.index, orientation='h', title=f"Регионы с наибольшими зарплатами", text_auto='.0f')
        st.plotly_chart(fig_sal, use_container_width=True)

# ========== ТОП-10 ПРОФЕССИЙ ==========
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
    fig_risky = px.bar(top_risky, x='automation_level', y='profession', orientation='h', title="🔴 Топ-10 профессий под угрозой", text_auto='.0%', color='category')
    st.plotly_chart(fig_risky, use_container_width=True)
with col2:
    fig_safe = px.bar(top_safe, x='automation_level', y='profession', orientation='h', title="🟢 Топ-10 безопасных профессий", text_auto='.0%', color='category')
    st.plotly_chart(fig_safe, use_container_width=True)

# ========== ПРОГНОЗ ДЛЯ IT ==========
st.header("🔮 Прогноз до 2030 года для IT-профессий")
it_professions = df[df['category'] == 'it']['profession'].unique()
default_it = [p for p in ['Senior-разработчик', 'ML-инженер', 'Golang-разработчик'] if p in it_professions]
selected_it = st.multiselect("Выберите IT-профессии для прогноза:", sorted(it_professions), default=default_it)
if selected_it:
    it_forecast_model = st.radio("Модель для прогноза IT-профессий:", ["Линейная регрессия", "LSTM (нейросеть)"], horizontal=True)
    forecast_data = []
    for prof in selected_it:
        prof_data = df[df['profession'] == prof].groupby('date').agg({'salary': 'mean', 'automation_level': 'mean'}).reset_index().sort_values('date')
        if len(prof_data) >= 3:
            y_salary = prof_data['salary'].values
            y_auto = prof_data['automation_level'].values
            if it_forecast_model == "Линейная регрессия":
                X = np.arange(len(y_salary)).reshape(-1, 1)
                lr_s = LinearRegression().fit(X, y_salary)
                lr_a = LinearRegression().fit(X, y_auto)
                pred_s = lr_s.predict(np.arange(len(y_salary), len(y_salary)+5).reshape(-1, 1))
                pred_a = lr_a.predict(np.arange(len(y_salary), len(y_salary)+5).reshape(-1, 1))
            else:
                pred_s = forecast_lstm(y_salary, n_forecast=5)
                pred_a = forecast_lstm(y_auto, n_forecast=5)
            future_years = np.arange(CURRENT_YEAR+1, CURRENT_YEAR+6)
            for j, year in enumerate(future_years):
                forecast_data.append({'profession': prof, 'year': year, 'salary': pred_s[j], 'automation': pred_a[j], 'type': 'Прогноз'})
            for _, row in prof_data.iterrows():
                forecast_data.append({'profession': prof, 'year': row['date'].year, 'salary': row['salary'], 'automation': row['automation_level'], 'type': 'Факт'})
    if forecast_data:
        forecast_df = pd.DataFrame(forecast_data)
        col1, col2 = st.columns(2)
        with col1:
            fig_s = px.line(forecast_df, x='year', y='salary', color='profession', line_dash='type', title=f"Прогноз зарплат ({it_forecast_model})")
            st.plotly_chart(fig_s, use_container_width=True)
        with col2:
            fig_a = px.line(forecast_df, x='year', y='automation', color='profession', line_dash='type', title=f"Прогноз автоматизации ({it_forecast_model})")
            st.plotly_chart(fig_a, use_container_width=True)
        st.subheader(f"📋 Прогноз на {forecast_df['year'].max()} год")
        forecast_last = forecast_df[forecast_df['year'] == forecast_df['year'].max()].sort_values('salary', ascending=False)
        for _, row in forecast_last.iterrows():
            st.markdown(f"**{row['profession']}**: {row['salary']:,.0f} ₽, автоматизация {row['automation']:.1%}")

# ========== ТОП ЗАРПЛАТ ==========
st.header(f"💰 Самые высокооплачиваемые профессии (последние данные)")
latest_salaries = df[df['date'] == df['date'].max()].groupby('profession')['salary'].mean().reset_index().nlargest(8, 'salary')
categories_map = df[['profession', 'category']].drop_duplicates().set_index('profession')['category'].to_dict()
latest_salaries['category'] = latest_salaries['profession'].map(categories_map)
fig_salary = px.bar(latest_salaries, x='salary', y='profession', color='category', orientation='h', title=f"Топ высокооплачиваемых профессий", text_auto='.0f')
st.plotly_chart(fig_salary, use_container_width=True)

# ========== ИНТЕРЕСНЫЕ ВЫВОДЫ ==========
st.header("🔍 Интересные выводы из данных")
if not filtered_df.empty:
    last_hist = filtered_df[filtered_df['date'] == filtered_df['date'].max()]
    prof_avg = last_hist.groupby('profession').agg({'salary': 'mean', 'automation_level': 'mean', 'category': 'first'}).reset_index()
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
            st.metric("🤖 Максимальная средняя автоматизация", f"{max_auto['profession']} — {max_auto['automation_level']:.1%}")
            min_auto = prof_avg.loc[prof_avg['automation_level'].idxmin()]
            st.metric("🛠️ Минимальная средняя автоматизация", f"{min_auto['profession']} — {min_auto['automation_level']:.1%}")
    region_salary = last_hist.groupby('region')['salary'].mean().sort_values(ascending=False)
    if not region_salary.empty:
        st.metric("🏙️ Регион-лидер по средним зарплатам", f"{region_salary.index[0]} — {region_salary.iloc[0]:,.0f} ₽")
else:
    st.warning("Нет данных для анализа. Измените фильтры.")

# ========== ТРЕНДЫ ==========
st.header(f"📌 Ключевые тренды рынка труда (на основе накопленных данных)")
last_year = df[df['year'] == df['year'].max()]
prev_year = df[df['year'] == df['year'].max() - 1]
if not prev_year.empty:
    demand_change = (last_year.groupby('category')['demand'].mean() / prev_year.groupby('category')['demand'].mean() - 1) * 100
    declining = demand_change[demand_change < -5].index.tolist()
    growing = demand_change[demand_change > 5].index.tolist()
    risky_profs = last_year[last_year['automation_level'] > 0.7]['profession'].unique()
    safe_profs = last_year[last_year['automation_level'] < 0.3]['profession'].unique()
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 📉 Снижение спроса")
        st.markdown(f"**Категории:** {', '.join(declining) if declining else 'Нет'}")
        st.markdown(f"**Примеры профессий под угрозой:** {', '.join(risky_profs[:5]) if len(risky_profs)>0 else 'Нет'}")
    with col2:
        st.markdown("### 📈 Рост спроса")
        st.markdown(f"**Категории:** {', '.join(growing) if growing else 'Нет'}")
        st.markdown(f"**Примеры безопасных профессий:** {', '.join(safe_profs[:5]) if len(safe_profs)>0 else 'Нет'}")

# ========== ДАННЫЕ (ТАБЛИЦА) ==========
with st.expander("📋 Исходные данные"):
    st.dataframe(filtered_df, use_container_width=True)
    csv = filtered_df.to_csv(index=False)
    st.download_button("📥 Скачать отфильтрованные данные как CSV", csv, "filtered_data.csv", "text/csv")
