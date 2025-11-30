import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Загрузка данных ---
try:
    allEnergy = pd.read_excel('Production of electricity.xlsx')
    gas = pd.read_excel('Natural gas production volume (stat kg).xlsx')
    coal = pd.read_excel('Volume of coal and lignite production (stat kg).xlsx')
    energyUse = pd.read_excel('Energy use in Kyrgyz Republic(WBOD).xlsx')
except FileNotFoundError as e:
    print(f"Ошибка: Не удалось найти файл. Убедитесь, что все файлы Excel находятся в рабочей директории. Ошибка: {e}")
    exit()

# --- 2. Функция для трансформации и очистки данных ---
def transform_and_clean_robust(df, value_col_name, row_index=0):
    """Преобразует данные из широкого формата в длинный, очищает 'Год' и 'Значение'."""
    data_series = df.iloc[row_index].copy()

    # Пытаемся взять столбцы, начиная с 4-го (индекс 3), как в предыдущей структуре
    year_columns = data_series.index[3:]

    df_long = data_series[year_columns].to_frame(name=value_col_name)
    df_long.index.name = 'Год'
    df_long = df_long.reset_index()

    df_long['Год'] = df_long['Год'].astype(str).str.replace(r'[^\d]+', '', regex=True).str.strip()
    df_long = df_long[df_long['Год'] != '']
    df_long['Год'] = df_long['Год'].astype(int)

    # Преобразование Значения с обработкой прочерков/нечисел
    df_long[value_col_name] = pd.to_numeric(df_long[value_col_name], errors='coerce')

    df_long = df_long.dropna(subset=[value_col_name, 'Год'])

    return df_long

# Применяем трансформацию
df_energy = transform_and_clean_robust(allEnergy, 'Потребление_МВтч', row_index=0)
df_gas = transform_and_clean_robust(gas, 'Производство_Газ', row_index=0)
df_coal = transform_and_clean_robust(coal, 'Производство_Уголь', row_index=0)
df_use = transform_and_clean_robust(energyUse, 'Использование_на_душу', row_index=0)

# --- 3. Объединение и КОРРЕКТНАЯ ИМПУТАЦИЯ ---

df_combined = df_energy.copy()
df_combined = pd.merge(df_combined, df_gas, on='Год', how='left') # left merge на df_energy
df_combined = pd.merge(df_combined, df_coal, on='Год', how='left')
df_combined = pd.merge(df_combined, df_use, on='Год', how='left')

# Оставляем только те года, для которых есть данные о целевой переменной
df_final = df_combined.dropna(subset=['Потребление_МВтч'])

# Сортируем по году для корректной импутации временных рядов
df_final = df_final.sort_values(by='Год')

# Замена агрессивного dropna() на надежную импутацию
feature_cols = ['Производство_Газ', 'Производство_Уголь', 'Использование_на_душу']

for col in feature_cols:
    # 1. Forward Fill: Заполнение пропусков последним известным значением
    df_final[col] = df_final[col].fillna(method='ffill')

    # 2. Backward Fill: Заполнение пропусков в начале ряда первым известным значением
    df_final[col] = df_final[col].fillna(method='bfill')

    # 3. Финальный Fallback: Если столбец полностью пуст (что маловероятно), заполняем нулем.
    mean_val = df_final[col].mean()
    fallback_value = mean_val if pd.notna(mean_val) else 0
    df_final[col] = df_final[col].fillna(fallback_value)


print(f"Готовые данные для обучения: {len(df_final)} строк (с {df_final['Год'].min()} по {df_final['Год'].max()}).")


# --- 4. Моделирование и Прогнозирование ---
if len(df_final) < 2:
    print("\nКРИТИЧЕСКАЯ ОШИБКА: Недостаточно данных для обучения после очистки.")
else:
    features = ['Год', 'Производство_Газ', 'Производство_Уголь', 'Использование_на_душу']
    target = 'Потребление_МВтч'

    # ТЕПЕРЬ X ГАРАНТИРОВАННО НЕ СОДЕРЖИТ NaN
    X = df_final[features]
    y = df_final[target]

    # Обучение модели
    model = LinearRegression()
    model.fit(X, y) # ЭТА СТРОКА БОЛЬШЕ НЕ ДОЛЖНА ВЫЗЫВАТЬ ОШИБКУ

    # 5. Создание данных для прогноза (5 лет)
    last_year = df_final['Год'].max()
    future_years = pd.DataFrame({
        'Год': range(last_year + 1, last_year + 6)
    })

    # Используем среднее значение из последних 5 лет наблюдаемых данных для будущих признаков
    for col in features[1:]:
        mean_val = df_final[col].tail(5).mean()
        future_years[col] = mean_val

    # Делаем прогноз
    forecast_X = future_years[features]
    forecast_y = model.predict(forecast_X)

    # Формируем финальный результат
    df_forecast = future_years[['Год']].copy()
    df_forecast['Прогноз_Потребление_МВтч'] = forecast_y.round(1)

    print("\n=============================================")
    print("Прогноз потребления электроэнергии на 5 лет:")
    print("=============================================")
    print(df_forecast)

    # 6. Визуализация результатов
    plt.figure(figsize=(12, 7))
    plt.plot(df_final['Год'], df_final['Потребление_МВтч'], label='Фактическое потребление', marker='o', color='blue')
    plt.plot(df_forecast['Год'], df_forecast['Прогноз_Потребление_МВтч'], label='Прогноз (5 лет)', marker='s', linestyle='--', color='red')
    plt.title('Прогноз потребления электроэнергии')
    plt.xlabel('Год')
    plt.ylabel('Потребление (МВт*ч)')
    plt.legend()
    plt.grid(True)
    plt.show()