# %% =============================================================================
# КРОК 1: ІМПОРТ НЕОБХІДНИХ ПАКЕТІВ ТА БІБЛІОТЕК (SENIOR СТАНДАРТ)
# =============================================================================

# Базові бібліотеки для маніпуляції даними та математичних обчислень
import numpy as np
import pandas as pd

# Інструменти Scikit-Learn для передобробки, масштабування та кодування
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Модель логістичної регресії та метрики оцінки якості класифікації
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Налаштування для вимикання некритичних системних попереджень
import warnings
warnings.filterwarnings("ignore")

print("КРОК 1 ВИКОНАНО: Всі бібліотеки імпортовано.")


# %% =============================================================================
# КРОК 2: ЗАВАНТАЖЕННЯ НАБОРУ ДАНИХ "RAIN IN AUSTRALIA"
# =============================================================================

data_path = 'weatherAUS.csv'
data = pd.read_csv(data_path)

# Первинний контроль розмірності
print("=== КРОК 2 ВИКОНАНО ===")
print(f"Розмірність завантаженого датасету: {data.shape}")
print(f"Кількість колонок: {len(data.columns)}")


# %% =============================================================================
# КРОК 3: ПЕРЕДОБРОБКА ТА ІСТОРИЧНЕ РОЗДІЛЕННЯ ВИБІРОК (OUT-OF-TIME)
# =============================================================================

# --- Крок 3.1: Видалення ознак з великою кількістю пропусків
# Залишаємо колонки, де пропусків менше 35%
data = data[data.columns[data.isna().mean().lt(0.35)]]

# Видаляємо рядки без цільової змінної та дублікати
data = data.dropna(subset=['RainTomorrow'])
data = data.drop_duplicates()

# Відділяємо таргет (y) від ознак (X) до створення підмножин
X = data.drop('RainTomorrow', axis=1)
y = data['RainTomorrow']

# --- Крок 3.2: Створюємо підмножини числових та категоріальних ознак
X_num = X.select_dtypes(include=[np.number]).copy()
X_cat = X.select_dtypes(include=['object']).copy()

# --- Крок 3.3: Змінюємо тип Date та створюємо Year і Month
# Оскільки Date була текстом, вона зараз знаходиться у X_cat
X_cat['Date'] = pd.to_datetime(X_cat['Date'])
X_cat['Year'] = X_cat['Date'].dt.year
X_cat['Month'] = X_cat['Date'].dt.month.astype(str) # Місяць як категорія

# Оригінальна колонка Date більше не потрібна
X_cat = X_cat.drop('Date', axis=1)

# --- Крок 3.4: Переміщаємо колонку Year
# Витягуємо Year з X_cat (за допомогою pop) і передаємо в X_num
X_num['Year'] = X_cat.pop('Year')

# --- Крок 3.5: Розбиття на Train та Test за останнім роком
max_year = X_num['Year'].max()

# Булева індексація
train_mask = X_num['Year'] < max_year
test_mask = X_num['Year'] == max_year

X_train_num = X_num[train_mask]
X_test_num = X_num[test_mask]

X_train_cat = X_cat[train_mask]
X_test_cat = X_cat[test_mask]

y_train = y[train_mask]
y_test = y[test_mask]

# Контрольний вивід
print("=== КРОК 3 ВИКОНАНО ===")
print(f"Максимальний рік (йде в Тест): {max_year}")
print(f"Розмірність Train (числові): {X_train_num.shape}, (категоріальні): {X_train_cat.shape}")
print(f"Розмірність Test (числові): {X_test_num.shape}, (категоріальні): {X_test_cat.shape}")


# %% =============================================================================
# КРОК-ПЕРЕВІРКА:
# =============================================================================
print("=== ПЕРЕВІРКА ОЧИЩЕНОГО ДАТАСЕТУ ===")
print(f"1. Всі роки, присутні в даних: {sorted(X_num['Year'].unique())}")
print(f"2. Загальна кількість днів (рядків) після очищення: {len(X)}")
print(f"3. Загальна кількість ознак (без таргету): {len(X.columns)}")
print(f"4. Список ознак: {list(X.columns)}")


# %% =============================================================================
# КРОК 4: ОБРОБКА ПРОПУСКІВ, МАСШТАБУВАННЯ ТА КОДУВАННЯ (БЕЗ ВИТОКУ ДАНИХ) - ДЗ пункти 4, 5, 6, 7 (перша частина)
# =============================================================================

# --- 4: Обробка числових ознак ---
# Заповнюємо пропуски медіаною
num_imputer = SimpleImputer(strategy='median')
X_train_num_imp = num_imputer.fit_transform(X_train_num)
X_test_num_imp = num_imputer.transform(X_test_num)

# --- 5: Масштабуємо числові дані (StandardScaler)
scaler = StandardScaler()
X_train_num_scaled = scaler.fit_transform(X_train_num_imp)
X_test_num_scaled = scaler.transform(X_test_num_imp)

# --- 6: Обробка категоріальних ознак ---
# Заповнюємо пропуски найчастішим значенням
cat_imputer = SimpleImputer(strategy='most_frequent')
X_train_cat_imp = cat_imputer.fit_transform(X_train_cat)
X_test_cat_imp = cat_imputer.transform(X_test_cat)

# Кодуємо категоріальні ознаки (OneHotEncoder)
# sparse_output=False повертає звичайний масив замість розрідженої матриці
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X_train_cat_encoded = encoder.fit_transform(X_train_cat_imp)
X_test_cat_encoded = encoder.transform(X_test_cat_imp)

# --- 7: Об'єднання оброблених ознак ---
# Збираємо числові та категоріальні масиви назад в єдиний набір даних
X_train_final = np.hstack((X_train_num_scaled, X_train_cat_encoded))
X_test_final = np.hstack((X_test_num_scaled, X_test_cat_encoded))

# Контрольний вивід
print("=== КРОК 4 ВИКОНАНО ===")
print(f"Фінальна розмірність Train: {X_train_final.shape}")
print(f"Фінальна розмірність Test: {X_test_final.shape}")



# #%% =============================================================================
# КРОК 7, 8: НАВЧАННЯ МОДЕЛІ ТА ОЦІНКА ЯКОСТІ (Пункти 7 та 8 ДЗ)
# =============================================================================

# Ініціалізуємо логістичну регресію.
# ЕКСПЕРИМЕНТ: зараз стоїть solver='liblinear'. Можеш потім змінити на 'saga' або 'lbfgs'
# class_weight='balanced' допомагає моделі звертати більше уваги на дощові дні, яких у природі менше.
log_reg = LogisticRegression(class_weight='balanced', solver='liblinear', random_state=42)

# Навчаємо модель ВИКЛЮЧНО на історичних (тренувальних) даних
print("Модель навчається... Це може зайняти кілька секунд.")
log_reg.fit(X_train_final, y_train)

# Робимо прогноз для нашого "майбутнього" (2017 рік - тестові дані)
y_pred = log_reg.predict(X_test_final)

# Виводимо фінальний звіт про якість класифікації
print("=== КРОК 5 ВИКОНАНО: ЗВІТ КЛАСИФІКАЦІЇ (OUT-OF-TIME TEST) ===")
print(classification_report(y_test, y_pred))































