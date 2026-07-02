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

