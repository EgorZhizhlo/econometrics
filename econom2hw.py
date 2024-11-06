# %% md
## Работу выполнил Осинский А.  группа - ПМ23-2

# %%
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import statsmodels.api as sm
import itertools

# %% md
## Предобработка Данных
# %%


file_path = 'econometricsHW2.xlsx'

try:
    df = pd.read_excel(file_path, header=[0, 1])
    print("Данные успешно загружены.")
except FileNotFoundError:
    print("Файл не найден. Убедитесь, что указали правильный путь к файлу.")
    df = None
except Exception as e:
    print(f"Ошибка при загрузке данных: {e}")
    df = None

if df is not None:
    # Просмотр первых 5 строк данных
    print("\nПервые 5 строк данных:")
    print(df.head())

    # Проверка типа столбцов
    print("\nТип столбцов:")
    print(df.columns)

    # Проверка, является ли df.columns MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        print("\nСтолбцы являются MultiIndex. Выполняем объединение уровней заголовков.")
        # Объединение уровней MultiIndex в одно имя столбца с пробелом
        df.columns = [' '.join([str(i) for i in col]).strip() for col in df.columns.values]
        print("Новые названия столбцов после объединения:")
        print(df.columns.tolist())
    else:
        print("\nСтолбцы являются обычным Index.")

    print("\nКолонки до очистки:")
    print(df.columns.tolist())

    df.columns = df.columns.str.replace(r'\s+', ' ', regex=True)

    df.columns = df.columns.str.replace(r'\s+[XYА-Яа-я]+\d*\.?\d*$', '', regex=True)

    print("\nКолонки после очистки:")
    print(df.columns.tolist())

    numeric_columns = [
        'Прибыль (убыток)',
        'Долгосрочные обязательства',
        'Краткосрочные обязательства',
        'Оборотные активы',
        'Основные средства',
        'Дебиторская задолженность (краткосрочная)',
        'Запасы готовой продукции и товаров для перепродажи'
    ]

    missing_cols = [col for col in numeric_columns if col not in df.columns]
    if missing_cols:
        print(f"\nОтсутствуют следующие столбцы: {missing_cols}")
        print("Проверьте названия столбцов после объединения и очистки.")
    else:
        print("\nВсе необходимые столбцы присутствуют.")

        print("\nТипы данных после обработки:")
        print(df[numeric_columns].dtypes)

# %% md

## Обнаружение и очистка  выбросов с помощью IQR

# %%

if df.index.duplicated().any():
    print("Обнаружены дубликаты в индексах. Выполняем сброс индекса.")
    df = df.reset_index(drop=True)


def detect_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return data[(data[column] < lower_bound) | (data[column] > upper_bound)]


outlier_indices = set()
outliers_summary = {}

for col in numeric_columns:
    outliers = detect_outliers_iqr(df, col)
    outliers_summary[col] = outliers.shape[0]
    outlier_indices.update(outliers.index.tolist())

print("\nКоличество выбросов в каждом столбце:")
for col, count in outliers_summary.items():
    print(f"{col}: {count}")

print(f"\nОбщее количество уникальных строк с выбросами: {len(outlier_indices)}")


def remove_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]


df_cleaned = df.copy()

for col in numeric_columns:
    initial_shape = df_cleaned.shape[0]
    df_cleaned = remove_outliers_iqr(df_cleaned, col)
    final_shape = df_cleaned.shape[0]
    removed = initial_shape - final_shape
    print(f"Удалено {removed} выбросов из столбца '{col}'. Осталось строк: {final_shape}")

# %% md
## Построение диаграмм рассеяния для очищенных данных
# %%
if df_cleaned is not None and not df_cleaned.empty:
    def plot_scatter_separately(data, dependent_var, independent_vars):
        for var in independent_vars:
            plt.figure(figsize=(20, 10))  # Большой размер для полноэкранного вывода
            sns.scatterplot(x=data[var], y=data[dependent_var], s=70)  # Размер маркеров
            plt.xlabel(var, fontsize=18)  # Увеличиваем размер шрифта подписей осей
            plt.ylabel(dependent_var, fontsize=18)
            plt.title(f'{dependent_var} vs {var}', fontsize=22)  # Увеличиваем размер шрифта заголовка
            plt.show()


    dependent_variable = 'Прибыль (убыток)'
    independent_variables = [
        'Долгосрочные обязательства',
        'Краткосрочные обязательства',
        'Оборотные активы',
        'Основные средства',
        'Дебиторская задолженность (краткосрочная)',
        'Запасы готовой продукции и товаров для перепродажи'
    ]

    # Построение диаграмм рассеяния
    plot_scatter_separately(df_cleaned, dependent_variable, independent_variables)

# %% md
## Вычисление корреляционной матрицы
# %%

corr_matrix = df_cleaned[numeric_columns].corr()

print("\nКорреляционная матрица:")
print(corr_matrix)

plt.figure(figsize=(10, 8))

sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)

plt.title('Корреляционная матрица числовых столбцов', fontsize=16)

plt.show()


# %% md
## Метод пошагового отбора признаков - Forward_selection

# %%
def forward_selection(data, response, predictors, significance_level=0.05):
    """
    Perform a forward selection based on p-value from statsmodels.api.OLS

    Arguments:
        data - pandas DataFrame with all possible predictors and response
        response - string, name of response column in data
        predictors - list of strings, names of predictor columns in data
        significance_level - p-value used to include a variable

    Returns:
        list of selected predictors
    """
    initial_features = []
    best_features = []
    remaining_features = set(predictors)
    while remaining_features:
        scores_with_candidates = []
        for candidate in remaining_features:
            model = sm.OLS(data[response], sm.add_constant(data[initial_features + [candidate]])).fit()
            p_value = model.pvalues[candidate]
            scores_with_candidates.append((candidate, p_value))
        # Select the candidate with the lowest p-value
        scores_with_candidates.sort(key=lambda x: x[1])
        best_candidate, best_p_value = scores_with_candidates[0]
        if best_p_value < significance_level:
            initial_features.append(best_candidate)
            remaining_features.remove(best_candidate)
            best_features = initial_features.copy()
            print(f"Добавлен фактор '{best_candidate}' с p-значением {best_p_value:.4f}")
        else:
            break
    return best_features


# %%
# Определение зависимой переменной и списка регрессоров
dependent_variable = 'Прибыль (убыток)'
independent_variables = [
    'Долгосрочные обязательства',
    'Краткосрочные обязательства',
    'Оборотные активы',
    'Основные средства',
    'Дебиторская задолженность (краткосрочная)',
    'Запасы готовой продукции и товаров для перепродажи'
]

# Применение метода пошагового отбора
selected_features = forward_selection(df_cleaned, dependent_variable, independent_variables)

print("\nОтобранные факторы после пошагового отбора:")
print(selected_features)

# %% md
## Построение Модели с отобранными факторами

# %%
# Построение модели с отобранными факторами
X = sm.add_constant(df_cleaned[selected_features])
y = df_cleaned[dependent_variable]

model = sm.OLS(y, X).fit()

# Вывод результатов модели
print(model.summary())

# %% md
### Выводы для результатов модели

1. ** R
squared(R ^ 2) **: 0.538
Интерпретация: Модель
объясняет
примерно
53.8 % вариации
прибыли
компании.Это
означает, что
выбранные
независимые
переменные(факторы)
относительно
хорошо
описывают
зависимую
переменную(прибыль), но
остаётся
около
46.2 % вариации, не
объяснённой
моделью.
2. ** F - статистика: 13.37(p - value: 0.000141) **: Очень
низкое
p - значение
свидетельствует
о
статистической
значимости
модели
в
целом.Это
означает, что
хотя
некоторые
предикторы
могут
быть
незначимыми, модель
как
совокупность
предикторов
существенно
предсказывает
прибыль.
3. ** Краткосрочные
обязательства(X1) **: Краткосрочные
обязательства
являются
статистически
значимым
положительным
предиктором
прибыли(p < 0.05).Это
указывает
на
то, что
увеличение
краткосрочных
обязательств
связано
с
увеличением
прибыли
компании.Для
каждого
увеличения
краткосрочных
обязательств
на
единицу
прибыль
увеличивается
в
среднем
на
0.5696
единиц, при
условии
фиксированных
остальных
факторов.
4. ** Запасы
готовой
продукции
и
товаров
для
перепродажи(Х6) **: Запасы
также
являются
статистически
значимым
положительным
предиктором
прибыли(p < 0.05).Это
означает, что
увеличение
запасов
готовой
продукции
и
товаров
для
перепродажи
связано
с
увеличением
прибыли
компании.Для
каждого
увеличения
запасов
на
единицу
прибыль
увеличивается
в
среднем
на
23.7531
единиц, при
условии
фиксированных
остальных
факторов.
5. ** Основные
выводы **:
Значимость
модели: Модель
статистически
значима(p < 0.05), что
подтверждается
низким
p - значением
F - статистики.Это
указывает
на
то, что
модель
существенно
предсказывает
прибыль.

Влияние
факторов:

Краткосрочные
обязательства
и
запасы
готовой
продукции
и
товаров
для
перепродажи
являются
значимыми
предикторами
прибыли.Их
увеличение
связано
с
ростом
прибыли.
Перехват
не
значим, что
подразумевает
отсутствие
значимого
среднего
уровня
прибыли
при
нулевых
значениях
независимых
переменных.
Проблемы
модели:

Мультиколлинеарность: Очень
высокий
коэффициент
условности
указывает
на
сильную
мультиколлинеарность
между
предикторами.Это
может
привести
к
нестабильности
оценок
коэффициентов
и
затруднить
интерпретацию
влияния
отдельных
факторов.
Нормальность
остатков: Хотя
тесты
показывают
пограничные
результаты, остатки
в
целом
не
отклоняются
существенно
от
нормального
распределения.
6. ** Важно
отметить **: у
нас
всего
26
наблюдений
и
этот
Относительно
небольшой
размер
выборки
может
ограничивать
обобщаемость
выводов
и
повышает
риск
переобучения
модели.