# -*- coding: utf-8 -*-
"""
Приложение Streamlit для предсказания стоимости страхового полиса.
Использует модель машинного обучения CatBoost для прогнозирования.

Автор: Aziz Rasulov
"""

# =============================================================================
# ИМПОРТ НЕОБХОДИМЫХ БИБЛИОТЕК
# =============================================================================

import numpy as np              # Для работы с числовыми массивами
import pickle                   # Для загрузки сохраненной модели
import pandas as pd             # Для работы с табличными данными
import streamlit as st          # Для создания веб-интерфейса
from pathlib import Path        # Для удобной работы с путями к файлам

# =============================================================================
# ЗАГРУЗКА МОДЕЛИ МАШИННОГО ОБУЧЕНИЯ
# =============================================================================

# Определяем путь к файлу с моделью
MODEL_PATH = Path(__file__).parent / "insurance_catboost.pkl"

@st.cache_resource
def load_model(model_path: Path):
    """
    Загружает предварительно обученную модель CatBoost из файла.
    
    Функция использует кэширование @st.cache_resource для того, чтобы
    модель загружалась только один раз при запуске приложения, а не
    при каждом взаимодействии пользователя с интерфейсом.
    
    Args:
        model_path: Путь к файлу с моделью (.pkl)
    
    Returns:
        Загруженная модель машинного обучения
    """
    try:
        with open(model_path, "rb") as pickle_file:
            model = pickle.load(pickle_file)
        return model
    except FileNotFoundError:
        st.error(f"Файл модели не найден: {model_path}")
        st.info("Убедитесь, что файл 'insurance_catboost.pkl' находится в той же директории.")
        return None
    except Exception as e:
        st.error(f"Ошибка при загрузке модели: {e}")
        return None

# Загружаем модель при старте приложения
classifier = load_model(MODEL_PATH)


# =============================================================================
# ФУНКЦИЯ ПРЕДСКАЗАНИЯ
# =============================================================================

def predict_insurance_cost(
    claim_amount: int,
    marital_status: int,
    age: int,
    tenure: int,
    no_of_family_members: int,
    risk_segmentation: int,
    house_type: int,
    social_class: int,
    customer_education_level: int,
    claim_status: int,
    incident_severity: int,
    authority_contacted: int,
    any_injury: int,
    police_report_available: int,
    incident_hour_of_the_day: int,
    health: int,
    life: int,
    mobile: int,
    motor: int,
    property: int,
    travel: int
) -> float:
    """
    Выполняет предсказание стоимости страховки на основе введенных пользователем данных.
    
    Эта функция принимает все параметры от пользователя, формирует из них
    единый вектор признаков и передает его модели для получения прогноза.
    
    Args:
        claim_amount: Сумма требования к страховой компании
        marital_status: Семейное положение (0 - есть семья, 1 - нет семьи)
        age: Возраст клиента
        tenure: Срок действия полиса в днях
        no_of_family_members: Количество членов семьи
        risk_segmentation: Сегмент риска (0 - низкий, 1 - средний, 2 - высокий)
        house_type: Тип жилья (0 - собственный, 1 - съемный, 2 - ипотечный)
        social_class: Социальный класс (0 - нижний, 1 - средний, 2 - высший)
        customer_education_level: Уровень образования (0-6)
        claim_status: Статус обращения (0 - A, 1 - D)
        incident_severity: Тяжесть инцидента (0 - незначительный, 1 - средний, 2 - критический)
        authority_contacted: Кому сообщили (0 - никто, 1 - другое, 2 - скорая, 3 - полиция)
        any_injury: Наличие травм (0 - да, 1 - нет)
        police_report_available: Отчет полиции (0 - да, 1 - нет)
        incident_hour_of_the_day: Час происшествия (0-23)
        health: Страховка здоровья (0 - да, 1 - нет)
        life: Страховка жизни (0 - да, 1 - нет)
        mobile: Страховка мобильного (0 - да, 1 - нет)
        motor: Автостраховка (0 - да, 1 - нет)
        property: Страховка имущества (0 - да, 1 - нет)
        travel: Туристическая страховка (0 - да, 1 - нет)
    
    Returns:
        Предсказанная стоимость страховки в долларах США
    """
    # Формируем список всех признаков в том порядке, в котором модель ожидает их получить
    # Важно: порядок признаков должен точно соответствовать порядку при обучении модели
    features = [
        claim_amount,           # Сумма требования
        marital_status,         # Семейное положение
        age,                    # Возраст
        tenure,                 # Срок полиса
        no_of_family_members,   # Количество членов семьи
        risk_segmentation,      # Сегмент риска
        house_type,             # Тип жилья
        social_class,           # Социальный класс
        customer_education_level,  # Образование
        claim_status,           # Статус обращения
        incident_severity,      # Тяжесть инцидента
        authority_contacted,    # Кому сообщили
        any_injury,             # Наличие травм
        police_report_available,# Отчет полиции
        incident_hour_of_the_day, # Время происшествия
        health,                 # Страховка здоровья (One-Hot encoded)
        life,                   # Страховка жизни (One-Hot encoded)
        mobile,                 # Страховка мобильного (One-Hot encoded)
        motor,                  # Автостраховка (One-Hot encoded)
        property,               # Страховка имущества (One-Hot encoded)
        travel                  # Туристическая страховка (One-Hot encoded)
    ]
    
    # Преобразуем список в двумерный массив NumPy (модель ожидает 2D массив)
    # [[...]] - двойные скобки создают массив формы (1, n_features)
    features_array = np.array([features])
    
    # Выполняем предсказание с помощью модели
    prediction = classifier.predict(features_array)
    
    # Возвращаем первое (и единственное) значение предсказания
    return prediction[0]


# =============================================================================
# ОСНОВНАЯ ФУНКЦИЯ ПРИЛОЖЕНИЯ (STREAMLIT UI)
# =============================================================================

def main():
    """
    Основная функция приложения Streamlit.
    Создает пользовательский интерфейс для ввода данных и отображения результатов.
    """
    
    # -------------------------------------------------------------------------
    # НАСТРОЙКА СТРАНИЦЫ И ЗАГОЛОВКОВ
    # -------------------------------------------------------------------------
    
    # Устанавливаем заголовок вкладки браузера
    st.set_page_config(
        page_title="Предсказание стоимости страховки",
        page_icon="🛡️",
        layout="centered"
    )
    
    # Главный заголовок приложения
    st.title("🛡️ Оценка стоимости страхового полиса")
    
    # Приветственный баннер с использованием HTML/CSS
    welcome_html = """
    <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
        <h2 style="color: #1f77b4; text-align: center;">
            Добро пожаловать в систему предсказания стоимости страхового полиса
        </h2>
        <p style="text-align: center; color: #666;">
            Заполните форму ниже, чтобы получить оценку стоимости страховки
        </p>
    </div>
    """
    st.markdown(welcome_html, unsafe_allow_html=True)
    
    # Проверка наличия модели перед показом формы
    if classifier is None:
        st.warning("⚠️ Модель не загружена. Пожалуйста, проверьте наличие файла модели.")
        return
    
    # -------------------------------------------------------------------------
    # БОКОВАЯ ПАНЕЛЬ С ИНФОРМАЦИЕЙ
    # -------------------------------------------------------------------------
    
    with st.sidebar:
        st.header("ℹ️ О приложении")
        st.info("""
        Это приложение использует модель машинного обучения 
        **CatBoost** для предсказания стоимости страхового полиса 
        на основе различных факторов.
        
        **Разработчик:** Aziz Rasulov
        
        **Технологии:**
        - Python
        - Streamlit
        - CatBoost
        - scikit-learn
        """)
        
        st.subheader("📋 Как использовать:")
        st.write("""
        1. Заполните все поля формы
        2. Нажмите кнопку **Predict**
        3. Получите результат в долларах США
        """)
    
    # -------------------------------------------------------------------------
    # ФОРМА ВВОДА ДАННЫХ ПОЛЬЗОВАТЕЛЕМ
    # -------------------------------------------------------------------------
    
    st.subheader("📝 Введите данные для расчета")
    
    # Создаем две колонки для более компактного размещения полей
    col1, col2 = st.columns(2)
    
    with col1:
        # ----- ФИНАНСОВЫЕ И ДЕМОГРАФИЧЕСКИЕ ДАННЫЕ -----
        
        claim_amount = st.number_input(
            '💰 Сумма требования (USD)',
            min_value=0,
            step=100,
            value=10000,
            help='Сколько вы хотите получить от страховой компании в случае несчастного случая'
        )
        
        age = st.number_input(
            '🎂 Ваш возраст (лет)',
            min_value=0,
            max_value=120,
            step=1,
            value=35,
            help='Введите ваш текущий возраст'
        )
        
        tenure = st.number_input(
            '📅 Срок полиса (дней)',
            min_value=1,
            step=1,
            value=365,
            help='На какой срок вы хотите оформить страховой полис'
        )
        
        no_of_family_members = st.number_input(
            '👨‍👩‍👧‍👦 Количество членов семьи',
            min_value=1,
            step=1,
            value=3,
            help='Сколько человек в вашей семье'
        )
        
        # ----- КАТЕГОРИАЛЬНЫЕ ПРИЗНАКИ (RADIO BUTTONS) -----
        
        marital_status = st.radio(
            '💍 Семейное положение',
            options=[0, 1],
            format_func=lambda x: "Есть семья (0)" if x == 0 else "Нет семьи (1)",
            index=0,
            help='0 - состоите в браке/есть семья, 1 - холост/нет семьи'
        )
        
        risk_segmentation = st.radio(
            '⚠️ Сегмент риска',
            options=[0, 1, 2],
            format_func=lambda x: ["Низкий риск (0)", "Средний риск (1)", "Высокий риск (2)"][x],
            index=1,
            help='Оценка вероятности наступления страхового случая'
        )
        
        house_type = st.radio(
            '🏠 Тип жилья',
            options=[0, 1, 2],
            format_func=lambda x: ["Собственный (0)", "Съемный (1)", "Ипотечный (2)"][x],
            index=0,
            help='Тип вашего текущего жилья'
        )
        
        social_class = st.radio(
            '👔 Социальный класс',
            options=[0, 1, 2],
            format_func=lambda x: ["Нижний (0)", "Средний (1)", "Высший (2)"][x],
            index=1,
            help='Ваш социальный класс'
        )
        
        customer_education_level = st.radio(
            '🎓 Уровень образования',
            options=[0, 1, 2, 3, 4, 5, 6],
            format_func=lambda x: [
                "Докторантура (0)",
                "Магистратура (1)",
                "PhD (2)",
                "Бакалавриат (3)",
                "Колледж (4)",
                "Старшая школа (5)",
                "Другое (6)"
            ][x],
            index=3,
            help='Ваш уровень образования'
        )
    
    with col2:
        # ----- ПАРАМЕТРЫ СТРАХОВОГО СЛУЧАЯ -----
        
        claim_status = st.radio(
            '📋 Статус обращения',
            options=[0, 1],
            format_func=lambda x: "Статус A (0)" if x == 0 else "Статус D (1)",
            index=0,
            help='Статус вашего обращения за страховым полисом'
        )
        
        incident_severity = st.radio(
            '🚨 Тяжесть инцидента',
            options=[0, 1, 2],
            format_func=lambda x: ["Незначительный (0)", "Средний (1)", "Критический (2)"][x],
            index=1,
            help='Оцененная тяжесть произошедшего инцидента'
        )
        
        authority_contacted = st.radio(
            '📞 Кому сообщили об инциденте',
            options=[0, 1, 2, 3],
            format_func=lambda x: ["Никто (0)", "Другое (1)", "Скорая помощь (2)", "Полиция (3)"][x],
            index=0,
            help='Какие службы были contacted после происшествия'
        )
        
        any_injury = st.radio(
            '🏥 Были ли травмы',
            options=[0, 1],
            format_func=lambda x: "Да (0)" if x == 0 else "Нет (1)",
            index=1,
            help='Получили ли вы какие-либо травмы в результате инцидента'
        )
        
        police_report_available = st.radio(
            '📄 Отчет полиции',
            options=[0, 1],
            format_func=lambda x: "Да, имеется (0)" if x == 0 else "Нет (1)",
            index=1,
            help='Имеется ли официальный отчет полиции по инциденту'
        )
        
        incident_hour_of_the_day = st.number_input(
            '🕐 Время инцидента (час)',
            min_value=0,
            max_value=23,
            step=1,
            value=12,
            help='Во сколько часов произошел инцидент (24-часовой формат)'
        )
        
        # ----- ТИПЫ СТРАХОВОК (ONE-HOT ENCODED) -----
        
        st.markdown("---")
        st.markdown("**📌 Какие типы страховок у вас уже имеются?**")
        
        health = st.radio(
            '❤️ Страховка здоровья',
            options=[0, 1],
            format_func=lambda x: "Да (0)" if x == 0 else "Нет (1)",
            index=1
        )
        
        life = st.radio(
            '💼 Страховка жизни',
            options=[0, 1],
            format_func=lambda x: "Да (0)" if x == 0 else "Нет (1)",
            index=1
        )
        
        mobile = st.radio(
            '📱 Страховка мобильного',
            options=[0, 1],
            format_func=lambda x: "Да (0)" if x == 0 else "Нет (1)",
            index=1
        )
        
        motor = st.radio(
            '🚗 Автостраховка',
            options=[0, 1],
            format_func=lambda x: "Да (0)" if x == 0 else "Нет (1)",
            index=1
        )
        
        property_ins = st.radio(
            '🏢 Страховка имущества',
            options=[0, 1],
            format_func=lambda x: "Да (0)" if x == 0 else "Нет (1)",
            index=1
        )
        
        travel = st.radio(
            '✈️ Туристическая страховка',
            options=[0, 1],
            format_func=lambda x: "Да (0)" if x == 0 else "Нет (1)",
            index=1
        )
    
    # -------------------------------------------------------------------------
    # КНОПКИ УПРАВЛЕНИЯ И ОТОБРАЖЕНИЕ РЕЗУЛЬТАТОВ
    # -------------------------------------------------------------------------
    
    st.markdown("---")
    
    # Создаем колонки для кнопок
    btn_col1, btn_col2 = st.columns([2, 1])
    
    with btn_col1:
        # Кнопка выполнения предсказания
        predict_button = st.button("🔮 Рассчитать стоимость страховки", type="primary", use_container_width=True)
    
    with btn_col2:
        # Кнопка информации о программе
        about_button = st.button("ℹ️ О программе", use_container_width=True)
    
    # Обработка нажатия кнопки предсказания
    if predict_button:
        try:
            # Вызываем функцию предсказания со всеми параметрами
            result = predict_insurance_cost(
                claim_amount=claim_amount,
                marital_status=marital_status,
                age=age,
                tenure=tenure,
                no_of_family_members=no_of_family_members,
                risk_segmentation=risk_segmentation,
                house_type=house_type,
                social_class=social_class,
                customer_education_level=customer_education_level,
                claim_status=claim_status,
                incident_severity=incident_severity,
                authority_contacted=authority_contacted,
                any_injury=any_injury,
                police_report_available=police_report_available,
                incident_hour_of_the_day=incident_hour_of_the_day,
                health=health,
                life=life,
                mobile=mobile,
                motor=motor,
                property=property_ins,
                travel=travel
            )
            
            # Отображаем результат в красивом формате
            st.success(f"""
            ### 💵 Результат предсказания
            
            **Ориентировочная стоимость страхового полиса:**
            
            # ${result:,.2f} USD
            
            *Расчет выполнен на основе модели машинного обучения CatBoost*
            """)
            
            # Добавляем визуальный разделитель
            st.balloons()
            
        except Exception as e:
            st.error(f"❌ Произошла ошибка при расчете: {e}")
            st.info("Пожалуйста, проверьте корректность введенных данных или обратитесь к разработчику.")
    
    # Обработка нажатия кнопки "О программе"
    if about_button:
        st.markdown("""
        ### 📖 О программе
        
        Данное приложение предназначено для оценки стоимости страхового полиса 
        с использованием современных алгоритмов машинного обучения.
        
        **Технические детали:**
        - **Модель:** CatBoost (градиентный бустинг над решающими деревьями)
        - **Интерфейс:** Streamlit
        - **Язык:** Python 3.x
        
        **Как работает модель:**
        1. Модель была обучена на исторических данных о страховых полисах
        2. При вводе новых данных модель анализирует 21 признак
        3. На основе паттернов в данных выдается прогноз стоимости
        
        **Автор:** Aziz Rasulov
        
        ---
        *Приложение создано в образовательных целях*
        """)


# =============================================================================
# ТОЧКА ВХОДА В ПРИЛОЖЕНИЕ
# =============================================================================

if __name__ == '__main__':
    # Запускаем основное приложение
    main()
