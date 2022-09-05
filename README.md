

# Yandex.Praktikum Data Science Projects
В данном репозитории находятся проекты, выполненные в процессе обучения по направлению 
"Специалист по Data Scince" в Яндекс.Практикум (ООО "ШАД")


### Содержание: / Content:
|№| Название | Общая информация | Стек технологий |
|:---|:-------------------|:----------------------------------------------------------|:-----------:|
|1  |[Подготовка данных для анализа](https://github.com/annapavlovads/yandex_praktikum/blob/main/1_data_preprocessing/1_data_preprocessing.ipynb)<br> | Нужно разобраться, влияет ли семейное положение и количество детей клиента на факт погашения кредита в срок. Результаты исследования будут учтены при построении модели кредитного скоринга — специальной системы, которая оценивает способность потенциального заёмщика вернуть кредит банку. |`seaborn` `matplotlib` `plotly` `pandas` `numpy` `Jupyter Notebook`|
|2  |[Исследовательский анализ данных](https://github.com/annapavlovads/yandex_praktikum/blob/main/2_data_exploration/2_data_exploration.ipynb)<br> | Нужно научиться определять рыночную стоимость объектов недвижимости, задача — установить параметры. Это позволит построить автоматизированную систему: она отследит аномалии и мошенническую деятельность. |`seaborn` `matplotlib` `plotly` `pandas` `numpy` `Jupyter Notebook`|
|3  |[Статистический анализ данных](https://github.com/annapavlovads/yandex_praktikum/blob/main/3_statistical_data_analysis/3_statistical_data_analysis.ipynb)<br> | Нужно проанализировать поведение клиентов оператора связи и сделать вывод — какой тариф лучше. |`seaborn` `scipy` `matplotlib` `plotly` `pandas` `numpy` `Jupyter Notebook`|
|4  |[Сборный проект по подготовке и исследованию данных: исследование рынка компьютерных игр](https://github.com/annapavlovads/yandex_praktikum/blob/main/4_game_investigation/4_game_investigation.ipynb)<br> | Выявить определяющие успешность игры закономерности, что позволит сделать ставку на потенциально популярный продукт и спланировать рекламные кампании. |`seaborn` `scipy` `matplotlib` `plotly` `pandas` `numpy` `Jupyter Notebook`|
|5  |[Введение в машинное обучение: поиск подходящего тарифа для клиентов телеком-компании](https://github.com/annapavlovads/yandex_praktikum/blob/main/5_ml_introduction/5_ml_introduction.ipynb)<br> | Цель: построить модель для задачи классификации, которая выберет подходящий тариф, модель должна иметь максимально большое значение accuracy, не менее 0.75 на тестовой выборке. |`seaborn` `scipy` `matplotlib` `plotly` `pandas` `numpy` `scikit-learn` `Jupyter Notebook`|
|6  |[Предсказание оттока клиентов банка](https://github.com/annapavlovads/yandex_praktikum/blob/main/6_bank_client_leaving_prediction/6_ml_bank_client_leaving_prediction.ipynb)<br> | Цель: необходимо построить модель, прогнозирующую, уйдёт клиент из банка в ближайшее время или нет; требуется модель с предельно большим значением F1-меры. |`seaborn` `scipy` `matplotlib` `plotly` `pandas` `numpy` `scikit-learn` `Jupyter Notebook`|
|7  |[Поиск лучшего местоположения нефтяной скважины](https://github.com/annapavlovads/yandex_praktikum/blob/main/7_lr_bootstrap_oil_model/7_lr_bootstrap_oil_model.ipynb)<br> | Цель: необходимо построить модель машинного обучения, которая поможет определить регион, где добыча принесёт наибольшую прибыль, проанализировать возможную прибыль и риски техникой Bootstrap. |`seaborn` `scipy` `matplotlib` `plotly` `pandas` `numpy` `scikit-learn` `Jupyter Notebook`|
|8  |[Модель для золотообрабатывающего предприятия](https://github.com/annapavlovads/yandex_praktikum/blob/main/8_gold_industry_model/8_gold_industry.ipynb)<br> | Цель: необходимо разработать прототип модели машинного обучения, которая предскажет коэффициент восстановления золота из золотосодержащей руды. Модель поможет оптимизировать производство, и не запускать предприятие с убыточными характеристиками. |`seaborn` `scipy` `matplotlib` `plotly` `pandas` `numpy` `scikit-learn` `Jupyter Notebook`|
|9  |[Линейная алгебра: шифрование данных клиентов страховой компании](https://github.com/annapavlovads/yandex_praktikum/blob/main/9_linear_algebra/9_linear_algebra.ipynb)<br> | Необходимо разработать такой метод преобразования данных, чтобы по ним было сложно восстановить персональную информацию и обосновать корректность работы метода. При преобразовании данных качество моделей машинного обучения не должно ухудшиться. |`sklearn` `pandas` `numpy` `math` `linalg` `Jupyter Notebook`|
|10  |[Предсказание цены автомобилей с пробегом](https://github.com/annapavlovads/yandex_praktikum/blob/main/10_car_price_prediction/10_car_price_prediction.ipynb)<br> | Цель: необходимо построить модель для определения цены автомобилей с пробегом (несколько моделей МL и сравненить их результативность). |`sklearn` `pandas` `numpy` `matplotlib` `plotly` `math` `time` `Jupyter Notebook`|
|11  |[Временные ряды: прогнозирование заказов такси](https://github.com/annapavlovads/yandex_praktikum/blob/main/11_time_series/11_time_series.ipynb)<br> | Цель: спрогнозировать количество заказов такси на следующий час, чтобы привлекать нужное количество водителей. |`catboost` `lightgbm` `sklearn` `pandas` `numpy` `matplotlib` `plotly` `math` `time` `Jupyter Notebook`|
|12  |[Обработка текстовых данных: классификация комментариев](https://github.com/annapavlovads/yandex_praktikum/blob/main/12_text/12_text.ipynb)<br> | Цель: модель, которая будет искать токсичные комментарии и отправлять их на модерацию, значение метрики F1 должно превышать 0,75. |`BERT` `spacy` `SVC` `pymystem3` `re` `sklearn` `pandas` `numpy` `matplotlib` `plotly` `math` `Jupyter Notebook`|
|13  |[Работа с нейронной сетью resnet](https://github.com/annapavlovads/yandex_praktikum/tree/main/13_resnet)<br> | Цель: познакомиться с библиотеками keras, tensorflow на основе работы с нейронной сетью ResNet50. |`Keras` `tensorflow` `Adam` `ResNet50` `ImageDataGenerator` `seaborn` `PIL` `pandas` `matplotlib` `plotly` `numpy`|

#### Дипломный проект: 
|№| Название | Общая информация | Стек технологий |
|:---|:-------------------|:----------------------------------------------------------|:-----------:|
|1  |[Вариант 1: предсказание оттока клиента телеком-компании](https://github.com/annapavlovads/yandex_praktikum/blob/main/14_1_final_project_telecom/YandexDiploma%20_telecom.ipynb)<br> | Необходимо научиться с высокой точностью прогнозировать отток клиентов телеком-оператора. Это позволит своевременно предложить промокоды и бонусы и сохранить клиента - задача классификации |`seaborn` `matplotlib` `plotly` `catboost` `lightgbm` `sklearn` `pandas` `numpy` `scikit-learn` `time` `Jupyter Notebook`|
|2  |[Вариант 2: предсказание температуры плавления стали](https://github.com/annapavlovads/yandex_praktikum/blob/main/14_2_final_project_steel/YandexDiploma%20_steel.ipynb)<br> | Необходимо построить модель, прогнозирующую температуру плавления стали (для снижения электропотребления и оптимизизации производственных расходов) - задача регрессии |`seaborn` `matplotlib` `plotly` `catboost` `lightgbm` `sklearn` `pandas` `numpy` `scikit-learn` `time` `Jupyter Notebook`|

