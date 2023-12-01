# mlops

# Описание задачи
В данной задаче решается задача классификации ирисов Фишера. Ирисы Фишера состоят из данных о 150 экземплярах ириса, по 50 экземпляров из трёх видов:

 - Ирис щетинистый (Iris setosa).
 - Ирис виргинский (Iris virginica).
 - Ирис разноцветный (Iris versicolor).

Для каждого экземпляра измерялись четыре характеристики (в сантиметрах):

 - Длина чашелистника (sepal length).
 - Ширина чашелистника (sepal width).
 - Длина лепестка (petal length).
 - Ширина лепестка (petal width).

# Запуск кода

```
poetry shell
python3 ./irises_classification/project/train.py
python3 ./irises_classification/project/infer.py
```

# Запуск сервера

```
export MLFLOW_TRACKING_URI=http://127.0.0.1:8080
mlflow models serve -m models:/log_reg/latest --no-conda  -h 127.0.0.1 -p 8005
```

# Пример обращения к серверу

```
curl -d '{"dataframe_split": {
"columns": ["sepal length (cm)","sepal width (cm)","petal length (cm)","petal width (cm)","target"],
"data": [[6.1,2.8,4.7,1.2,1]]}}' -H 'Content-Type: application/json' -X POST localhost:8005/invocations
```
