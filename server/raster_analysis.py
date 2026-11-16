# Расчёт растрового сервиса: справочник городов и граница города (OpenStreetMap), композит Sentinel-2, признаки и
# классификация в Earth Engine, карта классов массивом NumPy, площади, изменения, выгрузка в PNG, GeoTIFF и
# GeoJSON; сегментация нейронной сетью SegFormer-B0 в ONNX Runtime на процессоре.
import io, json, os, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import urllib3
import ee
import onnxruntime as ort
import osmnx as ox
import rasterio, rasterio.features
import shapely
from PIL import Image
from pyproj import Transformer
from rasterio.features import geometry_mask
from rasterio.transform import from_origin
from rasterio.warp import calculate_default_transform, reproject, Resampling
from shapely.geometry import box, shape


CITIES = {  # Город: объект OpenStreetMap с границей.
    "Астрахань": "R1853866",
    "Балашиха": "R3171497",
    "Барнаул": "R1550512",
    "Владивосток": "R1704857",
    "Волгоград": "R3374767",
    "Воронеж": "R3536492",
    "Екатеринбург": "R6564910",
    "Ижевск": "R1670935",
    "Иркутск": "R1430614",
    "Казань": "R3437391",
    "Калининград": "R1674442",
    "Кемерово": "R1312868",
    "Киров": "R2383150",
    "Краснодар": "R7373058",
    "Красноярск": "R1430616",
    "Липецк": "R3134925",
    "Махачкала": "R4632505",
    "Москва": "R102269",
    "Набережные Челны": "R3437417",
    "Нижний Новгород": "R1752948",
    "Новокузнецк": "R1437127",
    "Новосибирск": "R1751445",
    "Омск": "R3442814",
    "Оренбург": "W166155507",
    "Пенза": "R2552696",
    "Ростов-на-Дону": "R1285772",
    "Рязань": "R1782722",
    "Самара": "R3368701",
    "Санкт-Петербург": "R337422",
    "Саратов": "R3955288",
    "Ставрополь": "R6712557",
    "Тольятти": "R3338286",
    "Томск": "R7261645",
    "Тула": "R2538203",
    "Тюмень": "R2049854",
    "Ульяновск": "R2049867",
    "Уфа": "R1549169",
    "Хабаровск": "R2049848",
    "Чебоксары": "R3374548",
    "Челябинск": "R4442556",
    "Ярославль": "R1701435",
}


def place(city):
    """Граница города: из справочника – по номеру объекта, любой другой город – поиском по названию."""
    if city in CITIES:  # Город справочника – по номеру объекта.
        return ox.geocode_to_gdf(CITIES[city], by_osmid=True).geometry.iloc[0]  # Граница по номеру.
    try:
        found = ox.geocode_to_gdf(city)  # Поиск по названию в Nominatim.
    except Exception:  # Ничего не найдено.
        # Сообщение с подсказкой, как уточнить.
        raise ValueError(f"Город «{city}» не найден в OpenStreetMap. Уточните название, например «Томск, Россия».")
    polygon = found.geometry.iloc[0]  # Первый результат поиска.
    if polygon.geom_type not in ("Polygon", "MultiPolygon"):  # Найдена точка или линия, а не граница.
        raise ValueError(f"У объекта «{city}» в OpenStreetMap нет границы. Уточните название города.")  # Без границы карты нет.
    return polygon  # Граница в WGS 84.


MODEL_DIR = os.environ.get("MODEL_DIR", "models/raster_model")  # Папка моделей и карточки.
MAX_PARTS = int(os.environ.get("MAX_PARTS", 36))  # Предел частей 256 × 256 пикселей: квадрат 30 × 30 км при 30 м.
SCALE = 30  # Пиксель карты классов по умолчанию, м.
SCALES = [30, 60, 100, 250, 500]  # Разрешения режима «Авто»: от подробного к обзорному.
PIXEL_KM2 = SCALE * SCALE / 1e6  # Площадь такого пикселя, км².
BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]  # Каналы Sentinel-2.
# Цвета классов (HTML без «#»).
COLORS = {"деревья": "397d49", "леса": "397d49", "трава": "88b053", "травы": "88b053", "посевы": "e49635", "пашня": "e49635",
          "застройка": "c4281b", "вода": "419bdf", "открытый грунт": "a59b8f", "прочее": "dfc35a",
          "затопляемая растительность": "7a87c6", "кустарники": "dfc35a", "снег и лёд": "b39fe1", "травяные болота": "0096a0",
          "мангры": "00cf75", "мхи и лишайники": "fae6a0"}
# Карты меток для режима «обучение на лету»: исходные коды и классы (номер класса – позиция в списке).
LABELS = {"Dynamic World": {"codes": list(range(9)), "classes": ["вода", "деревья", "трава", "затопляемая растительность",
                                                                  "посевы", "кустарники", "застройка", "открытый грунт", "снег и лёд"]},
          "ESA WorldCover": {"codes": [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100],
                             "classes": ["деревья", "кустарники", "травы", "пашня", "застройка", "открытый грунт", "снег и лёд",
                                         "вода", "травяные болота", "мангры", "мхи и лишайники"]}}
EMBEDDING = [f"A{i:02d}" for i in range(64)]  # 64 канала эмбеддингов AlphaEarth.
SPECTRAL = BANDS + ["NDVI", "MNDWI", "NDVI_p10", "NDVI_p50", "NDVI_p90"]  # Каналы и индексы композита.
EE_PARALLEL = int(os.environ.get("EE_PARALLEL", 16))  # Одновременных запросов частей к Earth Engine.
MAX_VECTOR_KM2 = float(os.environ.get("MAX_VECTOR_KM2", 300))  # Предел площади векторизации карты классов в GeoJSON.
LAST_EMBEDDING_YEAR = int(os.environ.get("LAST_EMBEDDING_YEAR", 2025))  # Последний год эмбеддингов AlphaEarth в каталоге.


def connect():
    """Подключение к Earth Engine: секрет EE_SERVICE_ACCOUNT (JSON-ключ) или сохранённый вход; проект – EE_PROJECT."""
    key = os.environ.get("EE_SERVICE_ACCOUNT")  # Содержимое JSON-ключа сервисного аккаунта.
    if key:  # Ключ задан – вход сервисного аккаунта.
        info = json.loads(key)  # Почта аккаунта и проект ключа.
        credentials = ee.ServiceAccountCredentials(info["client_email"], key_data=key)  # Учётные данные из ключа.
        ee.Initialize(credentials, project=os.environ.get("EE_PROJECT", info["project_id"]))  # Проект из EE_PROJECT или из ключа.
    else:
        ee.Initialize(project=os.environ.get("EE_PROJECT"))  # Свой компьютер: вход earthengine authenticate.


def read_card(model_dir=MODEL_DIR):
    """Карточка модели card.json."""
    return json.load(open(os.path.join(model_dir, "card.json"), encoding="utf-8"))  # Классы, сезоны, признаки, метрики.


def read_classifier(model_dir, file):
    """Классификатор Earth Engine, восстановленный из описания JSON."""
    return ee.deserializer.fromJSON(open(os.path.join(model_dir, file)).read())  # Без запроса к Earth Engine.


def load_model(model_dir=MODEL_DIR):
    """Основной классификатор карточки и сама карточка."""
    card = read_card(model_dir)  # Карточка модели.
    return read_classifier(model_dir, card["classifier_file"]), card  # Файл классификатора указан в карточке.


def load_variants(model_dir=MODEL_DIR):
    """Варианты своей модели: «spectral» – каналы и индексы композита, «embedding» – эмбеддинги AlphaEarth."""
    card = read_card(model_dir)  # Описание вариантов.
    # Без вариантов в карточке единственный классификатор – вариант своего набора признаков.
    variants = card.get("variants") or {("embedding" if card["features"][0].startswith("A") else "spectral"):
                                        {"classifier": card["classifier"], "features": card["features"],
                                         "file": card["classifier_file"], "metrics": card.get("metrics", {})}}
    return {k: v | {"model": read_classifier(model_dir, v["file"])} for k, v in variants.items()}  # Плюс сам классификатор.


def season(collection, when, seasons):
    """Снимки периода: год (сезоны темы, последний день включается) или интервал дат (начало, день после конца)."""
    if isinstance(when, (tuple, list)):  # Интервал дат.
        return collection.filterDate(when[0], when[1])  # Конец интервала не включается.
    # Год: объединение сезонов темы этого года.
    return collection.filter(ee.Filter.Or(*[ee.Filter.date(ee.Date(f"{when}-{a}"), ee.Date(f"{when}-{b}").advance(1, "day"))
                                            for a, b in seasons]))


def embedding_year(when):
    """Год эмбеддингов для периода: год его начала, но не позже последнего года каталога."""
    return min(int(str(when[0])[:4]) if isinstance(when, (tuple, list)) else int(when), LAST_EMBEDDING_YEAR)  # Год из даты.


def composite(region, when, seasons):
    """Медиана ясных наблюдений Sentinel-2 за сезон, индексы NDVI и MNDWI, процентили NDVI за сезон."""
    # Снимки периода и к каждому – вероятность ясного неба Cloud Score+.
    s2 = season(ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(region), when, seasons) \
        .linkCollection(ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED"), ["cs_cdf"])
    # Отражение 0…1 и NDVI каждого снимка, облачные пиксели скрыты.
    clear = s2.map(lambda im: im.select(BANDS).divide(10000).addBands(im.normalizedDifference(["B8", "B4"]).rename("NDVI"))
                   .updateMask(im.select("cs_cdf").gte(0.6)))
    median = clear.select(BANDS).median()  # Медианный композит каналов.
    # Процентили NDVI за сезон.
    percentiles = clear.select("NDVI").reduce(ee.Reducer.percentile([10, 50, 90])).rename(["NDVI_p10", "NDVI_p50", "NDVI_p90"])
    # Индексы композита и процентили NDVI – дополнительные каналы.
    return median.addBands(median.normalizedDifference(["B8", "B4"]).rename("NDVI")) \
                 .addBands(median.normalizedDifference(["B3", "B11"]).rename("MNDWI")).addBands(percentiles)


def embeddings(region, when):
    """Годовая мозаика эмбеддингов AlphaEarth."""
    year = embedding_year(when)  # Год каталога для периода.
    # Снимки эмбеддингов года над областью – одной мозаикой.
    return ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL").filterDate(f"{year}-01-01", f"{year + 1}-01-01") \
        .filterBounds(region).mosaic()


def classify(region, when, classifier, card):
    """Карта классов темы за период в Earth Engine: номер класса 0, 1, 2… для каждого пикселя."""
    stack = composite(region, when, card["seasons"]).addBands(embeddings(region, when))  # Все возможные признаки.
    return stack.select(card["features"]).classify(classifier).rename("class")  # Признаки модели → класс.


def label_image(source, when, seasons):
    """Карта меток за год: Dynamic World – самый частый класс сезона, ESA WorldCover – карта 2021 г."""
    info = LABELS[source]  # Коды и классы карты меток.
    if source == "ESA WorldCover":  # WorldCover – одна карта на все годы.
        raw = ee.ImageCollection("ESA/WorldCover/v200").first()  # Единственная карта 2021 г.
    else:
        raw = season(ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1"), when, seasons).select("label").mode()  # Самый частый класс.
    return raw.remap(info["codes"], list(range(len(info["codes"])))).rename("class")  # Исходные коды → номера 0, 1, 2…


def features_image(region, when, seasons, features):
    """Признаки для классификации: каналы и индексы композита и (или) эмбеддинги AlphaEarth."""
    stack = composite(region, when, seasons)  # Каналы и индексы.
    if any(f.startswith("A") and f[1:].isdigit() for f in features):  # Среди признаков есть эмбеддинги.
        stack = stack.addBands(embeddings(region, when))  # Добавить эмбеддинги AlphaEarth.
    return stack.select(features)  # Только признаки модели.


def train_on_area(geometry, when, seasons, source, use_embeddings=False, per_class=200):
    """Обучение на лету: случайный лес по карте меток в области, проверка на 1/5 блоков 2 × 2 км; классификатор, признаки, метрики."""
    region = ee.Geometry(geometry)  # Область интереса в Earth Engine.
    features = SPECTRAL + (EMBEDDING if use_embeddings else [])  # Признаки модели.
    train_when = 2021 if source == "ESA WorldCover" else when  # WorldCover есть только за 2021 г.
    # Признаки и метка класса – одним изображением.
    stack = features_image(region, train_when, seasons, features).addBands(label_image(source, train_when, seasons))
    points = stack.stratifiedSample(per_class, "class", region, scale=SCALE, seed=42, geometries=True, tileScale=4)  # Точки по классам.

    def fold(f):  # Номер части по блоку 2 × 2 км в EPSG:6931.
        xy = f.geometry().transform("EPSG:6931", 1).coordinates()  # Координаты точки в метрах EPSG:6931.
        bx, by = [ee.Number(xy.get(i)).add(1e7).divide(2000).floor() for i in (0, 1)]  # Номер блока по x и y.
        return f.set("fold", bx.multiply(7).add(by.multiply(13)).mod(5))  # Номер части 0…4 по номеру блока.

    points = points.map(fold)  # Номер части каждой точки.
    # Проверочная модель – на точках всех частей, кроме 0.
    test = ee.Classifier.smileRandomForest(100, seed=42).train(points.filter(ee.Filter.neq("fold", 0)), "class", features)
    # Матрица ошибок проверочной модели на точках части 0.
    matrix = points.filter(ee.Filter.eq("fold", 0)).classify(test).errorMatrix("class", "classification",
                                                                             list(range(len(LABELS[source]["codes"]))))
    # Все метрики проверки – одним запросом getInfo.
    stats = ee.Dictionary({"accuracy": matrix.accuracy(), "f1": matrix.fscore(), "n": points.size(),
                           "present": points.aggregate_array("class").distinct()}).getInfo()
    present = sorted(stats["present"])  # Классы, встретившиеся в области.
    f1 = [v for i, v in enumerate(stats["f1"]) if i in present and isinstance(v, (int, float))]  # F1 только классов, встреченных в области.
    # Метрики для ответа; средняя F1 – по классам области.
    metrics = {"обучающих точек": stats["n"], "классов в области": len(present),
               "общая точность": round(stats["accuracy"], 3), "средняя F1-мера": round(float(np.mean(f1)), 3) if f1 else None}
    return ee.Classifier.smileRandomForest(100, seed=42).train(points, "class", features), features, metrics  # Итог – на всех точках.


def project(geometry, source, target):
    """Геометрия или список геометрий в другой системе координат: все вершины одним вызовом pyproj."""
    transformer = Transformer.from_crs(source, target, always_xy=True)  # Порядок осей: долгота, широта.

    def move(xy):  # Массив вершин N × 2 → новые координаты.
        return np.column_stack(transformer.transform(xy[:, 0], xy[:, 1]))  # Две колонки: новые x и y.

    return shapely.transform(geometry, move)  # Та же форма в новых координатах.


def utm_grid(geometry, step, max_frame_parts=None):
    """Зона UTM и сетка квадратных частей со стороной step м над полигоном GeoJSON в градусах (север вверху)."""
    area = shape(geometry)  # Полигон shapely в градусах.
    lon, lat = area.centroid.x, area.centroid.y  # Центр области.
    crs = f"EPSG:{(32600 if lat >= 0 else 32700) + int((lon + 180) // 6) + 1}"  # Зона UTM по центру области.
    utm = project(area, "EPSG:4326", crs)  # Область в метрах UTM.
    xmin, ymin, xmax, ymax = utm.bounds  # Рамка области.
    x0, y1 = np.floor(xmin / step) * step, np.ceil(ymax / step) * step  # Левый верхний угол сетки.
    nx, ny = int(np.ceil((xmax - x0) / step)), int(np.ceil((y1 - ymin) / step))  # Частей по x и y.
    # Мультиполигон с большой общей рамкой.
    if area.geom_type == "MultiPolygon" and max_frame_parts is not None and nx * ny > max_frame_parts:
        return None  # Маска и GeoTIFF занимают весь прямоугольник, включая пустоты между полигонами.
    to_geo = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)  # Обратно в градусы.
    (w, n), (e, s) = to_geo.transform(x0, y1), to_geo.transform(x0 + nx * step, y1 - ny * step)  # Углы сетки в градусах.
    cells = [(i, j) for j in range(ny) for i in range(nx)  # Только части, пересекающие область.
             if utm.intersects(box(x0 + i * step, y1 - (j + 1) * step, x0 + (i + 1) * step, y1 - j * step))]
    # Всё о сетке: зона, угол, размеры, части, область в UTM.
    return {"crs": crs, "x0": x0, "y1": y1, "nx": nx, "ny": ny, "step": step, "bounds": [[s, w], [n, e]], "cells": cells,
            "area": utm}


def processing_grid(geometry, scales, max_parts, size=256):
    """Сетка для самого подробного из допустимых разрешений, в порядке перебора scales."""
    for scale in scales:  # От подробного к обзорному.
        grid = utm_grid(geometry, size * scale, max_frame_parts=max_parts)  # Часть – size пикселей.
        if grid is not None and len(grid["cells"]) <= max_parts:  # Частей хватает – разрешение подходит.
            return grid | {"scale": scale, "pixel_km2": scale * scale / 1e6}  # Сетка, разрешение и площадь пикселя.
    if grid is None:  # Мешает рамка мультиполигона.
        # Ни одно разрешение не подходит из-за рамки.
        raise ValueError(f"Общая рамка полигонов слишком велика при разрешении {scale} м/пиксель. "
                         "Выберите один полигон или более крупный размер пикселя.")
    # Частей больше предела даже при последнем разрешении.
    raise ValueError(f"Область слишком велика при разрешении {scale} м/пиксель: нужно {len(grid['cells'])} фрагментов "
                     f"при пределе {max_parts}. Выберите более крупный размер пикселя или уменьшите область.")


ENCODED = {}  # Закодированные выражения карт: id выражения → (выражение, JSON для Earth Engine).
encode = ee.serializer.encode  # Кодировщик библиотеки: computePixels вызывает его в каждом запросе.


def encode_once(obj, *args, **kwargs):  # Готовый JSON для выражения карты, иначе обычное кодирование.
    known = ENCODED.get(id(obj))  # Выражение уже закодировано в class_maps.
    return known[1] if known and known[0] is obj and not (args or kwargs) else encode(obj, *args, **kwargs)


ee.serializer.encode = encode_once  # Классификатор кодируется секунды: один раз на период вместо каждой части.


def pixels(request, tries=4):
    """Запрос части карты к Earth Engine с повтором при временных сбоях (очередь, тайм-аут, перегрузка)."""
    for attempt in range(tries):  # До tries попыток.
        try:
            return ee.data.computePixels(request)  # Массив NumPy части.
        except ee.EEException as error:  # Ошибка Earth Engine.
            # Временная ли ошибка: очередь, тайм-аут, перегрузка.
            transient = any(w in str(error) for w in ("Too many", "timed out", "Deadline", "429", "503", "Internal"))
            if attempt == tries - 1 or not transient:  # Последняя попытка или постоянная ошибка.
                raise  # Ошибка передаётся вызывающему коду.
            time.sleep(3 * 2 ** attempt)  # Пауза 3, 6, 12 с.
        except (OSError, urllib3.exceptions.HTTPError) as error:  # Сетевой сбой (тайм-аут, разрыв соединения) – тоже повтор.
            if attempt == tries - 1:  # Попытки закончились.
                raise ee.EEException(f"Нет ответа Earth Engine: {type(error).__name__}") from error  # Как ошибка Earth Engine.
            time.sleep(3 * 2 ** attempt)  # Та же пауза.


def class_maps(images, geometry, scale=None, size=256, grid=None):
    """Карты классов {год: карта Earth Engine} массивами NumPy в UTM (−1 – вне области); части size × size – параллельно."""
    if grid is None:  # Сетку не передали – подобрать.
        # Самое подробное разрешение, при котором хватает частей.
        grid = processing_grid(geometry, [scale] if scale else SCALES, MAX_PARTS, size)
    scale, x0, y1, step = grid["scale"], grid["x0"], grid["y1"], grid["step"]  # Пиксель, угол и шаг частей.
    region = ee.Geometry(geometry)  # Область для обрезки.
    exprs = {k: im.add(1).clip(region).unmask(0).toByte().rename("class") for k, im in images.items()}  # 0 – вне области.

    def part(job):  # Одна часть карты одного года.
        k, i, j = job  # Год, столбец и строка части.
        # Положение части: пиксель scale м, левый верхний угол части.
        affine = {"scaleX": scale, "shearX": 0, "translateX": x0 + i * step, "shearY": 0, "scaleY": -scale, "translateY": y1 - j * step}
        # Часть size × size пикселей в зоне UTM – канал class.
        return pixels({"expression": exprs[k], "fileFormat": "NUMPY_NDARRAY",
                       "grid": {"dimensions": {"width": size, "height": size}, "crsCode": grid["crs"], "affineTransform": affine}})["class"]

    jobs = [(k, i, j) for k in exprs for i, j in grid["cells"]]  # Года × части, пересекающие область.
    ENCODED.update({id(e): (e, encode(e)) for e in exprs.values()})  # Каждый период кодируется один раз.
    try:
        with ThreadPoolExecutor(EE_PARALLEL) as pool:  # Не больше EE_PARALLEL запросов сразу.
            parts = dict(zip(jobs, pool.map(part, jobs)))  # Части всех периодов – параллельно.
    finally:
        for e in exprs.values():  # Готовый JSON больше не нужен.
            ENCODED.pop(id(e), None)  # Освободить память.
    empty = np.zeros((size, size), np.uint8)  # Часть вне области.
    # Части – в общий массив, номер класса – снова 0, 1, 2…
    maps = {k: np.block([[parts.get((k, i, j), empty) for i in range(grid["nx"])] for j in range(grid["ny"])]).astype(np.int16) - 1
            for k in exprs}
    return maps, grid  # Карты и сетка для выгрузки.


def areas(classes, card, pixel_km2=PIXEL_KM2):
    """Площадь каждого класса, км²."""
    counts = np.bincount(classes[classes >= 0], minlength=len(card["classes"]))  # Пикселей каждого класса.
    return {c: round(float(n * pixel_km2), 2) for c, n in zip(card["classes"], counts)}  # Площадь класса, км².


def transitions(first, second, card, pixel_km2=PIXEL_KM2):
    """Матрица переходов между годами (км²) и чистый прирост каждого класса."""
    k, both = len(card["classes"]), (first >= 0) & (second >= 0)  # Пиксели с данными в оба года.
    matrix = np.bincount(first[both] * k + second[both], minlength=k * k).reshape(k, k) * pixel_km2  # Строка – класс «было».
    net = {c: round(float(matrix[:, i].sum() - matrix[i, :].sum()), 2) for i, c in enumerate(card["classes"])}  # Пришло − ушло.
    return np.round(matrix, 2).tolist(), net  # Матрица списком для JSON и прирост.


def palette(card):
    """Цвета классов карточки массивом RGB."""
    # «397d49» → [57, 125, 73]; неизвестный класс – серый.
    return np.array([[int(c[i:i + 2], 16) for i in (0, 2, 4)] for c in [COLORS.get(k, "999999") for k in card["classes"]]], np.uint8)


def to_mercator(classes, grid):
    """Карта классов в Web Mercator, как у подложки, и углы [[юг, запад], [север, восток]]; площади и GeoTIFF – в UTM."""
    height, width = classes.shape  # Размер карты, пикселей.
    scale = grid["scale"]  # Пиксель, м.
    edges = (grid["x0"], grid["y1"] - height * scale, grid["x0"] + width * scale, grid["y1"])  # Рамка карты в UTM.
    target, w, h = calculate_default_transform(grid["crs"], "EPSG:3857", width, height, *edges)  # Сетка в Web Mercator.
    out = np.full((h, w), -1, np.int16)  # Пустая карта: −1 – нет данных.
    # Ближайший сосед не смешивает номера классов; copy=False – без копии карты, уже int16.
    reproject(classes.astype(np.int16, copy=False), out, src_transform=from_origin(grid["x0"], grid["y1"], scale, scale),
              src_crs=grid["crs"], dst_transform=target, dst_crs="EPSG:3857", resampling=Resampling.nearest, src_nodata=-1, dst_nodata=-1)
    to_geo = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)  # Web Mercator → градусы.
    # Углы новой сетки в градусах.
    (west, north), (east, south) = to_geo.transform(target.c, target.f), to_geo.transform(target.c + w * target.a, target.f + h * target.e)
    return out, [[south, west], [north, east]]  # Карта и углы для подложки.


def to_png(classes, card):
    """Карта классов картинкой PNG с прозрачным фоном вне области – для наложения на карту."""
    rgba = np.zeros(classes.shape + (4,), np.uint8)  # Прозрачный фон.
    inside = classes >= 0  # Пиксели области.
    rgba[inside, :3], rgba[inside, 3] = palette(card)[classes[inside]], 255  # Цвет класса, непрозрачно.
    png = io.BytesIO()  # Файл в памяти.
    Image.fromarray(rgba).save(png, format="PNG")  # Запись PNG в память.
    return png.getvalue()  # Байты PNG.


def to_geojson(classes, grid, card):
    """Карта классов полигонами GeoJSON в градусах; None – область больше предела векторизации."""
    inside = classes >= 0  # Пиксели области.
    if inside.sum() * grid["pixel_km2"] > MAX_VECTOR_KM2:  # Слишком много полигонов для ответа.
        return None  # Клиент построит GeoJSON из GeoTIFF.
    # Полигоны одинаковых соседних пикселей в метрах UTM.
    shapes = list(rasterio.features.shapes(classes.astype(np.int16, copy=False), mask=inside,
                                           transform=from_origin(grid["x0"], grid["y1"], grid["scale"], grid["scale"])))
    # Упрощение на полпикселя, затем все полигоны сразу – в градусы.
    polygons = project([shape(g).simplify(grid["scale"] / 2) for g, _ in shapes], grid["crs"], "EPSG:4326")
    # Объект GeoJSON на каждый полигон: код и название класса.
    features = [{"type": "Feature", "properties": {"код": int(v), "класс": card["classes"][int(v)]}, "geometry": p.__geo_interface__}
                for (_, v), p in zip(shapes, polygons)]
    return {"type": "FeatureCollection", "features": features}  # Слой для карты и файла.


def to_geotiff(maps, grid):
    """GeoTIFF карт классов: канал на каждый год, номер класса, 255 – вне области."""
    years = list(maps)  # Порядок каналов.
    height, width = maps[years[0]].shape  # Размер общий у всех карт.
    with rasterio.MemoryFile() as memory:  # Файл в памяти, без диска.
        # Файл GeoTIFF: каналы, сжатие, система координат.
        with memory.open(driver="GTiff", width=width, height=height, count=len(years), dtype="uint8", nodata=255, crs=grid["crs"],
                         transform=from_origin(grid["x0"], grid["y1"], grid["scale"], grid["scale"]), compress="deflate") as tif:
            for band, year in enumerate(years, 1):  # Каналы нумеруются с 1.
                tif.write(np.where(maps[year] >= 0, maps[year], 255).astype(np.uint8), band)  # −1 → 255.
                tif.set_band_description(band, str(year))  # Подпись канала – год.
        return memory.read()  # Байты GeoTIFF.


MAX_TILES = int(os.environ.get("MAX_TILES", 144))  # Предел фрагментов 256 × 256 пикселей на запрос.
SEG_SCALES = [10, 20, 30, 60]  # Разрешения сегментации, м: «Авто» берёт самое подробное подходящее.


def load_network(model_dir=MODEL_DIR):
    """Сеть SegFormer-B0 (файл model.onnx) и нормировка каналов (среднее, разброс)."""
    options = ort.SessionOptions()  # Настройки выполнения сети.
    options.enable_cpu_mem_arena = False  # Память окна сразу освобождается: сервер помещается в 512 МБ.
    # Сеть возвращает вероятности классов в размере окна: увеличение и softmax – внутри файла.
    net = ort.InferenceSession(os.path.join(model_dir, "segformer", "model.onnx"), options, providers=["CPUExecutionProvider"])
    norm = np.load(os.path.join(model_dir, "segformer", "normalization.npz"))  # Статистика каналов обучающей выборки.
    return net, norm["mean"], norm["std"]  # Сеть и нормировка.


def segment(geometry, when, card, net, mean, std, scale=None, pad=0):
    """Маска классов области (−1 – вне области) и сетка; окна с полем pad смешиваются плавными весами, без швов."""
    grid = processing_grid(geometry, [scale] if scale else SEG_SCALES, MAX_TILES)  # Пропускаются фрагменты вне области.
    s = grid["scale"]  # Пиксель, м.
    nx, ny, x0, y1, tile = grid["nx"], grid["ny"], grid["x0"], grid["y1"], grid["step"]  # Размеры, угол, шаг сетки.
    window = 256 + 2 * pad  # Фрагмент с полем: соседние окна перекрываются.
    # Рамка всех окон вместе с полями – в метрах UTM.
    region = ee.Geometry.Rectangle([x0 - pad * s, y1 - ny * tile - pad * s, x0 + nx * tile + pad * s, y1 + pad * s], grid["crs"], False)
    # Композит последнего года темы или периода запроса: каналы R, G, B.
    image = composite(region, card["years"][-1] if when is None else when, card["seasons"]).select(["B4", "B3", "B2"]).toFloat()

    def fragment(ij):  # Окно (i, j) с полем: каналы R, G, B.
        i, j = ij  # Столбец и строка.
        # Положение окна: пиксель s м, левый верхний угол с полем.
        affine = {"scaleX": s, "shearX": 0, "translateX": x0 + tile * i - pad * s, "shearY": 0, "scaleY": -s,
                  "translateY": y1 - tile * j + pad * s}
        # Окно window × window пикселей в зоне UTM.
        a = pixels({"expression": image, "fileFormat": "NUMPY_NDARRAY",
                    "grid": {"dimensions": {"width": window, "height": window}, "crsCode": grid["crs"], "affineTransform": affine}})
        return np.stack([a["B4"], a["B3"], a["B2"]])  # Массив 3 × окно × окно.

    ramp = np.minimum(1, (np.arange(window) + 0.5) / max(2 * pad, 1))  # Вес пикселя окна: плавно убывает к краю.
    edge = np.minimum(ramp, ramp[::-1])  # Одинаково с обеих сторон.
    weight = np.outer(edge, edge).astype(np.float32)  # Вес по двум осям.
    cells = grid["cells"]  # Только фрагменты, пересекающие область.
    probs = np.zeros((len(card["classes"]), ny * 256 + 2 * pad, nx * 256 + 2 * pad), np.float32)  # Сумма вероятностей с весами.
    total = np.zeros(probs.shape[1:], np.float32)  # Сумма весов.
    with ThreadPoolExecutor(8) as pool:  # Загрузка окон в 8 потоков.
        # Окна по порядку: сеть считает готовое окно, пока загружаются следующие.
        for (i, j), rgb in zip(cells, pool.map(fragment, cells)):
            x = (rgb[None].astype(np.float32, copy=False) - mean) / std  # Нормировка как при обучении.
            p = net.run(None, {"pixel_values": x})[0][0]  # Вероятности классов окна.
            probs[:, j * 256:j * 256 + window, i * 256:i * 256 + window] += p * weight  # Окно – на своё место.
            total[j * 256:j * 256 + window, i * 256:i * 256 + window] += weight  # И его вес.
    inner = (slice(pad, pad + ny * 256), slice(pad, pad + nx * 256))  # Сетка без внешних полей.
    mask = np.where(total[inner] > 0, probs[(slice(None),) + inner].argmax(0), -1).astype(np.int16)  # Самый вероятный класс.
    outside = geometry_mask([grid["area"]], out_shape=mask.shape, transform=from_origin(x0, y1, s, s))  # Пиксели вне полигона.
    mask[outside] = -1  # Обрезка по границе области, как у классификации.
    return mask, grid  # Маска и сетка для выгрузки.
