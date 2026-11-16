# Серверная часть растрового сервиса: программный интерфейс (API) карты классов, площадей, изменений и сегментации
# в двух режимах – «своя модель» (/classify) и «обучение на лету» по области интереса (/train).
# Отдельный запуск: uvicorn server.api:app --port 8110 (обычно серверная часть работает внутри страницы main.py).
import base64, datetime, json, time, traceback
from collections import OrderedDict
from functools import lru_cache
import ee
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from shapely.geometry import mapping
from server.raster_analysis import (CITIES, place, MAX_TILES, SEG_SCALES, load_network, segment,
                    connect, load_model, load_variants, features_image, class_maps, areas, transitions, to_png, to_geotiff,
                    to_geojson, to_mercator, train_on_area, COLORS, LABELS, LAST_EMBEDDING_YEAR, MAX_PARTS, SCALES,
                    MAX_VECTOR_KM2, processing_grid)
from extensions import extension

app = FastAPI(title="Геоинформационный сервис картографирования по снимкам Sentinel-2")  # Приложение API.
connect()  # Earth Engine – один раз при запуске.
_, CARD = load_model()  # Карточка модели; файл классификатора проверяется при запуске.
VARIANTS = load_variants()  # Своя модель на каналах композита и (или) на эмбеддингах AlphaEarth.
NET = load_network()  # Сеть SegFormer-B0 и нормировка каналов – один раз при запуске.
# Вариант по умолчанию – классификатор, указанный в карточке.
DEFAULT_VARIANT = next((k for k, v in VARIANTS.items() if v["file"] == CARD["classifier_file"]), next(iter(VARIANTS)))
CACHE = OrderedDict()  # Ответы /classify на одинаковые запросы (последние 4).


def b64(data):
    """Байты файла текстом base64 для ответа JSON."""
    return base64.b64encode(data).decode()  # Строка вместо байтов.


class Request(BaseModel):  # Тело запроса всех режимов.
    geometry: dict  # Область интереса: Polygon или MultiPolygon GeoJSON в градусах.
    year: int | None = None  # Год карты (сезоны темы); по умолчанию – последний год темы.
    compare_year: int | None = None  # Второй год для режима изменений.
    start: str | None = None  # Начало интервала дат «ГГГГ-ММ-ДД» вместо года.
    end: str | None = None  # Конец интервала, включительно.
    compare_start: str | None = None  # Начало второго интервала для режима изменений.
    compare_end: str | None = None  # Конец второго интервала.
    labels: str = "Dynamic World"  # Карта меток для обучения на лету: Dynamic World или ESA WorldCover.
    scale: int | None = None  # Размер пикселя, м; None – «Авто»: самый подробный, при котором хватает частей.
    overlap: int = 0  # Перекрытие фрагментов сегментации, пикселей поля (0 – без перекрытия).
    # Эмбеддинги AlphaEarth (точнее, но дольше) или только каналы; None – лучший вариант карточки.
    embeddings: bool | None = None
    model: dict | None = None  # Модель, обученная на лету (/apply): описание классификатора, признаки, классы.


def asset_ok():
    """Доступен ли сервису ассет обучающих точек из карточки (роль Earth Engine Resource Viewer)."""
    if not CARD.get("training_asset"):  # Ассет не указан.
        return None  # Точки хранятся в описании классификатора.
    try:
        ee.data.getAsset(CARD["training_asset"])  # Запрос сведений об ассете.
        return True  # Ассет читается.
    except ee.EEException:  # Нет доступа или ассета.
        return False  # Ассет не читается.


def colors(card):
    """Цвета классов карточки; неизвестный класс – серый."""
    return [COLORS.get(c, "999999") for c in card["classes"]]  # Порядок – как у классов.


@app.get("/cities")  # Справочник городов.
def cities():  # Названия городов справочника по алфавиту.
    return sorted(CITIES)  # Список для выбора на странице.


@app.get("/place")  # Граница города.
@lru_cache(maxsize=64)  # Повторный запрос того же города – из памяти.
def city_boundary(city: str):
    """Граница города для карты клиентской части: из справочника или поиском по названию в OpenStreetMap."""
    try:
        # Граница упрощается примерно до 50 м: меньше вершин в запросах к Earth Engine.
        return {"city": city, "geometry": mapping(place(city.strip()).simplify(0.0005))}
    except ValueError as error:  # Город не найден или у объекта нет границы.
        raise HTTPException(400, str(error))  # Причина – в ответе.


@app.get("/health")  # Состояние сервиса.
def health():  # Классы, цвета, варианты, метрики, пределы запроса.
    # Ключи и их порядок – часть ответа API; limits – пределы одного запроса.
    return {"status": "ok", "seasons": CARD["seasons"], "default_variant": DEFAULT_VARIANT,
            "training_asset": CARD.get("training_asset"), "asset_ok": asset_ok(), "topic": CARD["topic"], "classes": CARD["classes"],
            "years": CARD["years"], "colors": colors(CARD), "classifier": CARD["classifier"],
            "approbation": CARD.get("approbation", []), "focus": CARD.get("focus"), "metrics": CARD.get("metrics", {}),
            "segmentation": CARD.get("segmentation", {}).get("metrics", {}), "labels": list(LABELS),
            "last_embedding_year": LAST_EMBEDDING_YEAR,
            "limits": {"max_parts": MAX_PARTS, "scales": SCALES, "max_tiles": MAX_TILES, "segment_scales": SEG_SCALES,
                       "max_vector_km2": MAX_VECTOR_KM2},
            "variants": {k: {"classifier": v["classifier"], "metrics": v["metrics"]} for k, v in VARIANTS.items()}}


def periods(req):
    """Периоды карты и сравнения: ключ результата и значение для расчёта (год или интервал дат, конец не включается)."""
    def interval(start, end):  # «начало…конец» и интервал до дня после конца.
        # Конец сдвигается на день: filterDate его не включает.
        return f"{start}…{end}", (start, str(datetime.date.fromisoformat(end) + datetime.timedelta(days=1)))
    main = interval(req.start, req.end) if req.start and req.end else (req.year or CARD["years"][-1],) * 2  # Год – сам себе ключ.
    # Период сравнения: интервал, год или нет.
    other = interval(req.compare_start, req.compare_end) if req.compare_start and req.compare_end else \
        ((req.compare_year,) * 2 if req.compare_year else None)
    return main, other  # Второй – None без сравнения.


def class_images(region, main, other, features, classifier):
    """Карты классов Earth Engine {период: карта} для периода карты и периода сравнения."""
    when = dict([main] + ([other] if other else []))  # Ключ результата → период расчёта.
    # Одна карта классов на каждый период.
    return {k: features_image(region, w, CARD["seasons"], features).classify(classifier).rename("class") for k, w in when.items()}


def respond(images, req, card, mode, metrics, grid=None):
    """Карты классов частями, площади, переходы, собственная функция, GeoTIFF и GeoJSON – общий ответ режимов."""
    (year, _), other = periods(req)  # Ключ основного периода.
    compare = other[0] if other else None  # Ключ второго периода или None.
    try:
        maps, grid = class_maps(images, req.geometry, req.scale, grid=grid)  # Запросы частей к Earth Engine.
    except ValueError as error:  # Область интереса больше предела.
        raise HTTPException(400, str(error))  # 400 – область нужно уменьшить.
    except ee.EEException as error:  # Нет доступа к ассету точек, превышена квота, долгий ответ.
        raise HTTPException(502, f"Earth Engine: {error}")  # 502 – сбой внешнего сервиса.
    except Exception as error:  # Любой другой сбой – тоже ответом JSON с причиной.
        raise HTTPException(500, f"{type(error).__name__}: {error}")  # 500 – внутренняя ошибка сервиса.
    try:
        view = {k: to_mercator(v, grid) for k, v in maps.items()}  # Для наложения на подложку Web Mercator.
        pixel_km2 = grid["pixel_km2"]  # Площадь пикселя при выбранном разрешении.
        result = {"mode": mode, "metrics": metrics, "classes": card["classes"], "colors": colors(card),  # Карта PNG, площади и разрешение.
                  "year": year, "png": b64(to_png(view[year][0], card)), "bounds": view[year][1],
                  "areas": areas(maps[year], card, pixel_km2), "scale": grid["scale"],
                  "area_km2": round(float((maps[year] >= 0).sum() * pixel_km2), 1), "pixel_km2": pixel_km2}
        if compare:  # Режим изменений: второй период и переходы.
            matrix, net = transitions(maps[compare], maps[year], card, pixel_km2)  # Матрица переходов и чистый прирост.
            # Ключи второго периода – в конец ответа.
            result |= {"compare_year": compare, "compare_png": b64(to_png(view[compare][0], card)),
                       "compare_areas": areas(maps[compare], card, pixel_km2), "matrix": matrix, "net_change": net}
        try:
            result["extension"] = extension(maps, card, result)  # Собственная функция из extensions.py.
        except Exception as error:  # Ошибка собственной функции не ломает основной расчёт.
            # Текст ошибки вместо результата функции.
            result["extension"] = {"name": "Собственная функция", "text": f"Ошибка: {type(error).__name__}: {error}"}
        result["geotiff"] = b64(to_geotiff(maps, grid))  # Все периоды – каналами одного файла.
        result["geojson"] = to_geojson(maps[year], grid, card)  # Полигоны карты основного периода.
        return result  # Готовый ответ режима.
    except Exception as error:  # Сбой обработки – с кратким следом ошибки.
        # 500 – с последними строками следа ошибки.
        traceback.print_exc()  # Подробности – в журнал серверной части, не в ответ.
        raise HTTPException(500, f"{type(error).__name__}: {error}")  # Тип и текст ошибки.


@app.post("/classify")  # Своя модель.
def classify_area(req: Request):
    """Режим «своя модель»: обученный классификатор из models/ применяется к области интереса. Повторный такой же запрос берётся
    из памяти; в ответе seconds – время первого расчёта."""
    key = json.dumps(req.model_dump(), sort_keys=True)  # Ключ кэша – все поля запроса.
    if key not in CACHE:  # Такого запроса ещё не было.
        start = time.time()  # Начало расчёта.
        CACHE[key] = classify_new(req) | {"seconds": 0}  # Место для времени – последним ключом.
        CACHE[key]["seconds"] = round(time.time() - start)  # Время расчёта, с.
        while len(CACHE) > 4:  # Хранятся только последние 4 ответа.
            CACHE.popitem(last=False)  # Удаляется самый старый ответ.
    return CACHE[key]  # Новый или сохранённый ответ.


def classify_new(req):
    """Расчёт /classify: выбор варианта модели и общий ответ."""
    if req.embeddings is None:  # Вариант не задан – лучший по карточке модели.
        key = DEFAULT_VARIANT  # Классификатор из карточки.
    elif req.embeddings and "embedding" in VARIANTS:  # Заданы эмбеддинги, и такой вариант есть.
        key = "embedding"  # Вариант на эмбеддингах.
    else:  # Каналы, а без такого варианта – первый из карточки.
        key = "spectral" if "spectral" in VARIANTS else next(iter(VARIANTS))  # Вариант на каналах или единственный.
    variant = VARIANTS[key]  # Классификатор, признаки, метрики.
    region, (main, other) = ee.Geometry(req.geometry), periods(req)  # Область и периоды.
    images = class_images(region, main, other, variant["features"], variant["model"])  # Карты классов периодов.
    name = "эмбеддинги AlphaEarth" if key == "embedding" else "каналы и индексы"  # Признаки – в подписи режима.
    # В подписи режима – название классификатора.
    return respond(images, req, CARD, f"своя модель: {variant['classifier'].lower()}, {name}", variant["metrics"])


@app.post("/train")  # Обучение на лету.
def train_area(req: Request):
    """Режим «обучение на лету»: случайный лес по карте меток внутри области интереса, затем классификация."""
    if req.labels not in LABELS:  # Неизвестная карта меток.
        raise HTTPException(400, f"Карта меток: {', '.join(LABELS)}")  # Подсказка: допустимые карты.
    try:  # Сетка до обучения: большая область отклоняется сразу.
        # Самое подробное разрешение, при котором хватает частей.
        grid = processing_grid(req.geometry, [req.scale] if req.scale else SCALES, MAX_PARTS)
    except ValueError as error:  # Область больше предела.
        raise HTTPException(400, str(error)) from error  # 400 – область нужно уменьшить.
    region, (main, other) = ee.Geometry(req.geometry), periods(req)  # Область и периоды.
    try:  # Обучение на периоде карты.
        # Случайный лес по меткам области.
        classifier, features, metrics = train_on_area(req.geometry, main[1], CARD["seasons"], req.labels, bool(req.embeddings))
    except ee.EEException as error:  # Мало точек, нет снимков, квота.
        raise HTTPException(400, f"Обучение не удалось: {error}")  # Причина – в тексте ответа.
    # Карточка модели на лету: классы карты меток.
    card = {"classes": LABELS[req.labels]["classes"], "focus": None, "seasons": CARD["seasons"], "features": features}
    images = class_images(region, main, other, features, classifier)  # Карты классов периодов.
    result = respond(images, req, card, f"обучение на лету ({req.labels})", metrics, grid=grid)  # Сетка уже подобрана.
    # Модель в ответе – чтобы применить её к другой области (/apply).
    result["model"] = {"classifier": classifier.serialize(), "features": features, "classes": card["classes"],
                       "labels": req.labels, "metrics": metrics}
    return result  # Общий ответ и модель.


@app.post("/apply")  # Модель на лету – к другой области.
def apply_model(req: Request):
    """Применение модели, обученной на лету, к другой области или другому периоду."""
    if not req.model:  # Модель не передана.
        raise HTTPException(400, "Нет модели: сначала обучите её в режиме «Обучение на лету»")  # 400 с подсказкой, что делать.
    model = ee.deserializer.fromJSON(req.model["classifier"])  # Классификатор из описания.
    # Карточка из описания модели.
    card = {"classes": req.model["classes"], "focus": None, "seasons": CARD["seasons"], "features": req.model["features"]}
    region, (main, other) = ee.Geometry(req.geometry), periods(req)  # Область и периоды.
    images = class_images(region, main, other, card["features"], model)  # Карты классов периодов.
    return respond(images, req, card, f"модель, обученная на лету ({req.model['labels']})", req.model["metrics"])  # Метрики модели.


@app.post("/segment")  # Сегментация сетью SegFormer-B0.
def segment_area(req: Request):  # Маска классов, доли классов, PNG, GeoTIFF и GeoJSON.
    start = time.time()  # Начало расчёта.
    try:  # Поле перекрытия – от 0 до 64 пикселей.
        # Маска классов и сетка фрагментов.
        mask, grid = segment(req.geometry, periods(req)[0][1], CARD, *NET, scale=req.scale, pad=max(0, min(req.overlap, 64)))
    except ValueError as error:  # Область больше предела фрагментов.
        raise HTTPException(400, str(error))  # 400 – область нужно уменьшить.
    except ee.EEException as error:  # Квота, тайм-аут, нет снимков.
        raise HTTPException(502, f"Earth Engine: {error}")  # 502 – сбой Earth Engine.
    inside = mask[mask >= 0]  # Пиксели области.
    shares = {CARD["classes"][k]: round(100 * float((inside == k).mean()), 1) for k in range(len(CARD["classes"]))}  # Доли, %.
    merc, bounds = to_mercator(mask, grid)  # Для наложения на подложку.
    # Картинка, доли, число фрагментов 256 × 256, время расчёта и файлы маски.
    return {"png": b64(to_png(merc, CARD)), "bounds": bounds, "shares": shares, "scale": grid["scale"],
            "tiles": len(grid["cells"]), "seconds": round(time.time() - start),
            "geotiff": b64(to_geotiff({"сегментация": mask}, grid)), "geojson": to_geojson(mask, grid, CARD)}
