# Клиентская часть растрового сервиса Sentinel Classification: область (город, полигон, готовый квадрат), своя модель
# или обучение на лету, карта классов (PNG от серверной части), площади, изменения, сегментация, GeoTIFF и GeoJSON.
# Запуск: streamlit run main.py. Серверная часть работает внутри приложения; если она развёрнута
# отдельно, её адрес задаётся переменной окружения или секретом API_URL.
import base64, datetime, json, os, sys  # Декодирование файлов, даты, настройки.
from pathlib import Path  # Папка сервиса.
import requests, pandas as pd, folium, streamlit as st  # Запросы, таблицы, карта и страница.
from pyproj import Geod, Transformer  # Площадь на эллипсоиде и перепроецирование.
from shapely.geometry import shape  # Геометрия из GeoJSON.
from streamlit_folium import st_folium  # Карта folium на странице.
from client.maps import base_map, overlay, finish, class_legend, drawing_tools, drawing_mode, remember_view  # Карта и слои.
from client.maps import geojson_from_geotiff  # Полигоны классов из GeoTIFF.
from client.interface import page, fmt, map_height, drawn_area, upload_area  # Оформление страницы, границы из файлов и рисунков.

OVERLAPS = {"нет": 0, "малое": 16, "среднее": 32, "сильное": 64}  # Поле фрагмента сегментации, пикселей.
RESOLUTIONS = ["Авто", 10, 20, 30, 60, 100, 250, 500]  # Размер пикселя карты, м; «Авто» – по площади области.
# Настройки страницы и шапка сервиса.
page("Sentinel Classification", "Классификация земного покрова и сравнение периодов по снимкам Sentinel-2",
     "orbit", "Sentinel Classification")


def secret(name):  # Значение из секретов Streamlit; без файла секретов – пусто.
    try:
        return str(st.secrets.get(name, ""))  # Secrets на Streamlit Community Cloud или .streamlit/secrets.toml.
    except Exception:  # Файла секретов нет (локальный запуск).
        return ""  # Значение не задано.


# Адрес отдельно развёрнутой серверной части; пусто – серверная часть работает внутри приложения.
API_URL = (os.environ.get("API_URL") or secret("API_URL")).rstrip("/")


SERVICE = Path(__file__).resolve().parent  # Папка сервиса.


def code_version():  # Время последнего изменения кода серверной части и модели.
    # Код серверной части, собственная функция и файлы модели.
    files = [*SERVICE.glob("server/*.py"), SERVICE / "extensions.py", *SERVICE.glob("models/**/*.*")]
    return max(f.stat().st_mtime for f in files)  # Обновили репозиторий – значение изменилось.


@st.cache_resource(show_spinner="Запуск серверной части…", max_entries=1)  # Один раз на версию кода.
def local_api(version):
    """Серверная часть (FastAPI) внутри этого приложения: запросы к ней идут без сети."""
    for name in ("EE_SERVICE_ACCOUNT", "EE_PROJECT"):  # Ключ и проект Earth Engine – из секретов.
        if secret(name) and name not in os.environ:  # Задан в Secrets, но ещё не в окружении.
            os.environ[name] = secret(name)  # Серверная часть читает его из окружения.
    for name in [m for m in sys.modules if m == "extensions" or m.split(".")[0] == "server"]:  # Прежний код.
        del sys.modules[name]  # Обновлённый код импортируется заново.
    from fastapi.testclient import TestClient  # Запросы к приложению FastAPI без сети.
    from server.api import app  # Серверная часть сервиса.
    return TestClient(app, raise_server_exceptions=False)  # Сбой расчёта – ответ 500, как у отдельного сервера.


def call(method, path, **kwargs):  # Запрос к серверной части: по адресу API_URL или внутри приложения.
    if API_URL:  # Серверная часть развёрнута отдельно.
        return requests.request(method, API_URL + path, **kwargs)  # Запрос по сети.
    return local_api(code_version()).request(method, path, **kwargs)  # Тот же запрос без сети.


@st.cache_data(show_spinner=False)  # Сведения о модели – один раз на версию кода и модели.
def health(version):  # Модель, классы, сезоны и варианты серверной части; version – только для ключа кэша.
    response = call("GET", "/health", timeout=60)  # Ответ /health.
    response.raise_for_status()  # Ответ с ошибкой не попадает в кэш.
    return response.json()  # Модель, классы, сезоны.


@st.cache_data(ttl=3600, show_spinner=False)  # Одинаковые запросы – из кэша в течение часа.
def get(path, **params):  # Запрос GET к серверной части: справочник городов, граница города.
    response = call("GET", path, params=params, timeout=600)  # Параметры – в адресе.
    if response.status_code >= 400:  # Сервер сообщил об ошибке.
        raise RuntimeError(response.json().get("detail", response.text))  # Текст ошибки из поля detail.
    return response.json()  # Данные ответа.


def square(center, half=15000):
    """Квадрат 30 × 30 км вокруг центра, как участки обучения и контроля модели (EPSG:6931)."""
    to = Transformer.from_crs("EPSG:4326", "EPSG:6931", always_xy=True)  # В метры равновеликой проекции.
    back = Transformer.from_crs("EPSG:6931", "EPSG:4326", always_xy=True)  # Обратно в градусы.
    x, y = to.transform(*center)  # Центр в метрах.
    corners = [(-half, -half), (half, -half), (half, half), (-half, half), (-half, -half)]  # Замкнутый контур.
    ring = [back.transform(x + dx, y + dy) for dx, dy in corners]  # Углы в градусах.
    return {"type": "Polygon", "coordinates": [ring]}  # Полигон GeoJSON.


def km2(geometry):
    """Площадь полигона GeoJSON на эллипсоиде, км²."""
    return abs(Geod(ellps="WGS84").geometry_area_perimeter(shape(geometry))[0]) / 1e6  # м² → км²


def label(key):
    """Подпись периода: интервал дат «01.06.2026–31.08.2026» или год."""
    if isinstance(key, str) and "…" in key:  # Период задан датами.
        start, end = (datetime.date.fromisoformat(x) for x in key.split("…"))  # Начало и конец.
        return f"{start:%d.%m.%Y}–{end:%d.%m.%Y}"  # Даты через тире.
    return f"{key} г."  # Подпись года.


def request(path, body):  # Запрос POST к серверной части.
    try:
        response = call("POST", f"/{path}", json=body, timeout=3600)  # Тело запроса – JSON.
    except requests.RequestException:  # Серверная часть недоступна или не ответила.
        st.error(f"Нет связи с серверной частью {API_URL}. Обновите страницу, когда она запустится.")  # Вместо трассировки.
        return None  # Результата нет.
    if response.status_code >= 400:  # Сервер сообщил об ошибке.
        st.error(response.json().get("detail", response.text))  # Текст ошибки из поля detail.
        return None  # Результата нет.
    return response.json()  # Данные ответа.


def picture(png, bounds, show=True):  # Картинка PNG от серверной части – слой карты.
    # Слой по углам области.
    return folium.raster_layers.ImageOverlay("data:image/png;base64," + png, bounds=bounds, opacity=0.75, show=show,
                                             control=False)


try:
    info, cities = health(code_version()), get("/cities")  # Сведения о модели и справочник городов.
except requests.RequestException:  # Серверная часть не запущена или недоступна.
    st.error(f"Нет связи с серверной частью {API_URL}. Обновите страницу, когда она запустится.")  # Понятное сообщение вместо трассировки.
    st.stop()  # Без сведений о модели страница не строится.
except Exception as error:  # Серверная часть внутри приложения не запустилась.
    st.error(f"Серверная часть не запустилась: {error}. Проверьте ключ EE_SERVICE_ACCOUNT в Secrets.")  # Причина.
    st.stop()  # Без сведений о модели страница не строится.
state = st.session_state  # Состояние страницы.
result, seg, area = state.get("result"), state.get("segmentation"), state.get("area")  # Классификация, сегментация, область.


def reset_area():  # Убрать область и рассчитанные слои.
    for key in ("area", "result", "result_area", "segmentation"):  # Область и результаты.
        state.pop(key, None)  # Ключа может не быть.


def clear_map():  # Кнопка «Очистить карту».
    reset_area()  # Убрать область и слои.
    state["area_city"] = None  # Поле города – пустое.
    state["area_square"] = None  # Квадрат не выбран.
    state["polygon_upload_revision"] = state.get("polygon_upload_revision", 0) + 1  # Новое поле загрузки – без файлов.


def select_area(selected):  # Сделать область текущей.
    reset_area()  # Прежние результаты к ней не относятся.
    state["area"] = selected  # Новая область.
    state["map_geometry"] = selected["geometry"]  # Охват карты.
    st.rerun()  # Показать область на карте.


if result:  # Полоса показателей классификации.
    focus = info.get("focus") if info.get("focus") in result["areas"] else max(result["areas"], key=result["areas"].get)  # Класс.
    share = 100 * result["areas"][focus] / max(sum(result["areas"].values()), 1e-9)  # Доля класса, %.
    cols = st.columns(4)  # Четыре показателя в ряд.
    cols[0].metric(f"Область: {state.get('result_area', '')}", f"{fmt(result['area_km2'])} км²",  # Площадь области.
                   help=f"Пиксель карты {result['scale']} м; режим – {result['mode']}")
    cols[1].metric(f"{focus.capitalize()}, км²", fmt(result["areas"][focus]), help=label(result["year"]))  # Площадь класса.
    cols[2].metric(f"{focus.capitalize()}: доля области", f"{fmt(share)} %")  # Доля класса.
    if result.get("compare_areas"):  # Есть период сравнения.
        cols[3].metric(f"{focus.capitalize()}: изменение, км²", fmt(result["net_change"][focus], 2),  # Изменение площади.
                       help=f"С {label(result['compare_year'])} по {label(result['year'])}: переходы в класс минус переходы из него")
    else:
        cols[3].metric("Точность модели", fmt(result["metrics"].get("общая точность") or 0, 3),  # Точность модели.
                       help="Общая точность на отложенных пространственных блоках")

map_height(310 if result else 215)  # Шапка, полоса показателей, легенда под картой.
left, right = st.columns([2.5, 1])  # Карта слева, параметры справа.
with right:  # Правая колонка.
    with st.container(border=True, gap="xsmall"):  # Блок параметров.
        title, clear = st.columns([2.3, 1], gap="xsmall", vertical_alignment="center")  # Заголовок и кнопка очистки.
        title.subheader("Параметры расчёта")  # Заголовок блока.
        with clear, st.container(horizontal=True, horizontal_alignment="right"):  # Кнопка у правого края.
            st.button("Очистить карту", icon=":material/cancel:", width=40, key="clear_map_button",  # Кнопка очистки.
                      disabled=area is None, on_click=clear_map,
                      help="Очистить карту: убрать границу, рассчитанные слои и их файлы.")
        way = st.segmented_control("Способ", ["Город", "Полигон", "Квадрат"], default="Город",  # Способ задать область.
                                   label_visibility="collapsed",
                                   format_func=lambda value: "Участки" if value == "Квадрат" else value) or "Город"
        current = area or {}  # Текущая область или пустая.
        if way == "Город":  # Область – граница города.
            current_city = current.get("query")  # Город текущей области.
            options = cities if current_city is None or current_city in cities else [*cities, current_city]  # Свой – в конец.
            index = options.index(current_city) if current_city is not None else None  # Выбранный пункт.
            name = st.selectbox("Город", options, index=index, accept_new_options=True, label_visibility="collapsed",  # Город.
                                placeholder="Город из списка или своё название", key="area_city",
                                help="Граница населённого пункта загружается из OpenStreetMap; своё название можно уточнить "
                                     "регионом, например «Абакан, Хакасия»")
            if name and name != current_city:  # Выбран новый город.
                try:
                    geometry = get("/place", city=name)["geometry"]  # Граница города от серверной части.
                except (requests.RequestException, RuntimeError) as error:  # Сервер недоступен или города нет.
                    st.error(str(error))  # Причина под списком городов.
                else:
                    select_area({"geometry": geometry, "name": name, "query": name, "method": "Город"})  # Город – новая область.
            ready = bool(name and name == current_city)  # Граница города уже на карте.
        elif way == "Полигон":  # Область – свой полигон.
            st.caption("Полигон, квадрат и круг рисуются на карте.")  # Подсказка о рисовании.
            imported = upload_area(state.get("polygon_upload_revision", 0))  # Границы из файла.
            if imported and imported != area:  # Применены новые границы.
                select_area(imported)  # Границы – новая область.
            ready = current.get("method") == way  # Полигон уже на карте.
        else:
            options = {f"Квадрат {i + 1}": c for i, c in enumerate(info["approbation"])}  # Центры готовых квадратов.
            names = list(options)  # Названия квадратов.
            current_square = current.get("name") if current.get("method") == way else None  # Выбранный квадрат.
            index = names.index(current_square) if current_square in options else None  # Его пункт.
            choice, area_size = st.columns([2.8, 1], gap="xsmall", vertical_alignment="center")  # Выбор и площадь рядом.
            pick = choice.radio("Готовый квадрат", names, index=index, horizontal=True,  # Выбор квадрата.
                                label_visibility="collapsed", key="area_square")
            if pick and pick != current_square:  # Выбран новый квадрат.
                select_area({"geometry": square(options[pick]), "name": pick, "method": "Квадрат"})  # Квадрат – новая область.
            ready = bool(pick and pick == current_square)  # Квадрат уже на карте.
            if ready:  # Площадь квадрата.
                area_size.caption(f"{fmt(km2(area['geometry']), 0)} км²")  # Показать площадь.
        if area and (way != "Квадрат" or not ready):  # Подпись выбранной области.
            label_area = "Выбрано" if ready else "На карте"  # Готова ли область к расчёту.
            size_km2 = km2(area["geometry"])  # Площадь области.
            amount = f"{fmt(size_km2 * 1e6, 0)} м²" if size_km2 < 1 else f"{fmt(size_km2, 0)} км²"  # Малые – в м².
            st.caption(f"{label_area}: **{area['name']}**, {amount}")  # Название и площадь.

        trained = state.get("trained")  # Модель, обученная на лету.
        modes = ["Своя модель", "Обучение на лету"] + (["Обученная на лету"] if trained else [])  # Доступные режимы.
        mode_labels = {"Своя модель": "Своя", "Обучение на лету": "На лету", "Обученная на лету": "Обученная"}  # Подписи.
        a, b = st.columns([1.4, 1], gap="xsmall", vertical_alignment="center")  # Режим и вариант модели рядом.
        mode = a.segmented_control("Режим", modes, default="Своя модель", label_visibility="collapsed", wrap=True,  # Режим.
                                   format_func=lambda value: mode_labels.get(value, value)) or "Своя модель"
        variants = info.get("variants", {})  # Варианты своей модели.
        fixed_variant = mode == "Своя модель" and len(variants) < 2  # Выбора варианта нет.
        use_emb = b.checkbox("AlphaEarth", value=fixed_variant and info.get("default_variant") == "embedding",  # Эмбеддинги.
                             disabled=mode == "Обученная на лету" or fixed_variant,
                             help="Использовать годовые эмбеддинги AlphaEarth для классификации.")
        if mode == "Обучение на лету":  # Карта меток для обучения.
            # Источник меток для случайного леса.
            labels = st.selectbox("Карта меток", info["labels"], help="Метки для обучения случайного леса внутри области; "
                                                                       "точность – на отложенных блоках 2 × 2 км")

        a, b = st.columns(2, gap="xsmall", vertical_alignment="bottom")  # Разрешение и перекрытие рядом.
        resolution = a.selectbox("Разрешение", RESOLUTIONS,  # Размер пикселя.
                                 format_func=lambda v: "Авто" if v == "Авто" else f"{v} м/пиксель",
                                 help="Размер пикселя карты. В режиме «Авто» подбирается по площади области.")
        overlap = b.selectbox("Перекрытие", list(OVERLAPS),  # Перекрытие фрагментов.
                              help="Перекрытие соседних фрагментов при сегментации уменьшает видимость стыков.")
        today = datetime.date.today()  # Верхний предел дат.
        m1, d1 = map(int, info["seasons"][0][0].split("-"))  # Начало сезона темы: месяц, день.
        m2, d2 = map(int, info["seasons"][-1][1].split("-"))  # Конец сезона темы.
        summer = today.year if today >= datetime.date(today.year, m2, d2) else today.year - 1  # Последний завершённый сезон.
        past = info["years"][0] if len(info["years"]) == 2 else summer - 5  # Год сравнения.
        first_day = datetime.date(2017, 4, 1)  # Начало снимков Sentinel-2 L2A.
        a, b = st.columns([1, 2.5], gap="xsmall", vertical_alignment="center")  # Подпись и даты рядом.
        a.caption("Снимки", help=f"Для AlphaEarth берётся год начала периода. Последний доступный год – "  # Подпись периода.
                                 f"{info.get('last_embedding_year', 2025)}.")
        # Период снимков.
        period = b.date_input("Снимки", (datetime.date(summer, m1, d1), datetime.date(summer, m2, d2)), min_value=first_day,
                              max_value=today, format="DD.MM.YYYY", label_visibility="collapsed")
        a, b = st.columns([1, 2.5], gap="xsmall", vertical_alignment="center")  # Флажок и даты рядом.
        compare_on = a.checkbox("Сравнить", value=len(info["years"]) == 2)  # Сравнивать ли периоды.
        # Период сравнения.
        other = b.date_input("Период сравнения", (datetime.date(past, m1, d1), datetime.date(past, m2, d2)), min_value=first_day,
                             max_value=today, format="DD.MM.YYYY", disabled=not compare_on, label_visibility="collapsed")
        dates = {"start": str(period[0]), "end": str(period[-1])} if len(period) == 2 else None  # Даты, если выбраны обе.
        classification_dates = dates  # Даты классификации.
        if compare_on and dates:  # С периодом сравнения.
            # Без второй даты сравнения классификация недоступна.
            classification_dates = dates | {"compare_start": str(other[0]), "compare_end": str(other[1])} if len(other) == 2 else None
        scale = None if resolution == "Авто" else resolution  # Пиксель; None – подберёт сервер.
        a, b = st.columns(2, gap="xsmall")  # Две кнопки в ряд.
        # Запуск классификации.
        run_map = a.button("Классификация", type="primary", width="stretch", disabled=not ready or not classification_dates,
                           help="Карта классов, площади и сравнение периодов.")
        run_seg = b.button("Сегментация", type="primary", width="stretch", disabled=not ready or not dates,  # Запуск сегментации.
                           help="Сегментация SegFormer-B0 для первого периода.")
        if mode == "Своя модель":  # Сведения о своей модели.
            v = variants.get("embedding" if use_emb else "spectral") or next(iter(variants.values()), {})  # Выбранный вариант.
            detail = fmt((v.get('metrics') or info['metrics']).get('общая точность', 0), 3)  # Его точность.
            st.caption(f"{v.get('classifier', info['classifier'])} · {len(info['classes'])} классов",  # Модель и число классов.
                       help=f"Общая точность модели: {detail}.")
        elif mode == "Обученная на лету":  # Сведения о модели, обученной на лету.
            # Область, метки и точность модели.
            st.caption(f"Случайный лес, обученный на лету по области «{trained['area']}» ({trained['labels']}), общая "
                       f"точность {fmt(trained['metrics'].get('общая точность') or 0, 3)}. Задайте другую область или период.")
            # Модель файлом JSON.
            st.download_button("Скачать модель", json.dumps(trained, ensure_ascii=False), file_name="модель_на_лету.json",
                               width="stretch")
        if mode != "Своя модель":  # Модель можно загрузить из файла.
            uploaded = st.file_uploader("Загрузить модель из файла", type="json")  # Файл модели.
            if uploaded is not None and st.button("Загрузить", width="stretch"):  # Применить файл.
                state["trained"] = json.loads(uploaded.getvalue())  # Модель из файла.
                st.rerun()  # Показать режим «Обученная».
        if run_map:  # Классификация.
            body = {"geometry": area["geometry"], "scale": scale, "embeddings": use_emb} | classification_dates  # Тело запроса.
            with st.spinner("Классификация области: от минуты до нескольких минут для больших областей…"):  # Индикатор.
                if mode == "Своя модель":  # Готовая модель.
                    answer = request("classify", body)  # Классификация своей моделью.
                elif mode == "Обучение на лету":  # Обучение внутри области.
                    answer = request("train", body | {"labels": labels})  # Обучение и классификация.
                else:
                    answer = request("apply", body | {"model": trained})  # Применение обученной модели.
            if answer:  # Расчёт выполнен.
                if answer.get("model"):  # Модель, обученная на лету, – для применения.
                    state["trained"] = answer.pop("model") | {"area": area["name"]}  # Запомнить модель и область.
                state["result"], state["result_area"] = answer, area["name"]  # Результат и его область.
                st.rerun()  # Показать результат.
        if run_seg:  # Сегментация.
            with st.spinner("Сегментация нейронной сетью SegFormer-B0…"):  # Индикатор.
                # Сегментация первого периода.
                answer = request("segment", {"geometry": area["geometry"], "start": dates["start"], "end": dates["end"],
                                             "scale": scale if scale in (10, 20, 30, 60) else None, "overlap": OVERLAPS[overlap]})
            if answer:  # Сегментация выполнена.
                state["segmentation"] = answer  # Результат сегментации.
                st.rerun()  # Показать результат.

with left:  # Левая колонка: карта.
    map_geometry = state.get("map_geometry")  # Геометрия для охвата карты.
    view = shape(map_geometry).bounds if map_geometry is not None else None  # Охват карты.
    m = base_map(view)  # Карта с подложками Esri.
    if result:  # Слои классификации.
        if result.get("compare_png"):  # Есть период сравнения.
            # Карта периода сравнения, по умолчанию скрыта.
            overlay(m, picture(result["compare_png"], result["bounds"], show=False), f"Классы, {label(result['compare_year'])}")
        overlay(m, picture(result["png"], result["bounds"]), f"Классы, {label(result['year'])}")  # Карта основного периода.
    if seg:  # Слой сегментации.
        overlay(m, picture(seg["png"], seg["bounds"]), "Сегментация SegFormer-B0")  # Маска SegFormer.
    if area:  # Граница области.
        overlay(m, folium.GeoJson(area["geometry"], control=False,  # Белый пунктир без заливки.
                                  style_function=lambda f: {"color": "white", "weight": 2, "dashArray": "6 4", "fill": False}),
                "Граница области интереса")
    drawing_tools(m)  # Подписи инструментов рисования.
    classes, colors = (result["classes"], result["colors"]) if result else (info["classes"], info["colors"])  # Классы.
    class_legend(m, "Классы", [(f"#{c}", k) for k, c in zip(classes, colors)])  # Легенда на карте, в правом нижнем углу.
    finish(m)  # Кнопки управления картой.
    remember_view(m, view)  # Положение карты при смене слоёв.
    drawn = st_folium(m, height=640, use_container_width=True, returned_objects=["last_active_drawing"],  # Карта на странице.
                      key="landcover_map", feature_group_to_add=drawing_mode(way == "Полигон"))
    feature = (drawn or {}).get("last_active_drawing") or {}  # Последняя нарисованная фигура.
    draw_id = feature.get("properties", {}).get("area_draw_id")  # Её номер.
    if draw_id and draw_id != state.get("last_area_drawing"):  # Нарисована новая фигура.
        state["last_area_drawing"] = draw_id  # Запомнить номер.
        if way == "Полигон":  # Рисование доступно для полигона.
            try:
                select_area(drawn_area(feature))  # Фигура – новая область.
            except ValueError as error:  # Некорректная фигура.
                st.error(str(error))  # Например, самопересечение контура.
    # Подпись об источниках данных.
    st.markdown('<div class="source">Источники: снимки Copernicus Sentinel-2 и маска облаков Cloud Score+, эмбеддинги '
                "AlphaEarth, карты меток Dynamic World и ESA WorldCover – каталог Google Earth Engine; границы – © сообщество "
                "OpenStreetMap; подложка – Esri.</div>", unsafe_allow_html=True)

if not result and not seg:  # Результатов нет.
    st.stop()  # Дальше показывать нечего.


def downloads(answer, class_names, title, name, help_text):  # Строка с файлами GeoTIFF и GeoJSON.
    tiff = base64.b64decode(answer["geotiff"])  # GeoTIFF из ответа.
    geojson = answer.get("geojson")  # Готовый GeoJSON, если сервер его прислал.

    def geojson_data():  # Строится только при скачивании.
        if geojson is not None:  # Сервер прислал полигоны.
            return json.dumps(geojson, ensure_ascii=False)  # Текст GeoJSON.
        return geojson_from_geotiff(tiff, class_names)  # Иначе – из GeoTIFF.

    a, b, c = st.columns([1.35, 1, 1], gap="xsmall", vertical_alignment="center")  # Подпись и две кнопки.
    a.caption(title)  # Подпись строки.
    b.download_button("GeoTIFF", tiff, file_name=f"{name}.tif", key=f"{name}_tiff",  # Файл GeoTIFF.
                      mime="image/tiff", width="stretch", help=help_text, on_click="ignore")
    c.download_button("GeoJSON", geojson_data, file_name=f"{name}.geojson", key=f"{name}_geojson",  # Файл GeoJSON.
                      mime="application/geo+json", width="stretch", help="Полигоны классов первого периода",
                      on_click="ignore")


with right, st.container(border=True, gap="xsmall"):  # Блок файлов.
    st.subheader("Файлы")  # Заголовок блока.
    if result:  # Файлы классификации.
        # Строка файлов классификации.
        downloads(result, result["classes"], "Классификация", "классификация", "Один канал на каждый период")
    if seg:  # Файлы сегментации.
        downloads(seg, info["classes"], "Сегментация", "сегментация", "Маска классов SegFormer-B0")  # Строка файлов сегментации.

with right, st.container(border=True, gap="xsmall"):  # Блок таблиц.
    if result:  # Площади классов.
        st.subheader("Площади классов")  # Заголовок таблицы площадей.
        table = pd.DataFrame({label(result["year"]): result["areas"]})  # Площади основного периода.
        if result.get("compare_areas"):  # Есть период сравнения.
            table.insert(0, label(result["compare_year"]), pd.Series(result["compare_areas"]))  # Площади периода сравнения.
            table["Изменение"] = pd.Series(result["net_change"])  # Изменение площадей.
        table = table.fillna(0).rename_axis("Класс, км²")  # Пропуски – нули.
        st.dataframe(table, width="stretch", row_height=28,  # Таблица площадей.
                     column_config={c: st.column_config.NumberColumn(format="%.2f") for c in table})
        if result.get("matrix"):  # Есть матрица переходов.
            with st.expander("Матрица переходов, км²"):  # Свёрнутый блок.
                st.caption(f"Строки – {label(result['compare_year'])}, столбцы – {label(result['year'])}")  # Пояснение.
                # Матрица переходов.
                st.dataframe(pd.DataFrame(result["matrix"], index=result["classes"], columns=result["classes"]), width="stretch")
    if seg:  # Доли классов сегментации.
        st.subheader("Сегментация")  # Заголовок таблицы долей.
        shares = pd.DataFrame({"доля, %": seg["shares"]}).rename_axis("Класс")  # Доли классов.
        # Таблица долей.
        st.dataframe(shares, width="stretch", row_height=28, column_config={"доля, %": st.column_config.NumberColumn(format="%.1f")})
    if result:  # Собственная функция и модель.
        ext = result.get("extension") or {}  # Результат собственной функции.
        st.subheader(ext.get("name", "Собственная функция"))  # Название функции.
        if ext.get("text"):  # Есть пояснение.
            st.caption(ext["text"])  # Краткий вывод функции.
        if ext.get("table"):  # Есть таблица.
            st.dataframe(pd.DataFrame(ext["table"]), width="stretch", hide_index=True, row_height=28)  # Таблица функции.
        st.subheader("Модель")  # Заголовок сведений о модели.
        st.caption(f"{result['mode'][:1].upper() + result['mode'][1:]} · пиксель {result['scale']} м")  # Режим и пиксель.
        st.dataframe(pd.DataFrame({"значение": result["metrics"]}), width="stretch", row_height=28)  # Точность модели.
