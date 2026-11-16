# Оформление страницы и область интереса: стили, шапка сервиса, высота карты, формат чисел; границы из файлов
# (GeoJSON, GeoPackage, ZIP с SHP) и фигуры, нарисованные на карте. Отличается от векторного сервиса отступами
# показателей и закреплённой колонкой карты.
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import BadZipFile, ZipFile

import numpy as np
import pyogrio
import streamlit as st
from pyogrio.errors import DataSourceError, DataLayerError
from pyproj import Geod, Transformer
from pyproj.exceptions import ProjError
from shapely import from_wkb, orient_polygons, union_all
from shapely.errors import ShapelyError
from shapely.geometry import Polygon, mapping, shape
from shapely.ops import transform


ACCENT = "#087f73"  # Основной цвет сервиса.
ICONS = {  # Знак сервиса: шестиугольная сетка (векторный) и снимок со спутника (растровый).
    "hex": '<path d="M12 3l4.3 2.5v5L12 13l-4.3-2.5v-5zM6.3 12.5l4.3 2.5v5L6.3 22.5 2 20v-5zM17.7 12.5L22 15v5l-4.3 2.5-4.3-2.5v-5z" '
           'fill="none" stroke="white" stroke-width="1.6" stroke-linejoin="round"/>',
    "orbit": '<rect x="7" y="7" width="10" height="10" rx="1.5" fill="none" stroke="white" stroke-width="1.6"/>'
             '<path d="M3 12h4M17 12h4M12 3v4M12 17v4" stroke="white" stroke-width="1.6" stroke-linecap="round"/>'
             '<path d="M7 13.5l3-3 2 2 2.5-2.5L17 12.5" fill="none" stroke="white" stroke-width="1.4" stroke-linejoin="round"/>',
}
# Стили страницы: отступы, шапка, показатели, легенда, кнопка очистки карты, поле загрузки файлов.
CSS = """<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@600;700;800&display=swap');
.block-container {padding:1.4rem 2rem 2rem; max-width:100%;}
[data-testid="stAppDeployButton"], #MainMenu, [data-testid="stDecoration"] {display:none;}
[data-testid="stHeader"] {height:1.5rem; background:transparent;}
.brand {display:flex; align-items:center; gap:14px; margin:.2rem 0 1.2rem;}
.brand .mark {flex:none; width:44px; height:44px; border-radius:12px; background:#087f73; display:grid; place-items:center;}
.brand h1 {font-family:'Manrope', sans-serif; font-size:1.75rem !important; font-weight:800; letter-spacing:-.04em;
           margin:0; padding:0 !important; line-height:1.05; color:#182b3a;}
.brand p {margin:.3rem 0 0; color:#5b6b7b; font-size:.88rem;}
h3 {font-size:1.02rem !important; font-weight:650 !important; letter-spacing:-.015em; padding-top:.2rem !important;}
[data-testid="stVerticalBlockBorderWrapper"]:has(> div > [data-testid="stVerticalBlock"]) {background:#fff;}
div[data-testid="stMetric"] {background:#fff; padding:.7rem 1rem; border:1px solid #e1e9ed; border-radius:12px;}
div[data-testid="stMetricLabel"] p {color:#5b6b7b; font-size:.8rem;}
div[data-testid="stMetricValue"] {font-family:'Manrope', sans-serif; font-size:1.45rem; font-weight:700; color:#182b3a;}
.legend {display:flex; flex-wrap:wrap; gap:6px 18px; font-size:12.5px; color:#5b6b7b; margin:.5rem 0 .2rem;}
.swatch {display:inline-block; width:11px; height:11px; border-radius:3px; margin-right:6px; vertical-align:-1px; border:1px solid #0002;}
.source {font-size:12px; color:#5b6b7b; margin-top:.2rem;}
iframe {border-radius:12px;}
.st-key-clear_map_button button {
    position:relative; display:grid; place-items:center; height:40px; min-height:0; padding:0;
}
.st-key-clear_map_button button [data-testid="stIconMaterial"] {margin:0; font-size:20px; line-height:1;}
.st-key-clear_map_button button [data-testid="stMarkdownContainer"] {
    position:absolute; width:1px; height:1px; overflow:hidden; clip-path:inset(50%); white-space:nowrap;
}
[data-testid='stFileUploaderDropzone'] button p, [data-testid='stFileUploaderDropzoneInstructions'] span {font-size:0;}
[data-testid='stFileUploaderDropzone'] button p::after {content:'Выбрать файл'; font-size:14px;}
[data-testid='stFileUploaderDropzoneInstructions'] span::after {content:'файл JSON до 200 МБ'; font-size:14px;}
.st-key-polygon_upload [data-testid='stFileUploaderDropzone'] {
    padding:.45rem .6rem; gap:.4rem; align-items:center;
}
.st-key-polygon_upload [data-testid='stFileUploaderDropzone'] button p::after {content:'Выбрать файлы'; font-size:12px;}
.st-key-polygon_upload [data-testid='stFileUploaderDropzoneInstructions'] {justify-content:center; text-align:center;}
.st-key-polygon_upload [data-testid='stFileUploaderDropzoneInstructions'] span::after {
    content:'GeoJSON, GPKG, ZIP с SHP'; font-size:11px;
}
@media (max-width:768px) {.block-container {padding:1.2rem 1rem 2rem;}}
</style>"""


def page(title, subtitle, icon, browser_title):
    """Настройки страницы, стили и шапка сервиса: знак, английское название, пояснение по-русски."""
    st.set_page_config(page_title=browser_title, page_icon=Path(__file__).with_name("favicon.png"),  # Вкладка браузера, широкая страница.
                       layout="wide", initial_sidebar_state="collapsed")
    st.html(CSS)  # Стили страницы.
    # Шапка: знак, название, пояснение.
    st.markdown(f'<div class="brand"><div class="mark"><svg width="26" height="26" viewBox="0 0 24 24">{ICONS[icon]}</svg></div>'
                f'<div><h1>{title}</h1><p>{subtitle}</p></div></div>',
                unsafe_allow_html=True)


def map_height(offset):
    """Высота карты под доступное окно, не менее 460 пикселей."""
    frame = "iframe[title='streamlit_folium.st_folium']"  # Окно карты st_folium.
    css = f"{frame} {{display:block; height:max(460px, calc(100vh - {offset}px)) !important;}}"  # Высота окна без шапки и панелей.
    column = f"[data-testid='stColumn']:has({frame})"  # Колонка, в которой стоит карта.
    css += f"@media (min-width:1024px) {{{column} {{position:sticky; top:1.5rem; align-self:flex-start;}}}}"  # Карта не уезжает вверх.
    st.html(f"<style>{css}</style>")  # Стиль на странице.


def fmt(value, digits=1):
    """Число с пробелами между разрядами и десятичной запятой."""
    return f"{value:,.{digits}f}".replace(",", " ").replace(".", ",")  # 12,345.6 → 12 345,6


SHAPE_NAMES = {"circle": "Круг", "rectangle": "Квадрат", "polygon": "Нарисованный полигон"}  # Названия нарисованных фигур.


def read_polygons(filename, contents):
    """Вернуть полигональные слои файла в WGS 84. В ZIP читаются только слои SHP."""
    suffix = Path(filename).suffix.lower()  # Формат файла по расширению.
    try:
        with TemporaryDirectory(prefix="landcover-") as folder:  # Папка удаляется после чтения.
            path = Path(folder, "area" + suffix)  # GDAL читает файл с диска.
            path.write_bytes(contents)  # Содержимое загруженного файла.
            if suffix == ".zip":  # Архив ZIP: ищем в нём SHP.
                with ZipFile(path) as archive:  # Список файлов архива.
                    members = [name for name in archive.namelist() if Path(name).suffix.lower() == ".shp"]  # Файлы SHP.
                if not members:  # В архиве нет ни одного SHP.
                    raise ValueError("В архиве нет файла .shp.")  # Без SHP читать нечего.
                sources = [(member, f"/vsizip/{{{path.as_posix()}}}/{member}", None) for member in members]  # SHP внутри ZIP.
            else:
                sources = [(name, path, name) for name, kind in pyogrio.list_layers(path) if kind is not None]  # Слои с геометрией.
            results = {}  # Имя слоя → полигоны в WGS 84.
            for name, source, layer in sources:  # Каждый слой по очереди.
                meta, _, geometries, _ = pyogrio.raw.read(source, layer=layer, columns=[], force_2d=True)  # Только геометрия.
                polygons = [p for p in from_wkb(geometries)  # Непустые полигоны и мультиполигоны.
                            if p is not None and p.geom_type in ("Polygon", "MultiPolygon") and not p.is_empty]
                if not polygons:  # В слое нет полигонов.
                    continue  # Слой без полигонов пропускается.
                crs = meta["crs"]  # Система координат слоя.
                if not crs and suffix not in (".geojson", ".json"):  # Без системы координат пересчёт невозможен.
                    raise ValueError(f"У слоя «{name}» не указана система координат.")  # Ошибка с именем слоя.
                if crs and crs != "EPSG:4326":  # GeoJSON без системы координат – уже WGS 84.
                    to_wgs84 = Transformer.from_crs(crs, "EPSG:4326", always_xy=True).transform  # Пересчёт в долготу и широту.
                    polygons = [transform(to_wgs84, p) for p in polygons]  # Каждый полигон – в WGS 84.
                results[name] = [checked_polygon(p) for p in polygons]  # Проверка каждого полигона.
    except (BadZipFile, DataSourceError, DataLayerError, ProjError, ShapelyError) as error:
        # Понятная ошибка вместо сообщения GDAL.
        raise ValueError("Не удалось прочитать границы. Проверьте формат файла, геометрию и систему координат.") from error
    if not results:  # Ни в одном слое нет полигонов.
        raise ValueError("В файле нет полигонов.")  # Сообщение покажется под полем загрузки.
    return results  # Имя слоя → полигоны.


def checked_polygon(polygon):  # Проверить полигон и ориентировать кольца по правилу GeoJSON.
    # Не полигон, пустой или с самопересечением.
    if polygon.geom_type not in ("Polygon", "MultiPolygon") or polygon.is_empty or not polygon.is_valid:
        raise ValueError("Некорректная геометрия. Проверьте границы на самопересечения.")  # Расчёт по такой границе невозможен.
    west, south, east, north = polygon.bounds  # Охват полигона.
    if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):  # Долгота и широта в допустимых пределах.
        # Ошибка вместо неверной области на карте.
        raise ValueError("Координаты выходят за пределы WGS 84. Проверьте систему координат файла.")
    return orient_polygons(polygon)  # Внешнее кольцо против часовой стрелки.


def uploaded_geometry(polygons):
    """Объединить границы, чтобы перекрытия учитывались один раз."""
    return mapping(checked_polygon(union_all(polygons)))  # Объединение в GeoJSON.


def drawn_area(feature):  # Область интереса из фигуры, нарисованной на карте.
    props = feature["properties"]  # Свойства от скрипта рисования.
    if props["area_shape"] == "circle":  # Круг: строится из центра и радиуса.
        lon, lat = feature["geometry"]["coordinates"]  # Центр круга.
        bearings = np.arange(128) * (360 / 128)  # 128 направлений через равные углы.
        x, y, _ = Geod(ellps="WGS84").fwd(np.full(128, lon), np.full(128, lat), bearings,  # Вершины на радиусе по эллипсоиду.
                                          np.full(128, props["radius"]))
        polygon = Polygon(zip(x, y))  # Круг – многоугольник из 128 вершин.
    else:
        polygon = shape(feature["geometry"])  # Квадрат или полигон – как нарисованы.
    # Область с названием фигуры.
    return {"geometry": mapping(checked_polygon(polygon)), "name": SHAPE_NAMES[props["area_shape"]], "method": "Полигон"}


@st.cache_data(show_spinner=False, max_entries=8)
def upload_layers(name, contents):  # Кэш: файл не читается заново при каждом обновлении страницы.
    return read_polygons(name, contents)  # Чтение – только при новом файле.


def upload_area(revision):
    """Показать слои и применить выбранные полигоны как область интереса."""
    with st.container(key="polygon_upload", gap="xsmall"):  # Компактное поле загрузки.
        # Поле загрузки; новый ключ revision очищает его.
        files = st.file_uploader("Загрузить границы", type=["geojson", "json", "gpkg", "zip"],
                                 accept_multiple_files=True, label_visibility="collapsed",
                                 key=f"polygon_files_{revision}")
        selected, valid = [], True  # Выбранные полигоны и признак ошибок.
        for file in files:  # Каждый загруженный файл.
            try:
                layers = upload_layers(file.name, file.getvalue())  # Слои файла.
            except (ValueError, OSError) as error:
                st.error(f"{file.name} · {error}")  # Файл не прочитан – причина под полем загрузки.
                valid = False  # С ошибкой границы применить нельзя.
                continue  # Переход к следующему файлу.
            layer = next(iter(layers))  # Первый слой по умолчанию.
            if len(layers) > 1:  # Слоёв несколько – выбор слоя.
                # Список слоёв файла.
                layer = st.selectbox(f"Слой в файле «{file.name}»", list(layers), key=f"polygon_layer_{file.file_id}")
            selected.extend(layers[layer])  # Полигоны выбранного слоя.
        if not files or not valid:  # Нет файлов или есть ошибка.
            return None  # Кнопка «Применить границы» не показывается.
        if len(selected) > 1:  # Полигонов несколько – выбор.
            pick = st.selectbox("Полигоны", list(range(len(selected) + 1)), label_visibility="collapsed",  # Номер полигона; 0 – все вместе.
                                format_func=lambda i: f"Все полигоны ({len(selected)})" if i == 0 else f"Полигон {i}",
                                key="polygon_feature", help="Примените все полигоны вместе или выберите один.")
            if pick:  # Выбран один полигон.
                selected = [selected[pick - 1]]  # Один выбранный полигон.
        if st.button("Применить границы", width="stretch"):  # Границы применяются по кнопке.
            name = files[0].name if len(files) == 1 else f"Загруженные полигоны ({len(selected)})"  # Название области.
            try:
                return {"geometry": uploaded_geometry(selected), "name": name, "method": "Полигон"}  # Новая область интереса.
            except ValueError as error:
                st.error(str(error))  # Например, полигоны вне WGS 84.
        return None  # Границы ещё не применены.
