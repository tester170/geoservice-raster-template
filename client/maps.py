# Карты растрового сервиса: подложки Esri, кнопки «Подложка» и «Слои», легенда классов в углу карты, инструменты
# рисования, восстановление положения карты; векторная выгрузка карты – полигоны классов из GeoTIFF в GeoJSON.
import io
import json

import folium
from branca.element import Element, MacroElement
from folium.plugins import Draw
from jinja2 import Template
from pyproj import Transformer
from rasterio.features import shapes
from rasterio.io import MemoryFile
from shapely.geometry import mapping, shape
from shapely.ops import transform


ACCENT = "#087f73"  # Цвет нарисованных фигур.
# Подложки Esri: сервис и название в списке.
BASEMAPS = [("World_Imagery", "Спутник Esri"), ("World_Street_Map", "Карта Esri"), ("World_Topo_Map", "Топографическая Esri")]
# Стили карты: размер, кнопки слоёв, легенды (шкала и классы), номерные метки.
MAP_CSS = """<style>
#map_div {height:100vh !important;}                     /* Карта заполняет окно компонента, высоту задаёт страница. */
#map_div:not(.draw-enabled) .leaflet-draw {display:none;}
.leaflet-control-layers {border:none !important; border-radius:10px !important; box-shadow:0 2px 10px rgba(24,43,58,.18) !important;
                         font:13px/1.45 -apple-system,'Segoe UI',Roboto,sans-serif;}
.leaflet-control-layers-toggle {width:38px !important; height:38px !important; background-size:20px !important;}
.basemaps .leaflet-control-layers-toggle {background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23182b3a' stroke-width='1.8' stroke-linejoin='round'%3E%3Cpath d='M3 6.5l6-2.5 6 2.5 6-2.5v13.5l-6 2.5-6-2.5-6 2.5z'/%3E%3Cpath d='M9 4v13.5M15 6.5V20'/%3E%3C/svg%3E") !important;}
.leaflet-control-layers-expanded {padding:8px 12px 8px 8px !important;}
.map-legend {position:absolute; z-index:1000; right:10px; bottom:22px; background:rgba(255,255,255,.94); border-radius:10px;
             padding:8px 12px; box-shadow:0 2px 10px rgba(24,43,58,.18); font:12px/1.4 -apple-system,'Segoe UI',Roboto,sans-serif; color:#182b3a;}
.map-legend .bar {height:9px; width:190px; border-radius:5px; margin:5px 0 3px;}
.map-legend.classes .item {display:flex; align-items:center; gap:7px; margin-top:4px; font-size:12.5px;}
.map-legend.classes i {width:12px; height:12px; border-radius:3px; border:1px solid rgba(0,0,0,.15);}
.map-legend .ends {display:flex; justify-content:space-between; color:#5b6b7b; font-size:11px;}
.pin {width:26px; height:26px; border-radius:50%; border:2px solid #fff; box-shadow:0 2px 6px rgba(0,0,0,.45);
      display:grid; place-items:center; font:700 12px/1 -apple-system,'Segoe UI',Roboto,sans-serif; color:#fff; text-shadow:0 1px 2px rgba(0,0,0,.6);}
</style>"""


class Controls(MacroElement):
    """Две кнопки управления картой: «Подложка» (базовые карты) и «Слои» (слои расчёта)."""
    # Скрипт Leaflet: два переключателя слоёв в правом верхнем углу.
    _template = Template("""{% macro script(this, kwargs) %}
    var bases_{{ this.get_name() }} = L.control.layers({ {% for n, l in this.bases %}{{ n|tojson }}: {{ l.get_name() }},{% endfor %} }, {},
        {collapsed: true, position: 'topright'}).addTo({{ this._parent.get_name() }});
    bases_{{ this.get_name() }}.getContainer().classList.add('basemaps');
    bases_{{ this.get_name() }}.getContainer().title = 'Подложка';
    var layers_{{ this.get_name() }} = L.control.layers({}, { {% for n, l in this.overlays %}{{ n|tojson }}: {{ l.get_name() }},{% endfor %} },
        {collapsed: true, position: 'topright'}).addTo({{ this._parent.get_name() }});
    layers_{{ this.get_name() }}.getContainer().title = 'Слои';
    {% endmacro %}""")

    def __init__(self, bases, overlays):  # Списки для двух кнопок.
        super().__init__()  # Инициализация элемента Folium.
        self._name = "Controls"  # Начало имени переменной в скрипте.
        self.bases, self.overlays = bases, overlays  # Пары (название, слой).


def class_legend(m, title, items):
    """Легенда классов в правом нижнем углу карты: [(цвет, подпись), …]."""
    rows = "".join(f'<div class="item"><i style="background:{c}"></i>{t}</div>' for c, t in items)  # Квадрат цвета и название.
    m.get_root().html.add_child(Element(f'<div class="map-legend classes"><b>{title}</b>{rows}</div>'))  # Легенда – поверх карты.


def base_map(bounds=None):
    """Подложки Esri (по умолчанию – спутник) без строки атрибуции; bounds – запад, юг, восток, север."""
    # Карта без своей подложки и строки атрибуции, со шкалой расстояний.
    m = folium.Map(location=[55.7558, 37.6176], zoom_start=11, tiles=None, control_scale=True, attribution_control=False)
    m.bases = []  # Подложки для кнопки «Подложка».
    for service, name in BASEMAPS:  # Подложка за подложкой.
        # Тайлы Esri; виден только спутник.
        layer = folium.TileLayer(f"https://services.arcgisonline.com/ArcGIS/rest/services/{service}/MapServer/tile/{{z}}/{{y}}/{{x}}",
                                 attr="Esri", name=name, show=service == "World_Imagery", control=False).add_to(m)
        m.bases.append((name, layer))  # В список кнопки «Подложка».
    m.overlays = []  # Слои расчёта для кнопки «Слои».
    m.get_root().header.add_child(Element(MAP_CSS))  # Стили карты в заголовке страницы.
    if bounds is not None:  # Охват задан.
        m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])  # Углы: юго-запад и северо-восток.
    return m  # Карта пока без слоёв расчёта.


# Фокус по щелчку на карте и метках – без прокрутки страницы: иначе карта выше окна сдвигает страницу и метка уходит из-под курсора.
NO_SCROLL_FOCUS = """
        (function () { var focus = HTMLElement.prototype.focus;
          HTMLElement.prototype.focus = function (o) { return focus.call(this, Object.assign({preventScroll: true}, o || {})); };
          document.addEventListener('mousedown', function (e) {
            if (e.target.closest && e.target.closest('.leaflet-marker-icon')) e.preventDefault(); }, true); })();
    """


class Script(MacroElement):
    """Код JavaScript внутри скрипта карты: st_folium выполняет только его, а не вставки в HTML."""
    _template = Template("{% macro script(this, kwargs) %}{{ this.code }}{% endmacro %}")  # Код вставляется как есть.

    def __init__(self, code):  # Код JavaScript – текстом.
        super().__init__()  # Инициализация элемента Folium.
        self.code = code  # Текст скрипта.


def drawing_tools(m):
    """Подписи инструментов и квадрат с равными сторонами в проекции карты."""
    # Русские подписи, квадрат вместо прямоугольника, свойства фигуры (номер, вид, радиус) для передачи в Python.
    m.add_child(Script(f"""(() => {{
        const t = L.drawLocal.draw;
        t.toolbar.buttons.polygon = 'Нарисовать полигон';
        t.toolbar.buttons.rectangle = 'Нарисовать квадрат';
        t.toolbar.buttons.circle = 'Нарисовать круг';
        t.toolbar.actions.text = 'Отмена';
        t.toolbar.actions.title = 'Отменить рисование';
        t.toolbar.finish.text = 'Завершить';
        t.toolbar.finish.title = 'Завершить полигон';
        t.toolbar.undo.text = 'Удалить последнюю точку';
        t.toolbar.undo.title = 'Удалить последнюю точку';
        t.handlers.polygon.tooltip.start = 'Щёлкните, чтобы начать полигон';
        t.handlers.polygon.tooltip.cont = 'Щёлкните, чтобы продолжить полигон';
        t.handlers.polygon.tooltip.end = 'Щёлкните первую точку, чтобы замкнуть полигон';
        t.handlers.rectangle.tooltip.start = 'Нажмите и потяните от угла квадрата';
        t.handlers.circle.tooltip.start = 'Нажмите и потяните от центра круга';
        t.handlers.circle.radius = 'Радиус';
        t.handlers.simpleshape.tooltip.end = 'Отпустите кнопку, чтобы завершить';

        const drawRectangle = L.Draw.Rectangle.prototype._drawShape;
        L.Draw.Rectangle.include({{
            _drawShape(latlng) {{
                const start = this._map.project(this._startLatLng);
                const delta = this._map.project(latlng).subtract(start);
                const side = Math.max(Math.abs(delta.x), Math.abs(delta.y));
                const corner = start.add(L.point(delta.x < 0 ? -side : side, delta.y < 0 ? -side : side));
                drawRectangle.call(this, this._map.unproject(corner));
            }}
        }});

        {m.get_name()}.on(L.Draw.Event.CREATED, (e) => {{
            const properties = {{area_draw_id: Date.now() + '-' + L.stamp(e.layer), area_shape: e.layerType}};
            if (e.layerType === 'circle') properties.radius = e.layer.getRadius();
            e.layer.feature = {{type: 'Feature', properties}};
        }});
    }})();"""))
    style = {"shapeOptions": {"color": ACCENT}}  # Цвет фигур при рисовании.
    Draw(draw_options={"polygon": style, "rectangle": style, "circle": style,  # Только полигон, квадрат и круг, без правки и удаления.
                       "polyline": False, "marker": False, "circlemarker": False},
         edit_options={"edit": False, "remove": False}, show_geometry_on_click=False).add_to(m)


def drawing_mode(enabled):
    """Показать инструменты полигона без пересоздания карты."""
    group = folium.FeatureGroup(control=False)  # Группа, которую st_folium обновляет без перезагрузки карты.
    # Выключение отменяет начатое рисование и скрывает панель.
    group.add_child(Script(f"""(() => {{
        const enabled = {json.dumps(enabled)};
        const map = map_div.getContainer();
        if (!enabled) map.querySelector('.leaflet-draw-actions a[title="Отменить рисование"]')?.click();
        map.classList.toggle('draw-enabled', enabled);
    }})();"""))
    return group  # Группа для feature_group_to_add.


def remember_view(m, bounds):
    """Восстановить положение карты при смене слоёв той же области."""
    region = json.dumps(json.dumps(bounds))  # Область как строка JavaScript.
    # Центр и масштаб хранятся в sessionStorage, пока область та же.
    m.add_child(Script(f"""(() => {{
        const map = {m.get_name()};
        const region = {region};
        const key = 'landcover-map-view';
        try {{
            const saved = JSON.parse(sessionStorage.getItem(key));
            if (saved?.region === region) map.setView(saved.center, saved.zoom, {{animate: false}});
            const save = () => {{
                sessionStorage.setItem(key, JSON.stringify({{region, center: map.getCenter(), zoom: map.getZoom()}}));
            }};
            save();
            map.on('moveend', save);
        }} catch {{ /* Карта также работает при отключённом хранилище браузера. */ }}
    }})();"""))


def finish(m):
    """Кнопки «Подложка» и «Слои» – после всех слоёв, затем фокус карты без прокрутки страницы."""
    m.add_child(Controls(m.bases, m.overlays))  # Кнопки видят все добавленные слои.
    m.add_child(Script(NO_SCROLL_FOCUS))  # Скрипт – в коде карты, чтобы st_folium его выполнил.
    return m  # Готовая карта.


def overlay(m, layer, name):  # Слой на карте и в списке кнопки «Слои».
    layer.add_to(m)  # Слой на карте.
    m.overlays.append((name, layer))  # Название в переключателе.


def geojson_from_geotiff(data, class_names):
    """Собрать полигоны первого канала в WGS 84 с кодами и названиями классов."""
    with MemoryFile(data) as memory, memory.open() as raster, io.BytesIO() as output:  # Растр и ответ – в памяти.
        reproject = Transformer.from_crs(raster.crs, "EPSG:4326", always_xy=True).transform  # Пересчёт в долготу и широту.
        tolerance = raster.res[0] / 2  # Упрощение до половины пикселя.
        encoder = json.JSONEncoder(ensure_ascii=False, separators=(",", ":"))  # Компактный JSON с кириллицей.
        output.write(b'{"type":"FeatureCollection","features":[')  # Начало коллекции объектов.
        separator = b""  # Перед первым объектом запятой нет.
        for geometry, value in shapes(raster.read(1), mask=raster.read_masks(1), transform=raster.transform):  # Области одного кода.
            code = int(value)  # Код класса.
            polygon = transform(reproject, shape(geometry).simplify(tolerance))  # Упростить и перевести в WGS 84.
            # Объект GeoJSON с кодом и названием класса.
            feature = {"type": "Feature", "properties": {"код": code, "класс": class_names[code]}, "geometry": mapping(polygon)}
            output.write(separator + encoder.encode(feature).encode("utf-8"))  # Объект сразу в байты: словари не копятся.
            separator = b","  # Перед следующими объектами – запятая.
        output.write(b"]}")  # Конец коллекции.
        return output.getvalue()  # Байты GeoJSON.
