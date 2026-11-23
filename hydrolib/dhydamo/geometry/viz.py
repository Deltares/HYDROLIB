"""Interactive folium maps of a D-Flow FM network."""

import base64
import io

import branca.colormap as bcm
import folium
import geopandas as gpd
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import shapely
from branca.element import MacroElement
from jinja2 import Template
from matplotlib.figure import Figure
from shapely.geometry import LineString, MultiLineString

from hydrolib.core.dflowfm.net.models import Network


# --- helper for point layers ---
def point_layer(
    m: folium.Map,
    gdf: gpd.GeoDataFrame,
    name: str,
    color: str,
    symbol: str,
    size: int = 14,
    use_centroid: bool = False,
) -> None:
    """Add points to a folium map as a layer of text symbols.

    Parameters
    ----------
    m : folium.Map
        Map to add the layer to.
    gdf : gpd.GeoDataFrame
        Geometries to show, in EPSG:28992 (RD New).
    name : str
        Layer name, also shown as tooltip.
    color : str
        CSS color of the symbols.
    symbol : str
        Text or HTML entity to show at each point, e.g. ``"&#9650;"``.
    size : int, optional
        Font size of the symbols in pixels. Defaults to 14.
    use_centroid : bool, optional
        Show the centroid of each geometry, e.g. for lines or polygons. Defaults to False.
    """
    # One GeoJson layer with a shared marker template instead of a folium.Marker per point, keeps the HTML small.
    geoms = gdf.geometry.set_crs(28992)
    if use_centroid:
        geoms = geoms.centroid
    pts = shapely.set_precision(geoms.to_crs(4326).values, 1e-6, mode="pointwise")
    folium.GeoJson(
        gpd.GeoDataFrame(geometry=pts, crs=4326),
        name=name,
        marker=folium.Marker(
            icon=folium.DivIcon(
                html=f'<span style="font-size:{size}px;color:{color};line-height:1;">{symbol}</span>',
                icon_size=(size, size),
                icon_anchor=(size // 2, size // 2),
            )
        ),
        tooltip=name,
    ).add_to(m)


class TitleControl(MacroElement):
    """Title in the top right corner of a folium map.

    Parameters
    ----------
    title : str
        Text of the title. It is inserted as HTML.
    """

    def __init__(self, title: str) -> None:
        super().__init__()
        self.title = title
        self._template = Template("""
            {% macro script(this, kwargs) %}
            var title_{{this.get_name()}} = L.control({position: 'topright'});
            title_{{this.get_name()}}.onAdd = function(map) {
                var d = L.DomUtil.create('div');
                d.style.cssText = 'background:white;padding:4px 10px;border-radius:4px;font-size:14px;margin:10px;';
                d.innerHTML = '{{ this.title }}';
                return d;
            };
            title_{{this.get_name()}}.addTo({{this._parent.get_name()}});
            {% endmacro %}
        """)


def plot_network(
    network: Network,
    crs: int = 28992,
    mesh1d_kwargs: dict | None = None,
    mesh2d_kwargs: dict | None = None,
    links1d2d_kwargs: dict | None = None,
    face_z_kwargs: dict | None = None,
    background: bool = True,
) -> folium.Map:
    """Show the 1D mesh, the 2D mesh, the 1D2D links and the 2D face levels on a folium map.

    Each part is a separate layer; empty parts are left out. The map zooms to the
    extent of all layers.

    Parameters
    ----------
    network : Network
        Network to show.
    crs : int, optional
        EPSG code of the network coordinates. Defaults to 28992 (RD New).
    mesh1d_kwargs, mesh2d_kwargs, links1d2d_kwargs : dict, optional
        Line style of each layer, with the keys ``color`` and ``weight`` (``lw`` is
        accepted for ``weight``). Defaults to red, blue and black lines.
    face_z_kwargs : dict, optional
        Show the face levels of the 2D mesh as colored dots, with the keys ``vmin``,
        ``vmax``, ``cmap``, ``label`` (legend caption) and ``radius`` (in points).
        Missing keys get defaults; ``vmin`` and ``vmax`` default to the range of the
        levels. Defaults to None: no face levels.
    background : bool, optional
        Show OpenStreetMap as background. Defaults to True.

    Returns
    -------
    folium.Map
        The map.
    """
    if mesh1d_kwargs is None:
        mesh1d_kwargs = {"color": "#d62728", "weight": 1}
    if mesh2d_kwargs is None:
        mesh2d_kwargs = {"color": "#1f77b4", "weight": 0.5}
    if links1d2d_kwargs is None:
        links1d2d_kwargs = {"color": "black", "weight": 1}

    for d in [mesh1d_kwargs, mesh2d_kwargs, links1d2d_kwargs]:
        if "lw" in d and "weight" not in d:
            d["weight"] = d.pop("lw")
    # prefer_canvas: draw vectors on a canvas instead of one SVG element per edge, much faster for large meshes
    if background:
        m = folium.Map(tiles="OpenStreetMap", prefer_canvas=True)
    else:
        m = folium.Map(location=[0, 0], zoom_start=5, tiles=None, prefer_canvas=True)
    all_bounds = []
    nodes1d = faces2d = None

    def edges_to_lines(
        nodes: np.ndarray, edge_nodes: np.ndarray
    ) -> LineString | np.ndarray:
        coords = nodes[edge_nodes]
        return shapely.linestrings(
            coords.reshape(-1, 2),
            indices=np.repeat(np.arange(len(edge_nodes)), 2),
        )

    def add_lines(
        geoms: LineString | np.ndarray, name: str, color: str, weight: float
    ) -> None:
        # One MultiLineString instead of a feature per edge, with coordinates rounded to 1e-6 degrees (~0.1 m).
        # This keeps the HTML small enough to render a 2D mesh in a notebook.
        lines = gpd.GeoSeries(geoms, crs=crs).to_crs(4326)
        merged = shapely.multilinestrings(lines.to_numpy())
        if not isinstance(merged, MultiLineString):
            raise TypeError(f"Expected one MultiLineString, got {type(merged)}")
        merged = shapely.set_precision(merged, 1e-6, mode="pointwise")
        folium.GeoJson(
            gpd.GeoDataFrame(geometry=[merged], crs=4326),
            name=name,
            style_function=lambda _, c=color, w=weight: {"color": c, "weight": w},
        ).add_to(m)
        all_bounds.append(lines.total_bounds)

    if not network._mesh1d.is_empty():
        nodes1d = np.stack(
            [network._mesh1d.mesh1d_node_x, network._mesh1d.mesh1d_node_y], axis=1
        )
        add_lines(
            edges_to_lines(nodes1d, network._mesh1d.mesh1d_edge_nodes),
            "Mesh 1D",
            mesh1d_kwargs["color"],
            mesh1d_kwargs["weight"],
        )

    if not network._mesh2d.is_empty():
        nodes2d = np.stack(
            [network._mesh2d.mesh2d_node_x, network._mesh2d.mesh2d_node_y], axis=1
        )
        add_lines(
            # TODO: fix in hydrolib-core, mesh2d_edge_nodes is annotated np.ndarray[int, int] (like #1314):
            # https://github.com/Deltares/HYDROLIB-core/issues/1314
            edges_to_lines(nodes2d, network._mesh2d.mesh2d_edge_nodes),  # type: ignore[arg-type]
            "Mesh 2D",
            mesh2d_kwargs["color"],
            mesh2d_kwargs["weight"],
        )
        faces2d = np.stack(
            [network._mesh2d.mesh2d_face_x, network._mesh2d.mesh2d_face_y], axis=1
        )

    if not network._link1d2d.is_empty() and nodes1d is not None and faces2d is not None:
        link_idx = network._link1d2d.link1d2d
        coords = np.stack([nodes1d[link_idx[:, 0]], faces2d[link_idx[:, 1]]], axis=1)
        add_lines(
            shapely.linestrings(
                coords.reshape(-1, 2), indices=np.repeat(np.arange(len(link_idx)), 2)
            ),
            "Links 1D-2D",
            links1d2d_kwargs["color"],
            links1d2d_kwargs["weight"],
        )

    if face_z_kwargs is not None and not network._mesh2d.is_empty():
        face_z_kwargs = {
            "vmin": None,
            "vmax": None,
            "cmap": "viridis",
            "label": "Face level [m+NAP]",
            "radius": 3,
            **face_z_kwargs,
        }

        face_x = network._mesh2d.mesh2d_face_x
        face_y = network._mesh2d.mesh2d_face_y
        face_z = network._mesh2d.mesh2d_face_z

        vmin = (
            face_z_kwargs["vmin"]
            if face_z_kwargs["vmin"] is not None
            else float(face_z.min())
        )
        vmax = (
            face_z_kwargs["vmax"]
            if face_z_kwargs["vmax"] is not None
            else float(face_z.max())
        )

        # reproject face points
        pts = shapely.points(np.stack([face_x, face_y], axis=1))
        gdf_z = gpd.GeoDataFrame({"z": face_z}, geometry=pts, crs=crs).to_crs(4326)
        xs, ys = gdf_z.geometry.x.to_numpy(), gdf_z.geometry.y.to_numpy()
        lon0, lat0, lon1, lat1 = xs.min(), ys.min(), xs.max(), ys.max()

        # render scatter to transparent PNG
        fig = Figure(figsize=(10, 10))
        ax = fig.add_axes(
            (0, 0, 1, 1)
        )  # axes fills entire figure — ensures bounds align exactly
        ax.scatter(
            xs,
            ys,
            c=face_z,
            cmap=face_z_kwargs["cmap"],
            vmin=vmin,
            vmax=vmax,
            s=face_z_kwargs["radius"] ** 2,
            linewidths=0,
        )
        ax.set_xlim(lon0, lon1)
        ax.set_ylim(lat0, lat1)
        ax.set_axis_off()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", transparent=True, dpi=150)
        buf.seek(0)

        folium.raster_layers.ImageOverlay(
            image=f"data:image/png;base64,{base64.b64encode(buf.read()).decode()}",
            bounds=[[lat0, lon0], [lat1, lon1]],
            name="Face level",
            opacity=0.8,
        ).add_to(m)

        colormap = bcm.LinearColormap(
            colors=[
                mcolors.to_hex(c)
                for c in plt.get_cmap(face_z_kwargs["cmap"])(np.linspace(0, 1, 10))
            ],
            vmin=vmin,
            vmax=vmax,
        )
        colormap.caption = face_z_kwargs["label"]
        colormap.add_to(m)
        all_bounds.append(gdf_z.total_bounds)

    if all_bounds:
        b = np.array(all_bounds)
        m.fit_bounds([[b[:, 1].min(), b[:, 0].min()], [b[:, 3].max(), b[:, 2].max()]])
    return m
