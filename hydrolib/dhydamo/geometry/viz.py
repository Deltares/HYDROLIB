import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
import folium
from hydrolib.core.dflowfm.net.models import Network
import branca.colormap as bcm
import matplotlib.colors as mcolors
import io, base64
import shapely
import geopandas as gpd
from matplotlib.figure import Figure

# --- helper for point layers ---
def point_layer(gdf, name, color, symbol, size=14, use_centroid=False):
    fg = folium.FeatureGroup(name=name)
    for _, row in gdf.geometry.set_crs(28992).to_crs(4326).items():
        pt = row.centroid if use_centroid else row
        folium.Marker(
            location=[pt.y, pt.x],
            icon=folium.DivIcon(
                html=f'<span style="font-size:{size}px;color:{color};line-height:1;">{symbol}</span>',
                icon_size=(size, size),
                icon_anchor=(size // 2, size // 2),
            ),
            tooltip=name,
        ).add_to(fg)
    fg.add_to(m)

def plot_network_folium(
    network,
    crs: int = 28992,
    mesh1d_kwargs: dict = None,
    mesh2d_kwargs: dict = None,
    links1d2d_kwargs: dict = None,
    face_z_kwargs: dict = None,
) -> folium.Map:
    if mesh1d_kwargs is None:
        mesh1d_kwargs = {"color": "#d62728", "weight": 1}
    if mesh2d_kwargs is None:
        mesh2d_kwargs = {"color": "#1f77b4", "weight": 0.5}
    if links1d2d_kwargs is None:
        links1d2d_kwargs = {"color": "black", "weight": 1}

    for d in [mesh1d_kwargs, mesh2d_kwargs, links1d2d_kwargs]:
        if "lw" in d and "weight" not in d:
            d["weight"] = d.pop("lw")

    m = folium.Map(tiles="OpenStreetMap")
    all_bounds = []
    nodes1d = faces2d = None

    def edges_to_lines(nodes, edge_nodes):
        coords = nodes[edge_nodes]
        return shapely.linestrings(
            coords.reshape(-1, 2),
            indices=np.repeat(np.arange(len(edge_nodes)), 2),
        )

    def add_lines(geoms, name, color, weight):
        gdf = gpd.GeoDataFrame(geometry=geoms, crs=crs).to_crs(4326)
        folium.GeoJson(
            gdf, name=name,
            style_function=lambda _, c=color, w=weight: {"color": c, "weight": w},
        ).add_to(m)
        all_bounds.append(gdf.total_bounds)

    if not network._mesh1d.is_empty():
        nodes1d = np.stack([network._mesh1d.mesh1d_node_x, network._mesh1d.mesh1d_node_y], axis=1)
        add_lines(
            edges_to_lines(nodes1d, network._mesh1d.mesh1d_edge_nodes),
            "Mesh 1D", mesh1d_kwargs["color"], mesh1d_kwargs["weight"],
        )

    if not network._mesh2d.is_empty():
        nodes2d = np.stack([network._mesh2d.mesh2d_node_x, network._mesh2d.mesh2d_node_y], axis=1)
        add_lines(
            edges_to_lines(nodes2d, network._mesh2d.mesh2d_edge_nodes),
            "Mesh 2D", mesh2d_kwargs["color"], mesh2d_kwargs["weight"],
        )
        faces2d = np.stack([network._mesh2d.mesh2d_face_x, network._mesh2d.mesh2d_face_y], axis=1)

    if not network._link1d2d.is_empty() and nodes1d is not None and faces2d is not None:
        link_idx = network._link1d2d.link1d2d
        coords = np.stack([nodes1d[link_idx[:, 0]], faces2d[link_idx[:, 1]]], axis=1)
        add_lines(
            shapely.linestrings(coords.reshape(-1, 2), indices=np.repeat(np.arange(len(link_idx)), 2)),
            "Links 1D-2D", links1d2d_kwargs["color"], links1d2d_kwargs["weight"],
        )

    if face_z_kwargs is not None and not network._mesh2d.is_empty():
        face_z_kwargs = {"vmin": None, "vmax": None, "cmap": "viridis",
                         "label": "Face level [m+NAP]", "radius": 3, **face_z_kwargs}

        face_x = network._mesh2d.mesh2d_face_x
        face_y = network._mesh2d.mesh2d_face_y
        face_z = network._mesh2d.mesh2d_face_z

        vmin = face_z_kwargs["vmin"] if face_z_kwargs["vmin"] is not None else float(face_z.min())
        vmax = face_z_kwargs["vmax"] if face_z_kwargs["vmax"] is not None else float(face_z.max())

        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        cmap = plt.get_cmap(face_z_kwargs["cmap"])
        hex_colors = [mcolors.to_hex(c) for c in cmap(norm(face_z))]

        pts = shapely.points(np.stack([face_x, face_y], axis=1))
        gdf_z = gpd.GeoDataFrame({"color": hex_colors}, geometry=pts, crs=crs).to_crs(4326)

        # reproject face points
        pts = shapely.points(np.stack([face_x, face_y], axis=1))
        gdf_z = gpd.GeoDataFrame({"z": face_z}, geometry=pts, crs=crs).to_crs(4326)
        xs, ys = gdf_z.geometry.x.values, gdf_z.geometry.y.values
        lon0, lat0, lon1, lat1 = xs.min(), ys.min(), xs.max(), ys.max()
        
        # render scatter to transparent PNG
        fig = Figure(figsize=(10, 10))
        ax = fig.add_axes([0, 0, 1, 1])  # axes fills entire figure — ensures bounds align exactly
        ax.scatter(xs, ys, c=face_z, cmap=face_z_kwargs["cmap"],
                   vmin=vmin, vmax=vmax, s=face_z_kwargs["radius"] ** 2, linewidths=0)
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
            colors=[mcolors.to_hex(c) for c in plt.get_cmap(face_z_kwargs["cmap"])(np.linspace(0, 1, 10))],
            vmin=vmin,
            vmax=vmax,
        )
        colormap.caption = face_z_kwargs["label"]
        colormap.add_to(m)
        all_bounds.append(gdf_z.total_bounds)

    if all_bounds:
        b = np.array(all_bounds)
        m.fit_bounds([[b[:, 1].min(), b[:, 0].min()], [b[:, 3].max(), b[:, 2].max()]])

    folium.LayerControl().add_to(m)
    return m

def plot_network(
    network: Network,
    ax=None,
    mesh1d_kwargs: dict = None,
    mesh2d_kwargs: dict = None,
    links1d2d_kwargs: dict = None,
) -> None:
    if ax is None:
        fig, ax = plt.subplots()
        autoscale = True
    else:
        autoscale = True

    if mesh1d_kwargs is None:
        mesh1d_kwargs = {"color": "C3", "lw": 1.0}
    if mesh2d_kwargs is None:
        mesh2d_kwargs = {"color": "C0", "lw": 0.5}
    if links1d2d_kwargs is None:
        links1d2d_kwargs = {"color": "k", "lw": 1.0}

    # Mesh 1d
    if not network._mesh1d.is_empty():
        nodes1d = np.stack(
            [network._mesh1d.mesh1d_node_x, network._mesh1d.mesh1d_node_y], axis=1
        )
        edge_nodes = network._mesh1d.mesh1d_edge_nodes
        lc_mesh1d = LineCollection(nodes1d[edge_nodes], **mesh1d_kwargs)
        ax.add_collection(lc_mesh1d)

    # Mesh 2d
    if not network._mesh2d.is_empty():
        nodes2d = np.stack(
            [network._mesh2d.mesh2d_node_x, network._mesh2d.mesh2d_node_y], axis=1
        )
        edge_nodes = network._mesh2d.mesh2d_edge_nodes
        lc_mesh2d = LineCollection(nodes2d[edge_nodes], **mesh2d_kwargs)
        ax.add_collection(lc_mesh2d)

    # Links
    if not network._link1d2d.is_empty():
        faces2d = np.stack(
            [network._mesh2d.mesh2d_face_x, network._mesh2d.mesh2d_face_y], axis=1
        )
        link_coords = np.stack(
            [
                nodes1d[network._link1d2d.link1d2d[:, 0]],
                faces2d[network._link1d2d.link1d2d[:, 1]],
            ],
            axis=1,
        )
        lc_link1d2d = LineCollection(link_coords, **links1d2d_kwargs)
        ax.add_collection(lc_link1d2d)

    if autoscale:
        ax.autoscale_view()
