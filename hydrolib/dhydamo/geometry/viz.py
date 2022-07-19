import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from hydrolib.core.io.net.models import Network


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
        autoscale = False

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
