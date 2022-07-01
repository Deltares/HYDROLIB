# -*- coding: utf-8 -*-

import configparser
import logging

import geopandas as gpd
import hydromt.io
import numpy as np
import pandas as pd
import shapely
from hydromt import config
from scipy.spatial import distance
from shapely.geometry import LineString, Point

logger = logging.getLogger(__name__)


__all__ = ["generate_roughness"]


def generate_roughness(roughness, roughness_ini, logger=logger):
    """ """
    roughness["frictionId"] = roughness.apply(
        lambda x: "%s_%s" % (x["frictionType"], x["frictionValue"]), axis=1
    )
    roughness_only = roughness[
        [roughness.index.name, "geometry", "frictionType", "frictionValue"]
    ]

    return roughness_only, roughness
