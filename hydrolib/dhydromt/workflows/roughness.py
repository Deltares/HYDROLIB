# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import geopandas as gpd
import shapely
from shapely.geometry import LineString, Point
from scipy.spatial import distance
import configparser
import logging

from hydromt import config
import hydromt.io

logger = logging.getLogger(__name__)


__all__ = ["generate_roughness"]


def generate_roughness(roughness, roughness_ini, logger = logger):
    """ """
    roughness["frictionId"] = roughness.apply(lambda x: '%s_%s' % (x['frictionType'], x['frictionValue']), axis=1)
    roughness_only = roughness[[roughness.index.name, "geometry", "frictionType", "frictionValue"]]

    return roughness_only, roughness




