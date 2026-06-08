from __future__ import annotations

from hydrolib.dhydamo.io.damo_converters.base import BaseDamoConverter, LayerSpec


class Damo22Converter(BaseDamoConverter):
    """DAMO v2.2 converter.

    Defines the canonical layer-to-attribute mapping used as the baseline for
    all other version converters. Post-processing copies soortregelmiddel to
    soortafsluitmiddel on management_device to normalise the column name.
    """

    hydamo_version = "2.2"
    layer_specs = (
        LayerSpec("branches", "HydroObject", index_col="code"),
        LayerSpec(
            "profile",
            "ProfielPunt",
            groupby_column="profiellijnid",
            order_column="codevolgnummer",
        ),
        LayerSpec("profile_roughness", "RuwheidProfiel"),
        LayerSpec("profile_line", "profiellijn"),
        LayerSpec("profile_group", "profielgroep"),
        LayerSpec("param_profile", "NormgeparamProfiel"),
        LayerSpec("param_profile_values", "NormgeparamProfielWaarde"),
        LayerSpec("weirs", "Stuw"),
        LayerSpec("opening", "Kunstwerkopening"),
        LayerSpec("closing_device", "Afsluitmiddel"),
        LayerSpec("management_device", "Regelmiddel"),
        LayerSpec("bridges", "Brug", index_col="code"),
        LayerSpec("culverts", "DuikerSifonHevel", index_col="code"),
        LayerSpec("pumpstations", "Gemaal", index_col="code"),
        LayerSpec("pumps", "Pomp", index_col="code"),
        LayerSpec("management", "Sturing", index_col="code"),
        LayerSpec(
            "boundary_conditions",
            "hydrologischerandvoorwaarde",
            index_col="code",
        ),
        LayerSpec(
            "catchments",
            "afvoergebiedaanvoergebied",
            index_col="code",
            check_geotype=False,
        ),
        LayerSpec("laterals", "lateraleknoop"),
    )

    def postprocess(self, hydamo) -> None:
        if not hydamo.management_device.empty:
            hydamo.management_device = self._copy_column(
                hydamo.management_device, "soortregelmiddel", "soortafsluitmiddel"
            )
