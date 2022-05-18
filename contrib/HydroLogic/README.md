## Hisreader usage example:

    from hisreader import HisResults

    output_path = "...\dflowfm\output"

    Results = HisResults(inputdir=output_path, outputdir=output_path)
    Results.observation_point_list[0].simulated_plot("waterlevel")
    Results.<object_name>.simulated_plot("waterlevel")


## Purpose
This subdir is aimed at contributions from HydroLogic on various topics.
Consider this as a sandbox environment where related scripts can also
be placed, which can later be cleaned up and promoted into HYDROLIB-core
for example.

## Contact
[@Kreef]( https://github.com/Kreef )
[@marcel-ald]( https://github.com/marcel-ald )
