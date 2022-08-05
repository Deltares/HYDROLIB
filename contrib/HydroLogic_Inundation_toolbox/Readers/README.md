# Readers
readers for D-HYDRO model output

## flowmesreader usage example:
	see "../Examples".
	

## Hisreader usage example:

    from hisreader import HisResults

    output_path = "...\dflowfm\output"

    Results = HisResults(inputdir=output_path, outputdir=output_path)
    Results.<object_name>.simulated_plot("waterlevel")


