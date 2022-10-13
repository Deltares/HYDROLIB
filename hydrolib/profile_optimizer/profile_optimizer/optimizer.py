import glob
from pathlib import Path
from hydrolib.core.io.crosssection.models import CrossDefModel
from hydrolib.core.io.mdu.models import FMModel
from hydrolib.core.io.dimr.models import DIMR, FMComponent
import subprocess
import shutil
import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class ProfileOptimizer():
    def __init__(self, base_model_fn: Path, bat_file, work_dir: Path, output_dir: Path,
                 iteration_name='Iteration', iteration_start_count=1):
        """Framework for iterative cross-section changes and calculations with DHydro

        The profile optimizer is a class that supplies a framework for running iterative DHydro calculations to optimize
        a crosssection profile. During the initialization, the base model is copied to the temporary folder (including
        all files in the parent folder of the base MDU). The base model is read with hydrolib-core and settings are
        saved to the class to create iterations afterwards.

        Args:
            base_model_fn: Path to the MDU path of the base model (Pathlib-Path)
            bat_file: Path to a batch file that runs DIMR (Path or string)
            work_dir: Path to folder that does not yet exist, where the iterations can be saved temporarily (Pathlib-Path)
            output_dir: Path to folder that does not yet exist where the final model is saved (Pathlib-Path)
            iteration_name: Name for the iteration models. Will be used like: "{iteration_name}_12" for example (string)
            iteration_start_count: What should be the first number of the iterations? Default is 1. Can be changed when
                iterations are run in multiple phases and should be continued.

        Functions:
            create_iteration: main function, every use of this function creates a new iteration model.
            run_model: function with which a model can be run using DIMR
            run_latest: applies run_model, on the most recently created iteration.
            export_model: using this function will export an iteration (default: last) to the output_dir.
        """
        self.model_name = base_model_fn.name
        self.source_folder = base_model_fn.parent
        self.iteration_nr = iteration_start_count-1
        self.name = iteration_name
        self.bat_file = bat_file
        self._latest_bat = None
        self.work_dir = Path(work_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        shutil.copytree(self.source_folder, self.work_dir)
        self.base_model = FMModel(self.work_dir/self.model_name)


    def create_iteration(self, prof_ids: list, trapezium_pars: dict):
        """Creates a new model, changing the profiles and saving it in the temporary folder.

        Creates:
        - Iteration folder (number is incremental and counted via class)
        - New crossdef file in iteration folder
        - New MDU in the upper model folder
        - DIMR config file in the iteration folder
        - Batch file in the iteration folder

        Args:
            prof_ids: list of profiles that should be changed
            Let op! Als een profiel-def op meerdere locaties wordt gebruikt, wordt deze overal aangepast.
            trapezium_pars: dict of the new trapezium profile parameters (bottom_width, slope_l, slope_r, depth)
            # Let op! Hier is assymetrisch profiel mogelijk.

        Returns:
            filename to the batch file of this iteration
        """
        cross_def = pd.DataFrame([cs.__dict__ for cs in self.base_model.geometry.crossdeffile.definition])

        to_change_def = cross_def[cross_def['id'].isin(prof_ids)]
        bottom_levels = [min(zcoords) for zcoords in to_change_def['zcoordinates']]

        yz = [self._trapezium_coordinates(bl, **trapezium_pars) for bl in bottom_levels]

        cross_def.loc[to_change_def.index, 'ycoordinates'] = pd.Series([y for y, z in yz], index=to_change_def.index)
        cross_def.loc[to_change_def.index, 'zcoordinates'] = pd.Series([z for y, z in yz], index=to_change_def.index)
        cross_def.loc[to_change_def.index, 'frictionpositions'] = pd.Series([[0, y[-1]] for y, z in yz],
                                                                            index=to_change_def.index)
        cross_def.loc[to_change_def.index, 'yzcount'] = pd.Series([len(y) for y, z in yz], index=to_change_def.index)

        cross_def = cross_def.replace({np.nan: None})
        crossdef_new = CrossDefModel(definition=cross_def.to_dict("records"))

        self.iteration_nr += 1
        iteration_name = f'{self.name}_{self.iteration_nr}'
        iteration_folder = self.work_dir/iteration_name
        iteration_folder.mkdir(parents=True, exist_ok=True)
        # write new crossdef to iteration iteration_folder
        crossdef_name = f'crossdef_{self.iteration_nr}.ini'
        crossdef_new.save(iteration_folder/crossdef_name)
        crossdef_new.filepath = Path(f'{iteration_name}/{crossdef_name}')
        # write new mdu
        mdu_copy = self.base_model.copy()
        mdu_copy.geometry.crossdeffile = crossdef_new
        mdu_copy.filepath = self.work_dir/f'{self.name}_{self.iteration_nr}.mdu'
        mdu_copy.save(recurse=False)
        dimr = DIMR()
        dimr.component.append(FMComponent(name=f'{self.name}_{self.iteration_nr}',
                                          workingDir=iteration_folder.parent.absolute(),
                                          inputfile=mdu_copy.filepath.absolute(),
                                          model=mdu_copy))
        dimr_fn = iteration_folder/'dimr_config.xml'
        dimr.save(dimr_fn)
        with open(dimr_fn, 'r') as f:
            content = f.readlines()
        end_documentation_line = [l for l in content if '</documentation>' in l][0]
        end_documentation_loc = content.index(end_documentation_line)
        spaces = end_documentation_line.split('<')[0]
        content.insert(end_documentation_loc + 1, f'{spaces}</control>\n')
        content.insert(end_documentation_loc + 1, f'{spaces}{spaces}<start name="{self.name}_{self.iteration_nr}" />\n')
        content.insert(end_documentation_loc + 1, f'{spaces}<control>\n')
        with open(dimr_fn, 'w+') as f:
            content_string = ''.join(content)
            f.write(content_string)
        self._latest_bat = dimr_fn.parent/'run.bat'
        shutil.copy(self.bat_file, self._latest_bat)
        return iteration_folder

    @staticmethod
    def _trapezium_coordinates(bottom_level, bottom_width, slope_l, slope_r, depth):
        """General function to create y an z coords for a trapezium profile"""
        slope_width_l = depth * slope_l
        slope_width_r = depth * slope_r
        ycoords = [0, slope_width_r, slope_width_r + bottom_width, slope_width_r + bottom_width + slope_width_l]
        zcoords = [bottom_level + depth, bottom_level, bottom_level, bottom_level + depth]
        return ycoords, zcoords

    @staticmethod
    def run_model(bat_path, model_folder):
        """Runs DIMR for a model of choice

        Args:
            bat_path: Path to the desired bat file that runs DIMR
            model_folder: directory where the model is ran
        """
        print("Begin running model")
        subprocess.call([str(Path(bat_path).absolute())], cwd=str(Path(model_folder).absolute()))
        print("Done running model")

    def run_latest(self):
        """Runs DIMR for the most recently made iteration"""
        if self._latest_bat is not None:
            self.run_model(self._latest_bat, self._latest_bat.parent)
        else:
            raise ValueError(f'No latest run available. '
                             f'Use create_iteration() first, or run another model using run_model().')

    def export_model(self, specific_iteration='latest', cleanup=True):
        """Export a model iteration to the output directory.

        Args:
            specific_iteration: Default: 'latest' will export the last made iteration. Can also be a number-number.
            cleanup: bool, when True, the temp folder will be deleted afterwards.
        """
        if specific_iteration == 'latest':
           iteration = self.iteration_nr
        else:
            if type(specific_iteration) is int:
                iteration = specific_iteration
            else:
                raise TypeError(f"specific_iteration must be an interger or be 'latest'. "
                                f"Input was: {specific_iteration}")

        mdu_fn = self.work_dir/f"{self.name}_{iteration}.mdu"
        mdu = FMModel(mdu_fn)
        new_mdu = self.output_dir/f"{self.model_name.split('.')[0]}_Profile_Optimizer.mdu"
        mdu.save(filepath=new_mdu, recurse=True)
        shutil.copytree(self.work_dir/f"DFM_OUTPUT_{self.name}_{iteration}",
                        self.output_dir/f"DFM_OUTPUT_{self.model_name.strip('.mdu')}_Profile_Optimizer")

        other_files = os.listdir(self.work_dir)
        for filename in other_files:
            file = self.work_dir/filename
            if not filename.endswith('.mdu') and not os.path.isfile(file):
                destination = self.output_dir/Path(file).name
                if not destination.exists():
                    shutil.copy(file, destination)
                    print(f"Copied {destination.name} to destination")

        print(f"Exported iteration {iteration} to output folder as: {new_mdu}")

        if cleanup:
            shutil.rmtree(self.work_dir)
            print(f"Deleted working directory: {self.work_dir}")
            


def find_optimum(window_b, calculated_v_values, target_v, waterlevel):
    """ A function for the optimization of the bottom width of a trapezoidal cross section profile
        for the desired/required flow velocity
    Args:
        window_b: An array of the bottom widths within the optimalisation grid. 
        For each bottom width in this grid the model has been runned to extract the calculated flow velocity.
        
        target_v: desired flow velocity to achieve in the cross section profile (int).
        calculated_v_values: An array of the calculated flow velocities for the bottom widths in the optimalisation grid.
    Returns:
        geoptimaliseerde bodembreedte: The optimalised bottom width for the desired flow velocity.
    """
    if target_v < min(calculated_v_values) or target_v > max(calculated_v_values):
        raise ValueError("Velocity target is not in the range of the calculated velocities. "
                         "Please choose new bottom widths for iterations. /n"
                         f"Target velocity: {target_v}/n"
                         f"Range of calculated velocities: {min(calculated_v_values):.3f} - {max(calculated_v_values):.3f}"
                         f"Range of input bottom widths: {min(window_b):.3f} - {max(window_b):.3f}")

    # collect all the relevant data into a dataframe
    gewenste_u_array = np.ones(len(window_b)) * target_v
    data = {"bodembreedte": window_b, "berekende stroomsnelheid": calculated_v_values, "gewenste stroomsnelheid": gewenste_u_array, "berekende waterstand": waterlevel}
    df = pd.DataFrame(data=data)
    df['difference'] = df['berekende stroomsnelheid'] - df['gewenste stroomsnelheid']
    #print (df)

    # interpolate between the point just above the desired flow_velocity & the point just beneath the desired flow_velocity
    interpolation_point_u_max = df[(df.difference > 0)].sort_values(ascending=True, by='difference').iloc[0][
        'berekende stroomsnelheid']
    interpolation_point_u_min = df[(df.difference < 0)].sort_values(ascending=False, by='difference').iloc[0][
        'berekende stroomsnelheid']
    interpolation_point_width_max = df[(df.difference > 0)].sort_values(ascending=True, by='difference').iloc[0][
        'bodembreedte']
    interpolation_point_width_min = df[(df.difference < 0)].sort_values(ascending=False, by='difference').iloc[0][
        'bodembreedte']


    gewenste_stroomsnelheid = gewenste_u_array[0]
    x = [interpolation_point_width_min, interpolation_point_width_max]
    y = [interpolation_point_u_min, interpolation_point_u_max]
    geoptimaliseerde_bodembreedte = np.interp(gewenste_stroomsnelheid, y, x)

    # plotly figure relatie stroomsnelheid en bodembreedte
    fig = px.scatter(df, x='bodembreedte', y='berekende stroomsnelheid', text="bodembreedte")
    fig.update_traces(mode='markers', marker_line_width=2, marker_size=10)
    
    fig.add_trace(go.Scatter(x=[interpolation_point_width_min, interpolation_point_width_max], y=[interpolation_point_u_min, interpolation_point_u_max], mode='lines', name='geïnterpoleerde relatie', marker_color='blue'))
    fig.add_hline(y=gewenste_stroomsnelheid, line_width=1, line_dash='dash', line_color='black')
    fig.add_hrect(y0=interpolation_point_u_min, y1=interpolation_point_u_max,
                    fillcolor='grey', opacity=0.2, annotation_text='interpolatie gebied')
    fig.add_vline(x=geoptimaliseerde_bodembreedte, line_width=1, line_dash='dash', line_color='black')
    
    fig.add_trace(go.Scatter(x=[geoptimaliseerde_bodembreedte], y=[gewenste_stroomsnelheid], mode='markers', name='geoptimaliseerde bodembreedte', marker_color='green', marker_line_width=2, marker_size=10))
    fig.update_yaxes(title_text="<b>berekende stroomsnelheid (m/s)</b>")
    # Naming x-axis
    fig.update_xaxes(title_text="<b>bodembreedte (m)</b>")
    fig.update_layout(title="<br>Relatie tussen bodembreedte en stroomsnelheid bij het te optimaliseren profiel</b>")
    fig.show()
    
    # plotly figure relatie stroomsnelheid en bodembreedte en waterlevel
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=df['bodembreedte'], y=df['berekende stroomsnelheid'], name='bodembreedte'), secondary_y=False)
    
    fig.update_layout(title="Relatie tussen bodembreedte, stroomsnelheid en waterlevel bij het te optimaliseren profiel")
    fig.add_hline(y=gewenste_stroomsnelheid, line_width=1, line_dash='dash', line_color='black')
    fig.add_hrect(y0=interpolation_point_u_min, y1=interpolation_point_u_max,
                    fillcolor='grey', opacity=0.2, annotation_text='interpolatie gebied')
    fig.add_vline(x=geoptimaliseerde_bodembreedte, line_width=1, line_dash='dash', line_color='black')
    
    fig.add_trace(go.Scatter(x=[geoptimaliseerde_bodembreedte], y=[gewenste_stroomsnelheid], mode='markers', name='geoptimaliseerde bodembreedte', marker_color='green', marker_line_width=2, marker_size=10),secondary_y=False)
    
    #secondary y-axis
    fig.add_trace(go.Scatter(x=df['bodembreedte'], y=df['berekende waterstand'], name='waterstand'), secondary_y=True)
    # Naming x-axis
    fig.update_xaxes(title_text="<b>bodembreedte (m)</b>")
     
    # Naming y-axes
    fig.update_yaxes(title_text="<b>berekende stroomsnelheid (m/s)</b>", secondary_y=False)
    fig.update_yaxes(title_text="<b>berekende waterstand (m)</b>", secondary_y=True)
    fig.show()

    return df, geoptimaliseerde_bodembreedte

if __name__ == "__main__":
    model_folder = Path("playground/model")
    file_source = 'cross_section_definitions.ini'
    prof_ids = ['prof_04082014-DP11']


    prof = {'depth': 2, 'bottom_width': 3, 'slope_l': 2, 'slope_r': 2}

    optimizer = IterateProfile(Path(r'c:\Dev\Hydrolib_optimizer\src\moergestels_broek\moergestels_broek.mdu'),
                               work_dir=r"C:\local\new_work",
                               output_dir=r"C:\local\new_out",
                               bat_file = r'c:\Dev\Hydrolib_optimizer\src\moergestels_broek\run.bat',
                               iteration_name='TEST_py')

    iteration_folder = optimizer.create_iteration(prof_ids, prof)
    print(iteration_folder)

    optimizer.run_latest()

    optimizer.export_model()


    point = (141103, 393599.9)
    u = optimizer.read_result(model_folder, point, 'velocity')

    #TODO: Verzamel outputs zodat we ze kunnen beoordelen

    print(u)
