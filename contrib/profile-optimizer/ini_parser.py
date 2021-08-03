import configparser
import os

class Ini:
    def __init__(self, source, outputfolder):
        self._filename = os.path.basename(source)
        self._config = configparser.ConfigParser(default_section='General')
        self._config.optionxform = str  # Disables transformations to lowercase. This retains source capitalization.
        self._config.read(source)
        self._outputfolder = outputfolder
        # Use config.options('sectionname') to see the available option (keys)
        # Not sure if the default export from configparser is sufficient, as it removes the whitespaces seen in original

    @property
    def config(self):
        return self._config

    def adjust_and_export(self, change_where, change_what, value):
        if type(value) is not str:
            value = str(value)
        self.config.set(section=change_where, option=change_what, value=value)
        with open(os.path.join(self._outputfolder, self._filename), 'w') as f:
            self.config.write(f)


if __name__ == '__main__':
    # fn_source_ini = input("pathname for the source ini file")
    # value = input("what is the new fricion value?")
    # outputfolder = input("to which folder should the ini file be written?")
    #
    # ini = Ini(fn_source_ini, outputfolder)
    # ini.adjust_and_export(value)
    # print(f"Exported the ini file with new friction {value} succesfully to {outputfolder}")

    fn_source_ini = r'src\roughness-Channels.ini'

    ini = Ini(fn_source_ini, 'output')
    ini.config.set(section='Global', option='frictionValue', value='0.02')
    with open('output/test.ini', 'w') as f:
        ini.config.write(f, True)
    print(ini.config)
