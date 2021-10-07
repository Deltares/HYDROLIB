import shutil
import os
import fileinput

class OptimalizationRuns:
    def __init__(self, source_dir, new_model_dir):
        self.source_dir = source_dir
        self._new_model_dir = new_model_dir
        self.frictions = [0.019, 0.021, 0.023, 0.025, 0.027, 0.029]
        self.friction_fn = 'roughness-Channels.ini'
        self.caselist = []

    def create_cases(self, new_friction):
        target_dir = os.path.join(self._new_model_dir, str(f'friction_{new_friction}'))
        shutil.copytree(self.source_dir, target_dir, dirs_exist_ok=True)
        self.change_friction(os.path.join(target_dir, self.friction_fn), new_friction)
        self.caselist.append(target_dir)

    def change_friction(self, file_path, new_friction):
        with fileinput.FileInput(file_path, inplace=True) as file:
            for line in file:
                replacement_text = "<time>0 3600 " + str(129) + "</time>"
                print(line.replace("<time>0 300 86400</time>", replacement_text), end='')

if __name__ == '__main__':
    opt = OptimalizationRuns('output', 'output/new')
    opt.change_friction('src/roughness-Channels.ini', 0.025)