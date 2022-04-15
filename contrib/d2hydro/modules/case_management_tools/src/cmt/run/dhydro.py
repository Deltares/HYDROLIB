from pathlib import Path
import argparse
import logging
import os
import shutil
import subprocess
from multiprocessing.pool import ThreadPool
from typing import List, Optional

from ..utils.log import setup_basic_logging, add_logging_arguments

logger = logging.getLogger(__name__)

DIMR_BAT = Path(r"c:/Program Files/Deltares/D-HYDRO Suite 2022.03 1D2D/plugins/DeltaShell.Dimr/kernels/x64/dimr/scripts/run_dimr.bat")

def main():
    args = get_args()
    setup_basic_logging(level=args.loglevel, filename=args.logfile)
    p = run(
        work_dir=Path(args.case_dir),
        dimr_bat=Path(args.dimr_bat),
        num_threads=args.threads,
        stream_output=args.stream_output,
    )
    print(p.stdout)


def run(
    work_dir: Path,
    dimr_bat: Path = DIMR_BAT,
    num_threads: int = 1,
    stream_output: bool = False,
    returncode: bool = True
    ):
    f"""
    Run h2flo flow simulation

    Args:
        work_dir (Path): Path to the work_dir containing the 'dimr_config.xml'
        file
        dimr_bat (Path): Path to the 'run_dimr.bat' as part of the DHYDRO
        installation. Optional, defults to {DIMR_BAT}
        num_threads (int, optional): Number of threads to be used for 1
        model-run. Optional, defaults to 1.
        returncode (bool, optional): if True, the proc returncode will be returned
        stream_output (bool, optional): if True, print stdout line by line
        while process is running. Optional, defaults to True.

    """

    if type(work_dir) == str:
        work_dir = Path(work_dir)
    os.environ["OMP_NUM_THREADS"] = str(num_threads)
    env = os.environ.copy()

    logger.info("Start running DHYDRO")
    input = ""
    proc = subprocess.Popen(
        [dimr_bat.as_posix()],
        cwd=work_dir.as_posix(),
        env=env,
 #       creationflags=subprocess.CREATE_NEW_CONSOLE,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        encoding="ascii",
    )
    if stream_output:
        with proc:
            proc.stdin.write(input)
            proc.stdin.close()
            for line in proc.stdout:
                print(line, end="")
        outs = None
    else:
        outs, _ = proc.communicate(input)
    logger.info("Finished running DHYDRO")

    if returncode:
        return proc.returncode
    else:
        return outs


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DHYDRO")
    parser.add_argument(
        "case_dir",
        help="Path to the work_dir containing the 'dimr_config.xml' file",
    )
    parser.add_argument(
        "dimr_bat",
        help="Path to the 'run_dimr.bat' as part of the DHYDRO installation.",
        default=DIMR_BAT
    )
    parser.add_argument(
        "num_threads",
        help="Input number",
        type=int,
        default=1
    )
    parser.add_argument(
        "--stream_output",
        help="Stream output to stdout",
        type=bool,
        default=False
    )
    add_logging_arguments(parser)
    return parser.parse_args()

if __name__ == "__main__":
    main()
