@ echo off
set OMP_NUM_THREADS=1

call "C:\Program Files\Deltares\D-HYDRO Suite 2022.04 1D2D\plugins\DeltaShell.Dimr\kernels\x64\dimr\scripts\run_dimr.bat"

    rem To prevent the DOS box from disappearing immediately: remove the rem on the following line
pause
