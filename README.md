## dependencies
Eigen
PCL
Nlopt

## run
`cd TATOS/build`
`rm -rf *` (opt)
`../build_and_run.sh`

PCD Results are in the ./results/

## eval 
put the PCD of ./results into ./results/road_easy/ and ./results/road_hard/
`python eval.py`