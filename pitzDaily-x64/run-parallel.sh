#!/bin/sh
cd "${0%/*}" || exit                                # Run from this directory
. ${WM_PROJECT_DIR:?}/bin/tools/RunFunctions        # Tutorial run functions
#------------------------------------------------------------------------------
restore0Dir

runApplication decomposePar

runParallel $(getApplication)

runApplication reconstructPar

#------------------------------------------------------------------------------
