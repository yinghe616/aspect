#!/bin/bash
#
# This script compiles all plugins and runs all prm files in the subdirectories
# of the benchmarks folder

if [ "$#" -ne 1 ]; then
    echo "usage: $0 aspect-build-directory"
    exit 1
fi

BUILD=`cd $1;pwd`

if [[ ! -e $BUILD/AspectConfig.cmake || ! -e $BUILD/aspect  ]];
then
    echo "'$BUILD' doesn't look like a build directory"
    exit 1
fi

#
run_prm ()
{
prm=$1
    echo "Running '$prm' at `pwd` with '$BUILD' ..."
    cp $prm $prm.tmp
    echo "set End time=0" >> $prm.tmp
    $BUILD/aspect $prm.tmp >/dev/null || { rm -f $prm.tmp; return 2; }
    rm -f $prm.tmp
}

# run aspect on all .prm files in the current folder or any subdirectory
run_all_prms ()
{
    for prm in `find . -name "*prm"`;
    do
    if [ "`basename $prm`" = "parameters.prm" ];
    then
	continue;
    fi
    echo "Running '$prm' at `pwd` with '$BUILD' ..."
    cp $prm $prm.tmp
    echo "set End time=0" >> $prm.tmp
    $BUILD/aspect $prm.tmp >/dev/null || { rm -f $prm.tmp; return 2; }
    rm -f $prm.tmp
    done
    echo "... completed `pwd`"
    return 0;
}

# configure and compile the plugin in the current directory
make_lib ()
{
    echo "configuring in `pwd` ..."
    rm -rf CMakeCache.txt
    cmake -D Aspect_DIR=$BUILD . >/dev/null || { echo "cmake failed!"; return 1; }
    make >/dev/null || { echo "make failed!"; return 2; }
    return 0;
}


echo "Checking benchmarks using $BUILD/aspect"
echo "Please be patient..."

( (cd blankenbach/plugin; make_lib && cd .. && run_all_prms ) || { echo "FAILED"; exit 1; } ) &

( (cd burstedde; make_lib && run_all_prms ) || { echo "FAILED"; exit 1; } ) &

( (cd crameri_et_al/case_1 && make_lib && run_all_prms ) || { echo "FAILED"; exit 1; } ) &

( (cd crameri_et_al/case_2 && run_all_prms ) || { echo "FAILED"; exit 1; } ) &

( (cd davies_et_al; cd case-2.3-plugin; make_lib && cd .. && run_all_prms ) || { echo "FAILED"; exit 1; } ) &

( (cd finite_strain && make_lib && run_all_prms ) || { echo "FAILED"; exit 1; } ) &

( (cd inclusion; make_lib && run_all_prms ) || { echo "FAILED"; exit 1; } ) &

( (cd king2dcompressible/ala && make_lib && run_prm "ala.prm" && cat ala.prm tala.prm >mytala.prm && run_prm "mytala.prm" && rm mytala.prm ) || { echo "FAILED"; exit 1; } ) &

( (cd shear_bands; make_lib && run_all_prms ) || { echo "FAILED"; exit 1; } ) &

( (cd solcx; make_lib && run_all_prms ) || { echo "FAILED"; exit 1; } ) &

( (cd solcx/compositional_fields; make_lib && run_all_prms ) || { echo "FAILED"; exit 1; } ) &

( (cd solitary_wave; make_lib && run_all_prms ) || { echo "FAILED"; exit 1; } ) &

( (cd solkz; make_lib && run_all_prms ) || { echo "FAILED"; exit 1; } ) &

( (cd tangurnis; cd code; make_lib && cd .. && run_all_prms ) || { echo "FAILED"; exit 1; } ) &

wait

exit 0
