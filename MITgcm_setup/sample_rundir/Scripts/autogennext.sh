#!/bin/bash

for i in {1..100..1}
    do 
    # new file number
    fnamenew=$(($i+1))
    # start mtidy val
    tidyval=$(($i-1))
    tidyval=$(($tidyval*20))
    # start year end 
    yearend1=$(($tidyval+20))
    # new year end
    yearend2=$(($tidyval+20+20))
    # start timestep
    timestep1=$(($tidyval*69120))
    # new time step
    timestep2=$(($yearend1*69120))
    # sed -i 's/orignal_string/replacement_string/g' file
    sed -e s/*$timestep1/*$timestep2/g <next$i >next$fnamenew
    sed -i s/data.$timestep1/data.$timestep2/g next$fnamenew
    sed -i s/-$yearend1/-$yearend2/g next$fnamenew
    sed -i s/$tidyval-/$yearend1-/g next$fnamenew
    sed -i s/mtidy.$tidyval/mtidy.$yearend1/g next$fnamenew   
    done

