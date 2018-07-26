#!/bin/bash

# For loop Start to end , nTimeStep
for i in {0..207360000..691200}
    do 
       val=$(($i + 691200))
       sed -e s/nIter0=$i/nIter0=$val/g <data.$i >data.$val
    done
