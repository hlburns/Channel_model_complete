#!/bin/bash

for i in {0..300}
    do 
       val=$i
       val2=$(($i + 1))
       echo $val
       echo $val2
       sed -e s/next$val/next$val2/g <J$val >J$val2
       sed -i s/J$val/J$val2/g J$val2
       
    done

