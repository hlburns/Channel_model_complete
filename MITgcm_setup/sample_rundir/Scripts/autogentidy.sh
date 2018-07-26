#!/bin/bash

for i in {0..1000..20}
    do 
       val=$(($i + 20))
       val2=$(($val + 20))
       sed -e s/$i'-'$val/$val'-'$val2/g <mtidy.$i >mtidy.$val
    done

