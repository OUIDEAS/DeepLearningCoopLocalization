#!/bin/bash
while [ 1 ]
do
     cnt=`ps x | grep -v grep | grep -c Noise-Filter-NN.py`
     if  [[ "$cnt" -eq 0 ]]
     then
        break
     fi
     sleep 1
done
echo "Data Generation Complete"
sleep 15
python3 Noise-Filter-NN.py
