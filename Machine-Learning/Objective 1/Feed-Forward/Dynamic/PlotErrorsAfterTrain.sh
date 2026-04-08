#!/bin/bash
while [ 1 ]
do
     cnt=`ps x | grep -v grep | grep -c Noise-Filter-NN.py`
     if  [[ "$cnt" -eq 1 ]]
     then
        break
     fi
     sleep 20
done
echo "Data Generation Complete"
python3 PlotAnchorsVsAcc.py
