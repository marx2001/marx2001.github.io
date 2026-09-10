#!/bin/bash -x
mkdir result
for i in {0,5,10,15,20,25}
	do
for j in {0,100,200,300,400,500}
	do 
	cp B$i/T$j/*600000.ovf ./result/B$i-T$j-final.ovf
done
done
