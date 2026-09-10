
totalk=`awk '{if(NR==2)print $4}' PROCAR`
totalb=`awk '{if(NR==2)print $8}' PROCAR`
Ef=`awk '{if(NR==6)print $4}' DOSCAR`
l=`awk '/^band/{print $6}' PROCAR |wc -l`
# if ispin = 2 without of soc, we should use the following line, else comment it with # 
l=$((l/2))
awk '/^band/{print $5 - '$Ef'}' PROCAR |head -$l > bnd-up.dat
awk '/^band/{print $5 - '$Ef'}' PROCAR |tail -$l > bnd-dn.dat
echo $totalk $totalb >note-bnd
grep -A3 reciprocal OUTCAR |tail -3 |awk '{print $4,$5,$6}' >>note-bnd
cat k-points >>note-bnd
grep mesh kp.py |head -1 |cut -d= -f 2 >>note-bnd
