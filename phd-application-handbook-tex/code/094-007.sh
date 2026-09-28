
#ion      s     py     pz     px    dxy    dyz    dz2    dxz  x2-y2    tot
#1        2     3      4      5     6      7      8      9    10
grep ^tot PROCAR|awk '{print $3}' >py
grep ^tot PROCAR|awk '{print $4}' >pz
grep ^tot PROCAR|awk '{print $5}' >px
grep ^tot PROCAR|awk '{print $6}' >xy
grep ^tot PROCAR|awk '{print $7}' >yz
grep ^tot PROCAR|awk '{print $8}' >z2
grep ^tot PROCAR|awk '{print $9}' >xz
grep ^tot PROCAR|awk '{print $10}' >x2
