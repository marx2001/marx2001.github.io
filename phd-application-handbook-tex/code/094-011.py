
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc 
rc('text',usetex=False)
plt.rc('font', family='DejaVu Sans', size=20)

################
min=-2 
max=1
step = (max - min)/10
fonts = 15
spin = 'dn' #spin = 'up/dn/no' default value: 'up'
###############
orbs=['px','py','pz','z2','x2','xy','yz','xz']
bndfile = 'bnd-up.dat'
figname = "Project_upbnd.pdf"

if spin == 'dn':
   bndfile = 'bnd-dn.dat'
   figname ="Project_dnbnd.pdf"

if spin == 'no':
   bndfile = 'bnd-up.dat'
   figname ="Project_nospinbnd.pdf"

col='cyan'
kp = ['Γ','M','K','Γ','K']
m=open("note-bnd",'r+')
ml=m.readlines()
ml=[ml[i].strip().split() for i in range(len(ml))]
knum=int(ml[0][0])
bnum=int(ml[0][1])

base=[ml[1],ml[2],ml[3]]
base=np.array(base)
dkvec=[]
for k in range(4,len(ml)-1):
 ml[k]=[float(ml[k][j]) for j in range(3)]
for k in range(5,len(ml)-1): 
 dkvec.append((np.array(ml[k])-np.array(ml[k-1])).tolist())
def r(m,n):
 global rfinal
 global r1
 global r2
 global r3
 r1=0
 r2=0
 r3=0
 r1=[float(m[0])*float(n[0][i]) for i in range(3)]
 r2=[float(m[1])*float(n[1][i]) for i in range(3)]
 r3=[float(m[2])*float(n[2][i]) for i in range(3)]
 x=r1[0]+r2[0]+r3[0]
 y=r1[1]+r2[1]+r3[1]
 z=r1[2]+r2[2]+r3[2]
 rfinal=np.sqrt(x**2+y**2+z**2)
sum=0
k=[0 for i in range(len(ml)-6)]
for i in range(len(ml)-6):
 r(dkvec[i],base)
 k[i]=rfinal
 sum=k[i]+sum
mesh=int(ml[len(ml)-1][0])
x=[0]
a0=0
for i in range(len(ml)-6):
 for j in range(mesh):
  a0=a0+k[i]/(mesh+0.0)
  x.append(a0)

plt.figure(figsize=(20,16))

def plt_pro(i,orb,col,lab):
 plt.subplot(2,4,i)
#######up###########
 a = np.loadtxt(bndfile)
# file=open(bndfile,'r+')
# lines=file.readlines()
# a=[float(lines[i].strip()) for i in range(len(lines))]
# a=np.array(a)
 a.resize(knum,bnum)
 b=np.transpose(a)
 plt.ylim(min, max)
 plt.xlim(0,sum)
 plt.plot([0,sum],[0,0],'r-.',linewidth=1 )
 hline=0
 xlab_val = [0]
 for i in range(len(ml)-6):
  hline=hline+k[i]
  xlab_val.append(hline)
 hline=0
 for i in range(len(ml)-7):
  hline=hline+k[i]
  plt.plot([hline,hline],[min,max+step],'b-.', linewidth=1)
 for i in range(bnum):
  plt.plot(x,b[i],'k--',linewidth = 1)
 a2 = np.loadtxt(orb) 
 lens = len(a2)//2
 if spin == 'up':
  a2 = a2[0:lens]
 else:
  a2 = a2[lens:]
 a2.resize(knum,bnum)
 b2=np.transpose(a2)
 for i in range(bnum):
  ax=plt.scatter(x,b[i],c=col, s=b2[i]*300,lw=0, alpha=0.7)
  plt.legend([ax],[lab],frameon= False,fontsize = 20,loc='upper right')
###Fermi-and-high-sym-line####

 plt.ylabel('Energy (eV)')
 
 plt.xticks(xlab_val, kp)
 plt.tight_layout()
 plt.yticks(np.arange(min,max+step,step))
 ax = plt.gca()
 plt.subplots_adjust(left=0.15, right=0.9, top=0.9, bottom=0.1)
plt_pro(1,orbs[0],"gray",r'$p_x$')
plt_pro(2,orbs[1],"gray",r'$p_y$')
plt_pro(3,orbs[2],"gray",r'$p_z$')
plt_pro(4,orbs[3],"cyan",r'$d_{z^2}$')
plt_pro(5,orbs[4],"red",r'$d_{x^2-y^2}$')
plt_pro(6,orbs[5],"red",r'$d_{xy}$')
plt_pro(7,orbs[6],"blue",r'$d_{yz}$')
plt_pro(8,orbs[7],"blue",r'$d_{xz}$')
plt.subplots_adjust(wspace =0.25, hspace =0.1)
plt.savefig(figname,format='pdf', bbox_inches='tight', dpi=300)
