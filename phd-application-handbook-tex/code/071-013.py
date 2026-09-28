
# v1.8 code
path=[[0.,0.],[2./3.,1./3.],[.5,.5],[1./3.,2./3.], [0.,0.]]
label=(r'$\Gamma $',r'$K$', r'$M$', r'$K^\prime$', r'$\Gamma $')

(k_vec,k_dist,k_node) = my_model.k_path(path,101)
evals = my_model.solve_all(k_vec)

fig, ax = plt.subplots()
ax.set_xlim(k_node[0],k_node[-1])
ax.set_xticks(k_node)
ax.set_xticklabels(label)
for n in range(len(k_node)):
  ax.axvline(x=k_node[n],linewidth=0.5, color='k')
ax.set_ylabel("Band energy")

ax.plot(k_dist,evals[0])
ax.plot(k_dist,evals[1])
