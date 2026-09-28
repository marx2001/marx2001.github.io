
# 可视化
(fig, ax) = fin_model_half.visualize(
    proj_plane=[0, 1], eig_dr=evecs_half[ed, :], draw_hoppings=False
)
ax.set_title("Edge state for finite model periodic in one direction")
ax.set_xlabel("x coordinate")
ax.set_ylabel("y coordinate")
fig.tight_layout()
plt.show()
