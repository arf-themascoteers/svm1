import numpy as np
import matplotlib.pyplot as plt

normal = np.array([3.0, 4.0])
bias = -2.0

test_points = np.array([
    [2.0, 1.0],
    [-1.0, 3.0],
    [0.0, 0.5],
    [1.0, -1.0],
    [-2.0, -1.0],
])

def signed_distance(point, normal, bias):
    return (normal @ point + bias) / np.linalg.norm(normal)

fig, ax = plt.subplots(figsize=(7, 7))

xs = np.linspace(-4, 4, 400)
hyperplane_ys = -(normal[0] * xs + bias) / normal[1]
ax.plot(xs, hyperplane_ys, "k-", linewidth=2, label="w'x + b = 0")

origin_on_line = np.array([0.0, -bias / normal[1]])
unit_normal = normal / np.linalg.norm(normal)
ax.annotate("", xy=origin_on_line + unit_normal * 1.5, xytext=origin_on_line,
            arrowprops=dict(arrowstyle="->", color="red", lw=2))
ax.text(origin_on_line[0] + unit_normal[0] * 1.7, origin_on_line[1] + unit_normal[1] * 1.7,
        "w (normal)", color="red", fontsize=11)

for point in test_points:
    dist = signed_distance(point, normal, bias)
    color = "blue" if dist > 0 else "orange"
    ax.plot(point[0], point[1], "o", color=color, markersize=10)
    ax.annotate(f"d={dist:.2f}", xy=point, xytext=(point[0] + 0.15, point[1] + 0.25), fontsize=9)

    foot = point - dist * unit_normal
    ax.plot([point[0], foot[0]], [point[1], foot[1]], "--", color="gray", linewidth=0.8)

ax.set_xlim(-4, 4)
ax.set_ylim(-3, 5)
ax.set_aspect("equal")
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_title("Hyperplane w'x + b = 0, normal vector, signed distances")
ax.legend()
fig.savefig("c1.png", dpi=150, bbox_inches="tight")
print("saved c1.png")
plt.show()
