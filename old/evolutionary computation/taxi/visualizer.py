import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from VRParser import VRParser

def get_node_coords(parser):
    coords = []
    if hasattr(parser, 'node_coords') and parser.node_coords:
        coords = parser.node_coords
    else:
        import math
        n = parser.dimension
        r = 50
        for i in range(n):
            angle = 2 * math.pi * i / n
            coords.append((r * math.cos(angle), r * math.sin(angle)))
    return coords
def animate_routes(routes, parser, output_file="routes_animation.gif", last_frame_file="last_frame.png"):
    coords = get_node_coords(parser)
    depot_idx = parser.depot - 1 if parser.depot else 0

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = plt.cm.get_cmap('tab20', len(routes))

    xs, ys = zip(*coords)
    ax.scatter(xs, ys, c='gray', s=40, label='Nodes')

    depot_x, depot_y = coords[depot_idx]
    ax.scatter([depot_x], [depot_y], c='red', s=100, marker='*', label='Depot')

    for i, (x, y) in enumerate(coords):
        ax.text(x, y, str(i), fontsize=8, ha='right', va='bottom')

    ax.set_title("Best Solution Routes")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend()
    ax.set_xlim(min(xs) - 10, max(xs) + 10)
    ax.set_ylim(min(ys) - 10, max(ys) + 10)

    lines = []
    for idx in range(len(routes)):
        line, = ax.plot([], [], marker='o', color=colors(idx), label=f'Route {idx+1}')
        lines.append(line)

    def init():
        for line in lines:
            line.set_data([], [])
        return lines

    def update(frame):
        for idx, route in enumerate(routes):
            if frame < len(route) - 1:
                route_xs = [coords[route[i]][0] for i in range(frame + 1)]
                route_ys = [coords[route[i]][1] for i in range(frame + 1)]
                lines[idx].set_data(route_xs, route_ys)
        return lines

    max_frames = max(len(route) for route in routes)
    ani = FuncAnimation(fig, update, frames=max_frames, init_func=init, blit=True, repeat=False)

    ani.save(output_file, writer='pillow', fps=2)

    update(max_frames - 1)  # Update to the last frame
    plt.savefig(last_frame_file)  # Save the last frame as an image
    plt.close(fig)


def plot_fitness_evolution(fitness_values, output_file="fitness_evolution.png"):
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(fitness_values)), fitness_values, marker='o', label='Fitness')
    plt.yscale('log')  # Set y-axis to log scale
    plt.title("Fitness Evolution")
    plt.xlabel("Generation")
    plt.ylabel("Fitness (log scale)")
    plt.legend()
    plt.grid(True, which="both", linestyle='--', linewidth=0.5)
    plt.savefig(output_file)
    plt.close()

if __name__ == "__main__":
    parser = VRParser('A045-03f.dat')

    routes = [
        [44, 39, 35, 31, 30, 29, 22, 15, 17, 14, 19, 21, 0, 1, 2, 44],
        [44, 4, 5, 20, 7, 9, 8, 40, 38, 44],
        [44, 3, 41, 6, 10, 43, 11, 12, 13, 16, 18, 42, 23, 24, 25, 26, 27, 28, 32, 33, 34, 36, 37, 44]
    ]

    fitness_values = [1000, 800, 600, 400, 300, 250, 200, 180, 150, 120]
    parser.display_info()
    animate_routes(routes, parser, output_file="routes_animation.gif")
    plot_fitness_evolution(fitness_values, output_file="fitness_evolution.png")