def plot_fitness_history(history, save_path=None):
    plt.figure(figsize=(10, 5))
    generations = [i * 10 for i in range(len(history))]
    plt.plot(generations, history, label="Fitness", color='blue', linewidth=2)
    plt.xlabel("Generación")
    plt.ylabel("Fitness")
    plt.title("Evolución del Fitness a lo largo del tiempo")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, format='png')
    plt.show()
