from pathlib import Path

from src.mgloc.experiment import run_and_visualize


def main():
    """
    Main entry point for the project.
    Runs the experiment and saves the resulting figures.
    """
    print("Running M-GLOC experiment...")
    figures = run_and_visualize()
    print("Experiment finished. Saving figures...")

    # Define the output directory for figures relative to the current script
    figures_dir = Path("figures")
    figures_dir.mkdir(exist_ok=True)

    for name, fig in figures.items():
        file_path = figures_dir / f"{name}.png"
        fig.savefig(file_path)
        print(f"Saved figure to {file_path}")

    print("All figures saved.")


if __name__ == "__main__":
    main()
