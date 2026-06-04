"""
Entry point for the BO+CDU benchmark suite.

Run with:
    python main.py

Outputs:
    figures/01..05_*.png
    results/01..06_*.csv
"""
from bo_cdu import (run_experiments, save_csv_results,
                    generate_figures, print_summary)


def main():
    main_results, stat_results = run_experiments()
    save_csv_results(main_results, stat_results)
    generate_figures(main_results, stat_results)
    print_summary(main_results, stat_results)


if __name__ == "__main__":
    main()
