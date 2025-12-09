#!/usr/bin/env python3

"""
Quick Pipeline Verification Test

This script verifies that all required scripts exist and are properly configured
for the full run_all.py pipeline, WITHOUT running time-consuming simulations.

Usage:
    python3 evaluation/test_pipeline.py
"""

from pathlib import Path
import sys

# Color codes
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'
BOLD = '\033[1m'

def print_header(text):
    print()
    print("=" * 80)
    print(f"{BLUE}{BOLD}{text}{RESET}")
    print("=" * 80)
    print()

def print_success(text):
    print(f"{GREEN}✅ {text}{RESET}")

def print_error(text):
    print(f"{RED}❌ {text}{RESET}")

def print_info(text):
    print(f"{BLUE}ℹ️  {text}{RESET}")

def main():
    script_dir = Path(__file__).parent
    base_dir = script_dir.parent

    print_header("🔍 PIPELINE VERIFICATION TEST")

    # Check main execution scripts
    print_info("Checking main execution scripts...")

    required_scripts = {
        'Master Script': 'run_all.py',
        'Graph Generator': 'generate_all_graphs.py',
        'Extension 1 Results': 'generate_extension1_results.py',
        'Extension 2 Results': 'generate_extension2_results.py',
        'Extension 3 Results': 'generate_extension3_results.py',
        'Extension 4 Results': 'generate_extension4_results.py',
    }

    all_exist = True
    for name, script in required_scripts.items():
        script_path = script_dir / script
        if script_path.exists():
            print_success(f"{name}: {script}")
        else:
            print_error(f"{name}: {script} NOT FOUND")
            all_exist = False

    # Check plotting scripts
    print()
    print_info("Checking graph plotting scripts...")

    plotting_scripts = [
        'plot_extension1_analysis.py',
        'plot_extension2_analysis.py',
        'plot_extension3_analysis.py',
        'plot_extension4_analysis.py',
        'generate_extension3_complete_graphs.py',
    ]

    for script in plotting_scripts:
        script_path = script_dir / script
        if script_path.exists():
            print_success(f"{script}")
        else:
            print_error(f"{script} NOT FOUND")
            all_exist = False

    # Check source files
    print()
    print_info("Checking source implementation files...")

    src_dir = base_dir / "src"
    source_files = [
        'pcc_vivace_baseline.py',
        'pcc_vivace_extension1.py',
        'pcc_vivace_extension2.py',
        'pcc_vivace_extension3.py',
        'pcc_vivace_extension4.py',
        'network_simulator.py',
    ]

    for src_file in source_files:
        src_path = src_dir / src_file
        if src_path.exists():
            print_success(f"{src_file}")
        else:
            print_error(f"{src_file} NOT FOUND")
            all_exist = False

    # Check directories
    print()
    print_info("Checking directory structure...")

    results_dir = base_dir / "results"
    plots_dir = base_dir / "plots"

    if results_dir.exists():
        print_success(f"Results directory: {results_dir}")

        # Check result subdirectories
        result_subdirs = [
            'extension1',
            'extensions_1_2',
            'extensions_1_2_3',
            'extensions_1_2_3_4'
        ]

        for subdir in result_subdirs:
            subdir_path = results_dir / subdir
            if subdir_path.exists():
                json_count = len(list(subdir_path.glob('*.json')))
                print_success(f"  {subdir}/: {json_count} result files")
            else:
                print_error(f"  {subdir}/: NOT FOUND")
    else:
        print_error(f"Results directory not found: {results_dir}")
        all_exist = False

    if plots_dir.exists():
        png_count = len(list(plots_dir.glob('*.png')))
        print_success(f"Plots directory: {plots_dir} ({png_count} graphs)")
    else:
        print_error(f"Plots directory not found: {plots_dir}")
        all_exist = False

    # Final summary
    print_header("📋 VERIFICATION SUMMARY")

    if all_exist:
        print("╔" + "=" * 78 + "╗")
        print("║" + f"{GREEN}{BOLD}✅ ALL PIPELINE COMPONENTS VERIFIED{RESET}".center(88) + "║")
        print("╚" + "=" * 78 + "╝\n")

        print(f"{GREEN}The pipeline is ready to run!{RESET}")
        print()
        print("To run the complete pipeline:")
        print(f"  {BLUE}python3 evaluation/run_all.py{RESET}")
        print()
        print("Expected runtime: 15-25 minutes")
        print("Expected output:")
        print("  - 120+ simulation results")
        print("  - 16 graphs")
        print()
        return 0
    else:
        print("╔" + "=" * 78 + "╗")
        print("║" + f"{RED}{BOLD}❌ SOME COMPONENTS MISSING{RESET}".center(88) + "║")
        print("╚" + "=" * 78 + "╝\n")

        print(f"{RED}Please fix the missing components before running the pipeline.{RESET}")
        print()
        return 1

if __name__ == '__main__':
    sys.exit(main())
