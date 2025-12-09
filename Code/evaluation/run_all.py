#!/usr/bin/env python3

"""
MASTER EXECUTION SCRIPT - Run All Extensions and Generate Graphs

This script orchestrates the complete workflow:
1. Generate Extension 1 results
2. Generate Extension 2 results
3. Generate Extension 3 results
4. Generate Extension 4 results
5. Generate all publication-ready graphs

Usage:
    python3 evaluation/run_all.py

Output:
    - Results: results/ (120+ simulations across 4 extension sets)
    - Graphs: plots/ (16 PNG visualizations)

Total Runtime: ~15-25 minutes depending on hardware
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime
import time

# Color codes
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'
BOLD = '\033[1m'

class ExecutionManager:
    """Manages execution of all scripts"""

    def __init__(self):
        self.script_dir = Path(__file__).parent
        self.base_dir = self.script_dir.parent
        self.results_dir = self.base_dir / "results"
        self.plots_dir = self.base_dir / "plots"
        self.start_time = None
        self.results = {}

    def print_header(self, text):
        """Print a formatted header"""
        print()
        print("=" * 80)
        print(f"{BLUE}{BOLD}{text}{RESET}")
        print("=" * 80)
        print()

    def print_success(self, text):
        """Print success message"""
        print(f"{GREEN}✅ {text}{RESET}")

    def print_error(self, text):
        """Print error message"""
        print(f"{RED}❌ {text}{RESET}")

    def print_warning(self, text):
        """Print warning message"""
        print(f"{YELLOW}⚠️  {text}{RESET}")

    def print_info(self, text):
        """Print info message"""
        print(f"{BLUE}ℹ️  {text}{RESET}")

    def run_script(self, script_name, description):
        """Run a Python script and return success status"""
        script_path = self.script_dir / script_name

        if not script_path.exists():
            self.print_error(f"Script not found: {script_path}")
            return False

        print(f"\n{description}")
        print("-" * 80)

        try:
            # Longer timeout for simulation scripts (10 minutes per extension)
            timeout_seconds = 600 if 'generate_extension' in script_name else 300

            result = subprocess.run(
                ["python3", str(script_path)],
                cwd=self.script_dir,
                capture_output=True,
                text=True,
                timeout=timeout_seconds
            )

            if result.returncode == 0:
                print(result.stdout)
                self.print_success(f"{description} completed successfully")
                return True
            else:
                print(result.stdout)
                if result.stderr:
                    self.print_warning(f"Warnings: {result.stderr[:200]}")
                self.print_error(f"{description} failed with return code {result.returncode}")
                return False

        except subprocess.TimeoutExpired:
            timeout_min = 10 if 'generate_extension' in script_name else 5
            self.print_error(f"{description} timed out (>{timeout_min} minutes)")
            return False
        except Exception as e:
            self.print_error(f"{description} error: {str(e)}")
            return False

    def verify_results(self):
        """Verify all result files were created"""
        self.print_header("📊 VERIFYING RESULT FILES")

        result_dirs = {
            'extension1': 1,      # Minimum 1 file per directory
            'extensions_1_2': 1,
            'extensions_1_2_3': 1,
            'extensions_1_2_3_4': 1
        }

        all_found = True
        total_files = 0
        total_size = 0

        for dir_name, min_count in result_dirs.items():
            dir_path = self.results_dir / dir_name

            if dir_path.exists():
                json_files = list(dir_path.glob('*.json'))
                file_count = len(json_files)
                dir_size = sum(f.stat().st_size for f in json_files) / 1024  # KB

                if file_count >= min_count:
                    self.print_success(f"{dir_name}: {file_count} files ({dir_size:.1f} KB)")
                    total_files += file_count
                    total_size += dir_size
                else:
                    self.print_error(f"{dir_name}: {file_count} files - Expected at least {min_count}")
                    all_found = False
            else:
                self.print_error(f"{dir_name}: Directory not found")
                all_found = False

        print(f"\n{BLUE}Total Result Files: {total_files}{RESET}")
        print(f"{BLUE}Total Results Size: {total_size:.1f} KB{RESET}")

        return all_found

    def verify_graphs(self):
        """Verify all graph files were created"""
        self.print_header("📈 VERIFYING GRAPH FILES")

        if not self.plots_dir.exists():
            self.print_error(f"Plots directory not found: {self.plots_dir}")
            return False

        # Get all PNG files
        png_files = list(self.plots_dir.glob("*.png"))

        if not png_files:
            self.print_error("No graph files found!")
            return False

        found_count = 0
        total_size = 0

        for graph_path in sorted(png_files):
            size_kb = graph_path.stat().st_size / 1024
            self.print_success(f"{graph_path.name} ({size_kb:.0f} KB)")
            found_count += 1
            total_size += size_kb

        print(f"\n{BLUE}Total Graphs Generated: {found_count}{RESET}")
        print(f"{BLUE}Total Graphs Size: {total_size:.1f} KB{RESET}")

        # We expect at least 15 graphs (could be more with additional analyses)
        expected_minimum = 15
        if found_count >= expected_minimum:
            self.print_success(f"Graph count verification passed ({found_count} >= {expected_minimum})")
            return True
        else:
            self.print_error(f"Expected at least {expected_minimum} graphs, found {found_count}")
            return False

    def print_summary(self):
        """Print final execution summary"""
        elapsed_time = time.time() - self.start_time
        minutes = int(elapsed_time) // 60
        seconds = int(elapsed_time) % 60

        self.print_header("📋 EXECUTION SUMMARY")

        print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Duration: {minutes}m {seconds}s")
        print()

        passed = sum(1 for v in self.results.values() if v)
        failed = len(self.results) - passed

        print(f"Scripts Executed: {len(self.results)}")
        print(f"{GREEN}Passed: {passed}{RESET}")
        print(f"{RED}Failed: {failed}{RESET}")
        print()

        for script, success in self.results.items():
            status = f"{GREEN}✅ PASS{RESET}" if success else f"{RED}❌ FAIL{RESET}"
            print(f"  {script}: {status}")

    def run_all(self):
        """Run all scripts in sequence"""
        self.start_time = time.time()

        print("\n" + "╔" + "=" * 78 + "╗")
        print("║" + f"{GREEN}{BOLD}MASTER EXECUTION SCRIPT - PCC VIVACE WITH 4 EXTENSIONS{RESET}".center(78) + "║")
        print("╚" + "=" * 78 + "╝\n")

        self.print_header("🚀 PHASE 1: GENERATING EXTENSION RESULTS")

        # Run all extension generators
        extensions = [
            ("generate_extension1_results.py", "Extension 1: Application-Aware Utilities"),
            ("generate_extension2_results.py", "Extension 2: Wireless Loss Differentiation"),
            ("generate_extension3_results.py", "Extension 3: Distributed Fairness"),
            ("generate_extension4_results.py", "Extension 4: Multipath Rate Allocation"),
        ]

        for script, description in extensions:
            self.results[description] = self.run_script(script, description)

        # Verify results
        results_ok = self.verify_results()

        # Generate graphs
        self.print_header("🚀 PHASE 2: GENERATING GRAPHS")
        self.results["Graph Generation"] = self.run_script(
            "generate_all_graphs.py",
            "Generating all graphs (16 visualizations)"
        )

        # Verify graphs
        graphs_ok = self.verify_graphs()

        # Print summary
        self.print_summary()

        # Final status - Check if main scripts passed
        all_passed = all(self.results.values())

        print()
        if all_passed:
            print("╔" + "=" * 78 + "╗")
            print("║" + f"{GREEN}{BOLD}✅ ALL TASKS COMPLETED SUCCESSFULLY{RESET}".center(78) + "║")
            print("╚" + "=" * 78 + "╝\n")
            return 0
        else:
            print("╔" + "=" * 78 + "╗")
            print("║" + f"{RED}{BOLD}❌ SOME TASKS FAILED{RESET}".center(78) + "║")
            print("╚" + "=" * 78 + "╝\n")
            return 1

def main():
    """Main entry point"""
    manager = ExecutionManager()
    sys.exit(manager.run_all())

if __name__ == '__main__':
    main()
