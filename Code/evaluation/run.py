#!/usr/bin/env python3
"""
Run all simulations without generating graphs - faster for iterative development

Usage: python3 evaluation/run.py

Takes about 10-20 minutes depending on your hardware. If you want graphs too,
use run_all.py instead (adds ~5 minutes for graph generation).
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime
import time

# Terminal colors for nicer output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'
BOLD = '\033[1m'

class SimulationManager:
    """Handles running all the simulation scripts in sequence"""

    def __init__(self):
        self.script_dir = Path(__file__).parent
        self.base_dir = self.script_dir.parent
        self.results_dir = self.base_dir / "results"
        self.start_time = None
        self.results = {}  # Track which scripts passed/failed

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
        """Run a simulation script and track if it succeeded"""
        script_path = self.script_dir / script_name

        if not script_path.exists():
            self.print_error(f"Script not found: {script_path}")
            return False

        print(f"\n{description}")
        print("-" * 80)

        try:
            # Give each simulation 10 minutes max - they usually finish in 2-5
            timeout_seconds = 600

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
            self.print_error(f"{description} timed out (>10 minutes)")
            return False
        except Exception as e:
            self.print_error(f"{description} error: {str(e)}")
            return False

    def verify_results(self):
        """Make sure all the result files got created properly"""
        self.print_header("📊 VERIFYING RESULT FILES")

        # Each extension should have at least one result file
        result_dirs = {
            'extension1': 1,
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
                dir_size = sum(f.stat().st_size for f in json_files) / 1024

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

        print(f"Simulations Executed: {len(self.results)}")
        print(f"{GREEN}Passed: {passed}{RESET}")
        print(f"{RED}Failed: {failed}{RESET}")
        print()

        for script, success in self.results.items():
            status = f"{GREEN}✅ PASS{RESET}" if success else f"{RED}❌ FAIL{RESET}"
            print(f"  {script}: {status}")

    def run_simulations(self):
        """Run all the extension simulations in sequence"""
        self.start_time = time.time()

        print("\n" + "╔" + "=" * 78 + "╗")
        print("║" + f"{GREEN}{BOLD}SIMULATION EXECUTION - PCC VIVACE WITH 4 EXTENSIONS{RESET}".center(78) + "║")
        print("╚" + "=" * 78 + "╝\n")

        self.print_header("🚀 GENERATING EXTENSION RESULTS")

        # Run each extension's simulations
        # These run sequentially since they're CPU-intensive
        extensions = [
            ("generate_extension1_results.py", "Extension 1: Application-Aware Utilities"),
            ("generate_extension2_results.py", "Extension 2: Wireless Loss Differentiation"),
            ("generate_extension3_results.py", "Extension 3: Distributed Fairness"),
            ("generate_extension4_results.py", "Extension 4: Multipath Rate Allocation"),
        ]

        for script, description in extensions:
            self.results[description] = self.run_script(script, description)

        # Double-check all the result files got written
        results_ok = self.verify_results()

        # Show what happened
        self.print_summary()

        # All done!
        all_passed = all(self.results.values()) and results_ok

        print()
        if all_passed:
            print("╔" + "=" * 78 + "╗")
            print("║" + f"{GREEN}{BOLD}✅ ALL SIMULATIONS COMPLETED SUCCESSFULLY{RESET}".center(78) + "║")
            print("╚" + "=" * 78 + "╝")
            print()
            self.print_info("To generate graphs, run: python3 evaluation/generate_all_graphs.py")
            print()
            return 0
        else:
            print("╔" + "=" * 78 + "╗")
            print("║" + f"{RED}{BOLD}❌ SOME SIMULATIONS FAILED{RESET}".center(78) + "║")
            print("╚" + "=" * 78 + "╝\n")
            return 1

def main():
    """Main entry point"""
    manager = SimulationManager()
    sys.exit(manager.run_simulations())

if __name__ == '__main__':
    main()
