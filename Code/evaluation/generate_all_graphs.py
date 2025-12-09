#!/usr/bin/env python3

"""
MASTER GRAPH GENERATION SCRIPT

This script generates ALL graphs for all 4 extensions:
- Extension 1: 3 graphs (utility, throughput, performance tradeoff)
- Extension 2: 4 graphs (loss differentiation, throughput, utility)
- Extension 3: 5 graphs (fairness, convergence, multi-flow metrics, timeline, dashboard)
- Extension 4: 4 graphs (utility, throughput, path distribution, latency)

Total: 16 visualizations

Usage:
    python3 generate_all_graphs.py

Output:
    All graphs saved to ../plots/
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime

# Color codes for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'
BOLD = '\033[1m'


class GraphGenerator:
    """Manages generation of all graphs"""

    def __init__(self):
        self.script_dir = Path(__file__).parent
        self.base_dir = self.script_dir.parent
        self.plots_dir = self.base_dir / "plots"
        self.results = {}

    def print_header(self, text):
        """Print formatted header"""
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

    def print_info(self, text):
        """Print info message"""
        print(f"{BLUE}ℹ️  {text}{RESET}")

    def run_script(self, script_name, description):
        """Run a plotting script and return success status"""
        script_path = self.script_dir / script_name

        if not script_path.exists():
            self.print_error(f"Script not found: {script_path}")
            return False

        print(f"\n{description}")
        print("-" * 80)

        try:
            result = subprocess.run(
                ["python3", str(script_path)],
                cwd=self.script_dir,
                capture_output=True,
                text=True,
                timeout=120
            )

            if result.returncode == 0:
                print(result.stdout)
                self.print_success(f"{description} completed successfully")
                return True
            else:
                print(result.stdout)
                if result.stderr:
                    print(f"{YELLOW}Warnings: {result.stderr[:300]}{RESET}")
                self.print_error(f"{description} failed with return code {result.returncode}")
                return False

        except subprocess.TimeoutExpired:
            self.print_error(f"{description} timed out (>2 minutes)")
            return False
        except Exception as e:
            self.print_error(f"{description} error: {str(e)}")
            return False

    def verify_graphs(self):
        """Verify generated graph files"""
        self.print_header("📈 VERIFYING GENERATED GRAPHS")

        if not self.plots_dir.exists():
            self.print_error(f"Plots directory not found: {self.plots_dir}")
            return False

        # Get all PNG files
        png_files = list(self.plots_dir.glob("*.png"))

        if not png_files:
            self.print_error("No graph files found!")
            return False

        total_size = 0
        for graph_file in sorted(png_files):
            size_kb = graph_file.stat().st_size / 1024
            self.print_success(f"{graph_file.name} ({size_kb:.0f} KB)")
            total_size += size_kb

        print(f"\n{BLUE}Total Graphs: {len(png_files)}{RESET}")
        print(f"{BLUE}Total Size: {total_size:.1f} KB{RESET}")

        return len(png_files) > 0

    def generate_all(self):
        """Generate all graphs"""
        self.print_header("🎨 GENERATING ALL GRAPHS")

        # Define all plotting scripts
        plotting_scripts = [
            ("plot_extension1_analysis.py", "Extension 1 Graphs (Application-Aware Utilities)"),
            ("plot_extension2_analysis.py", "Extension 2 Graphs (Wireless Loss Differentiation)"),
            ("generate_extension3_complete_graphs.py", "Extension 3 Graphs (Distributed Fairness)"),
            ("plot_extension4_analysis.py", "Extension 4 Graphs (Multipath Rate Allocation)"),
        ]

        # Run each plotting script
        for script, description in plotting_scripts:
            self.results[description] = self.run_script(script, description)

        # Verify all graphs were created
        graphs_ok = self.verify_graphs()

        # Print summary
        self.print_header("📋 GRAPH GENERATION SUMMARY")

        passed = sum(1 for v in self.results.values() if v)
        failed = len(self.results) - passed

        print(f"Plotting Scripts Executed: {len(self.results)}")
        print(f"{GREEN}Passed: {passed}{RESET}")
        print(f"{RED}Failed: {failed}{RESET}")
        print()

        for script, success in self.results.items():
            status = f"{GREEN}✅ PASS{RESET}" if success else f"{RED}❌ FAIL{RESET}"
            print(f"  {script}: {status}")

        print()

        # Final status
        all_passed = all(self.results.values()) and graphs_ok

        if all_passed:
            print("╔" + "=" * 78 + "╗")
            print("║" + f"{GREEN}{BOLD}✅ ALL GRAPHS GENERATED SUCCESSFULLY{RESET}".center(88) + "║")
            print("╚" + "=" * 78 + "╝")
            print()
            return 0
        else:
            print("╔" + "=" * 78 + "╗")
            print("║" + f"{RED}{BOLD}❌ SOME GRAPHS FAILED TO GENERATE{RESET}".center(88) + "║")
            print("╚" + "=" * 78 + "╝")
            print()
            return 1


def main():
    """Main entry point"""
    generator = GraphGenerator()
    sys.exit(generator.generate_all())


if __name__ == '__main__':
    main()
