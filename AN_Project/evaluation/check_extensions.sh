#!/bin/bash

################################################################################
#                     CHECK ALL EXTENSIONS SCRIPT                             #
#                                                                              #
#  Run all extensions and verify results are created                          #
#  Usage: bash check_extensions.sh                                            #
################################################################################

echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║                    EXTENSION VERIFICATION SCRIPT                          ║"
echo "║                   Check all 3 extensions at once                          ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"

# Color codes for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Track results
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0
EXTENSIONS_PASSED=0
EXTENSIONS_FAILED=0

# Function to print section headers
print_header() {
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo -e "${BLUE}$1${NC}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
}

# Function to run extension tests
run_extension() {
    local ext_name=$1
    local test_files=$2
    local output_dir=$3

    print_header "🧪 RUNNING: $ext_name"

    # Run the tests
    echo -e "${YELLOW}Command:${NC} pytest $test_files -v"
    echo ""

    pytest $test_files -v --tb=short
    TEST_RESULT=$?

    # Check if tests passed
    if [ $TEST_RESULT -eq 0 ]; then
        echo -e "${GREEN}✅ $ext_name PASSED${NC}"
        ((EXTENSIONS_PASSED++))

        # Check if results file was created
        if [ -f "results/$output_dir/aggregate_results.json" ]; then
            echo -e "${GREEN}✅ Results file created: results/$output_dir/aggregate_results.json${NC}"

            # Show summary from JSON
            python3 << EOF
import json
try:
    with open('results/$output_dir/aggregate_results.json') as f:
        data = json.load(f)
        if isinstance(data, list):
            data = data[0]
        print(f"\n   📊 Test Summary:")
        print(f"      Total Tests: {data.get('total_tests', 'N/A')}")
        print(f"      Passed: {data.get('passed_tests', 'N/A')}")
        print(f"      Failed: {data.get('failed_tests', 'N/A')}")
        print(f"      Success Rate: {data.get('success_rate', 'N/A')}%")
        print(f"      Duration: {data.get('total_duration_seconds', 'N/A'):.2f}s")
        print(f"      Extension: {data.get('extension', 'N/A')}")
except Exception as e:
    print(f"   Error reading results: {e}")
EOF
        else
            echo -e "${RED}⚠️  Results file not found in results/$output_dir/${NC}"
        fi
    else
        echo -e "${RED}❌ $ext_name FAILED${NC}"
        ((EXTENSIONS_FAILED++))
    fi

    echo ""
}

# Function to check results files
check_results_file() {
    local ext_name=$1
    local file_path=$2

    if [ -f "$file_path" ]; then
        echo -e "${GREEN}✅${NC} $ext_name: $file_path ($(du -h "$file_path" | cut -f1))"
        return 0
    else
        echo -e "${RED}❌${NC} $ext_name: $file_path (NOT FOUND)"
        return 1
    fi
}

# Start checking
echo ""
echo -e "${YELLOW}This script will:${NC}"
echo "  1. Run Extension 1 tests"
echo "  2. Run Extension 2 tests"
echo "  3. Run Extension 3 tests"
echo "  4. Verify all results are saved"
echo ""

# Check Python and pytest
echo -e "${YELLOW}Checking prerequisites...${NC}"
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python 3 not found${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Python 3 found${NC}"

if ! python3 -m pytest --version &> /dev/null; then
    echo -e "${RED}❌ pytest not installed${NC}"
    exit 1
fi
echo -e "${GREEN}✅ pytest found${NC}"

# Create results directories if they don't exist
mkdir -p results/extension1
mkdir -p results/extensions_1_2
mkdir -p results/extensions_1_2_3
mkdir -p results/extensions_1_2_3_4

echo ""
echo -e "${GREEN}✅ Prerequisites satisfied${NC}"

################################################################################
# EXTENSION 1: Application-Aware Utilities
################################################################################

run_extension \
    "Extension 1: Application-Aware Utilities" \
    "tests/test_traffic_classifier.py tests/test_utility_bank.py tests/test_meta_controller.py" \
    "extension1"

################################################################################
# EXTENSION 2: Wireless Loss Differentiation
################################################################################

run_extension \
    "Extension 2: Wireless Loss Differentiation" \
    "tests/test_loss_classifier.py tests/test_adaptive_loss_coefficient.py" \
    "extensions_1_2"

################################################################################
# EXTENSION 3: Distributed Fairness
################################################################################

run_extension \
    "Extension 3: Distributed Fairness" \
    "tests/test_fairness_controller_quick.py" \
    "extensions_1_2_3"

################################################################################
# FINAL SUMMARY
################################################################################

print_header "📋 FINAL VERIFICATION SUMMARY"

echo ""
echo -e "${YELLOW}Checking result files...${NC}"
echo ""

FILES_FOUND=0
check_results_file "Extension 1" "results/extension1/aggregate_results.json" && ((FILES_FOUND++))
check_results_file "Extension 2" "results/extensions_1_2/aggregate_results.json" && ((FILES_FOUND++))
check_results_file "Extension 3" "results/extensions_1_2_3/aggregate_results.json" && ((FILES_FOUND++))

echo ""
echo -e "${YELLOW}Test Summary:${NC}"
echo "  Extensions Passed: $EXTENSIONS_PASSED"
echo "  Extensions Failed: $EXTENSIONS_FAILED"
echo "  Result Files Found: $FILES_FOUND/3"

echo ""
if [ $EXTENSIONS_FAILED -eq 0 ] && [ $FILES_FOUND -eq 3 ]; then
    echo "╔════════════════════════════════════════════════════════════════════════════╗"
    echo -e "║${GREEN}                    ✅ ALL EXTENSIONS PASSED!${NC}                          ║"
    echo "╚════════════════════════════════════════════════════════════════════════════╝"
    echo ""
    echo -e "${GREEN}All extensions are working correctly!${NC}"
    echo "All results have been automatically saved to JSON files in results/"
    echo ""
    echo -e "${YELLOW}Next steps:${NC}"
    echo "  1. View results: python -m json.tool results/extension1/aggregate_results.json"
    echo "  2. Generate plots: python evaluation/plot_extension_results.py"
    echo "  3. Ready for submission!"
else
    echo "╔════════════════════════════════════════════════════════════════════════════╗"
    echo -e "║${RED}                   ⚠️  SOME TESTS FAILED${NC}                              ║"
    echo "╚════════════════════════════════════════════════════════════════════════════╝"
    echo ""
    echo -e "${RED}Please review the output above for errors${NC}"
fi

echo ""
