#!/bin/bash

# Script to generate all plots for the paper
# Usage: ./generate_plots.sh

# Configuration
CONDA_ENV_NAME=$1
OUTPUT_DIR="../plots"
SCRIPT_DIR="./"
CONFIG_DIR="./configs"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
  echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if conda is available
if ! command -v conda &> /dev/null; then
    print_error "Conda is not available. Please install conda or ensure it's in your PATH."
    exit 1
fi

# Initialize conda for bash (needed for conda activate to work in scripts)
eval "$(conda shell.bash hook)"

print_status "Starting plot generation for paper..."

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Activate conda environment
print_status "Activating conda environment: $CONDA_ENV_NAME"
if conda activate "$CONDA_ENV_NAME"; then
  print_success "Successfully activated conda environment: $CONDA_ENV_NAME"
else
  print_error "Failed to activate conda environment: $CONDA_ENV_NAME"
  print_error "Make sure the environment exists: conda env list"
  exit 1
fi

# The following function runs a script with error handling
run_script() {
  local script_name="$1"
  local script_args="$2"
  local description="$3"

    print_status "Running: $description"
    print_status "Command: python $SCRIPT_DIR/$script_name $script_args"

    if python "$SCRIPT_DIR/$script_name" $script_args; then
        print_success "Completed: $description"
    else
        print_error "Failed: $description"
        print_error "Script: $script_name"
        print_error "Arguments: $script_args"
        exit 1
    fi
    echo ""  # Add spacing between scripts
}

# =============================================================================
# PLOT GENERATION COMMANDS
# =============================================================================
print_status "Starting plot generation..."

run_script "ntk_initialisation.py" \
  "--plot-dir $OUTPUT_DIR" \
  "Plot of the NTK at initialisation"

run_script "kernel_recursion.py" \
  "--plot-dir $OUTPUT_DIR" \
  "Plot of the kernel recursion"

run_script "delta_ntk_plots.py" \
  "--plot-dir $OUTPUT_DIR" \
  "Plot of delta ntk"

run_script "eigvals_single_plot.py" \
  "--plot-dir $OUTPUT_DIR" \
  "Single plot of the eigenvalues"

run_script "ntk_eigvals_plots.py" \
  "--plot-dir $OUTPUT_DIR" \
  "Plot of ntk eigvals for L0 L1 and L2 data"

run_script "ntk_alignment.py" \
  "--plot-dir $OUTPUT_DIR" \
  "Plot of the alignment of the NTK with M"

run_script "expval_u_f0.py" \
  "--plot-dir $OUTPUT_DIR" \
  "Plot of the expectation value of U and f0"

run_script "gibbs.py" \
  "--plot-dir $OUTPUT_DIR" \
  "Plot of the Gibbs kernel and covariance of f0"

run_script "u_v_fluctuations.py" \
  "--plot-dir $OUTPUT_DIR" \
  "Plot of the fluctuations of U and V"

run_script "pdf_plots.py" \
  "--plot-dir $OUTPUT_DIR" \
  "PDF plots with full and split contributions"


# =============================================================================
# COMPLETION
# =============================================================================

print_success "All plots generated successfully!"
print_status "Plots saved to: $OUTPUT_DIR"
# print_status "Files generated:"
# ls -la "$OUTPUT_DIR"

print_success "Plot generation complete! 🎉"
