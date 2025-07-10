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

# NTK pheno section
# ~~~~~~~~~~~~~~~~~~~~~
NTK_PHENO_DIR="$OUTPUT_DIR/ntk_pheno"
mkdir -p "$NTK_PHENO_DIR"
run_script "delta_ntk_plots.py" \
  "$CONFIG_DIR/delta_ntk.yaml --plot-dir $NTK_PHENO_DIR --filename delta_ntk.pdf" \
  "Plot of delta ntk for LO L1 and L2 data"

run_script "ntk_eigvals_plots.py" \
  "$CONFIG_DIR/ntk_eigvals_L0_L1_L2.yaml --plot-dir $NTK_PHENO_DIR --filename ntk_eigvals_L0_L1_L2" \
  "Plot of ntk eigvals for LO L1 and L2 data"

run_script "delta_ntk_plots.py" \
  "$CONFIG_DIR/delta_ntk_arch.yaml --plot-dir $NTK_PHENO_DIR --filename delta_ntk_arch.pdf" \
  "Plot of delta ntk for different architectures"

run_script "eigvals_single_plot.py" \
  "$CONFIG_DIR/ntk_eigvals_arch_single_plot.yaml --plot-dir $NTK_PHENO_DIR --filename ntk_eigvals_single_plot_arch.pdf" \
  "Single plot of the ntk eigvals for different architectures"

run_script "ntk_alignment.py" \
  "$CONFIG_DIR/ntk_alignment.yaml --plot-dir $NTK_PHENO_DIR --filename ntk_alignment.pdf" \
  "Plot of the alignment of the NTK with M"

# Studies on U and V
# ~~~~~~~~~~~~~~~~~~~~~
U_V_STUDIES_DIR="$OUTPUT_DIR/u_v_studies"
mkdir -p "$U_V_STUDIES_DIR"
run_script "u_v_fluctuations.py" \
  "$CONFIG_DIR/u_v_fluctuations.yaml --plot-dir $U_V_STUDIES_DIR" \
  "Plot of the fluctuations of U and V"

# H section
# ~~~~~~~~~~~~~~~~~~~~~
run_script "eigvals_single_plot.py" \
  "$CONFIG_DIR/h_eigvals_single_plot.yaml --plot-dir $OUTPUT_DIR --filename h_eigvals_single_plot.pdf" \
  "Single plot of the H eigvals"


# Analytical evolution
# ~~~~~~~~~~~~~~~~~~~~~
AS_DIR="$OUTPUT_DIR/analytical_solution"
mkdir -p "$AS_DIR"
# PDF plots with L0 data
run_script "pdf_plots.py" \
  "250604-ac-01-L0 --ref-epoch 20000 --plot-dir $AS_DIR" \
  "PDF plots with L0 data"

# PDF plots with L2 data
run_script "pdf_plots.py" \
  "250604-ac-03-L2 --ref-epoch 20000 --plot-dir $AS_DIR" \
  "PDF plots with L2 data"


# =============================================================================
# COMPLETION
# =============================================================================

print_success "All plots generated successfully!"
print_status "Plots saved to: $OUTPUT_DIR"
print_status "Files generated:"
ls -la "$OUTPUT_DIR"

print_success "Plot generation complete! 🎉"
