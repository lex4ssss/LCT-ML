set -euo pipefail
ML_PYTHON="${ML_PYTHON:-.venv/bin/python}"
export PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
for suite in ml-stage1 ml-stage2 ml-dataset ml-baseline; do
  "$ML_PYTHON" -m unittest discover -s "outputs/$suite" -p 'test_*.py'
done
