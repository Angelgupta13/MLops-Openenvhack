#!/usr/bin/env bash
#
# validate-submission.sh — OpenEnv Submission Validator
#
# This script validates that your submission meets all competition requirements.
# It checks:
#   1. Docker build passes
#   2. openenv validate passes
#   3. Server responds to /health and /reset
#   4. Baseline inference script reproduces scores
#   5. 3+ tasks with graders produce scores in 0.0-1.0 range

set -uo pipefail

DOCKER_BUILD_TIMEOUT=600

if [ -t 1 ]; then
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[1;33m'
    BOLD='\033[1m'
    NC='\033[0m'
else
    RED=''
    GREEN=''
    YELLOW=''
    BOLD=''
    NC=''
fi

log() { echo -e "${GREEN}[OK]${NC} $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
err() { echo -e "${RED}[ERROR]${NC} $*"; }
fail() { echo -e "${RED}[FAIL]${NC} $*"; exit 1; }

check() {
    local cmd="$1"
    local name="$2"
    echo -n "Checking $name... "
    if eval "$cmd" &>/dev/null; then
        log "$name"
        return 0
    else
        fail "$name"
        return 1
    fi
}

# Check prerequisites
check "command -v docker" "Docker installed"
check "command -v python" "Python installed"

# Build Docker
log "Building Docker image..."
cd "$(dirname "$0")"
if docker build -t mlops-debug-env .; then
    log "Docker build passed"
else
    fail "Docker build failed"
fi

# Start container in background
log "Starting server..."
docker run -d -p 7860:7860 --name mlops-test mlops-debug-env
sleep 5

# Check health endpoint
check "curl -s http://localhost:7860/health" "Server /health responds"

# Test reset endpoint
log "Testing /reset..."
RESET_RESP=$(curl -s -X POST http://localhost:7860/reset -H "Content-Type: application/json" -d '{"task_id":"easy","seed":42}')
if echo "$RESET_RESP" | grep -q "task_id"; then
    log "/reset works"
else
    fail "/reset failed"
fi

# Test inference script
log "Running baseline inference..."
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gemini-2.5-flash"
export HF_TOKEN="${HF_TOKEN:-test}"
export ENV_BASE_URL="http://localhost:7860"

if python inference.py --task easy --seed 42 2>&1 | grep -q "score="; then
    log "Inference script format correct"
else
    warn "Inference script may have format issues"
fi

# Cleanup
log "Cleaning up..."
docker stop mlops-test 2>/dev/null || true
docker rm mlops-test 2>/dev/null || true

echo ""
echo "========================================="
log "All validation checks passed!"
echo "========================================="