#!/bin/bash
# Quick deployment script for Fly.io
# Usage: ./scripts/deploy_flyio.sh

set -e  # Exit on error

echo "🚀 Phone Cough Classifier - Fly.io Deployment"
echo "=============================================="
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if fly is installed
if ! command -v fly &> /dev/null; then
    echo -e "${RED}❌ Fly CLI not found${NC}"
    echo "Install it with:"
    echo "  curl -L https://fly.io/install.sh | sh"
    exit 1
fi

echo -e "${GREEN}✓ Fly CLI found${NC}"
echo ""

# Check if logged in
if ! fly auth whoami &> /dev/null; then
    echo -e "${YELLOW}⚠ Not logged in to Fly.io${NC}"
    echo "Running: fly auth login"
    fly auth login
fi

echo -e "${GREEN}✓ Authenticated with Fly.io${NC}"
echo ""

# Read app name from fly.toml or use default
APP_NAME=$(grep "^app = " fly.toml | cut -d'"' -f2 || echo "phone-cough-classifier")

echo "App name: ${APP_NAME}"
echo ""

# Check if app exists
if fly apps list | grep -q "^${APP_NAME}"; then
    echo -e "${GREEN}✓ App '${APP_NAME}' exists${NC}"
else
    echo -e "${YELLOW}⚠ App '${APP_NAME}' not found. Creating...${NC}"
    fly apps create "${APP_NAME}"
    echo -e "${GREEN}✓ App created${NC}"
fi

echo ""

# Check if volume exists
REGION=$(grep "^primary_region = " fly.toml | cut -d'"' -f2 || echo "sjc")

if fly volumes list -a "${APP_NAME}" | grep -q "cough_data"; then
    echo -e "${GREEN}✓ Volume 'cough_data' exists${NC}"
else
    echo -e "${YELLOW}⚠ Volume 'cough_data' not found. Creating in region: ${REGION}${NC}"
    fly volumes create cough_data --size 1 --region "${REGION}" -a "${APP_NAME}"
    echo -e "${GREEN}✓ Volume created${NC}"
fi

echo ""
echo "=============================================="
echo "⚠️  IMPORTANT: Set your secrets before deploying!"
echo "=============================================="
echo ""
echo "Required secrets (run these commands):"
echo ""
echo -e "${YELLOW}fly secrets set TWILIO_ACCOUNT_SID=\"your_account_sid\" -a ${APP_NAME}${NC}"
echo -e "${YELLOW}fly secrets set TWILIO_AUTH_TOKEN=\"your_auth_token\" -a ${APP_NAME}${NC}"
echo -e "${YELLOW}fly secrets set TWILIO_PHONE_NUMBER=\"+1234567890\" -a ${APP_NAME}${NC}"
echo -e "${YELLOW}fly secrets set BASE_URL=\"https://${APP_NAME}.fly.dev\" -a ${APP_NAME}${NC}"
echo ""
echo "Optional secrets:"
echo -e "${YELLOW}fly secrets set OPENAI_API_KEY=\"sk-...\" -a ${APP_NAME}${NC}"
echo -e "${YELLOW}fly secrets set DOCTOR_HELPLINE_NUMBER=\"+919999999999\" -a ${APP_NAME}${NC}"
echo ""

# Check if secrets are set
echo "Checking secrets..."
if fly secrets list -a "${APP_NAME}" | grep -q "TWILIO_ACCOUNT_SID"; then
    echo -e "${GREEN}✓ Secrets configured${NC}"
    DEPLOY=true
else
    echo -e "${YELLOW}⚠ No secrets found${NC}"
    echo ""
    read -p "Do you want to deploy anyway? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        DEPLOY=true
    else
        DEPLOY=false
        echo -e "${RED}Deployment cancelled. Set secrets first.${NC}"
        exit 0
    fi
fi

echo ""

if [ "$DEPLOY" = true ]; then
    echo "=============================================="
    echo "🚀 Deploying to Fly.io..."
    echo "=============================================="
    echo ""

    # Deploy
    fly deploy -a "${APP_NAME}"

    echo ""
    echo -e "${GREEN}=============================================="
    echo "✅ Deployment Complete!"
    echo "=============================================="
    echo ""
    echo "Your app is live at:"
    echo -e "  https://${APP_NAME}.fly.dev"
    echo ""
    echo "Useful commands:"
    echo "  fly status -a ${APP_NAME}        # Check app status"
    echo "  fly logs -a ${APP_NAME}          # View logs"
    echo "  fly open -a ${APP_NAME}          # Open in browser"
    echo "  fly dashboard -a ${APP_NAME}     # Open dashboard"
    echo ""
    echo "Next steps:"
    echo "  1. Configure Twilio webhooks to:"
    echo "     https://${APP_NAME}.fly.dev/india/voice/router"
    echo "  2. Test by calling your Twilio number"
    echo "  3. Monitor logs: fly logs -a ${APP_NAME}"
    echo -e "${NC}"
fi
