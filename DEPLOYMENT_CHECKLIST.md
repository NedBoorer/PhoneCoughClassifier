# Deployment Checklist

Complete this checklist before deploying to Fly.io production.

## Pre-Deployment

### 1. Local Testing ✓
- [ ] Application runs locally without errors
  ```bash
  python3 -m uvicorn app.main:app --port 8000
  ```
- [ ] Health endpoint responds: `curl http://localhost:8000/health`
- [ ] All ML models load successfully
- [ ] Database initializes correctly
- [ ] Core tests pass: `pytest tests/ -v`

### 2. Environment Configuration ✓
- [ ] `.env` file configured with all required variables
- [ ] ML model files present in `models/` directory
  - [ ] `cough_classifier.joblib`
  - [ ] `parkinsons_classifier.joblib`
  - [ ] `depression_classifier.joblib`
- [ ] External service credentials ready:
  - [ ] Twilio Account SID
  - [ ] Twilio Auth Token
  - [ ] Twilio Phone Number
  - [ ] OpenAI API Key (optional)

### 3. Fly.io Account Setup ✓
- [ ] Fly.io account created at https://fly.io/signup
- [ ] Fly CLI installed: `fly version`
- [ ] Logged in: `fly auth login`
- [ ] Credit card added (required for deployment)

### 4. Code Repository ✓
- [ ] Code committed to git
- [ ] `.gitignore` excludes sensitive files (.env, secrets)
- [ ] README.md updated with production URL
- [ ] Version tagged (optional): `git tag v1.0.0`

---

## Deployment Steps

### 5. Create Fly App ✓
```bash
# Create app (or use deployment script)
fly apps create phone-cough-classifier

# Or use custom name
fly apps create your-custom-name
```
- [ ] App created successfully
- [ ] App name matches `fly.toml`

### 6. Create Persistent Volume ✓
```bash
# Create 1GB volume in your region
fly volumes create cough_data --size 1 --region sjc
```
- [ ] Volume created
- [ ] Region matches `fly.toml` primary_region

### 7. Set Secrets ✓
```bash
# Required secrets
fly secrets set TWILIO_ACCOUNT_SID="AC..."
fly secrets set TWILIO_AUTH_TOKEN="..."
fly secrets set TWILIO_PHONE_NUMBER="+1234567890"
fly secrets set BASE_URL="https://your-app.fly.dev"

# Optional secrets
fly secrets set OPENAI_API_KEY="sk-..."
fly secrets set DOCTOR_HELPLINE_NUMBER="+919999999999"
fly secrets set TWILIO_WHATSAPP_FROM="whatsapp:+14155238886"
```
- [ ] All required secrets set
- [ ] Secrets verified: `fly secrets list`

### 8. Deploy Application ✓
```bash
# Deploy using script
./scripts/deploy_flyio.sh

# Or deploy manually
fly deploy
```
- [ ] Build completes successfully
- [ ] Deployment succeeds
- [ ] Health checks pass

### 9. Verify Deployment ✓
```bash
# Check status
fly status

# View logs
fly logs

# Test health endpoint
curl https://your-app.fly.dev/health
```
- [ ] App status: `running`
- [ ] Health check: `healthy`
- [ ] No errors in logs
- [ ] API docs accessible: `https://your-app.fly.dev/docs`

---

## Post-Deployment

### 10. Configure Twilio Webhooks ✓
In Twilio Console (https://console.twilio.com/):
- [ ] Navigate to Phone Numbers > Active Numbers
- [ ] Select your phone number
- [ ] Voice Configuration:
  - [ ] A call comes in: `https://your-app.fly.dev/india/voice/router` (POST)
- [ ] Messaging Configuration:
  - [ ] A message comes in: `https://your-app.fly.dev/twilio/sms/incoming` (POST)
- [ ] WhatsApp Configuration (if enabled):
  - [ ] When a message comes in: `https://your-app.fly.dev/whatsapp/incoming` (POST)
- [ ] Save all changes

### 11. Test Phone System ✓
- [ ] Call your Twilio number
- [ ] Hear greeting: "Namaste. Welcome to the health screening service..."
- [ ] Test language selection (Press 1 for English, 2 for Hindi)
- [ ] Test recording flow
- [ ] Verify SMS/WhatsApp reports are sent
- [ ] Check logs for successful call: `fly logs`

### 12. Test API Endpoints ✓
```bash
# Get your app URL
APP_URL="https://your-app.fly.dev"

# Test health
curl $APP_URL/health

# Test model status
curl $APP_URL/test/screening-models

# View API docs
open $APP_URL/docs
```
- [ ] All endpoints respond
- [ ] Models show as loaded
- [ ] No 500 errors

### 13. Monitoring Setup ✓
- [ ] Fly.io dashboard accessible: `fly dashboard`
- [ ] Metrics visible (CPU, memory, requests)
- [ ] Set up alerts (optional):
  - [ ] Email alerts for crashes
  - [ ] Memory usage alerts
- [ ] Log monitoring configured

### 14. Security Review ✓
- [ ] Secrets not in code or git
- [ ] HTTPS enforced (`force_https = true` in fly.toml)
- [ ] Debug mode disabled in production
- [ ] Twilio webhook signature validation enabled (if applicable)
- [ ] No sensitive data in logs
- [ ] Database backups scheduled

### 15. Documentation ✓
- [ ] Update README.md with production URL
- [ ] Document Twilio webhook URLs
- [ ] Add deployment notes (date, version, changes)
- [ ] Update team on deployment

---

## Performance Tuning (Optional)

### 16. Optimize Resources ✓
```bash
# Check current resources
fly scale show

# Adjust if needed
fly scale memory 512   # Reduce for cost savings
fly scale memory 2048  # Increase for performance
```
- [ ] Memory usage monitored
- [ ] CPU usage acceptable
- [ ] Response times < 2s

### 17. Database Optimization ✓
```bash
# Check database size
fly ssh console -C "du -sh /app/data"

# Backup database
fly ssh console -C "sqlite3 /app/data/cough_classifier.db '.backup /app/data/backup.db'"
```
- [ ] Database size reasonable
- [ ] Backup created
- [ ] Old data archived (if needed)

---

## Rollback Plan

### In Case of Issues
```bash
# View recent releases
fly releases

# Rollback to previous version
fly releases rollback

# Or rollback to specific version
fly releases rollback v5

# Check logs for errors
fly logs --filter ERROR
```

Keep previous version info:
- Previous version: `v____`
- Deployed at: `____/____/____`
- Rollback command: `fly releases rollback v____`

---

## Success Criteria

Deployment is successful when:
- ✅ App accessible at `https://your-app.fly.dev`
- ✅ Health endpoint returns `{"status": "healthy"}`
- ✅ Phone calls connect and play greeting
- ✅ Voice recordings are processed
- ✅ ML models classify audio correctly
- ✅ WhatsApp reports are sent
- ✅ No errors in logs
- ✅ Response times < 2 seconds
- ✅ Auto-scaling works (scales to 0 when idle)

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| App won't start | Check `fly logs` for errors, verify secrets set |
| Health check fails | Verify `/health` endpoint works, check model loading |
| Twilio webhooks fail | Verify BASE_URL matches app URL, check logs |
| Out of memory | Scale up: `fly scale memory 2048` |
| Slow performance | Add region, increase VM size, or add instance |
| Database errors | Check volume mounted: `fly ssh console -C "ls /app/data"` |

---

## Maintenance

### Regular Tasks
- [ ] **Daily**: Monitor logs for errors
- [ ] **Weekly**: Check metrics (CPU, memory, disk usage)
- [ ] **Monthly**: Review costs, optimize resources
- [ ] **Quarterly**: Update dependencies, security patches

### Backup Schedule
```bash
# Weekly database backup
fly ssh console -C "sqlite3 /app/data/cough_classifier.db '.backup /app/data/backup-$(date +%Y%m%d).db'"

# Download backup
fly ssh sftp get /app/data/backup-*.db ./backups/
```

---

## Deployment Complete! 🎉

Your Phone Cough Classifier is now live at:
```
https://your-app-name.fly.dev
```

Next steps:
1. Share the Twilio number with users
2. Monitor initial usage and logs
3. Collect feedback and iterate
4. Scale as needed

---

**Deployment Date**: ________________
**Deployed By**: ________________
**Version**: ________________
**App URL**: ________________
**Twilio Number**: ________________
