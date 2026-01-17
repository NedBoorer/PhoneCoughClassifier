# Deploying to Fly.io

This guide walks you through deploying the Phone Cough Classifier to Fly.io.

## Prerequisites

1. **Fly.io Account**: Sign up at [fly.io/signup](https://fly.io/signup)
2. **Fly CLI Installed**:
   ```bash
   # macOS/Linux
   curl -L https://fly.io/install.sh | sh

   # Windows (PowerShell)
   powershell -Command "iwr https://fly.io/install.ps1 -useb | iex"
   ```
3. **Twilio Account** (optional, for phone features)
4. **OpenAI API Key** (optional, for voice agent)

---

## Step 1: Install & Authenticate with Fly CLI

```bash
# Verify installation
fly version

# Login to Fly.io
fly auth login
```

This will open your browser for authentication.

---

## Step 2: Create Your Fly App

```bash
# Navigate to project directory
cd PhoneCoughClassifier

# Create the app (this generates fly.toml if it doesn't exist)
# Note: fly.toml is already included, so you can skip 'fly launch'
# Just create the app:
fly apps create phone-cough-classifier

# Or use a custom name:
fly apps create your-custom-name
```

**Important**: If you use a custom name, update the `app` field in `fly.toml`

---

## Step 3: Create Persistent Volume

The app needs persistent storage for:
- Database (`data/cough_classifier.db`)
- Audio recordings (`recordings/`)
- Generated health cards (`data/health_cards/`)

```bash
# Create a 1GB volume
fly volumes create cough_data --size 1 --region sjc

# Verify volume was created
fly volumes list
```

**Regions**: Change `sjc` (San Jose) to your preferred region:
- `sjc` - San Jose, California
- `iad` - Ashburn, Virginia
- `lhr` - London, UK
- `fra` - Frankfurt, Germany
- `syd` - Sydney, Australia
- `sin` - Singapore

Update the `primary_region` in `fly.toml` to match your volume region.

---

## Step 4: Set Environment Secrets

**Never commit secrets to git!** Use Fly's secrets management:

```bash
# Required for Twilio integration
fly secrets set TWILIO_ACCOUNT_SID="ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
fly secrets set TWILIO_AUTH_TOKEN="your_auth_token_here"
fly secrets set TWILIO_PHONE_NUMBER="+1234567890"

# Required for WhatsApp
fly secrets set TWILIO_WHATSAPP_FROM="whatsapp:+14155238886"

# Optional: For voice agent features
fly secrets set OPENAI_API_KEY="sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

# Optional: For triage/doctor referrals
fly secrets set DOCTOR_HELPLINE_NUMBER="+919999999999"

# Set BASE_URL to your Fly app URL
fly secrets set BASE_URL="https://phone-cough-classifier.fly.dev"
```

**Get your app URL:**
```bash
fly status
# Look for "Hostname:" in the output
```

---

## Step 5: Deploy the Application

```bash
# Deploy to Fly.io
fly deploy

# This will:
# 1. Build the Docker image
# 2. Push to Fly's registry
# 3. Deploy to your machine
# 4. Run health checks
```

**Deployment output:**
```
==> Building image
...
==> Pushing image to registry
...
==> Deploying
--> v0 deployed successfully
```

---

## Step 6: Verify Deployment

```bash
# Check app status
fly status

# View logs
fly logs

# Open the app in browser
fly open

# Check health endpoint
fly open /health

# SSH into the machine (if needed)
fly ssh console
```

**Test the API:**
```bash
# Get your app URL
APP_URL=$(fly status --json | jq -r '.Hostname')

# Test health endpoint
curl https://$APP_URL/health

# View API docs
open https://$APP_URL/docs
```

---

## Step 7: Configure Twilio Webhooks

Now that your app is deployed, configure Twilio to use your Fly.io URL:

1. **Go to Twilio Console**: https://console.twilio.com/
2. **Navigate to**: Phone Numbers → Manage → Active Numbers
3. **Select your number**
4. **Configure Voice & Messaging:**

   | Setting | Value |
   |---------|-------|
   | **Voice: A CALL COMES IN** | Webhook, `https://your-app.fly.dev/india/voice/router`, HTTP POST |
   | **Messaging: A MESSAGE COMES IN** | Webhook, `https://your-app.fly.dev/twilio/sms/incoming`, HTTP POST |

5. **Save** your changes

**Test the phone system:**
- Call your Twilio number
- You should hear: "Namaste. Welcome to the health screening service..."

---

## Step 8: Monitor Your App

### View Logs
```bash
# Live tail logs
fly logs

# Filter by level
fly logs --filter ERROR

# Get recent logs
fly logs --lines 100
```

### Check Metrics
```bash
# View metrics dashboard
fly dashboard

# Check resource usage
fly status

# View VM metrics
fly vm status
```

### Scale Your App
```bash
# Scale VMs
fly scale count 2  # Run 2 instances

# Scale memory
fly scale memory 2048  # Increase to 2GB

# Auto-scale (already configured in fly.toml)
# - Scales to 0 when idle
# - Auto-starts on requests
```

---

## Step 9: Database Management

### Backup Database
```bash
# SSH into the machine
fly ssh console

# Create backup
sqlite3 /app/data/cough_classifier.db ".backup /app/data/backup.db"

# Exit
exit

# Download backup to local machine
fly ssh sftp get /app/data/backup.db ./backup.db
```

### View Database
```bash
# SSH and query
fly ssh console -C "sqlite3 /app/data/cough_classifier.db 'SELECT COUNT(*) FROM health_assessments;'"
```

---

## Troubleshooting

### Issue: App won't start
```bash
# Check logs for errors
fly logs

# Common issues:
# - Missing secrets (TWILIO_*, OPENAI_*)
# - Volume not mounted
# - Database initialization failed
```

### Issue: Volume not persisting data
```bash
# Verify volume is mounted
fly ssh console
ls -la /app/data

# If empty, check fly.toml [mounts] section
```

### Issue: Out of memory
```bash
# Check memory usage
fly vm status

# Scale up memory
fly scale memory 2048

# Or reduce ML model memory usage:
fly secrets set USE_HEAR_EMBEDDINGS="false"
```

### Issue: Slow cold starts
```bash
# Keep at least 1 machine running
# Edit fly.toml:
# min_machines_running = 1  # Change from 0

fly deploy
```

### Issue: Webhooks timing out
```bash
# Increase timeout in fly.toml [http_service.checks.health]
# Or optimize ML model loading in app startup
```

---

## Cost Optimization

Fly.io pricing: https://fly.io/docs/about/pricing/

### Free Tier Includes:
- Up to 3 shared-cpu-1x VMs (256MB RAM each)
- 3GB persistent volume storage
- 160GB outbound data transfer

### Your App Usage:
- **Current config**: 1 VM × 1GB RAM = ~$2-3/month
- **Volume**: 1GB = Free (within 3GB limit)
- **Auto-scale to 0**: Saves money when idle

### Reduce Costs:
```bash
# Option 1: Use smaller VM (if app fits)
fly scale memory 512

# Option 2: Disable auto-start during low-usage hours
# (manually start/stop)
fly scale count 0  # Stop
fly scale count 1  # Start

# Option 3: Use free tier limits
# Change fly.toml:
# memory = '256mb'
# min_machines_running = 0
```

---

## Updating Your Deployment

```bash
# After making code changes:
git add .
git commit -m "Update feature X"

# Deploy new version
fly deploy

# Rollback if needed
fly releases
fly releases rollback <version>
```

---

## Environment Variables Reference

Set via: `fly secrets set KEY=VALUE`

| Secret | Required | Description |
|--------|----------|-------------|
| `TWILIO_ACCOUNT_SID` | ✅ Yes | Twilio account identifier |
| `TWILIO_AUTH_TOKEN` | ✅ Yes | Twilio authentication token |
| `TWILIO_PHONE_NUMBER` | ✅ Yes | Your Twilio phone number |
| `BASE_URL` | ✅ Yes | Your Fly.io app URL |
| `TWILIO_WHATSAPP_FROM` | ⚠️ Optional | WhatsApp sender (if using) |
| `OPENAI_API_KEY` | ⚠️ Optional | For voice agent features |
| `DOCTOR_HELPLINE_NUMBER` | ⚠️ Optional | Triage/referral number |

---

## Useful Commands

```bash
# App management
fly apps list                    # List all your apps
fly apps destroy phone-cough     # Delete app (careful!)

# Deployment
fly deploy                       # Deploy changes
fly deploy --local-only         # Build locally (faster for slow internet)
fly deploy --no-cache           # Force rebuild

# Monitoring
fly logs                        # Tail logs
fly status                      # App status
fly dashboard                   # Open web dashboard

# Scaling
fly scale count 2               # Run 2 instances
fly scale memory 2048           # 2GB RAM
fly scale vm shared-cpu-2x      # More powerful VM

# Volumes
fly volumes list                # List volumes
fly volumes extend cough_data --size 5  # Expand to 5GB

# Secrets
fly secrets list                # List secret names (not values)
fly secrets set KEY=VALUE       # Set secret
fly secrets unset KEY           # Remove secret

# SSH & Debugging
fly ssh console                 # SSH into running machine
fly ssh sftp shell             # SFTP access
fly proxy 8000                 # Proxy local port to app

# Cleanup
fly apps destroy APP_NAME       # Delete app
fly volumes delete VOLUME_ID    # Delete volume
```

---

## Next Steps

1. ✅ **Test the phone system** - Call your number
2. ✅ **Monitor initial usage** - Check logs for errors
3. ✅ **Set up alerts** - Configure Fly.io metrics alerts
4. ✅ **Backup strategy** - Schedule regular DB backups
5. ✅ **Update README** - Document your production URL

---

## Support

- **Fly.io Docs**: https://fly.io/docs/
- **Fly.io Community**: https://community.fly.io/
- **Project Issues**: https://github.com/YourUsername/PhoneCoughClassifier/issues

---

## Security Checklist

- [x] Secrets stored in Fly.io secrets (not in code)
- [x] HTTPS enforced (`force_https = true`)
- [x] Non-root user in Docker (`USER appuser`)
- [x] Debug mode disabled in production
- [ ] Set up Fly.io IP allowlisting (if needed)
- [ ] Configure Twilio webhook validation
- [ ] Review Fly.io access logs regularly

Your app is now live at: **https://phone-cough-classifier.fly.dev** 🚀
