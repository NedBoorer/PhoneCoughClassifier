# Nginx Configuration

This directory contains the Nginx configuration for the Phone Cough Classifier.

## SSL Certificates

To enable HTTPS (port 443), you need to generate SSL certificates and place them in an `ssl` subdirectory.

1. Create the `ssl` directory:
   ```bash
   mkdir ssl
   ```

2. Generate a self-signed certificate (for development/testing):
   ```bash
   openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
     -keyout ssl/server.key \
     -out ssl/server.crt
   ```

3. Uncomment the HTTPS section in `nginx.conf`.
