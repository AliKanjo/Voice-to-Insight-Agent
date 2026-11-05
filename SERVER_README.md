# Deploying Qwen2.5-1.5B-Instruct with llama.cpp on DigitalOcean

Production deployment of a quantized open LLM on a small Ubuntu droplet, served via `llama-server`, managed by `systemd`, and exposed through `nginx`.

## Stack

- **Model:** Qwen2.5-1.5B-Instruct, GGUF format, Q4_K_M quantization
- **Engine:** llama.cpp (`llama-server`, OpenAI-compatible API)
- **Host:** DigitalOcean droplet, Ubuntu, 1 vCPU / 1–2GB RAM
- **Process manager:** systemd (auto-restart, memory cap)
- **Reverse proxy:** nginx (TLS via certbot, basic API key check)

## Prerequisites

- A provisioned Ubuntu droplet (1–2GB RAM recommended minimum)
- SSH access with a sudo-capable user
- (Optional, for TLS) a domain name pointed at the droplet's IP

---

## 1. System setup

```bash
apt update && apt upgrade -y
apt install -y build-essential cmake git curl python3-pip nginx certbot python3-certbot-nginx
```

### Add a swap file (safety net against OOM)

```bash
fallocate -l 1G /swapfile
chmod 600 /swapfile
mkswap /swapfile
swapon /swapfile
echo '/swapfile none swap sw 0 0' >> /etc/fstab
```

---

## 2. Build llama.cpp from source

```bash
cd /opt
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp

cmake -B build -DGGML_NATIVE=ON
cmake --build build --config Release -j $(nproc)
```

The binary you need is at `/opt/llama.cpp/build/bin/llama-server`.

> **Note:** `-DGGML_NATIVE=ON` optimizes for this droplet's specific CPU. If you ever migrate this build to different underlying hardware, rebuild rather than copying the binary.

---

## 3. Download the model

Download once to local disk — never use `-hf` pull-on-boot in production (adds a network dependency, no version pinning, slower restarts).

```bash
pip install --break-system-packages -U "huggingface_hub[cli]"

mkdir -p /opt/llama/models
huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct-GGUF \
  qwen2.5-1.5b-instruct-q4_k_m.gguf \
  --local-dir /opt/llama/models
```

---

## 4. Manual test run

Before wiring up systemd, confirm it runs:

```bash
/opt/llama.cpp/build/bin/llama-server \
  -m /opt/llama/models/qwen2.5-1.5b-instruct-q4_k_m.gguf \
  -c 4096 --host 127.0.0.1 --port 8080 --parallel 2 --cont-batching
```

In a second terminal:

```bash
curl http://127.0.0.1:8080/health
```

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-1.5b-instruct",
    "messages": [
      {"role": "user", "content": "Say hi in one sentence"}
    ]
  }'
```

Plain-text completion alternative (non-chat format):

```bash
curl http://127.0.0.1:8080/completion \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Say hi in one sentence:",
    "n_predict": 50
  }'
```

If both respond correctly, stop the process (`Ctrl+C`) and move on.

---

## 5. systemd service

Find your username if needed:

```bash
whoami
```

Create the service file, replacing `YOUR_USERNAME`:

```bash
cat > /etc/systemd/system/llama-server.service << 'EOF'
[Unit]
Description=llama.cpp server
After=network.target

[Service]
Type=simple
ExecStart=/opt/llama.cpp/build/bin/llama-server \
  -m /opt/llama/models/qwen2.5-1.5b-instruct-q4_k_m.gguf \
  -c 4096 --host 127.0.0.1 --port 8080 \
  --parallel 2 --cont-batching --threads 1
Restart=always
RestartSec=3
User=YOUR_USERNAME
MemoryMax=900M

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable --now llama-server
systemctl status llama-server
```

### Verify auto-restart

```bash
kill -9 $(pgrep -f llama-server)
sleep 5
systemctl status llama-server   # should already be back up
```

> Running as a regular sudo user is simpler but a slightly weaker security posture than a restricted service account — a fine tradeoff for a demo/workshop box, worth revisiting if this ever handles sensitive data.

---

## 6. nginx reverse proxy

```bash
cat > /etc/nginx/sites-available/llama << 'EOF'
server {
    listen 80;
    server_name 90.2.2.2;

    location / {
        proxy_pass http://127.0.0.1:8080;
        proxy_set_header Host $host;
        proxy_read_timeout 300s;
    }
}
EOF

curl http://90.2.2.2/completion \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Say hi in one sentence:",
    "n_predict": 50
  }'


ln -s /etc/nginx/sites-available/llama /etc/nginx/sites-enabled/
nginx -t
systemctl restart nginx
```

Test from your local machine:

```bash
curl http://YOUR_DROPLET_IP/health
```

---

## 7. TLS (requires a domain pointed at the droplet)

```bash
certbot --nginx -d your-domain.com
```

---

## 8. Basic API key protection

nginx has no built-in auth, so add a header check inside the `location /` block, before `proxy_pass`:

```nginx
if ($http_authorization != "Bearer YOUR_SECRET_KEY") {
    return 401;
}
```

Reload:

```bash
nginx -t && systemctl reload nginx
```

---

## 9. Final verification

```bash
curl https://your-domain.com/health -H "Authorization: Bearer YOUR_SECRET_KEY"

curl https://your-domain.com/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_SECRET_KEY" \
  -d '{
    "model": "qwen2.5-1.5b-instruct",
    "messages": [{"role": "user", "content": "Say hi in one sentence"}]
  }'
```

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `systemctl status` shows failed | Wrong path in `ExecStart` or model file missing | `journalctl -u llama-server -n 50` for the real error |
| OOM / process killed | RAM too tight for context size + parallel slots | Lower `-c` or `--parallel`, confirm swap is active (`swapon --show`) |
| `curl` to `/health` hangs | nginx not proxying, or llama-server not listening on 127.0.0.1:8080 | Check `systemctl status llama-server` and `nginx -t` |
| 401 on every request | API key header missing or mismatched | Confirm `Authorization: Bearer ...` matches nginx config exactly |
| Slow first response | Model load time, not a bug | Expected — subsequent requests are faster |

## Useful commands

```bash
systemctl status llama-server      # service state
journalctl -u llama-server -f      # live logs
curl localhost:8080/metrics        # Prometheus metrics
free -h                            # RAM + swap usage
```