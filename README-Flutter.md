# AgriSmart Mobile App 🌾 (Flutter + FastAPI)

A modern, cross-platform mobile agricultural application built with **Flutter** (Android, iOS, Web, macOS) paired with a high-performance **FastAPI backend**.

It offers **100% functional parity** with the original Streamlit application:
- 🌾 **Multi-Crop Selection** across all 10 Jalgaon commodities (*Bajra, Banana, Bengal Gram, Cotton, Groundnut, Jowar, Lime, Mousambi, Soyabean, Wheat*).
- 🌤️ **Live Agronomic Weather Advisory (14 Days)** via Open-Meteo with 6 key agronomic indices (Avg Temp, Humidity, Rain & Peak Probability, Wind Speed, Total Evapotranspiration ET₀, Soil Moisture at 0-1cm) plus an expandable date-wise 14-day daily breakdown.
- 📈 **Recursive Multi-Month Crop Price Forecasting** powered by pre-trained Scikit-Learn `.pkl` pipelines and live + historical Jalgaon weather features.
- 🏪 **Recent Market Data Bridge**: Toggling custom recent prices (1 month ago, 7 days ago, current price, year, month) to bridge dataset dates up to the present date for maximum prediction accuracy.
- ⏱️ **Target Forecast Horizon**: Customizable prediction year (up to 2028) and target month.
- 📊 **Dual Visualization Modes**: Interactive touch-enabled trend line charts (Historical vs Predicted) + tabular monthly breakdown data grids with status badge pills.
- 💬 **Farm AI Assistant**: Powered by Groq LLM (`qwen/qwen3.8-27b` with automatic fallback to `llama-3.3-70b-versatile`), providing tailored advice in the farmer's language based on real-time weather and forecast data.
- 🌐 **Full Bilingual Support**: Complete 1:1 translation for every screen element between **English** and **मराठी (Marathi)**.
- 🛡️ **Intelligent Network Resilience & Demo Mode**: Auto-detects server addresses (`10.0.2.2` for Android emulators, `127.0.0.1` for Web/iOS/Desktop, or custom Wi-Fi LAN IPs) with a live connection tester and an interactive Demo Mode fallback if the server is offline.

---

## Architecture Overview

```
AgriSmart/
├── backend/                  # Python FastAPI Backend
│   ├── main.py               # REST API endpoints with CORS & ML prediction logic
│   ├── requirements.txt      # Python dependencies (FastAPI, scikit-learn, groq, etc.)
│   └── run.sh                # Launches uvicorn on 0.0.0.0:8000 with auto-reload
├── data/                     # Historical Jalgaon monthly commodity & weather datasets
├── models/                   # Pre-trained scikit-learn .pkl models per crop
└── mobile/                   # Cross-Platform Flutter Application
    ├── lib/main.dart         # Material 3 Mobile UI with fl_chart, chat, & translations
    └── pubspec.yaml          # Flutter dependencies
```

---

## Quick Start Guide

### Step 1: Start the FastAPI Backend
Ensure your Python environment is active, then launch the backend:
```bash
cd backend
./run.sh
```
The server will start on `http://0.0.0.0:8000`. You can verify by visiting:
```
http://127.0.0.1:8000/api/health
```

### Step 2: Run the Flutter App
In a new terminal window:
```bash
cd mobile
flutter pub get
```

Run on your desired platform:
- **macOS Desktop**:
  ```bash
  flutter run -d macos
  ```
- **Web Browser (Chrome)**:
  ```bash
  flutter run -d chrome
  ```
- **Android Emulator**:
  ```bash
  flutter run
  ```
  *(The app automatically points to `http://10.0.2.2:8000/api` on Android)*
- **Physical Phone (via Wi-Fi or USB)**:
  Tap the top-right Settings gear ⚙️ icon and set the Server URL to your computer's LAN IP (e.g. `http://10.226.135.62:8000/api`), then tap **Test Connection**.

---

## Network Connection Presets in Settings
Inside the app Settings dialog (⚙️ icon in the top right), you can quickly pick:
1. **Android Emulator**: `http://10.0.2.2:8000/api`
2. **Localhost (iOS / Web / macOS)**: `http://127.0.0.1:8000/api`
3. **Wi-Fi LAN (Physical Devices)**: `http://<your-computer-ip>:8000/api`
4. **Test Connection Button**: Immediately tests latency and connectivity to the backend.
