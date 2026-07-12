# Run the Canonical Winner Web App Locally

This is the canonical localhost winner runtime. It uses the same browser FaceMesh bundle as the deployed app, sends the 20-landmark API payload to the Flask feature pipeline, and runs the winner model through `camera_hybrid` with the `recommended` profile as deployed.

## Windows

Double-click `run_local.cmd` from this directory, or run it from Command Prompt or PowerShell:

```powershell
.\run_local.cmd
```

The first run creates `.venv` with Python 3.12 and installs `requirements.txt`, so initial setup needs internet access unless those dependencies are already available locally. After setup, the self-hosted assets, model, config, and audio can operate without external network access.

The app opens `http://127.0.0.1:5000/`. If port 5000 is busy, the launcher tries 5001 through 5010.

## Cross-platform

On macOS or Linux, create and activate a Python 3.12 virtual environment the first time:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Initial setup needs internet access unless the dependencies are already available locally. Later runs can be offline because the app uses self-hosted assets, model, config, and audio.

Then use Python directly when you are not using the Windows wrapper:

```bash
python local_app.py
python local_app.py --no-browser
python local_app.py --port 5099
python local_app.py --lan
```

By default the app binds to loopback only at `127.0.0.1`. `--lan` binds to the private network with no authentication, so use it only on a trusted private network. Remote camera access over plain HTTP may be blocked by browsers because camera APIs require a secure context, usually HTTPS, outside localhost.

## Privacy

Video stays in the browser. Only normalized landmark JSON goes to local Flask. Raw landmarks, images, and video are not persisted by the Flask app. Flask keeps transient in-memory session state while active for runtime smoothing and decisions. The browser may persist derived event history locally, not raw media or landmarks.

Camera mode runs live in the browser. File mode uses a browser object URL for local playback; the selected file is not uploaded.

## Stop

Press `Ctrl+C` in the terminal that launched the app.

## Boundary

The localhost web app is the canonical offline winner runtime after initial dependencies are installed. Older OpenCV and Python desktop paths are research or legacy tools, not the deployed winner runtime.
