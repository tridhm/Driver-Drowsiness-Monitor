# Driver Drowsiness Monitor Winner Web

This repository contains the canonical winner web runtime for driver drowsiness monitoring. The browser runs the self-hosted MediaPipe FaceMesh bundle, keeps camera and video pixels client-side, and sends only normalized 20-landmark JSON to Flask.

Flask applies the protected feature pipeline and serves the winner model through `camera_hybrid` with the deployed `recommended` profile. The web app is the production runtime; older OpenCV and Python desktop entry points are research or legacy paths.

## Run locally

On Windows, double-click `run_local.cmd`. For command-line options, ports, LAN mode, privacy notes, and first-run Python 3.12 setup, see [README_RUN.md](README_RUN.md).

## Deploy

Production deploys run `web_server:app`. Render deployment notes are in [README_DEPLOY_RENDER.md](README_DEPLOY_RENDER.md).
