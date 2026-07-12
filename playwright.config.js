const { defineConfig, devices } = require('@playwright/test');
const fs = require('fs');
const path = require('path');

const winVenvPython = path.join(__dirname, '.venv', 'Scripts', 'python.exe');
const posixVenvPython = path.join(__dirname, '.venv', 'bin', 'python');
const python = process.env.PYTHON
  || (fs.existsSync(winVenvPython) ? winVenvPython : null)
  || (fs.existsSync(posixVenvPython) ? posixVenvPython : null)
  || (process.platform === 'win32' ? 'python' : 'python3');

const port = Number(process.env.DMS_PLAYWRIGHT_PORT || 5011);
if (!Number.isInteger(port) || port < 1024 || port > 65535) {
  throw new Error('DMS_PLAYWRIGHT_PORT must be an integer between 1024 and 65535.');
}
const baseURL = `http://127.0.0.1:${port}`;
const quoteArg = (value) => JSON.stringify(value);

module.exports = defineConfig({
  testDir: 'tests/browser',
  timeout: 60_000,
  workers: 1,
  fullyParallel: false,
  reporter: 'line',
  use: {
    baseURL,
    permissions: ['camera'],
    ...devices['Desktop Chrome'],
    launchOptions: {
      args: [
        '--use-fake-device-for-media-stream',
        '--use-fake-ui-for-media-stream',
      ],
    },
  },
  webServer: {
    command: `${quoteArg(python)} local_app.py --no-browser --port ${port}`,
    url: `${baseURL}/api/healthz`,
    timeout: 120_000,
    reuseExistingServer: false,
  },
});
