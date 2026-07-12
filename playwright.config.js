const { defineConfig, devices } = require('@playwright/test');
const fs = require('fs');
const path = require('path');

const venvPython = path.join(__dirname, '.venv', 'Scripts', 'python.exe');
const python = process.platform === 'win32' && fs.existsSync(venvPython) ? venvPython : 'python';

module.exports = defineConfig({
  testDir: 'tests/browser',
  timeout: 60_000,
  workers: 1,
  fullyParallel: false,
  reporter: 'line',
  use: {
    baseURL: 'http://127.0.0.1:5011',
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
    command: `"${python}" local_app.py --no-browser --port 5011`,
    url: 'http://127.0.0.1:5011/api/healthz',
    timeout: 120_000,
    reuseExistingServer: !process.env.CI,
  },
});
