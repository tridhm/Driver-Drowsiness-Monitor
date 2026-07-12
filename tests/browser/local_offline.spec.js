const { test, expect } = require('@playwright/test');

function isLocalRequest(url) {
  if (url.startsWith('blob:') || url.startsWith('data:')) return true;
  const parsed = new URL(url);
  return ['127.0.0.1', 'localhost', '[::1]', '::1'].includes(parsed.hostname);
}

function expectNoMediaBody(payload) {
  const json = JSON.stringify(payload).toLowerCase();
  expect(json).not.toContain('jpeg');
  expect(json).not.toContain('image');
  expect(json).not.toContain('video');
  expect(json).not.toContain('base64');
  expect(json).not.toContain('data:');
}

test.beforeEach(async ({ page }) => {
  page.consoleErrors = [];
  page.externalRequests = [];
  page.faceMeshAssetRequests = [];
  page.framePayloads = [];
  page.requestBodies = [];

  page.on('console', (message) => {
    if (message.type() === 'error') page.consoleErrors.push(message.text());
  });

  await page.route('**/*', async (route) => {
    const request = route.request();
    const url = request.url();
    if (!isLocalRequest(url)) {
      page.externalRequests.push(url);
      await route.abort();
      return;
    }
    const parsed = new URL(url);
    if (parsed.pathname.startsWith('/static/vendor/face_mesh/')) {
      page.faceMeshAssetRequests.push(url);
    }
    const postData = request.postData();
    if (postData) {
      page.requestBodies.push({ url, body: postData });
    }
    if (request.method() === 'POST' && /\/api\/v1\/sessions\/[^/]+\/frames$/.test(new URL(url).pathname)) {
      page.framePayloads.push(request.postDataJSON());
    }
    await route.continue();
  });
});

test('fake camera uses self-hosted MediaPipe and sends landmark-only JSON offline', async ({ page }) => {
  await page.goto('/');
  await expect(page.locator('#runtimeLocation')).toHaveText('LOCAL');

  await page.locator('#startBtn').click();
  await expect(page.locator('#topStatus')).toContainText('MediaPipe ready', { timeout: 45_000 });
  await expect.poll(() => page.framePayloads.length, { timeout: 30_000 }).toBeGreaterThan(0);

  expect(page.faceMeshAssetRequests).toEqual(expect.arrayContaining([
    expect.stringMatching(/\/static\/vendor\/face_mesh\/face_mesh\.js$/),
  ]));
  expect(page.faceMeshAssetRequests.some((url) => /\.(wasm|binarypb|data)$/.test(url))).toBe(true);

  for (const payload of page.framePayloads) {
    expect(Object.keys(payload).sort()).toEqual(['batch_seq', 'frames']);
    expect(Array.isArray(payload.frames)).toBe(true);
    expect(payload.frames.length).toBeGreaterThan(0);
    expectNoMediaBody(payload);

    for (const frame of payload.frames) {
      const expectedKeys = ['face_detected', 'height', 'seq', 'timestamp_ms', 'width'];
      if (frame.face_detected) expectedKeys.push('landmarks');
      expect(Object.keys(frame).sort()).toEqual(expectedKeys.sort());
      if (frame.face_detected) {
        expect(Object.keys(frame.landmarks)).toHaveLength(20);
      } else {
        expect(frame).not.toHaveProperty('landmarks');
      }
    }
  }

  expect(page.externalRequests).toEqual([]);
  expect(page.consoleErrors).toEqual([]);
});

test('file mode creates a local object URL and never uploads media bodies', async ({ page }) => {
  const fixtureMarker = 'local-browser-only-fixture-7f3a';

  await page.route('**/static/vendor/face_mesh/face_mesh.js', async (route) => {
    await route.fulfill({
      contentType: 'application/javascript',
      body: `
        window.FaceMesh = class {
          constructor() { this.callback = () => {}; }
          setOptions() {}
          onResults(callback) { this.callback = callback; }
          async initialize() {}
          async send() { this.callback({ multiFaceLandmarks: [] }); }
        };
      `,
    });
  });

  await page.addInitScript(() => {
    window.__objectUrls = [];
    const nativeCreateObjectURL = URL.createObjectURL.bind(URL);
    URL.createObjectURL = (value) => {
      const objectUrl = nativeCreateObjectURL(value);
      window.__objectUrls.push(objectUrl);
      return objectUrl;
    };

    Object.defineProperty(HTMLMediaElement.prototype, 'readyState', { configurable: true, get: () => 4 });
    Object.defineProperty(HTMLVideoElement.prototype, 'videoWidth', { configurable: true, get: () => 320 });
    Object.defineProperty(HTMLVideoElement.prototype, 'videoHeight', { configurable: true, get: () => 240 });
    HTMLMediaElement.prototype.load = function load() {
      setTimeout(() => {
        this.dispatchEvent(new Event('loadedmetadata'));
        this.dispatchEvent(new Event('canplay'));
      }, 0);
    };
    HTMLMediaElement.prototype.play = async function play() {
      this.dispatchEvent(new Event('loadedmetadata'));
      this.dispatchEvent(new Event('canplay'));
    };
    HTMLMediaElement.prototype.pause = function pause() {};
  });

  await page.goto('/mobile?mode=file');
  await expect(page.locator('#runtimeLocation')).toHaveText('LOCAL');
  await page.locator('#videoFile').setInputFiles({
    name: 'synthetic.mp4',
    mimeType: 'video/mp4',
    buffer: Buffer.from(fixtureMarker),
  });

  await expect.poll(() => page.evaluate(() => window.__objectUrls.length), { timeout: 15_000 }).toBeGreaterThan(0);
  for (const requestBody of page.requestBodies) {
    expect(requestBody.url).not.toContain('/api/upload');
    expect(requestBody.body.toLowerCase()).not.toContain('video');
    expect(requestBody.body.toLowerCase()).not.toContain('base64');
    expect(requestBody.body).not.toContain(fixtureMarker);
  }
  expect(page.externalRequests).toEqual([]);
  expect(page.consoleErrors).toEqual([]);
});

for (const viewport of [
  { name: 'desktop', width: 1280, height: 720 },
  { name: 'mobile', width: 375, height: 844 },
]) {
  test(`${viewport.name} layout does not overflow horizontally`, async ({ page }) => {
    await page.setViewportSize({ width: viewport.width, height: viewport.height });
    await page.goto('/');
    await expect(page.locator('#runtimeLocation')).toHaveText('LOCAL');
    await expect(page.locator('#runtimeLocation')).toBeVisible();

    const layout = await page.evaluate(() => {
      const controls = document.querySelector('.controls');
      const badge = document.querySelector('#runtimeLocation');
      const rect = (element) => {
        const box = element.getBoundingClientRect();
        return { left: box.left, right: box.right, top: box.top, bottom: box.bottom, width: box.width, height: box.height };
      };
      return {
        innerWidth: window.innerWidth,
        bodyScrollWidth: document.body.scrollWidth,
        controlsScrollWidth: controls.scrollWidth,
        controlsClientWidth: controls.clientWidth,
        badge: rect(badge),
      };
    });

    expect(layout.bodyScrollWidth).toBeLessThanOrEqual(layout.innerWidth);
    expect(layout.controlsScrollWidth).toBeLessThanOrEqual(layout.controlsClientWidth);
    expect(layout.badge.width).toBeGreaterThan(0);
    expect(layout.badge.height).toBeGreaterThan(0);
    expect(layout.badge.left).toBeGreaterThanOrEqual(0);
    expect(layout.badge.right).toBeLessThanOrEqual(layout.innerWidth);
    expect(page.externalRequests).toEqual([]);
    expect(page.consoleErrors).toEqual([]);
  });
}
