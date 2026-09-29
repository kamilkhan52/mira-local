/**
 * Generate a PDF from an HTML file using headless Chromium (n8n "Generate PDF").
 *
 * Usage:
 *   npx tsx generate-pdf.ts <html-path> <pdf-path>
 *
 * Browser: PUPPETEER_EXECUTABLE_PATH (or CHROME_PATH) when set, else the first
 * of /usr/bin/chromium, /usr/bin/chromium-browser, /usr/bin/google-chrome,
 * Google Chrome.app, Chromium.app. Exits non-zero on any failure; the Python
 * caller (mira.report.generate_pdf) treats that as "no PDF" and carries on.
 */

import puppeteer from 'puppeteer-core';
import { existsSync } from 'fs';
import { resolve } from 'path';

const [, , htmlPath, pdfPath] = process.argv;

if (!htmlPath || !pdfPath) {
  console.error('Usage: tsx generate-pdf.ts <html-path> <pdf-path>');
  process.exit(1);
}

function findBrowser(): string {
  const fromEnv = [process.env.PUPPETEER_EXECUTABLE_PATH, process.env.CHROME_PATH];
  for (const p of fromEnv) {
    if (p && existsSync(p)) return p;
  }
  const candidates = [
    '/usr/bin/chromium',
    '/usr/bin/chromium-browser',
    '/usr/bin/google-chrome',
    '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
    '/Applications/Chromium.app/Contents/MacOS/Chromium',
  ];
  for (const p of candidates) {
    if (existsSync(p)) return p;
  }
  throw new Error('Chrome/Chromium not found. Set PUPPETEER_EXECUTABLE_PATH.');
}

async function main(): Promise<void> {
  if (!existsSync(htmlPath)) throw new Error(`HTML file not found: ${htmlPath}`);
  const browser = await puppeteer.launch({
    executablePath: findBrowser(),
    args: ['--no-sandbox', '--disable-setuid-sandbox'],
  });
  try {
    const page = await browser.newPage();
    await page.setViewport({ width: 900, height: 1200 });
    await page.goto(`file://${resolve(htmlPath)}`, { waitUntil: 'networkidle0', timeout: 120_000 });
    await page.pdf({
      path: pdfPath,
      format: 'Letter',
      margin: { top: '0.5in', right: '0.5in', bottom: '0.5in', left: '0.5in' },
      printBackground: true,
    });
  } finally {
    await browser.close();
  }
  console.log(`PDF saved to ${pdfPath}`);
}

main().catch((err) => {
  console.error(`generate-pdf failed: ${err instanceof Error ? err.message : String(err)}`);
  process.exit(1);
});
