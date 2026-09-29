/**
 * Generate a PDF from an HTML file using headless Chromium.
 *
 * Usage:
 *   pnpm tsx generate-pdf.ts <html-path> <pdf-path>
 */

import puppeteer from 'puppeteer-core';
import { resolve } from 'path';

const [,, htmlPath, pdfPath] = process.argv;

if (!htmlPath || !pdfPath) {
  console.error('Usage: tsx generate-pdf.ts <html-path> <pdf-path>');
  process.exit(1);
}

const browser = await puppeteer.launch({
  executablePath: '/usr/bin/chromium',
  args: ['--no-sandbox', '--disable-setuid-sandbox'],
});

const page = await browser.newPage();
await page.setViewport({ width: 900, height: 1200 });
await page.goto(`file://${resolve(htmlPath)}`, { waitUntil: 'networkidle0' });

await page.pdf({
  path: pdfPath,
  format: 'Letter',
  margin: { top: '0.5in', right: '0.5in', bottom: '0.5in', left: '0.5in' },
  printBackground: true,
});

await browser.close();
console.log(`PDF saved to ${pdfPath}`);
