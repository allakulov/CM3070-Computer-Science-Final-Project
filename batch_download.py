"""Download documents for multiple procurements from eis.gov.lv.

Reads procurement page URLs from the output of get_procurement_ids.py.

Usage::

    python get_procurement_ids.py --date 2024-06-15
    python batch_download.py
    python batch_download.py --file data/document_urls_2024-06-15.json
    python batch_download.py --urls "https://www.eis.gov.lv/EKEIS/Supplier/Procurement/98405"
"""

import asyncio
import json
import random
import re
import zipfile
from pathlib import Path
from playwright.async_api import async_playwright

OUTPUT_DIR = Path("downloads")
HEADLESS = False


def load_urls(filepath=None, urls_arg=None):
    """Load procurement URLs from a JSON file or command line argument."""
    if urls_arg:
        return [u.strip() for u in urls_arg.split(",")]

    if filepath:
        path = Path(filepath)
    else:
        # Find the most recent document_urls file in data/
        data_dir = Path("data")
        candidates = sorted(data_dir.glob("document_urls_*.json"), reverse=True)
        if not candidates:
            print("No document_urls_*.json found in data/. Run get_procurement_ids.py first.")
            return []
        path = candidates[0]

    print(f"Reading URLs from {path}")
    with open(path) as f:
        return json.load(f)


def extract_id_from_url(url):
    """Extract the numeric procurement ID from an EIS URL."""
    match = re.search(r"/Procurement/(\d+)", url)
    return match.group(1) if match else url.split("/")[-1]


async def download_procurement(page, url):
    """Download all documents for one procurement page."""
    eis_id = extract_id_from_url(url)
    out = OUTPUT_DIR / eis_id
    out.mkdir(parents=True, exist_ok=True)

    try:
        response = await page.goto(url, wait_until="networkidle", timeout=30000)
        if not response or response.status >= 400:
            print(f"page error (status {response.status if response else 'None'})")
            return []
    except Exception as e:
        print(f"navigation error: {e}")
        return []

    # Remove cookie banner
    await page.evaluate("""() => {
        document.querySelectorAll(
            '[class*="cookie"], [id*="cookie"], [class*="consent"], [id*="consent"]'
        ).forEach(el => el.remove());
        for (const btn of document.querySelectorAll('button, a'))
            if (btn.textContent.trim() === 'Got it') { btn.click(); break; }
    }""")
    await page.wait_for_timeout(1000)

    # Expand documents section
    for selector in [
        'text=Documents (actuals)',
        'text=Dokumenti (aktuālie)',
        'text=Documents',
        'text=Dokumenti',
    ]:
        try:
            el = await page.wait_for_selector(selector, timeout=2000)
            await el.click()
            await page.wait_for_timeout(2000)
            break
        except Exception:
            continue
    else:
        try:
            await page.evaluate("""() => {
                for (const el of document.querySelectorAll('*')) {
                    const t = el.textContent.toLowerCase().trim();
                    if ((t.startsWith('documents') || t.startsWith('dokumenti'))
                         && el.offsetParent !== null && el.offsetHeight > 0) {
                        el.click(); return true;
                    }
                }
                return false;
            }""")
            await page.wait_for_timeout(2000)
        except Exception:
            print("could not expand documents section")
            await page.screenshot(path=out / "debug_no_expand.png", full_page=True)
            return []

    # Find download icons
    icons = await page.query_selector_all('a[title*="Download"]')
    visible = [ic for ic in icons if await ic.is_visible()]
    if not visible:
        icons = await page.query_selector_all('a[title*="Lejupielādēt"]')
        visible = [ic for ic in icons if await ic.is_visible()]
    if not visible:
        print("no download icons found")
        await page.screenshot(path=out / "debug_no_icons.png", full_page=True)
        return []

    total = len(visible)
    downloaded = []

    for i in range(total):
        icons = await page.query_selector_all('a[title*="Download"]')
        vis = [ic for ic in icons if await ic.is_visible()]
        if not vis:
            icons = await page.query_selector_all('a[title*="Lejupielādēt"]')
            vis = [ic for ic in icons if await ic.is_visible()]
        if i >= len(vis):
            continue

        try:
            await vis[i].click()
        except Exception:
            continue
        await page.wait_for_timeout(2000)

        zip_btn = None
        for sel in [
            'button:has-text("Download all files")',
            'button:has-text("Lejupielādēt visas datnes")',
            'button#uxDownloadZip',
            'button:has-text(".zip")',
        ]:
            try:
                btn = await page.wait_for_selector(sel, timeout=3000)
                if btn and await btn.is_visible():
                    zip_btn = btn
                    break
            except Exception:
                continue

        if zip_btn:
            try:
                async with page.expect_download(timeout=30000) as dl_info:
                    await zip_btn.click()
                download = await dl_info.value
                filename = download.suggested_filename or f"doc_{i}.zip"
                await download.save_as(out / filename)
                downloaded.append(filename)
                await page.wait_for_timeout(1000)
            except Exception:
                pass

        try:
            close = await page.query_selector('.modal .close, button[data-dismiss="modal"]')
            if close and await close.is_visible():
                await close.click()
                await page.wait_for_timeout(500)
        except Exception:
            await page.keyboard.press("Escape")
            await page.wait_for_timeout(500)

    return downloaded


async def main():
    import sys
    filepath = None
    urls_arg = None
    for i, arg in enumerate(sys.argv[1:]):
        if arg == "--file" and i + 1 < len(sys.argv[1:]):
            filepath = sys.argv[i + 2]
        elif arg == "--urls" and i + 1 < len(sys.argv[1:]):
            urls_arg = sys.argv[i + 2]

    urls = load_urls(filepath, urls_arg)
    if not urls:
        return

    print(f"Batch download: {len(urls)} procurements")
    print(f"Headless: {HEADLESS}")
    print()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {}

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=HEADLESS)
        context = await browser.new_context(
            accept_downloads=True,
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
        )
        page = await context.new_page()

        for idx, url in enumerate(urls, 1):
            eis_id = extract_id_from_url(url)
            print(f"[{idx}/{len(urls)}] {eis_id}... ", end="", flush=True)
            downloaded = await download_procurement(page, url)
            results[eis_id] = downloaded

            if downloaded:
                files = 0
                for f in downloaded:
                    fpath = OUTPUT_DIR / eis_id / f
                    if zipfile.is_zipfile(fpath):
                        with zipfile.ZipFile(fpath) as z:
                            files += len(z.namelist())
                print(f"{len(downloaded)} ZIPs, {files} files")
            else:
                print("no files")

            if idx < len(urls):
                delay = random.uniform(3, 7)
                await asyncio.sleep(delay)

        await browser.close()

    ok = sum(1 for d in results.values() if d)
    fail = len(results) - ok
    print(f"\nDone: {ok} succeeded, {fail} failed")
    if fail:
        failed = [eid for eid, d in results.items() if not d]
        print(f"Failed: {failed}")
        print("Check downloads/{id}/debug_*.png for screenshots")


if __name__ == "__main__":
    asyncio.run(main())