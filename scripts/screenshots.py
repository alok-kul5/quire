"""Capture README screenshots from a running server (uses the installed Edge, no browser download).

  python scripts/screenshots.py --url http://localhost:8000
"""

from __future__ import annotations

import argparse
from pathlib import Path

from playwright.sync_api import sync_playwright

OUT = Path(__file__).resolve().parent.parent / "docs"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--question", default="What does PEP 572 say about using := at the top level of an expression statement?")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="msedge")
        page = browser.new_page(viewport={"width": 1440, "height": 900}, device_scale_factor=1)
        page.goto(args.url)
        page.wait_for_timeout(2600)  # opening curtain + reveal
        page.screenshot(path=str(OUT / "screenshot-hero.png"))

        page.fill("#q", args.question)
        page.click("#askBtn")
        page.wait_for_selector(".src", timeout=400_000)
        page.wait_for_function("!document.querySelector('#askBtn').disabled", timeout=400_000)
        page.wait_for_timeout(2500)
        page.set_viewport_size({"width": 1440, "height": 1560})
        page.evaluate("window.scrollTo(0, 0)")
        page.wait_for_timeout(500)
        page.screenshot(path=str(OUT / "screenshot-answer.png"))
        page.set_viewport_size({"width": 1440, "height": 900})
        first = page.locator(".cite").first
        if first.count():
            first.hover()
            page.wait_for_timeout(400)
            page.locator("#result").screenshot(path=str(OUT / "screenshot-citations.png"))

        page.locator("#evals").scroll_into_view_if_needed()
        page.wait_for_timeout(1600)
        page.locator("#evals").screenshot(path=str(OUT / "screenshot-evals.png"))

        mobile = browser.new_page(viewport={"width": 390, "height": 844}, device_scale_factor=2)
        mobile.goto(args.url)
        mobile.wait_for_timeout(2600)
        mobile.screenshot(path=str(OUT / "screenshot-mobile.png"))
        browser.close()
    print("saved to", OUT)


if __name__ == "__main__":
    main()
