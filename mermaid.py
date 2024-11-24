from playwright.sync_api import sync_playwright
def generate_mermaid_quadrantChart(mermaid_code: str):
    html_template = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
    </head>
    <body>
        <div id="diagram" class="mermaid" style="width: auto; height: auto; justify-content: center;">
            {mermaid_code}
        </div>
        <script>
            mermaid.initialize({{ startOnLoad: true }});
        </script>
    </body>
    </html>
    """
    with sync_playwright() as temphtml:
        browser = temphtml.chromium.launch()
        page = browser.new_page()
        page.set_content(html_template)
        page.wait_for_selector("#diagram")
        diagram = page.locator("#diagram")
        diagram_bytes = diagram.screenshot()
        browser.close()
        return diagram_bytes