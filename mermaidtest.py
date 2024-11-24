from playwright.sync_api import sync_playwright
import uuid
def save_mermaid_diagram(mermaid_code: str, output_dir: str = "./") -> str:
    """
    將 Mermaid 語法渲染成圖片並保存到本地。

    Args:
        mermaid_code (str): Mermaid 語法內容。
        output_dir (str): 圖片保存的資料夾。

    Returns:
        str: 生成的圖片路徑。
    """
    # 確保保存的資料夾存在
    import os
    os.makedirs(output_dir, exist_ok=True)

    # 生成唯一的圖片名稱
    image_path = f"{output_dir}/{uuid.uuid4().hex}.png"

    # Mermaid 的 HTML 模板
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

    # 使用 Playwright 渲染
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.set_content(html_template)
        # 截圖並保存
        # 等待圖表渲染完成
        # 等待圖表渲染完成
        page.wait_for_selector("#diagram")

        # # 獲取圖表的邊界框大小
        # bounding_box = page.locator("#diagram").bounding_box()
        # if bounding_box:
        #     # 設置視窗大小匹配圖表大小
        #     page.set_viewport_size({
        #         "width": int(bounding_box["width"]),
        #         "height": int(bounding_box["height"]),
        #     })

        #     # 截圖並保存
        #     page.locator("#diagram").screenshot(path=image_path)
        #     print(f"Diagram saved to: {image_path}")
        # else:
        #     print("Failed to determine diagram size.")

        # 對特定元素截圖
        element = page.locator("#diagram")
        element.screenshot(path=image_path)

        browser.close()

    print(f"Diagram saved to: {image_path}")
    return image_path


# 測試範例
mermaid_code_example = """
quadrantChart
    title Reach and engagement of campaigns
    x-axis Low Reach --> High Reach
    y-axis Low Engagement --> High Engagement
    quadrant-1 We should expand
    quadrant-2 Need to promote
    quadrant-3 Re-evaluate
    quadrant-4 May be improved
    Campaign A: [0.3, 0.6]
    Campaign B: [0.45, 0.23]
    Campaign C: [0.57, 0.69]
    Campaign D: [0.78, 0.34]
    Campaign E: [0.40, 0.34]
    Campaign F: [0.35, 0.78]
"""

# 呼叫函數並保存圖片
save_mermaid_diagram(mermaid_code_example)