import json
import time
import os
from DrissionPage import ChromiumPage

class XiaoHongShuSpider:
    def __init__(self):
        self.page = ChromiumPage()
        # self.start_url = "https://www.xiaohongshu.com/explore?channel_id=homefeed.travel_v3"
        self.start_url = "https://www.xiaohongshu.com/search_result?keyword=%25E6%2594%25BB%25E7%2595%25A5&source=web_explore_feed"
        self.json_file = "xiaohongshu_data.json"
        self.visited_urls, self.results = self.load_existing_data()

    def load_existing_data(self):
        """加载已有的 JSON 数据，避免重复爬取"""
        if os.path.exists(self.json_file):
            with open(self.json_file, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                    visited_urls = {item["url"] for item in data}
                    print(f"🔄 已加载 {len(data)} 条数据，将继续爬取新的帖子...")
                    return visited_urls, data
                except json.JSONDecodeError:
                    print("⚠️ JSON 文件格式错误，重建数据...")
                    return set(), []
        return set(), []

    def save_data(self, data):
        """实时保存数据到 JSON 文件"""
        with open(self.json_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"💾 数据已实时保存，共 {len(data)} 条记录")

    def scroll_down_and_parse(self, max_scrolls=20):
        self.parse_page()
        """向下滚动页面，并解析新加载的数据"""
        for i in range(max_scrolls):
            print(f"🔄 正在滚动第 {i + 1} 次...")
            self.page.scroll.down(500)
            time.sleep(3)
            self.parse_page()

    def parse_post(self, post_element, post_url):
        """点击帖子封面图，打开详情页，获取内容，并关闭页面"""
        try:
            print("🖼️ 点击封面图打开帖子详情...")
            post_element.click()
            time.sleep(3)  # 等待详情弹窗加载

            # **等待详情内容加载**
            content_xpath = '//*[@id="detail-desc"]'
            content_ele = self.page.ele(f'xpath:{content_xpath}')
            content = content_ele.text if content_ele else "🚨 未找到帖子内容"
            print(f"📖 帖子内容: {content[:50]}...")
            post_data = {"content": content, "url": post_url}
            self.visited_urls.add(post_url)
            self.results.append(post_data)
            self.save_data(self.results)
            # **正确选择关闭按钮**
            close_xpath = '//div[contains(@class, "close-mask-dark")]'
            close_ele = self.page.ele(f'xpath:{close_xpath}')
            if close_ele:
                print("❌ 关闭帖子弹窗...")
                close_ele.click()
                time.sleep(2)

        except Exception as e:
            print(f"❌ 解析帖子出错: {e}")
            content = "❌ 解析失败"

        return content

    def parse_page(self):
        """解析当前页面的所有帖子"""
        posts = self.page.eles('xpath://a[contains(@class, "cover")]')
        print(f"🔍 当前页面找到 {len(posts)} 篇帖子")

        for post in posts:
            try:
                post_url = post.attr('href') if post else None

                if not post_url or post_url in self.visited_urls:
                    print(f"🚨 {post_url} 已经爬取过，跳过...")
                    continue


                self.parse_post(post, post_url)  # 传入封面图片的 element，进行点击

                # post_data = {"content": content, "url": post_url}
                # self.results.append(post_data)
                # self.save_data(self.results)

            except Exception as e:
                print(f"❌ 解析帖子出错: {e}")

    def run(self):
        """执行爬虫"""
        print("🚀 直接打开已登录的 Chrome 会话...")
        self.page.get(self.start_url)
        time.sleep(5)

        print("🔄 开始滚动页面并解析数据...")
        self.scroll_down_and_parse(max_scrolls=100)

        print(f"✅ 爬取完成，最终共 {len(self.results)} 条数据")

if __name__ == "__main__":
    spider = XiaoHongShuSpider()
    spider.run()
