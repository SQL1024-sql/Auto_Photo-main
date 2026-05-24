"""
壓力測試腳本 — 針對 Auto_Photo Flask 應用

安裝: pip install locust Pillow
執行: locust -f locustfile.py --host=http://localhost:5000
然後開瀏覽器: http://localhost:8089
"""
import io
import logging
from locust import HttpUser, task, between
from PIL import Image

# 抑制 Windows 上 gevent 關閉時的 greenlet 噪音
logging.getLogger("gevent").setLevel(logging.CRITICAL)


def _make_image_bytes(width, height):
    img = Image.new("RGB", (width, height), color=(100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# 啟動時預先產生，所有 user 共用同一份 bytes（避免每次 task 重新編碼 block event loop）
_STRIP_BYTES = _make_image_bytes(2556, 1200)
_COVER_BYTES = _make_image_bytes(3000, 1500)
_ANCHOR_BYTES = _make_image_bytes(300, 100)


class PhotoUser(HttpUser):
    """模擬一位使用者的操作流程"""
    wait_time = between(1, 3)  # 每個 task 之間等 1~3 秒

    def on_start(self):
        self.client.post(
            "/upload_anchor",
            files={"anchor": ("anchor.png", io.BytesIO(_ANCHOR_BYTES), "image/png")},
            data={"width": "2556"},
            name="/upload_anchor [setup]",
        )

    @task(1)
    def get_homepage(self):
        """輕量請求：首頁"""
        self.client.get("/")

    @task(1)
    def get_sort_tags(self):
        """輕量請求：取得標籤列表"""
        self.client.get("/sort_tags_list")

    @task(5)
    def upload_strip(self):
        self.client.post(
            "/upload_strip",
            files={"strip": ("strip.png", io.BytesIO(_STRIP_BYTES), "image/png")},
            data={"y_top": "300", "width": "2556"},
            name="/upload_strip",
        )

    @task(2)
    def upload_cover(self):
        resp = self.client.post(
            "/upload_cover",
            files={"cover": ("cover.png", io.BytesIO(_COVER_BYTES), "image/png")},
            name="/upload_cover",
        )
        if resp.ok:
            try:
                self._last_cover = resp.json().get("filename")
            except Exception:
                pass

    @task(3)
    def generate(self):
        """重量請求：合成最終圖片"""
        self.client.post(
            "/generate",
            json={
                "cover": getattr(self, "_last_cover", None),
                "cells": [None] * 5,
                "grid_rows": 1,
                "grid_cols": 5,
                "width": 2556,
            },
            name="/generate",
        )
