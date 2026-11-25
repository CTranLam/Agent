"""Kiểm tra giao diện trên Chrome thật; tự bỏ qua nếu máy không có playwright/Chrome."""
import threading, time
import pytest
import uvicorn
from diabetes_mas.agents.orchestrator import Orchestrator
from diabetes_mas.api import create_app
from diabetes_mas.kb.rag import RAG
from diabetes_mas.kb.store import load_kb
from tests.fakes import FakePredictor

sync_api = pytest.importorskip("playwright.sync_api")
PORT = 8765


class Fixed:
    def suggest(self, g, t, i): return 2.0


@pytest.fixture(scope="module")
def page():
    kb = load_kb(); rag = RAG(kb)
    server = uvicorn.Server(uvicorn.Config(create_app(Orchestrator(FakePredictor(), kb, rag, Fixed()), rag), port=PORT, log_level="error"))
    threading.Thread(target=server.run, daemon=True).start()
    while not server.started:
        time.sleep(.05)
    with sync_api.sync_playwright() as p:
        try:
            b = p.chromium.launch(channel="chrome", headless=True)
        except Exception as e:
            pytest.skip(f"không mở được Chrome: {e}")
        pg = b.new_page(viewport={"width": 1280, "height": 900})
        pg.goto(f"http://127.0.0.1:{PORT}")
        pg.wait_for_selector("#scenario option:nth-child(9)", state="attached")
        yield pg
        b.close()
    server.should_exit = True


def run_case(pg, key):
    pg.select_option("#scenario", key)
    pg.wait_for_function("document.querySelector('.big') && !document.querySelector('#go').disabled")
    pg.wait_for_timeout(150)


def test_tabs_hide_each_other(page):
    page.click("#tab-g"); page.wait_for_selector("#graph svg")
    assert not page.is_visible("#form")
    page.click("#tab-a")
    assert page.is_visible("#form") and not page.is_visible("#graph")


def test_editing_current_overrides_hidden_series(page):  # I10
    run_case(page, "4")
    page.fill("#current", "60"); page.click("#go")
    page.wait_for_function("document.querySelector('.big').firstChild.textContent === '60'")
    assert page.input_value("#series") == ""


def test_decimal_comma_series_in_browser(page):  # C1
    page.evaluate("document.querySelector('details').open = true")
    page.select_option("#unit", "mmol/L")
    page.fill("#series", "2,9"); page.click("#go")
    page.wait_for_function("document.querySelector('.big').firstChild.textContent === '52'")
    page.select_option("#unit", "mg/dL"); page.fill("#series", "")


def test_blocked_dose_shows_reason_without_tile(page):  # I1
    run_case(page, "4")
    page.evaluate("document.querySelector('details').open = true")
    page.fill("#series", ""); page.fill("#current", "300"); page.click("#go")
    page.wait_for_function("document.querySelector('.big').firstChild.textContent === '300'")
    page.wait_for_timeout(100)
    text = page.inner_text(".agents")
    assert "Không đề xuất liều bù" in text and "RLAgent đề xuất" not in text and "RuleAgent quyết định: không bù" in text


def test_kb_unreviewed_flag_visible(page):  # I13
    page.click("#fab")
    assert "chưa thẩm định" in page.inner_text("#chat")
    page.click("#close")
    run_case(page, "1")
    assert "chưa thẩm định" in page.inner_text("#out")


def test_diagnosis_card_shown_first(page):
    run_case(page, "3")
    assert page.locator(".agents .agent h3").first.inner_text().split() == "DiagnosisAgent Chẩn đoán Deep learning + Rule".split()
    assert "DiabetesNet" in page.locator(".agents .agent").first.inner_text()
