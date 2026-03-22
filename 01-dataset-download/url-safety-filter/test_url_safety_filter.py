from __future__ import annotations

import importlib.util
import ipaddress
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image
from requests.exceptions import TooManyRedirects


MODULE_DIR = Path(__file__).resolve().parent


def load_module(filename: str, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, MODULE_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


filter_mod = load_module("filter_unsafe_urls.py", "filter_unsafe_urls")
validator_mod = load_module("validate_downloaded_images.py", "validate_downloaded_images")
redirect_mod = load_module("revalidate_url_redirects.py", "revalidate_url_redirects")


def test_parent_domain_blocking_applies_to_subdomains():
    blocked_domains = {"example.com"}

    assert filter_mod.should_block_host("example.com", blocked_domains, [])
    assert filter_mod.should_block_host("img.example.com", blocked_domains, [])
    assert not filter_mod.should_block_host("goodexample.com", blocked_domains, [])


def test_direct_ip_urls_are_blocked_for_non_public_and_matching_cidrs():
    blocked_cidrs = [ipaddress.ip_network("8.8.8.0/24")]

    assert filter_mod.should_block_url("http://127.0.0.1/image.jpg", set(), [])
    assert filter_mod.should_block_url("http://[::1]/image.jpg", set(), [])
    assert filter_mod.should_block_url("http://8.8.8.8/image.jpg", set(), blocked_cidrs)
    assert not filter_mod.should_block_url("http://1.1.1.1/image.jpg", set(), blocked_cidrs)
    assert filter_mod.should_block_url("ftp://safe.example/image.jpg", set(), blocked_cidrs)


def test_blocklist_round_trip_preserves_cidrs(tmp_path):
    blocklist_path = tmp_path / "blocked.txt"
    blocked_domains = {"example.com"}
    blocked_cidrs = [ipaddress.ip_network("8.8.8.0/24")]

    filter_mod.save_blocklist(blocklist_path, blocked_domains, blocked_cidrs)
    loaded_domains, loaded_cidrs = filter_mod.load_blocklist(blocklist_path)

    assert loaded_domains == blocked_domains
    assert {str(net) for net in loaded_cidrs} == {"8.8.8.0/24"}


def test_download_all_blocklists_fails_closed(monkeypatch):
    monkeypatch.setattr(
        filter_mod,
        "BLOCKLIST_SOURCES",
        {
            "bad": {"url": "https://bad.example", "type": "domain"},
            "ok": {"url": "https://ok.example", "type": "domain"},
        },
    )

    def fake_download_blocklist(name, info):
        if name == "bad":
            return name, set(), [], "network down"
        return name, {"ok.example"}, [], None

    monkeypatch.setattr(filter_mod, "download_blocklist", fake_download_blocklist)

    with pytest.raises(RuntimeError, match="bad"):
        filter_mod.download_all_blocklists()


def test_single_file_filter_uses_explicit_output_path(tmp_path):
    input_path = tmp_path / "input.parquet"
    output_path = tmp_path / "custom-name.parquet"

    table = pa.table(
        {
            "url": [
                "https://good.example/image.jpg",
                "http://127.0.0.1/blocked.png",
            ]
        }
    )
    pq.write_table(table, input_path)

    rel, original_rows, kept_rows = filter_mod.filter_parquet(
        (input_path, input_path.parent, None, output_path, "url", set(), [])
    )

    assert rel == "input.parquet"
    assert original_rows == 2
    assert kept_rows == 1
    assert output_path.exists()
    assert pq.read_table(output_path).column("url").to_pylist() == ["https://good.example/image.jpg"]


def test_discover_input_files_scans_mislabeled_files_and_skips_ignored_metadata(tmp_path):
    html_path = tmp_path / "payload.html"
    html_path.write_text("<html>not an image</html>")
    tmp_image_path = tmp_path / "photo.tmp"
    tmp_image_path.write_text("not really an image")
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text("{}")

    discovered = validator_mod.discover_input_files(
        tmp_path,
        ignored_extensions={".json"},
        expected_extensions={".png"},
    )

    assert html_path in discovered
    assert tmp_image_path in discovered
    assert metadata_path not in discovered


def test_validate_image_flags_extension_mismatch(tmp_path):
    image_path = tmp_path / "photo.tmp"
    Image.new("RGB", (128, 128), "red").save(image_path, format="PNG")

    _, reason = validator_mod.validate_image(image_path)

    assert reason is not None
    assert reason.startswith("extension_mismatch")


def test_validate_image_rejects_decompression_bomb_like_inputs(tmp_path, monkeypatch):
    image_path = tmp_path / "bomb.png"
    Image.effect_noise((256, 256), 100).save(image_path, format="PNG")

    monkeypatch.setattr(validator_mod, "MAX_IMAGE_PIXELS", 100)
    _, reason = validator_mod.validate_image(image_path)

    assert reason is not None
    assert reason.startswith("decompression_bomb")


def test_validate_redirect_chain_rejects_unsafe_final_hop():
    ok, reason = redirect_mod.validate_redirect_chain(
        ["https://safe.example/a", "https://bad.example/final"],
        blocked_domains={"bad.example"},
        blocked_cidrs=[],
    )

    assert not ok
    assert reason == "unsafe_redirect_hop_2"


def test_probe_redirect_target_returns_final_url_and_content_type(monkeypatch):
    class FakeResponse:
        def __init__(self, url, headers=None, history=None):
            self.url = url
            self.headers = headers or {}
            self.history = history or []

        def close(self):
            pass

    class FakeSession:
        def __init__(self, response):
            self.response = response

        def get(self, *args, **kwargs):
            return self.response

    response = FakeResponse(
        "https://cdn.safe.example/image.png",
        headers={"Content-Type": "image/png; charset=binary"},
        history=[FakeResponse("https://safe.example/start")],
    )
    monkeypatch.setattr(
        redirect_mod,
        "get_session",
        lambda max_redirects, user_agent: FakeSession(response),
    )

    keep, final_url, content_type, reason = redirect_mod.probe_redirect_target(
        "https://safe.example/start",
        blocked_domains=set(),
        blocked_cidrs=[],
        timeout=(1, 1),
        require_image_content_type=True,
        max_redirects=5,
        user_agent="test-agent",
    )

    assert keep
    assert final_url == "https://cdn.safe.example/image.png"
    assert content_type == "image/png"
    assert reason is None


def test_probe_redirect_target_rejects_unsafe_redirect_hop(monkeypatch):
    class FakeResponse:
        def __init__(self, url, headers=None, history=None):
            self.url = url
            self.headers = headers or {}
            self.history = history or []

        def close(self):
            pass

    class FakeSession:
        def __init__(self, response):
            self.response = response

        def get(self, *args, **kwargs):
            return self.response

    response = FakeResponse(
        "https://bad.example/final.png",
        headers={"Content-Type": "image/png"},
        history=[FakeResponse("https://safe.example/start")],
    )
    monkeypatch.setattr(
        redirect_mod,
        "get_session",
        lambda max_redirects, user_agent: FakeSession(response),
    )

    keep, final_url, content_type, reason = redirect_mod.probe_redirect_target(
        "https://safe.example/start",
        blocked_domains={"bad.example"},
        blocked_cidrs=[],
        timeout=(1, 1),
        require_image_content_type=False,
        max_redirects=5,
        user_agent="test-agent",
    )

    assert not keep
    assert final_url == "https://bad.example/final.png"
    assert content_type == "image/png"
    assert reason == "unsafe_redirect_hop_2"


def test_probe_redirect_target_surfaces_http_errors(monkeypatch):
    class FakeSession:
        def get(self, *args, **kwargs):
            raise TooManyRedirects("loop")

    monkeypatch.setattr(
        redirect_mod,
        "get_session",
        lambda max_redirects, user_agent: FakeSession(),
    )

    keep, final_url, content_type, reason = redirect_mod.probe_redirect_target(
        "https://safe.example/start",
        blocked_domains=set(),
        blocked_cidrs=[],
        timeout=(1, 1),
        require_image_content_type=False,
        max_redirects=5,
        user_agent="test-agent",
    )

    assert not keep
    assert final_url is None
    assert content_type == ""
    assert reason is not None
    assert reason.startswith("http_error")
