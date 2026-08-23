import argparse
import contextlib
import io
import json
import os
import tempfile
import unittest
from unittest import mock

import main
import feed_parsers


class BuilderTests(unittest.TestCase):
    def test_build_writes_lists_and_manifest(self):
        test_feed = {
            "name": "test_feed",
            "url": "https://feed.invalid/list",
            "parser": feed_parsers.parse_generic_url_list,
        }

        def fake_fetch(feed, download_dir, timeout, allow_plain_http_fallback):
            self.assertEqual(feed, test_feed)
            self.assertEqual(timeout, 10)
            self.assertFalse(allow_plain_http_fallback)
            path = os.path.join(download_dir, "test_feed.download")
            with open(path, "wb") as handle:
                handle.write(b"https://bad.example/payload\nhttp://192.0.2.5/payload\n")
            return {
                "path": path,
                "fetched_url": test_feed["url"],
                "transport": "https",
                "warnings": [],
            }

        with tempfile.TemporaryDirectory() as tmp_dir:
            with contextlib.ExitStack() as stack:
                stack.enter_context(mock.patch.object(main, "FEEDS", [test_feed]))
                stack.enter_context(
                    mock.patch.object(main, "fetch_feed", side_effect=fake_fetch)
                )
                args = argparse.Namespace(
                    output_dir=tmp_dir,
                    timeout=10,
                    allow_plain_http_fallback=False,
                    skip_squid_parse=True,
                    squid_binary="unused",
                )
                with contextlib.redirect_stdout(io.StringIO()):
                    main.build_blacklist(args)

                with open(os.path.join(tmp_dir, "blocked_domains.txt")) as handle:
                    self.assertEqual(handle.read(), ".bad.example\n")
                with open(os.path.join(tmp_dir, "manifest.json")) as handle:
                    manifest = json.load(handle)

                self.assertNotIn("output_dir", manifest)
                self.assertEqual(manifest["feeds"][0]["name"], "test_feed")
                self.assertEqual(manifest["totals"]["blocked_domains.txt"], 1)


if __name__ == "__main__":
    unittest.main()
