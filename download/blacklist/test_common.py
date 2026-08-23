import os
import re
import tempfile
import unittest

from common import (
    add_exact_url,
    add_url_path,
    collapse_ip_networks,
    empty_indicators,
    normalize_domain,
    render_domains,
    write_text_file_atomic,
)


class CommonTests(unittest.TestCase):
    def test_exact_https_url_is_promoted_to_domain_acl(self):
        indicators = empty_indicators()
        add_exact_url(indicators, "HTTPS://Example.COM/a?b=1#fragment")

        self.assertEqual(indicators["domains"], {"example.com"})
        self.assertEqual(
            indicators["url_regexes"],
            {"^{}$".format(re.escape("https://example.com/a?b=1"))},
        )

    def test_exact_ip_url_is_promoted_to_ip_acl(self):
        indicators = empty_indicators()
        add_exact_url(indicators, "https://192.0.2.8/payload")

        self.assertEqual(indicators["ip_networks"], {"192.0.2.8"})

    def test_domains_are_idna_normalized_and_collapsed(self):
        self.assertEqual(normalize_domain("BÜCHER.example."), "xn--bcher-kva.example")
        self.assertEqual(
            render_domains({"example.com", "a.example.com", "other.example"}),
            [".example.com", ".other.example"],
        )

    def test_ip_networks_are_normalized_and_collapsed(self):
        self.assertEqual(
            collapse_ip_networks({"192.0.2.0/25", "192.0.2.128/25", "2001:db8::1"}),
            ["192.0.2.0/24", "2001:db8::1"],
        )

    def test_url_path_is_escaped(self):
        indicators = empty_indicators()
        add_url_path(indicators, "/a+b")
        self.assertEqual(
            indicators["url_path_regexes"],
            {"^{}([?#].*)?$".format(re.escape("/a+b"))},
        )

    def test_atomic_writer_creates_parent_and_trailing_newline(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, "nested", "output.txt")
            write_text_file_atomic(path, ["a", "b"])
            with open(path, encoding="utf-8") as handle:
                self.assertEqual(handle.read(), "a\nb\n")


if __name__ == "__main__":
    unittest.main()
