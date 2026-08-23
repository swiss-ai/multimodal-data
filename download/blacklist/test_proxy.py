import os
import subprocess
import tempfile
import unittest


ROOT = os.path.dirname(os.path.abspath(__file__))
PROXY_SCRIPT = os.path.join(ROOT, "proxy.sh")


class ProxyConfigTests(unittest.TestCase):
    def test_rendered_config_has_blacklists_annotations_and_port_policy(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            for name, contents in (
                ("blocked_ip.txt", "192.0.2.1\n"),
                ("blocked_domains.txt", ".bad.example\n"),
                ("blocked_urls.txt", "^http://bad\\.example/$\n"),
                ("blocked_url_paths.txt", "^/payload$\n"),
            ):
                with open(os.path.join(tmp_dir, name), "w", encoding="utf-8") as handle:
                    handle.write(contents)

            work_dir = os.path.join(tmp_dir, "work")
            os.mkdir(work_dir)
            config_path = os.path.join(work_dir, "squid.conf")
            subprocess.check_call(
                [
                    PROXY_SCRIPT,
                    "render-config",
                    tmp_dir,
                    config_path,
                    os.path.join(work_dir, "access.log"),
                    "127.0.0.1",
                    "43128",
                ]
            )

            with open(config_path, encoding="utf-8") as handle:
                config = handle.read()

            self.assertIn("http_port 127.0.0.1:43128", config)
            self.assertIn("http_access deny CONNECT !SSL_ports", config)
            self.assertIn("CSCS_BLACKLIST_HIT", config)
            self.assertIn("acl blocked_url_paths urlpath_regex", config)

    def test_empty_optional_path_file_is_omitted(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = os.path.join(tmp_dir, "squid.conf")
            open(os.path.join(tmp_dir, "blocked_url_paths.txt"), "w").close()
            subprocess.check_call(
                [
                    PROXY_SCRIPT,
                    "render-config",
                    tmp_dir,
                    config_path,
                    os.path.join(tmp_dir, "access.log"),
                    "127.0.0.1",
                    "43128",
                ]
            )
            with open(config_path, encoding="utf-8") as handle:
                self.assertNotIn("acl blocked_url_paths urlpath_regex", handle.read())


if __name__ == "__main__":
    unittest.main()
