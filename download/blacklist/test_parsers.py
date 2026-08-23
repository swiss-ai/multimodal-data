import unittest

import feed_parsers


class FeedParserTests(unittest.TestCase):
    def test_generic_url_feed_promotes_hosts(self):
        parsed = feed_parsers.parse_generic_url_list(
            b"https://bad.example/payload\nhttp://192.0.2.5/a\ninvalid\n"
        )

        self.assertEqual(parsed["domains"], {"bad.example"})
        self.assertEqual(parsed["ip_networks"], {"192.0.2.5"})
        self.assertEqual(len(parsed["url_regexes"]), 2)

    def test_suricata_parser_reads_header_addresses_only(self):
        parsed = feed_parsers.parse_suricata_ip_rules(
            b"alert ip [192.0.2.1,198.51.100.0/24] any -> $HOME_NET any "
            b'(msg:"contains 203.0.113.9";)\n'
        )

        self.assertEqual(parsed["ip_networks"], {"192.0.2.1", "198.51.100.0/24"})

    def test_hosts_file_ignores_invalid_rows(self):
        parsed = feed_parsers.parse_hostfile_domains(
            b"# comment\n0.0.0.0 bad.example\nmalformed\n127.0.0.1 localhost\n"
        )

        self.assertEqual(parsed["domains"], {"bad.example"})


if __name__ == "__main__":
    unittest.main()
