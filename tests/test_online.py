from __future__ import annotations

import json
import unittest
from urllib.parse import parse_qs, urlsplit

from offline_voice_chat.online import OnlineLookup, OnlineLookupError


def query_parameters(url: str) -> dict[str, list[str]]:
    return parse_qs(urlsplit(url).query)


class OnlineLookupTests(unittest.TestCase):
    def test_weather_fetches_a_city_and_current_conditions(self) -> None:
        fetched_urls = []

        def fetch(url: str) -> bytes:
            fetched_urls.append(url)
            if urlsplit(url).hostname == "geocoding-api.open-meteo.com":
                return json.dumps(
                    {
                        "results": [
                            {
                                "name": "Zurich",
                                "country": "Switzerland",
                                "latitude": 47.37,
                                "longitude": 8.54,
                            }
                        ]
                    }
                ).encode()

            return json.dumps(
                {
                    "current": {
                        "time": "2026-09-28T14:15",
                        "temperature_2m": 17.2,
                        "weather_code": 2,
                        "wind_speed_10m": 12,
                    }
                }
            ).encode()

        answer = OnlineLookup(fetch).check("the weather in Zurich, Switzerland")

        self.assertEqual(
            answer,
            "Open-Meteo reports for Zurich, Switzerland at 14:15 local time: "
            "17.2 degrees Celsius, partly cloudy, and wind at "
            "12 kilometers per hour.",
        )
        self.assertEqual(len(fetched_urls), 2)
        self.assertEqual(
            query_parameters(fetched_urls[0])["name"],
            ["Zurich, Switzerland"],
        )
        self.assertEqual(query_parameters(fetched_urls[1])["timezone"], ["auto"])

    def test_news_returns_three_credited_headlines(self) -> None:
        fetched_urls = []

        def fetch(url: str) -> bytes:
            fetched_urls.append(url)
            return (
                "<rss><channel>"
                "<item><title>First story</title></item>"
                "<item><title>Second story</title></item>"
                "<item><title>Third story</title></item>"
                "<item><title>Fourth story</title></item>"
                "</channel></rss>"
            ).encode()

        answer = OnlineLookup(fetch).check("news in Switzerland")

        self.assertEqual(
            answer,
            "Google News results for news in Switzerland: First story; Second story; Third story",
        )
        self.assertEqual(urlsplit(fetched_urls[0]).hostname, "news.google.com")
        self.assertEqual(
            query_parameters(fetched_urls[0])["q"], ["news in Switzerland"]
        )

    def test_empty_and_long_queries_do_not_fetch(self) -> None:
        fetched_urls = []

        def fetch(url: str) -> bytes:
            fetched_urls.append(url)
            raise OnlineLookupError("I could not reach the online source.")

        lookup = OnlineLookup(fetch)
        self.assertIn("what you want", lookup.check(""))
        self.assertIn("shorter", lookup.check("a" * 201))
        self.assertEqual(fetched_urls, [])


if __name__ == "__main__":
    unittest.main()
