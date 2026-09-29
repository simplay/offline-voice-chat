"""Bounded online lookups after an explicit spoken search request."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from datetime import datetime
from html import unescape
from http.client import HTTPException
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from xml.etree import ElementTree

_GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
_WEATHER_URL = "https://api.open-meteo.com/v1/forecast"
_NEWS_URL = "https://news.google.com/rss/search"
_SEARCH_URL = "https://www.bing.com/search"
_MAX_RESPONSE_BYTES = 256 * 1024
_REQUEST_TIMEOUT_SECONDS = 2.5
_WEATHER_REQUEST = re.compile(
    r"(?:(?:what is|what's) )?(?:the )?(?:current )?weather (?:in|for) (.+)",
    re.I,
)
_WEATHER_DESCRIPTIONS = {
    0: "clear sky",
    1: "mostly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "fog",
    48: "freezing fog",
    51: "light drizzle",
    53: "drizzle",
    55: "heavy drizzle",
    56: "freezing drizzle",
    57: "freezing drizzle",
    61: "light rain",
    63: "rain",
    65: "heavy rain",
    66: "freezing rain",
    67: "freezing rain",
    71: "light snow",
    73: "snow",
    75: "heavy snow",
    77: "snow grains",
    80: "light rain showers",
    81: "rain showers",
    82: "heavy rain showers",
    85: "snow showers",
    86: "heavy snow showers",
    95: "thunderstorms",
    96: "thunderstorms with hail",
    97: "heavy thunderstorms",
    99: "thunderstorms with hail",
}


class OnlineLookupError(Exception):
    """
    An online source was unavailable or returned unusable data.
    """


def _fetch_url(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": "offline-voice-chat/0.1"})
    try:
        with urlopen(request, timeout=_REQUEST_TIMEOUT_SECONDS) as response:
            data = response.read(_MAX_RESPONSE_BYTES + 1)
    # OSError covers URLError and socket timeouts. HTTPException covers a
    # connection that drops mid-response, which raises IncompleteRead.
    except (OSError, HTTPException) as exception:
        raise OnlineLookupError("I could not reach the online source.") from exception

    if len(data) > _MAX_RESPONSE_BYTES:
        raise OnlineLookupError("The online source sent too much data.")

    return data


def _url(base: str, parameters: dict[str, object]) -> str:
    return f"{base}?{urlencode(parameters)}"


def _clean_text(text: str | None, max_chars: int) -> str:
    return " ".join(unescape(text or "").split())[:max_chars]


class OnlineLookup:
    """
    Takes a spoken subject, never a URL.
    """

    def __init__(self, fetch: Callable[[str], bytes] = _fetch_url) -> None:
        self._fetch = fetch

    def check(self, request: str) -> str:
        subject = request.strip(" .,:;!?").replace("’", "'")
        if not subject:
            return "Please say what you want me to check online."

        if len(subject) > 200:
            return "Please use a shorter online search query."

        weather = _WEATHER_REQUEST.fullmatch(subject)
        wants_news = re.search(r"\b(news|headlines)\b", subject, re.I) is not None

        try:
            if weather is not None:
                city = weather.group(1).strip()
                if len(city) > 80:
                    return "Please name a shorter city or city and country."

                return self._weather(city)

            if wants_news:
                return self._news(subject)

            return self._search(subject)
        except OnlineLookupError as exception:
            return str(exception)

    def _weather(self, city: str) -> str:
        geocoding_url = _url(
            _GEOCODING_URL,
            {"name": city, "count": 1, "language": "en", "format": "json"},
        )

        try:
            geocoding = json.loads(self._fetch(geocoding_url))
            # Open-Meteo leaves out "results" when no place matches.
            places = geocoding.get("results") or []
            if not places:
                return f"I could not find a location named {city}."

            place = places[0]
            latitude = float(place["latitude"])
            longitude = float(place["longitude"])
            name = place["name"].strip()[:80]
            country = place.get("country", "").strip()[:80]
        except (ValueError, TypeError, KeyError, AttributeError) as exception:
            raise OnlineLookupError(
                "I could not read the location response."
            ) from exception

        weather_url = _url(
            _WEATHER_URL,
            {
                "latitude": latitude,
                "longitude": longitude,
                "current": "temperature_2m,weather_code,wind_speed_10m",
                "timezone": "auto",
            },
        )

        try:
            forecast = json.loads(self._fetch(weather_url))
            current = forecast["current"]
            temperature = float(current["temperature_2m"])
            wind = float(current["wind_speed_10m"])
            code = int(current["weather_code"])
            observed_time = datetime.fromisoformat(current["time"])
        except (ValueError, TypeError, KeyError) as exception:
            raise OnlineLookupError(
                "I could not read the weather response."
            ) from exception

        observed_at = observed_time.strftime("%H:%M")
        location = ", ".join(part for part in (name, country) if part)
        conditions = _WEATHER_DESCRIPTIONS.get(code, "unknown conditions")
        return (
            f"Open-Meteo reports for {location} at {observed_at} local time: "
            f"{temperature:g} degrees Celsius, {conditions}, and wind at "
            f"{wind:g} kilometers per hour."
        )

    def _news(self, subject: str) -> str:
        items = self._rss_items(
            _url(_NEWS_URL, {"q": subject, "hl": "en", "gl": "CH", "ceid": "CH:en"}),
            error="I could not read the news feed.",
        )
        titles = (_clean_text(item.findtext("title"), 160) for item in items)
        headlines = [title for title in titles if title][:3]
        if not headlines:
            return f"I found no news results for {subject}."

        return f"Google News results for {subject}: " + "; ".join(headlines)

    def _search(self, subject: str) -> str:
        items = self._rss_items(
            _url(_SEARCH_URL, {"q": subject, "format": "rss"}),
            error="I could not read the search results.",
        )
        results = []

        for item in items[:3]:
            title = _clean_text(item.findtext("title"), 140)
            description_html = item.findtext("description") or ""
            description_text = re.sub(r"<[^>]*>", " ", description_html)
            description = _clean_text(description_text, 220)

            if not title:
                continue

            if description:
                results.append(f"{title}: {description}")
            else:
                results.append(title)

        if not results:
            return f"I found no search results for {subject}."

        return f"Bing search results for {subject}: " + "; ".join(results)

    def _rss_items(self, url: str, *, error: str) -> list[ElementTree.Element]:
        """
        Raises OnlineLookupError with the given spoken message if the feed is
        unreadable.
        """

        try:
            feed = ElementTree.fromstring(self._fetch(url))
        except (ElementTree.ParseError, ValueError) as exception:
            raise OnlineLookupError(error) from exception

        return feed.findall("./channel/item")
