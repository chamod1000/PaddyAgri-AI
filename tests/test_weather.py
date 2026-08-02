"""
Unit & Rule Engine Tests for Weather & Seasonal Intelligence (tests/test_weather.py)
"""

import unittest
from core.weather_service import (
    WeatherService, WeatherContext, SeasonalAdvisory, SeasonAdvisor, MockSriLankanWeatherProvider
)


class TestWeather(unittest.TestCase):

    def test_weather_service_ingestion(self):
        """Unit test for weather service context retrieval."""
        ws = WeatherService()
        w_ctx, s_adv = ws.get_context_and_advisory("Anuradhapura")
        self.assertIsInstance(w_ctx, WeatherContext)
        self.assertIsInstance(s_adv, SeasonalAdvisory)
        self.assertGreater(w_ctx.temperature_c, 0.0)
        self.assertGreater(w_ctx.humidity_pct, 0.0)

    def test_season_advisor_high_humidity_rule(self):
        """Unit test verifying high humidity fungal alert rule."""
        high_rh_weather = WeatherContext(humidity_pct=88.0, rainfall_probability_pct=20.0, temperature_c=30.0)
        advisory = SeasonAdvisor.evaluate(high_rh_weather)
        self.assertIn("HIGH RISK", advisory.fungal_risk_alert)

    def test_season_advisor_heavy_rain_rule(self):
        """Unit test verifying heavy rain fertilizer delay rule."""
        rainy_weather = WeatherContext(humidity_pct=70.0, rainfall_probability_pct=80.0, temperature_c=30.0)
        advisory = SeasonAdvisor.evaluate(rainy_weather)
        self.assertIn("DELAY FERTILIZER", advisory.fertilizer_timing_advice)

    def test_season_advisor_heat_stress_rule(self):
        """Unit test verifying heat stress irrigation review rule."""
        hot_weather = WeatherContext(humidity_pct=70.0, rainfall_probability_pct=20.0, temperature_c=35.0)
        advisory = SeasonAdvisor.evaluate(hot_weather)
        self.assertIn("IRRIGATION REVIEW", advisory.irrigation_advice)


if __name__ == "__main__":
    unittest.main()
