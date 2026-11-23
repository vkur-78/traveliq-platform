"""
TRAVELIQ — HTTP Client Helpers for Connecting Streamlit to Flask Microservices.
"""

from typing import Any, Dict, Optional, Tuple

import requests

from .config import (
    API_TIMEOUT_SECONDS,
    FLIGHT_API_URL,
    GENDER_API_URL,
    HEALTH_TIMEOUT_SECONDS,
)


def _extract_error_message(response: requests.Response) -> str:
    """Extract readable error details from a non-200 API response."""
    try:
        body = response.json()
        if isinstance(body, dict):
            err = body.get("error", f"HTTP {response.status_code}")
            msg = body.get("message") or body.get("details")
            if msg:
                return f"{err} ({msg})"
            if "missing_fields" in body:
                return f"{err}: {', '.join(body['missing_fields'])}"
            return str(err)
    except ValueError:
        pass
    return f"API returned HTTP {response.status_code}: {response.text[:200]}"


def check_api_health(
    base_url: str,
    timeout: float = HEALTH_TIMEOUT_SECONDS,
) -> Dict[str, Any]:
    """Query a microservice GET /health endpoint and return normalized status info."""
    url = f"{base_url.rstrip('/')}/health"
    try:
        resp = requests.get(url, timeout=timeout)
        if resp.status_code == 200:
            payload = resp.json()
            return {
                "online": True,
                "status": payload.get("status", "healthy"),
                "details": payload,
                "error": None,
            }
        return {
            "online": False,
            "status": "unhealthy",
            "details": {},
            "error": _extract_error_message(resp),
        }
    except requests.exceptions.ConnectionError:
        return {
            "online": False,
            "status": "offline",
            "details": {},
            "error": f"Unable to reach {url}. Ensure the API service is running.",
        }
    except requests.exceptions.Timeout:
        return {
            "online": False,
            "status": "timeout",
            "details": {},
            "error": f"Health check timed out for {url}.",
        }
    except Exception as exc:
        return {
            "online": False,
            "status": "error",
            "details": {},
            "error": str(exc),
        }


def request_flight_prediction(
    payload: Dict[str, Any],
    base_url: Optional[str] = None,
    timeout: float = API_TIMEOUT_SECONDS,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Send a flight price prediction request to POST /predict."""
    target_base = (base_url or FLIGHT_API_URL).rstrip("/")
    url = f"{target_base}/predict"
    try:
        resp = requests.post(url, json=payload, timeout=timeout)
        if resp.status_code == 200:
            return resp.json(), None
        return None, _extract_error_message(resp)
    except requests.exceptions.ConnectionError:
        return None, (
            f"Unable to connect to the Flight Price API at {target_base}. "
            "Start the service with `python -m app.app` or check FLIGHT_API_URL."
        )
    except requests.exceptions.Timeout:
        return None, f"Flight Price API timed out after {timeout:.0f}s ({url})."
    except ValueError:
        return None, "Flight Price API returned an invalid JSON response."
    except Exception as exc:
        return None, f"Unexpected Flight API error: {exc}"


def request_gender_prediction(
    payload: Dict[str, Any],
    base_url: Optional[str] = None,
    timeout: float = API_TIMEOUT_SECONDS,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Send a gender classification request to POST /predict-gender."""
    target_base = (base_url or GENDER_API_URL).rstrip("/")
    url = f"{target_base}/predict-gender"
    try:
        resp = requests.post(url, json=payload, timeout=timeout)
        if resp.status_code == 200:
            return resp.json(), None
        return None, _extract_error_message(resp)
    except requests.exceptions.ConnectionError:
        return None, (
            f"Unable to connect to the Gender Classification API at {target_base}. "
            "Start the service with `python -m app.gender_api` or check GENDER_API_URL."
        )
    except requests.exceptions.Timeout:
        return None, f"Gender Classification API timed out after {timeout:.0f}s ({url})."
    except ValueError:
        return None, "Gender Classification API returned an invalid JSON response."
    except Exception as exc:
        return None, f"Unexpected Gender API error: {exc}"
