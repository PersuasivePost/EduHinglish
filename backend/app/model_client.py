"""HTTP client for the EduHinglish model server (src/model_server.py on Colab)."""

import httpx

from . import config


class ModelServerError(Exception):
    def __init__(self, status_code: int, detail: str):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


class ModelClient:
    def __init__(self):
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(config.MODEL_TIMEOUT_SEC, connect=10.0),
            headers={
                "X-API-Key": config.MODEL_API_KEY,
                # ngrok free tunnels show an HTML warning page unless this header is sent
                "ngrok-skip-browser-warning": "1",
            },
        )

    async def close(self):
        await self._client.aclose()

    def _url(self, path: str) -> str:
        if not config.MODEL_SERVER_URL:
            raise ModelServerError(503, "MODEL_SERVER_URL is not set. Start notebook 07 and copy its URL to backend/.env")
        return f"{config.MODEL_SERVER_URL}{path}"

    async def _request(self, method: str, path: str, **kwargs) -> dict:
        try:
            r = await self._client.request(method, self._url(path), **kwargs)
        except httpx.TimeoutException:
            raise ModelServerError(504, "Model server timed out (it may be busy with other questions)")
        except httpx.RequestError as e:
            raise ModelServerError(503, f"Model server unreachable ({type(e).__name__}). Is the Colab notebook running?")
        if r.status_code == 401:
            raise ModelServerError(502, "Model server rejected the API key (check MODEL_API_KEY)")
        if r.status_code >= 400:
            try:
                detail = r.json().get("detail", r.text)
            except ValueError:
                detail = r.text[:300]
            raise ModelServerError(502 if r.status_code >= 500 else r.status_code, f"Model server: {detail}")
        try:
            return r.json()
        except ValueError:
            # e.g. a tunnel's HTML error page after Colab stopped
            raise ModelServerError(503, "Model server returned a non-JSON page. Is the Colab notebook still running?")

    async def health(self) -> dict:
        return await self._request("GET", "/health")

    async def ask(self, payload: dict) -> dict:
        return await self._request("POST", "/ask", json=payload)

    async def preprocess(self, sentence: str) -> dict:
        return await self._request("POST", "/preprocess", json={"sentence": sentence})
