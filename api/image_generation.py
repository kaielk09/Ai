"""Paid OpenAI GPT Image endpoint for Purple AI."""
from __future__ import annotations

import hmac
import os
from typing import Optional

from fastapi import APIRouter, Header, HTTPException
from openai import OpenAI, APIError, RateLimitError, AuthenticationError
from pydantic import BaseModel, Field

router = APIRouter(tags=["image-generation"])


class ImageGenerationRequest(BaseModel):
    prompt: str = Field(min_length=3, max_length=6000)
    size: str = Field(default="1024x1024", pattern="^(1024x1024|1536x1024|1024x1536)$")
    quality: str = Field(default="xhigh", pattern="^(low|medium|high|xhigh|max|auto)$")
    output_format: str = Field(default="png", pattern="^(png|jpeg|webp)$")


def _authorize(authorization: Optional[str]) -> None:
    api_key = os.getenv("PURPLE_API_KEY", "").strip()
    if not api_key:
        raise HTTPException(status_code=503, detail="Klucz Purple API nie jest skonfigurowany.")
    expected = "Bearer " + api_key
    if not authorization or not hmac.compare_digest(authorization, expected):
        raise HTTPException(status_code=401, detail="Brak lub nieprawidłowy klucz API.")


@router.post("/v1/images/generations")
def generate_image(
    body: ImageGenerationRequest,
    authorization: Optional[str] = Header(default=None),
):
    """Generate one image with OpenAI GPT Image. Billing is handled by the OpenAI API account."""
    _authorize(authorization)

    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not openai_key:
        raise HTTPException(
            status_code=503,
            detail="Generowanie obrazów nie jest skonfigurowane. Administrator musi ustawić OPENAI_API_KEY po stronie serwera.",
        )

    model = os.getenv("PURPLE_IMAGE_MODEL", "gpt-image-2.6-sunburst").strip()
    client = OpenAI(api_key=openai_key, timeout=180.0, max_retries=1)

    try:
        result = client.images.generate(
            model=model,
            prompt=body.prompt,
            size=body.size,
            quality=body.quality,
            output_format=body.output_format,
            n=1,
        )
    except AuthenticationError as exc:
        raise HTTPException(status_code=502, detail="Dostawca odrzucił klucz OpenAI skonfigurowany na serwerze.") from exc
    except RateLimitError as exc:
        raise HTTPException(status_code=429, detail="Limit lub środki API dostawcy są chwilowo niedostępne.") from exc
    except APIError as exc:
        raise HTTPException(status_code=502, detail="Dostawca generowania obrazów zwrócił błąd. Spróbuj później.") from exc

    if not result.data or not result.data[0].b64_json:
        raise HTTPException(status_code=502, detail="Dostawca nie zwrócił obrazu.")

    return {
        "id": "purple-image",
        "model": model,
        "created": result.created,
        "output_format": body.output_format,
        "size": body.size,
        "quality": body.quality,
        "image_base64": result.data[0].b64_json,
        "mime_type": "image/" + ("jpeg" if body.output_format == "jpeg" else body.output_format),
    }
