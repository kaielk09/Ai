"""High-quality paid GPT Image generation endpoint for Purple AI.

The image model is a separate specialist model. Purple AI's text model and its
existing chat endpoints remain unchanged.
"""
from __future__ import annotations

import hmac
import os
from typing import Optional

from fastapi import APIRouter, Header, HTTPException
from openai import OpenAI, APIError, RateLimitError, AuthenticationError
from pydantic import BaseModel, Field, model_validator

router = APIRouter(tags=["image-generation"])


class ImageGenerationRequest(BaseModel):
    prompt: str = Field(min_length=3, max_length=6000)
    size: str = Field(
        default="1024x1024",
        pattern="^(auto|1024x1024|1536x1024|1024x1536)$",
    )
    quality: str = Field(
        default="xhigh",
        pattern="^(low|medium|high|xhigh|max|auto)$",
    )
    output_format: str = Field(default="png", pattern="^(png|jpeg|webp)$")
    background: str = Field(
        default="auto",
        pattern="^(auto|opaque|transparent)$",
    )
    complexity: str = Field(
        default="high",
        pattern="^(low|medium|high)$",
    )
    n: int = Field(default=1, ge=1, le=4)

    @model_validator(mode="after")
    def validate_background_format(self):
        if self.background == "transparent" and self.output_format not in {"png", "webp"}:
            raise ValueError("Przezroczyste tło wymaga formatu PNG lub WebP.")
        return self


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
    """Generate high-quality images using the configured GPT Image model.

    Billing is handled by the provider account configured by the server owner.
    Provider credentials are never accepted from or returned to API clients.
    """
    _authorize(authorization)

    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not openai_key:
        raise HTTPException(
            status_code=503,
            detail=(
                "Generowanie obrazów nie jest skonfigurowane. "
                "Administrator musi ustawić OPENAI_API_KEY jako sekret serwera."
            ),
        )

    model = os.getenv("PURPLE_IMAGE_MODEL", "gpt-image-2.6-sunburst").strip()
    if not model:
        raise HTTPException(status_code=503, detail="Nie ustawiono modelu generowania obrazów.")

    client = OpenAI(api_key=openai_key, timeout=180.0, max_retries=1)
    try:
        result = client.images.generate(
            model=model,
            prompt=body.prompt,
            size=body.size,
            quality=body.quality,
            output_format=body.output_format,
            background=body.background,
            moderation="auto",
            n=body.n,
            complexity=body.complexity,
        )
    except AuthenticationError as exc:
        raise HTTPException(
            status_code=502,
            detail="Dostawca odrzucił klucz API skonfigurowany na serwerze.",
        ) from exc
    except RateLimitError as exc:
        raise HTTPException(
            status_code=429,
            detail="Limit lub środki API dostawcy są chwilowo niedostępne.",
        ) from exc
    except APIError as exc:
        # Do not expose provider internals, credentials, or raw request data.
        raise HTTPException(
            status_code=502,
            detail="Dostawca generowania obrazów zwrócił błąd. Spróbuj później.",
        ) from exc

    if not result.data:
        raise HTTPException(status_code=502, detail="Dostawca nie zwrócił żadnego obrazu.")

    images = []
    for item in result.data:
        if not item.b64_json:
            continue
        images.append({
            "image_base64": item.b64_json,
            "mime_type": "image/" + ("jpeg" if body.output_format == "jpeg" else body.output_format),
            "revised_prompt": getattr(item, "revised_prompt", None),
        })

    if not images:
        raise HTTPException(status_code=502, detail="Dostawca nie zwrócił danych obrazu.")

    # image_base64 is retained at the top level for compatibility with the
    # first version of this endpoint. New clients can use the images array.
    return {
        "id": "purple-image",
        "model": model,
        "created": result.created,
        "output_format": body.output_format,
        "size": body.size,
        "quality": body.quality,
        "complexity": body.complexity,
        "background": body.background,
        "images": images,
        "image_base64": images[0]["image_base64"],
        "mime_type": images[0]["mime_type"],
    }
