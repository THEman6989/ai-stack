"""ODF-to-OOXML converter via OnlyOffice DocumentServer REST API.

Only used when ALPHARAVIS_ENABLE_ODF_UPLOAD=true and the OnlyOffice
container is running (docker compose --profile odf up).
"""

from __future__ import annotations

import logging
import os

import httpx

logger = logging.getLogger(__name__)

ONLYOFFICE_URL = os.getenv("ALPHARAVIS_ONLYOFFICE_URL", "http://onlyoffice:80")

# ODF → OOXML mapping (OnlyOffice output format parameter)
_ODF_TO_OOXML: dict[str, str] = {
    "application/vnd.oasis.opendocument.text": "docx",
    "application/vnd.oasis.opendocument.presentation": "pptx",
    "application/vnd.oasis.opendocument.spreadsheet": "xlsx",
}

# Target content-type for each output format
_FORMAT_TO_MIME: dict[str, str] = {
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
}

# Extension mapping
_FORMAT_TO_EXT: dict[str, str] = {
    "docx": ".docx",
    "pptx": ".pptx",
    "xlsx": ".xlsx",
}


def _target_format(mime_type: str) -> str:
    """Map ODF MIME type to target OOXML format."""
    lowered = (mime_type or "").split(";", 1)[0].strip().lower()
    return _ODF_TO_OOXML.get(lowered, "docx")


def _target_mime(mime_type: str) -> str:
    return _FORMAT_TO_MIME.get(_target_format(mime_type), _FORMAT_TO_MIME["docx"])


def _target_ext(mime_type: str) -> str:
    return _FORMAT_TO_EXT.get(_target_format(mime_type), ".docx")


async def convert_odf_to_ooxml(
    input_path: str,
    mime_type: str,
    output_dir: str,
    *,
    timeout: float = 120.0,
) -> dict[str, object]:
    """Convert an ODF file to OOXML via OnlyOffice DocumentServer.

    Args:
        input_path: Absolute path to the ODF file.
        mime_type: MIME type of the input file (e.g. "application/vnd.oasis.opendocument.text").
        output_dir: Directory where the converted file will be written.
        timeout: HTTP timeout for the conversion request.

    Returns:
        dict with keys: output_path, output_format, output_mime, output_ext.

    Raises:
        RuntimeError: If conversion fails or OnlyOffice is unreachable.
    """
    target = _target_format(mime_type)
    output_ext = _target_ext(mime_type)
    output_mime = _target_mime(mime_type)

    # Derive output filename
    base = os.path.basename(input_path)
    stem = os.path.splitext(base)[0]
    output_filename = f"{stem}_converted{output_ext}"
    output_path = os.path.join(output_dir, output_filename)

    logger.info(
        "Converting ODF → %s: %s → %s (OnlyOffice at %s)",
        target.upper(),
        input_path,
        output_path,
        ONLYOFFICE_URL,
    )

    # OnlyOffice Conversion API:
    # POST /ConvertService.ashx
    # Multipart form: file=<input>, outputtype=docx|pptx|xlsx
    convert_url = f"{ONLYOFFICE_URL}/ConvertService.ashx"

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            with open(input_path, "rb") as fh:
                files = {"file": (base, fh, mime_type)}
                data = {"outputtype": target}
                response = await client.post(convert_url, files=files, data=data)
    except httpx.ConnectError as exc:
        raise RuntimeError(
            f"OnlyOffice is not reachable at {ONLYOFFICE_URL}. "
            f"Start it with: docker compose --profile odf up -d onlyoffice"
        ) from exc
    except httpx.TimeoutException as exc:
        raise RuntimeError(f"OnlyOffice conversion timed out after {timeout}s") from exc

    if response.status_code != 200:
        body = response.text[:500]
        raise RuntimeError(
            f"OnlyOffice conversion failed with HTTP {response.status_code}: {body}"
        )

    # Write the converted file
    os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "wb") as fh:
        fh.write(response.content)

    file_size = os.path.getsize(output_path)
    logger.info(
        "Converted ODF → %s: %s (%d bytes)",
        target.upper(),
        output_path,
        file_size,
    )

    return {
        "output_path": output_path,
        "output_format": target,
        "output_mime": output_mime,
        "output_ext": output_ext,
        "output_size": file_size,
    }


def is_odf(mime_type: str) -> bool:
    """Check whether a MIME type is an ODF format."""
    lowered = (mime_type or "").split(";", 1)[0].strip().lower()
    return lowered in _ODF_TO_OOXML
