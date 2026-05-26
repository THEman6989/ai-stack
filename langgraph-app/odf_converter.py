"""ODF-to-OOXML converter via OnlyOffice DocumentServer REST API.

Only used when ALPHARAVIS_ENABLE_ODF_UPLOAD=true and the OnlyOffice
container is running (docker compose --profile odf up).
"""

from __future__ import annotations

import hashlib
import logging
import os
import xml.etree.ElementTree as ET

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

_ONLYOFFICE_ERROR_MESSAGES: dict[str, str] = {
    "-1": "unknown conversion error",
    "-2": "conversion timeout",
    "-3": "conversion error",
    "-4": "OnlyOffice could not download the source document URL",
    "-5": "incorrect password",
    "-6": "conversion result database error",
    "-7": "input error",
    "-8": "invalid token",
}


def _parse_conversion_response(response: httpx.Response) -> dict[str, object]:
    """Parse OnlyOffice ConvertService response.

    DocumentServer versions/configurations may return JSON or XML from
    ConvertService.ashx. Normalize both shapes to lower-camel JSON-like keys.
    """
    text = response.text.strip()
    content_type = (response.headers.get("content-type") or "").lower()
    if "json" in content_type or text.startswith("{"):
        return response.json()

    if text.startswith("<"):
        root = ET.fromstring(text)
        data: dict[str, object] = {}
        for child in root:
            key = child.tag[0].lower() + child.tag[1:] if child.tag else ""
            value = child.text or ""
            if value.lower() == "true":
                data[key] = True
            elif value.lower() == "false":
                data[key] = False
            else:
                data[key] = value
        return data

    raise ValueError(f"Unsupported OnlyOffice conversion response: {text[:500]}")


def _format_onlyoffice_error(error: object) -> str:
    code = str(error)
    message = _ONLYOFFICE_ERROR_MESSAGES.get(code)
    return f"{code} ({message})" if message else code


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
    source_url: str,
    timeout: float = 120.0,
) -> dict[str, object]:
    """Convert an ODF file to OOXML via OnlyOffice DocumentServer.

    OnlyOffice DocumentServer's conversion service does not accept raw file
    multipart uploads. It expects JSON containing a URL that the DocumentServer
    container can fetch. `source_url` must therefore be reachable from the
    OnlyOffice container, for example `http://media-gallery:8130/media/...`.

    Args:
        input_path: Absolute path to the ODF file, used for stable title/key and
            output filename.
        mime_type: MIME type of the input file.
        output_dir: Directory where the converted file will be written.
        source_url: Internal HTTP URL from which OnlyOffice can fetch the file.
        timeout: HTTP timeout for the conversion and download requests.

    Returns:
        dict with keys: output_path, output_format, output_mime, output_ext,
        output_size.

    Raises:
        RuntimeError: If conversion fails or OnlyOffice is unreachable.
    """
    target = _target_format(mime_type)
    output_ext = _target_ext(mime_type)
    output_mime = _target_mime(mime_type)

    base = os.path.basename(input_path)
    stem = os.path.splitext(base)[0]
    output_filename = f"{stem}_converted{output_ext}"
    output_path = os.path.join(output_dir, output_filename)
    file_type = os.path.splitext(base)[1].lstrip(".").lower() or "odt"

    logger.info(
        "Converting ODF → %s: %s via %s → %s (OnlyOffice at %s)",
        target.upper(),
        input_path,
        source_url,
        output_path,
        ONLYOFFICE_URL,
    )

    convert_url = f"{ONLYOFFICE_URL}/ConvertService.ashx"
    conversion_key = hashlib.sha256(f"{source_url}|{target}".encode("utf-8")).hexdigest()[:32]
    payload = {
        "async": False,
        "filetype": file_type,
        "key": f"alpharavis-{conversion_key}",
        "outputtype": target,
        "title": base,
        "url": source_url,
    }

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(convert_url, json=payload)
            if response.status_code != 200:
                body = response.text[:500]
                raise RuntimeError(
                    f"OnlyOffice conversion failed with HTTP {response.status_code}: {body}"
                )
            data = _parse_conversion_response(response)
            error = data.get("error")
            if error:
                raise RuntimeError(
                    f"OnlyOffice conversion failed with error {_format_onlyoffice_error(error)}: {data}"
                )
            if not data.get("endConvert", True):
                raise RuntimeError(f"OnlyOffice conversion did not finish synchronously: {data}")
            file_url = data.get("fileUrl") or data.get("fileurl")
            if not file_url:
                raise RuntimeError(f"OnlyOffice conversion response had no fileUrl: {data}")
            converted = await client.get(str(file_url))
            if converted.status_code != 200:
                raise RuntimeError(
                    f"OnlyOffice converted file download failed with HTTP {converted.status_code}: "
                    f"{converted.text[:500]}"
                )
    except httpx.ConnectError as exc:
        raise RuntimeError(
            f"OnlyOffice is not reachable at {ONLYOFFICE_URL}. "
            f"Start it with: docker compose --profile odf up -d onlyoffice"
        ) from exc
    except httpx.TimeoutException as exc:
        raise RuntimeError(f"OnlyOffice conversion timed out after {timeout}s") from exc

    os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "wb") as fh:
        fh.write(converted.content)

    file_size = os.path.getsize(output_path)
    logger.info("Converted ODF → %s: %s (%d bytes)", target.upper(), output_path, file_size)

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
