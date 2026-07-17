from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "plugins" / "beatdrop_outfit"))

import beatdrop_outfit.outfit_sorter as outfit_sorter  # noqa: E402
from beatdrop_outfit.outfit_sorter import (  # noqa: E402
    _validate_vision_endpoint,
    classify_with_openai_vision,
    sort_outfits,
)


PNG_1X1 = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
    b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00"
    b"\x00\x0cIDAT\x08\xd7c\xf8\xcf\xc0\x00\x00\x03\x01\x01\x00"
    b"\x18\xdd\x8d\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
)


def test_sort_outfits_requires_configured_allowed_roots_before_filesystem_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "look.png").write_bytes(b"not-an-image")
    output = tmp_path / "output"
    monkeypatch.setenv("ALPHARAVIS_BEATDROP_OUTFIT_ALLOW_LOCAL_PATHS", "true")
    monkeypatch.delenv("ALPHARAVIS_BEATDROP_OUTFIT_ALLOWED_ROOTS", raising=False)

    with pytest.raises(ValueError, match="allowed root"):
        sort_outfits(str(source), str(output))

    assert not output.exists()


def test_sort_outfits_defaults_to_side_effect_free_dry_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "look.png").write_bytes(PNG_1X1)
    output = tmp_path / "output"
    monkeypatch.setenv("ALPHARAVIS_BEATDROP_OUTFIT_ALLOW_LOCAL_PATHS", "true")
    monkeypatch.setenv("ALPHARAVIS_BEATDROP_OUTFIT_ALLOWED_ROOTS", str(tmp_path))

    result = sort_outfits(str(source), str(output))

    assert result["summary"]["dry_run"] is True
    assert not output.exists()


def test_sort_outfits_symlink_mode_refuses_to_unlink_existing_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    original = source / "look.png"
    original.write_bytes(PNG_1X1)
    output = tmp_path / "output"
    destination = output / "90_unclear_or_bad_quality" / original.name
    destination.parent.mkdir(parents=True)
    protected_target = tmp_path / "protected.png"
    protected_target.write_bytes(b"protected")
    destination.symlink_to(protected_target)
    monkeypatch.setattr(outfit_sorter, "Image", None)
    monkeypatch.setenv("ALPHARAVIS_BEATDROP_OUTFIT_ALLOW_LOCAL_PATHS", "true")
    monkeypatch.setenv("ALPHARAVIS_BEATDROP_OUTFIT_ALLOWED_ROOTS", str(tmp_path))

    with pytest.raises(FileExistsError, match="refusing to replace"):
        sort_outfits(str(source), str(output), mode="symlink", dry_run=False)

    assert destination.is_symlink()
    assert destination.resolve() == protected_target


def test_sort_outfits_copy_mode_refuses_existing_destination_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    original = source / "look.png"
    original.write_bytes(PNG_1X1)
    output = tmp_path / "output"
    destination = output / "90_unclear_or_bad_quality" / original.name
    destination.parent.mkdir(parents=True)
    protected_target = tmp_path / "protected.png"
    protected_target.write_bytes(b"protected")
    destination.symlink_to(protected_target)
    monkeypatch.setattr(outfit_sorter, "Image", None)
    monkeypatch.setenv("ALPHARAVIS_BEATDROP_OUTFIT_ALLOW_LOCAL_PATHS", "true")
    monkeypatch.setenv("ALPHARAVIS_BEATDROP_OUTFIT_ALLOWED_ROOTS", str(tmp_path))

    with pytest.raises(FileExistsError):
        sort_outfits(str(source), str(output), mode="copy", dry_run=False)

    assert destination.is_symlink()
    assert protected_target.read_bytes() == b"protected"


def test_sort_outfits_rejects_symlinked_original_outside_allowed_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    allowed = tmp_path / "allowed"
    source = allowed / "source"
    source.mkdir(parents=True)
    outside = tmp_path / "outside.png"
    outside.write_bytes(PNG_1X1)
    (source / "look.png").symlink_to(outside)
    monkeypatch.setenv("ALPHARAVIS_BEATDROP_OUTFIT_ALLOW_LOCAL_PATHS", "true")
    monkeypatch.setenv("ALPHARAVIS_BEATDROP_OUTFIT_ALLOWED_ROOTS", str(allowed))

    with pytest.raises(ValueError, match=r"(?i)symlink"):
        sort_outfits(str(source), str(allowed / "output"))


def test_sort_outfits_rejects_symlinked_cache_image_outside_allowed_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    allowed = tmp_path / "allowed"
    source = allowed / "source"
    cache = allowed / "cache"
    source.mkdir(parents=True)
    cache.mkdir()
    (source / "look.png").write_bytes(PNG_1X1)
    outside = tmp_path / "outside.png"
    outside.write_bytes(PNG_1X1)
    (cache / "look.png").symlink_to(outside)
    monkeypatch.setenv("ALPHARAVIS_BEATDROP_OUTFIT_ALLOW_LOCAL_PATHS", "true")
    monkeypatch.setenv("ALPHARAVIS_BEATDROP_OUTFIT_ALLOWED_ROOTS", str(allowed))

    with pytest.raises(ValueError, match=r"(?i)symlink"):
        sort_outfits(
            str(source), str(allowed / "output"), cache_dir=str(cache)
        )


def test_vision_endpoint_rejects_non_http_scheme_before_reading_image(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image = tmp_path / "secret.png"
    image.write_bytes(PNG_1X1)
    monkeypatch.setenv("ALPHARAVIS_BEATDROP_OUTFIT_VISION_ALLOWED_HOSTS", "localhost")
    monkeypatch.setenv("ALPHARAVIS_BEATDROP_OUTFIT_ALLOW_PRIVATE_URLS", "true")

    with pytest.raises(ValueError, match="http/https"):
        classify_with_openai_vision(image, "file:///tmp/collector", "qwen")


def test_vision_endpoint_requires_explicit_host_allowlist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(
        "ALPHARAVIS_BEATDROP_OUTFIT_VISION_ALLOWED_HOSTS", raising=False
    )

    with pytest.raises(ValueError, match="allowlist"):
        _validate_vision_endpoint("https://vision.example.test/v1")


def test_vision_endpoint_blocks_allowlisted_private_host_without_private_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "ALPHARAVIS_BEATDROP_OUTFIT_VISION_ALLOWED_HOSTS", "127.0.0.1"
    )
    monkeypatch.delenv("ALPHARAVIS_BEATDROP_OUTFIT_ALLOW_PRIVATE_URLS", raising=False)

    with pytest.raises(ValueError, match="private"):
        _validate_vision_endpoint("http://127.0.0.1:8080/v1")


def test_vision_endpoint_allows_explicit_local_qwen_without_redirects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, dict]] = []

    class Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {
                "choices": [
                    {
                        "message": {
                            "content": '{"framing_category":"10_upper_body_head_chest",'
                            '"clothing_category":"10_covered","confidence":1}'
                        }
                    }
                ]
            }

    class Requests:
        @staticmethod
        def post(url: str, **kwargs):
            calls.append((url, kwargs))
            return Response()

    image = tmp_path / "look.png"
    image.write_bytes(PNG_1X1)
    monkeypatch.setattr(outfit_sorter, "requests", Requests)
    monkeypatch.setenv(
        "ALPHARAVIS_BEATDROP_OUTFIT_VISION_ALLOWED_HOSTS", "127.0.0.1"
    )
    monkeypatch.setenv("ALPHARAVIS_BEATDROP_OUTFIT_ALLOW_PRIVATE_URLS", "true")

    result = classify_with_openai_vision(
        image, "http://127.0.0.1:8080", "Qwen3.6-35B-A3B-MTP"
    )

    assert result["confidence"] == 1
    assert calls[0][0] == "http://127.0.0.1:8080/v1/chat/completions"
    assert calls[0][1]["allow_redirects"] is False
