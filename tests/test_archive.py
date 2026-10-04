from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest

from baystfirm import archive
from baystfirm.models import Side


def make_zip(contents: str) -> bytes:
    target = io.BytesIO()
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as output:
        output.writestr("BTCUSDT-aggTrades-2025-01-01.csv", contents)
    return target.getvalue()


@pytest.mark.parametrize(
    ("header", "raw_timestamp", "expected_timestamp"),
    [
        (True, "1735689600123", 1735689600123),
        (False, "1735689600123000", 1735689600123),
    ],
)
def test_archive_trade_rows_parse_header_and_timestamp_units(
    tmp_path: Path, header: bool, raw_timestamp: str, expected_timestamp: int
) -> None:
    prefix = (
        "agg_trade_id,price,quantity,first_trade_id,last_trade_id,transact_time,is_buyer_maker,is_best_match\n"
        if header
        else ""
    )
    contents = prefix + f"123,101.25,0.5,120,121,{raw_timestamp},true,true\n"
    archive_path = tmp_path / "trades.zip"
    archive_path.write_bytes(make_zip(contents))

    assert list(archive.iter_archive_trades(archive_path)) == [
        (expected_timestamp, 101.25, 0.5, Side.SELL, 123)
    ]


def test_archive_side_maps_non_maker_to_buy(tmp_path: Path) -> None:
    archive_path = tmp_path / "trades.zip"
    archive_path.write_bytes(make_zip("124,100,2,122,123,1735689600123,false,true\n"))
    assert list(archive.iter_archive_trades(archive_path))[0][3] is Side.BUY


def test_checksum_mismatch_fails_before_archive_is_cached(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    zip_payload = make_zip("123,101,0.5,120,121,1735689600123,false,true\n")

    class Response(io.BytesIO):
        def __enter__(self) -> Response:
            return self

        def __exit__(self, *_args: object) -> None:
            self.close()

    def fake_urlopen(url: str, timeout: int) -> Response:
        del timeout
        if url.endswith(".CHECKSUM"):
            return Response(b"0" * 64 + b"  BTCUSDT-aggTrades-2025-01-01.zip\n")
        return Response(zip_payload)

    monkeypatch.setattr(archive, "urlopen", fake_urlopen)
    with pytest.raises(ValueError, match="checksum mismatch"):
        archive.fetch_archive("BTCUSDT", "2025-01-01", tmp_path / "cache")
    assert not (tmp_path / "cache" / "BTCUSDT-aggTrades-2025-01-01.zip").exists()
