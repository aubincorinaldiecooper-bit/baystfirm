from __future__ import annotations

import base64
import json
import struct
from pathlib import Path
from typing import Any

import httpx
import pytest
from solders.pubkey import Pubkey

import baystfirm.solana_tokens as solana_module
from baystfirm.config import Settings
from baystfirm.solana_tokens import (
    METADATA_PROGRAM,
    SPL_TOKEN_PROGRAM,
    TOKEN_2022_PROGRAM,
    NotTokenMint,
    SolanaTokenClient,
    SolanaTokenEngine,
    SourceError,
    classify_pool_type,
    decode_account_data,
    market_value,
    metadata_pda,
    parse_dex_pairs,
    parse_metaplex_metadata,
    parse_token_mint,
    select_main_pool,
    unavailable_fact,
    validate_mint,
)

MINT = "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263"
RATE_LIMITS_OFF = {
    "solana_rpc": 0.0,
    "solana_largest": 0.0,
    "dexscreener": 0.0,
    "geckoterminal": 0.0,
    "raydium": 0.0,
    "rugcheck": 0.0,
}


def _settings(*, rugcheck_api_key: str | None = None) -> Settings:
    return Settings(
        database_path=Path("unused.db"),
        enabled_venues=(),
        shadow_mode=True,
        symbols=(),
        solana_rpc_url="https://rpc.example.invalid/?api-key=hidden",
        rugcheck_api_key=rugcheck_api_key,
        solana_tokens_enabled=False,
    )


def test_solana_settings_load_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BAYST_SOLANA_RPC_URL", raising=False)
    monkeypatch.delenv("BAYST_RUGCHECK_API_KEY", raising=False)
    monkeypatch.delenv("BAYST_SOLANA_TOKENS", raising=False)
    defaults = Settings.from_env()
    assert defaults.solana_rpc_url == "https://api.mainnet-beta.solana.com"
    assert defaults.rugcheck_api_key is None
    assert defaults.solana_tokens_enabled is True

    monkeypatch.setenv("BAYST_SOLANA_RPC_URL", "https://rpc.example.invalid/key")
    monkeypatch.setenv("BAYST_RUGCHECK_API_KEY", "rugcheck-secret")
    monkeypatch.setenv("BAYST_SOLANA_TOKENS", "no")
    configured = Settings.from_env()
    assert configured.solana_rpc_url == "https://rpc.example.invalid/key"
    assert configured.rugcheck_api_key == "rugcheck-secret"
    assert configured.solana_tokens_enabled is False


def test_validate_mint_and_metadata_pda_use_solders_pubkeys() -> None:
    assert str(validate_mint(MINT)) == MINT
    assert str(metadata_pda(MINT)) == str(
        Pubkey.find_program_address(
            [
                b"metadata",
                bytes(Pubkey.from_string(METADATA_PROGRAM)),
                bytes(Pubkey.from_string(MINT)),
            ],
            Pubkey.from_string(METADATA_PROGRAM),
        )[0]
    )
    with pytest.raises(ValueError, match="32-byte base58"):
        validate_mint("not-a-mint")


def test_parse_spl_mint_with_revoked_authorities() -> None:
    parsed = parse_token_mint(
        {
            "owner": SPL_TOKEN_PROGRAM,
            "data": {
                "parsed": {
                    "info": {
                        "mintAuthority": None,
                        "freezeAuthority": None,
                        "extensions": [],
                    }
                }
            },
        }
    )

    assert parsed.token_program == "spl-token"
    assert parsed.mint_authority is None
    assert parsed.freeze_authority is None
    assert parsed.risky_extensions == []
    assert parsed.transfer_fee_bps is None
    assert parsed.token_metadata is None


def test_parse_token_2022_extensions_and_transfer_fee() -> None:
    update_authority = str(Pubkey.new_unique())
    parsed = parse_token_mint(
        {
            "owner": TOKEN_2022_PROGRAM,
            "data": {
                "parsed": {
                    "info": {
                        "mintAuthority": "mint-authority",
                        "freezeAuthority": None,
                        "extensions": [
                            {
                                "extension": "transferFeeConfig",
                                "state": {"newerTransferFee": {"transferFeeBasisPoints": 375}},
                            },
                            {
                                "extension": "permanentDelegate",
                                "state": {"delegate": "delegate"},
                            },
                            {
                                "extension": "tokenMetadata",
                                "state": {
                                    "updateAuthority": update_authority,
                                    "name": "Example Token",
                                    "symbol": "EX",
                                    "uri": "https://example.invalid/token.json",
                                },
                            },
                        ],
                    }
                }
            },
        }
    )

    assert parsed.token_program == "token-2022"
    assert parsed.mint_authority == "mint-authority"
    assert parsed.risky_extensions == ["transferFeeConfig", "permanentDelegate"]
    assert parsed.transfer_fee_bps == 375
    assert parsed.token_metadata == {
        "update_authority": update_authority,
        "name": "Example Token",
        "symbol": "EX",
        "uri": "https://example.invalid/token.json",
    }


def test_non_token_account_is_rejected() -> None:
    with pytest.raises(NotTokenMint, match="not owned"):
        parse_token_mint({"owner": str(Pubkey.new_unique()), "data": {}})
    with pytest.raises(NotTokenMint, match="not a token mint"):
        parse_token_mint(None)
    with pytest.raises(NotTokenMint, match="parsed token mint"):
        parse_token_mint(
            {
                "owner": SPL_TOKEN_PROGRAM,
                "data": {"parsed": {"type": "account", "info": {}}},
            }
        )


def _borsh_string(value: str) -> bytes:
    encoded = value.encode()
    return struct.pack("<I", len(encoded)) + encoded


def _metadata_account(is_mutable: bool, creators: bool) -> bytes:
    default = bytes(Pubkey.default())
    content = b"\x04" + default + default
    content += _borsh_string("Name\x00") + _borsh_string("SYM\x00")
    content += _borsh_string("https://example.invalid/meta.json")
    content += struct.pack("<H", 250)
    if creators:
        content += b"\x01" + struct.pack("<I", 1)
        content += default + b"\x01\x64"
    else:
        content += b"\x00"
    return content + bytes([0, int(is_mutable)])


@pytest.mark.parametrize(
    ("mutable", "has_creators"),
    [(True, False), (False, True)],
)
def test_metaplex_metadata_decode_handles_creators_and_mutability(
    mutable: bool, has_creators: bool
) -> None:
    encoded = base64.b64encode(_metadata_account(mutable, has_creators)).decode()
    metadata = parse_metaplex_metadata(decode_account_data(encoded))

    assert metadata["name"] == "Name"
    assert metadata["symbol"] == "SYM"
    assert metadata["uri"] == "https://example.invalid/meta.json"
    assert metadata["is_mutable"] is mutable
    if has_creators:
        assert metadata["creators"] == [
            {"address": str(Pubkey.default()), "verified": True, "share": 100}
        ]
    else:
        assert metadata["creators"] is None


def test_pool_type_mapping_covers_known_dex_and_label_pairs() -> None:
    cases = [
        ("pumpfun", None, "launch_curve"),
        ("launchlab", [], "launch_curve"),
        ("launchlab", ["LaunchLab"], "launch_curve"),
        ("raydium", None, "lp_token"),
        ("raydium", ["CPMM"], "lp_token"),
        ("raydium", ["Standard"], "lp_token"),
        ("raydium", ["CLMM"], "position_based"),
        ("raydium", ["unknown"], "unknown"),
        ("orca", ["wp"], "position_based"),
        ("orca", ["Whirlpool"], "position_based"),
        ("meteora", ["DLMM"], "position_based"),
        ("meteora", ["DYN2"], "position_based"),
        ("meteora", ["DYN"], "unknown"),
        ("meteora", ["DAMM v2"], "position_based"),
        ("pumpswap", None, "unknown"),
    ]
    for dex_id, labels, expected in cases:
        assert classify_pool_type(dex_id, labels) == expected


def test_main_pool_uses_highest_liquidity_and_missing_liquidity_is_zero() -> None:
    body = [
        {
            "baseToken": {"address": MINT},
            "pairAddress": "no-liquidity",
            "dexId": "pumpfun",
        },
        {
            "baseToken": {"address": MINT},
            "pairAddress": "main",
            "dexId": "raydium",
            "priceUsd": "2.5",
            "liquidity": {"usd": 25},
            "volume": {"h24": 75},
            "priceChange": {"h24": -1.5},
            "pairCreatedAt": 1_700_000_000_000,
            "labels": ["CPMM"],
        },
        {
            "baseToken": {"address": str(Pubkey.new_unique())},
            "pairAddress": "other-token",
            "liquidity": {"usd": 500},
        },
    ]

    pairs = parse_dex_pairs(body, MINT)
    assert len(pairs) == 2
    assert select_main_pool(pairs) == body[1]
    value = market_value(pairs, pools_checked_at="2026-10-04T00:00:00+00:00")
    assert value is not None
    assert value["main_pool"] == {"dex": "raydium", "address": "main", "labels": ["CPMM"]}
    assert value["price_usd"] == 2.5
    assert value["liquidity_usd"] == 25
    assert value["volume_24h_usd"] == 75
    assert value["price_change_24h_pct"] == -1.5
    assert value["pool_count"] == 2
    assert value["total_liquidity_usd"] == 25
    assert value["total_volume_24h_usd"] == 75
    assert value["pools_checked_at"] == "2026-10-04T00:00:00+00:00"


@pytest.mark.asyncio
async def test_raydium_burn_percent_populates_liquidity_lock() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"data": [{"burnPercent": 99.75}]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        checker = SolanaTokenClient(_settings(), http, rate_limits=RATE_LIMITS_OFF)
        fact = await checker.liquidity_lock(
            {"dexId": "raydium", "pairAddress": "pool-address", "labels": None}
        )

    assert fact.status == "ok"
    assert fact.value == {
        "pool_type": "lp_token",
        "dex": "raydium",
        "pool": "pool-address",
        "burned_pct": 99.75,
    }
    assert requests[0].url.path == "/pools/info/ids"
    assert requests[0].url.params["ids"] == "pool-address"


def _token_account(owner: str) -> dict[str, Any]:
    return {"data": {"parsed": {"info": {"owner": owner}}}}


@pytest.mark.asyncio
async def test_top10_excludes_pool_owned_accounts_from_concentration() -> None:
    owners = ["pool-address", "wallet-a", "wallet-b"]

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "rpc.example.invalid":
            body = request.read()
            payload = json.loads(body)
            method = payload["method"]
            if method == "getTokenLargestAccounts":
                result: Any = {
                    "value": [
                        {"address": "account-1", "uiAmountString": "60"},
                        {"address": "account-2", "uiAmountString": "25"},
                        {"address": "account-3", "uiAmountString": "15"},
                    ]
                }
            elif method == "getTokenSupply":
                result = {"value": {"uiAmountString": "100"}}
            else:
                result = {
                    "value": [_token_account(owner) for owner in owners],
                }
            return httpx.Response(
                200,
                json={"jsonrpc": "2.0", "id": payload["id"], "result": result},
            )
        return httpx.Response(404)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        checker = SolanaTokenClient(_settings(), http, rate_limits=RATE_LIMITS_OFF)
        fact = await checker.top10_share(MINT, {"pool-address"})

    assert fact.status == "ok"
    assert fact.value["pct"] == 40.0
    assert fact.value["holders"] == [
        {"owner": "pool-address", "pct": 60.0, "is_pool": True},
        {"owner": "wallet-a", "pct": 25.0, "is_pool": False},
        {"owner": "wallet-b", "pct": 15.0, "is_pool": False},
    ]


@pytest.mark.asyncio
async def test_top10_429_returns_unavailable_fact() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        checker = SolanaTokenClient(_settings(), http, rate_limits=RATE_LIMITS_OFF)
        fact = await checker.top10_share(MINT, set())

    assert fact.status == "unavailable"
    assert fact.detail == "Couldn't check right now (Solana connection rate-limited)"


@pytest.mark.asyncio
async def test_rpc_failure_logs_hostname_without_credential_bearing_url(
    caplog: pytest.LogCaptureFixture,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        checker = SolanaTokenClient(_settings(), http, rate_limits=RATE_LIMITS_OFF)
        with pytest.raises(SourceError):
            await checker.get_mint_account(MINT)

    assert "rpc.example.invalid" in caplog.text
    assert "api-key=hidden" not in caplog.text
    assert "https://rpc.example.invalid/" not in caplog.text


@pytest.mark.asyncio
async def test_rugcheck_failure_does_not_hide_other_card_facts() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "rpc.example.invalid":
            payload = json.loads(request.read())
            method = payload["method"]
            if method == "getAccountInfo" and payload["params"][0] == MINT:
                account = {
                    "owner": SPL_TOKEN_PROGRAM,
                    "data": {
                        "parsed": {
                            "info": {
                                "mintAuthority": None,
                                "freezeAuthority": None,
                                "extensions": [],
                            }
                        }
                    },
                }
            else:
                account = None
            result: Any = {"value": account}
            return httpx.Response(
                200,
                json={"jsonrpc": "2.0", "id": payload["id"], "result": result},
            )
        if request.url.host == "api.dexscreener.com":
            return httpx.Response(
                200,
                json=[
                    {
                        "baseToken": {"address": MINT, "name": "Bonk", "symbol": "BONK"},
                        "pairAddress": "pump-pool",
                        "dexId": "pumpfun",
                        "labels": [],
                        "liquidity": {"usd": 1234},
                    }
                ],
            )
        if request.url.host == "api.geckoterminal.com":
            return httpx.Response(
                200,
                json={"data": {"attributes": {"reserve_in_usd": "1200"}}},
            )
        if request.url.host == "api.rugcheck.xyz":
            return httpx.Response(500)
        return httpx.Response(404)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        checker = SolanaTokenClient(_settings(), http, rate_limits=RATE_LIMITS_OFF)
        engine = SolanaTokenEngine(checker)
        card = await engine.build_card(MINT)

    assert card.name == "Bonk"
    assert card.symbol == "BONK"
    assert card.token_program == "spl-token"
    assert card.facts.mint_authority.status == "ok"
    assert card.facts.mint_authority.value is None
    assert card.facts.token_extensions.value == {"risky": [], "transfer_fee_bps": None}
    assert card.facts.metadata_mutable.status == "unavailable"
    assert card.facts.metadata_mutable.detail == "No metadata account"
    assert card.facts.market.status == "ok"
    assert card.facts.market.value["geckoterminal_liquidity_usd"] == 1200
    assert card.facts.liquidity_lock.status == "not_applicable"
    assert card.second_opinion.status == "unavailable"


@pytest.mark.asyncio
async def test_full_card_uses_all_pools_for_market_and_holder_classification() -> None:
    pairs = [
        {
            "baseToken": {"address": MINT, "name": "Bonk", "symbol": "BONK"},
            "pairAddress": "pool-a",
            "dexId": "pumpfun",
            "labels": [],
            "liquidity": {"usd": 100},
            "volume": {"h24": 10},
        },
        {
            "baseToken": {"address": MINT, "name": "Bonk", "symbol": "BONK"},
            "pairAddress": "pool-b",
            "dexId": "raydium",
            "labels": ["CPMM"],
            "priceUsd": "2",
            "liquidity": {"usd": 400},
            "volume": {"h24": 40},
        },
        {
            "baseToken": {"address": MINT, "name": "Bonk", "symbol": "BONK"},
            "pairAddress": "pool-c",
            "dexId": "orca",
            "labels": ["wp"],
            "liquidity": {"usd": 200},
            "volume": {"h24": 20},
        },
    ]
    accounts = [
        {"address": "account-a", "uiAmountString": "40"},
        {"address": "account-b", "uiAmountString": "30"},
        {"address": "account-c", "uiAmountString": "20"},
        {"address": "account-wallet", "uiAmountString": "10"},
    ]
    owners = ["pool-a", "pool-b", "pool-c", "wallet"]
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.host == "rpc.example.invalid":
            payload = json.loads(request.read())
            method = payload["method"]
            if method == "getAccountInfo":
                account = (
                    {
                        "owner": SPL_TOKEN_PROGRAM,
                        "data": {
                            "parsed": {
                                "info": {
                                    "mintAuthority": None,
                                    "freezeAuthority": None,
                                    "extensions": [],
                                }
                            }
                        },
                    }
                    if payload["params"][0] == MINT
                    else None
                )
                result: Any = {"value": account}
            elif method == "getTokenLargestAccounts":
                result = {"value": accounts}
            elif method == "getTokenSupply":
                result = {"value": {"uiAmountString": "100"}}
            else:
                result = {"value": [_token_account(owner) for owner in owners]}
            return httpx.Response(
                200,
                json={"jsonrpc": "2.0", "id": payload["id"], "result": result},
            )
        if request.url.host == "api.dexscreener.com":
            assert request.url.path == f"/token-pairs/v1/solana/{MINT}"
            return httpx.Response(200, json=pairs)
        if request.url.host == "api.geckoterminal.com":
            return httpx.Response(
                200,
                json={"data": {"attributes": {"reserve_in_usd": "390"}}},
            )
        if request.url.host == "api-v3.raydium.io":
            return httpx.Response(200, json={"data": [{"burnPercent": 99.5}]})
        if request.url.host == "api.rugcheck.xyz":
            return httpx.Response(500)
        return httpx.Response(404)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        checker = SolanaTokenClient(_settings(), http, rate_limits=RATE_LIMITS_OFF)
        card = await SolanaTokenEngine(checker).build_card(MINT)

    market = card.facts.market.value
    assert market["pool_count"] == 3
    assert market["main_pool"]["address"] == "pool-b"
    assert market["total_liquidity_usd"] == 700
    assert market["total_volume_24h_usd"] == 70
    assert market["pools_checked_at"]
    assert card.facts.liquidity_lock.value["pool"] == "pool-b"
    assert card.facts.top10_share.value["pct"] == 10.0
    assert card.facts.top10_share.value["holders"] == [
        {"owner": "pool-a", "pct": 40.0, "is_pool": True},
        {"owner": "pool-b", "pct": 30.0, "is_pool": True},
        {"owner": "pool-c", "pct": 20.0, "is_pool": True},
        {"owner": "wallet", "pct": 10.0, "is_pool": False},
    ]
    assert any(request.url.path == f"/token-pairs/v1/solana/{MINT}" for request in requests)


@pytest.mark.asyncio
async def test_market_refresh_preserves_full_pool_totals_and_unions_pool_addresses() -> None:
    previous_check = "2026-10-04T00:00:00+00:00"
    previous_market = {
        "price_usd": 1.0,
        "liquidity_usd": 400,
        "volume_24h_usd": 40,
        "price_change_24h_pct": 1.0,
        "pool_created_at": None,
        "pool_count": 3,
        "total_liquidity_usd": 700,
        "total_volume_24h_usd": 70,
        "pools_checked_at": previous_check,
        "main_pool": {"dex": "raydium", "address": "old-main", "labels": ["CPMM"]},
        "geckoterminal_liquidity_usd": 390,
    }
    new_pair = {
        "baseToken": {"address": MINT, "name": "Bonk", "symbol": "BONK"},
        "pairAddress": "new-main",
        "dexId": "raydium",
        "labels": ["CPMM"],
        "priceUsd": "2",
        "liquidity": {"usd": 900},
        "volume": {"h24": 90},
    }
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.host == "api.dexscreener.com":
            assert request.url.path == f"/tokens/v1/solana/{MINT}"
            return httpx.Response(200, json=[new_pair])
        if request.url.host == "api-v3.raydium.io":
            return httpx.Response(200, json={"data": [{"burnPercent": 80.0}]})
        return httpx.Response(404)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        checker = SolanaTokenClient(_settings(), http, rate_limits=RATE_LIMITS_OFF)
        engine = SolanaTokenEngine(checker)
        card = solana_module._pending_card(MINT, "Bonk", "BONK", previous_check)
        facts = card.facts.model_copy(
            update={
                "market": solana_module.Fact(
                    status="ok",
                    value=previous_market,
                    source="dexscreener",
                    fetched_at=previous_check,
                ),
                "liquidity_lock": solana_module.Fact(
                    status="ok",
                    value={
                        "pool_type": "lp_token",
                        "dex": "raydium",
                        "pool": "old-main",
                        "burned_pct": 99.0,
                    },
                    source="raydium",
                    fetched_at=previous_check,
                ),
            }
        )
        engine.tokens[MINT] = card.model_copy(update={"facts": facts})
        engine._pool_addresses[MINT] = {"pool-a", "old-main", "pool-c"}

        await engine.refresh_markets_once()

    refreshed = engine.tokens[MINT]
    market = refreshed.facts.market.value
    assert market["main_pool"]["address"] == "new-main"
    assert market["liquidity_usd"] == 900
    assert market["volume_24h_usd"] == 90
    assert market["pool_count"] == 3
    assert market["total_liquidity_usd"] == 700
    assert market["total_volume_24h_usd"] == 70
    assert market["pools_checked_at"] == previous_check
    assert engine._pool_addresses[MINT] == {"pool-a", "old-main", "pool-c", "new-main"}
    assert refreshed.facts.liquidity_lock.value["pool"] == "new-main"
    raydium_request = next(
        request for request in requests if request.url.host == "api-v3.raydium.io"
    )
    assert raydium_request.url.params["ids"] == "new-main"


@pytest.mark.asyncio
async def test_token_2022_metadata_and_rugcheck_api_key_are_reported() -> None:
    rugcheck_headers: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "rpc.example.invalid":
            payload = json.loads(request.read())
            if payload["method"] == "getAccountInfo":
                result: Any = {
                    "value": {
                        "owner": TOKEN_2022_PROGRAM,
                        "data": {
                            "parsed": {
                                "info": {
                                    "mintAuthority": None,
                                    "freezeAuthority": "freeze-authority",
                                    "extensions": [
                                        {
                                            "extension": "transferFeeConfig",
                                            "state": {
                                                "newerTransferFee": {
                                                    "transferFeeBasisPoints": 500,
                                                }
                                            },
                                        },
                                        {
                                            "extension": "permanentDelegate",
                                            "state": {"delegate": "permanent-delegate"},
                                        },
                                        {
                                            "extension": "tokenMetadata",
                                            "state": {
                                                "updateAuthority": "metadata-authority",
                                                "name": "Token 2022",
                                                "symbol": "T22",
                                            },
                                        },
                                    ],
                                }
                            }
                        },
                    }
                }
            elif payload["method"] == "getTokenLargestAccounts":
                return httpx.Response(429)
            else:
                result = {"value": {"uiAmountString": "100"}}
            return httpx.Response(
                200,
                json={"jsonrpc": "2.0", "id": payload["id"], "result": result},
            )
        if request.url.host == "api.dexscreener.com":
            return httpx.Response(200, json=[])
        if request.url.host == "api.rugcheck.xyz":
            rugcheck_headers.append(request.headers.get("X-API-KEY"))
            return httpx.Response(
                200,
                json={
                    "score_normalised": 42,
                    "lpLockedPct": 50.0,
                    "risks": [{"name": "example", "level": "warn", "description": "Example"}],
                },
            )
        return httpx.Response(404)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        checker = SolanaTokenClient(
            _settings(rugcheck_api_key="rugcheck-key"),
            http,
            rate_limits=RATE_LIMITS_OFF,
        )
        card = await SolanaTokenEngine(checker).build_card(MINT, gecko_cross_check=False)

    assert card.token_program == "token-2022"
    assert card.name == "Token 2022"
    assert card.symbol == "T22"
    assert card.facts.metadata_mutable.status == "ok"
    assert card.facts.metadata_mutable.value is True
    assert card.facts.token_extensions.value == {
        "risky": ["transferFeeConfig", "permanentDelegate"],
        "transfer_fee_bps": 500,
    }
    assert rugcheck_headers == ["rugcheck-key"]
    assert card.second_opinion.status == "ok"
    assert card.second_opinion.score_normalised == 42
    assert card.facts.top10_share.status == "unavailable"


@pytest.mark.asyncio
async def test_holder_retries_are_limited_and_rugcheck_retries_once_after_ten_minutes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mints = [str(Pubkey.new_unique()) for _ in range(12)]
    body = {
        "data": [
            {
                "relationships": {
                    "base_token": {"data": {"id": f"solana_{mint}"}},
                },
            }
            for mint in mints
        ],
        "included": [
            {
                "id": f"solana_{mint}",
                "attributes": {"name": "Token", "symbol": "TKN"},
            }
            for mint in mints
        ],
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=body)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        engine = SolanaTokenEngine(
            SolanaTokenClient(_settings(), http, rate_limits=RATE_LIMITS_OFF)
        )
        await engine.discover_once()

    while not engine._queue.empty():
        queued_kind, queued_mint = engine._queue.get_nowait()
        engine._queue.task_done()
        engine._queued.discard((queued_kind, queued_mint))
    for mint in mints:
        card = engine.tokens[mint]
        facts = card.facts.model_copy(
            update={
                "top10_share": unavailable_fact(
                    "solana_rpc",
                    "Couldn't check right now",
                )
            }
        )
        engine.tokens[mint] = card.model_copy(update={"facts": facts})
    engine._rug_retry_at[mints[0]] = 1_600.0
    now = 1_000.0
    monkeypatch.setattr(solana_module, "monotonic", lambda: now)

    engine.schedule_retries()
    assert engine._queue.qsize() == 10
    assert ("rugcheck", mints[0]) not in engine._queued

    now = 1_600.0
    engine.schedule_retries()
    assert engine._queue.qsize() == 11
    assert ("rugcheck", mints[0]) in engine._queued

    engine._queued.discard(("rugcheck", mints[0]))
    engine._rug_retry_attempted.add(mints[0])
    engine.schedule_retries()
    assert engine._queue.qsize() == 11
    assert ("rugcheck", mints[0]) not in engine._queued
