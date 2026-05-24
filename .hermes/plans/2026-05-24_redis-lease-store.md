# Redis Context Lease Store — Multi-Worker Promotion

## Goal

Optionally promote `ContextScheduler` leases from process-local to Redis-backed,
so multiple `langgraph-api` workers can coordinate context admission against the
same llama.cpp server without overbooking.

**Default: OFF** (`local`). Single-worker setups pay zero cost.

## Current State

- `leases.py`: `LeaseStore` class with `asyncio.Lock`-protected `dict`
  (`_leases: dict[str, ContextLease]`). Stores leases in process memory.
- `scheduler.py`: `ContextScheduler.__init__` accepts `lease_store: LeaseStore | None`.
  Defaults to `GLOBAL_LEASE_STORE` (module-level singleton).
- `docker-compose.yml`: Redis container already exists (`redis:alpine` on port 6379).
  `langgraph-api` and `api-bridge` already have `REDIS_URL` and `redis` dependency.
- No Redis client import exists anywhere in `ai_stack/`.

## Approach

### 1. Refactor `LeaseStore` into abstract base

Current:

```python
class LeaseStore:
    def __init__(self):
        self._leases: dict = {}
        self._lock = asyncio.Lock()
```

New:

```python
class LeaseStore(ABC):
    @abstractmethod
    async def active_for_instance(self, instance_id: str) -> list[ContextLease]: ...
    @abstractmethod
    async def active_required_tokens(self, instance_id: str) -> int: ...
    @abstractmethod
    async def try_add(self, lease: ContextLease, *, capacity_tokens: int) -> tuple[bool, int]: ...
    @abstractmethod
    async def release(self, lease_id: str, *, status: str = "released") -> ContextLease | None: ...

class LocalLeaseStore(LeaseStore):
    # Same implementation as current LeaseStore
    ...
```

This is zero-behavior-change refactoring. `GLOBAL_LEASE_STORE` stays a `LocalLeaseStore()`.

### 2. Add `RedisLeaseStore` (optional import)

```python
class RedisLeaseStore(LeaseStore):
    def __init__(self, redis_url: str, prefix: str = "alpharavis:lease:", ttl: int = 600):
        ...
```

Uses `redis.asyncio` (async Redis client). Isolates the import so `redis` is
not required at module load time — only when this class is constructed.

**How it works:**

- `active_for_instance(instance_id)` — `HGETALL <prefix>leases` filtered by `instance_id` + `status=active`
- `active_required_tokens(instance_id)` — `SUM` over same filtered set
- `try_add(lease, capacity_tokens)` — Atomic via **Lua script**:
  1. Calculate current active sum for instance
  2. If `active + lease.required > capacity`: return `(false, active)`
  3. Else: `HSET` lease with TTL, return `(true, active)`
- `release(lease_id, status)` — `HSET lease_id status=<status>`
- Stale leases auto-expire via Redis TTL. If a worker crashes, its leases evaporate
  after `ttl` seconds instead of leaking forever.

**Lease serialization:** `ContextLease.to_dict()` → `json.dumps()` → Redis hash field.
Already exists as `asdict(self)`.

**Safety:** Lua script runs atomically — no race condition between read and write.
Multiple Workers A and B both see "80K free" → only one `try_add` wins atomically.

### 3. Feature flag / ENV

```
ALPHARAVIS_CONTEXT_LEASE_BACKEND=local   # default: process-local dict (current behavior)
ALPHARAVIS_CONTEXT_LEASE_BACKEND=redis   # use shared Redis store
ALPHARAVIS_REDIS_URL=redis://redis:6379  # default, already in docker-compose
ALPHARAVIS_CONTEXT_LEASE_TTL=600         # seconds until stale lease expires
```

**No change to existing behavior when `ALPHARAVIS_CONTEXT_LEASE_BACKEND=local`**
or when the variable is unset (defaults to `local`).

### 4. Wire into `ContextScheduler.from_env()`

```python
@classmethod
def from_env(cls) -> "ContextScheduler | None":
    manager = UbuntuLlamaManagerClient.from_env()
    if manager is None:
        return None
    lease_store = _make_lease_store_from_env()
    return cls(manager_client=manager, lease_store=lease_store, ...)


def _make_lease_store_from_env() -> LeaseStore:
    backend = os.getenv("ALPHARAVIS_CONTEXT_LEASE_BACKEND", "local").strip().lower()
    if backend == "redis":
        redis_url = os.getenv("ALPHARAVIS_REDIS_URL", "redis://redis:6379")
        ttl = int(os.getenv("ALPHARAVIS_CONTEXT_LEASE_TTL", "600"))
        try:
            return RedisLeaseStore(redis_url=redis_url, ttl=ttl)
        except Exception as exc:
            LOGGER.warning("RedisLeaseStore init failed, falling back to LocalLeaseStore", exc_info=exc)
            return LocalLeaseStore()
    return LocalLeaseStore()
```

Graceful fallback: if Redis is unreachable at startup, log warning and fall back
to local. The system doesn't crash.

## Files

| File | Action |
|------|--------|
| `ai_stack/context_budget/leases.py` | Refactor: abstract `LeaseStore`, move impl to `LocalLeaseStore`, add `RedisLeaseStore` |
| `ai_stack/context_budget/scheduler.py` | Add `_make_lease_store_from_env()`, call from `from_env()` |
| `tests/test_llama_context_scheduler.py` | Add tests: `LocalLeaseStore` behavior unchanged, `RedisLeaseStore` atomic admission, Lua script edge cases |
| `.env(exaple)` | Document new keys |
| `docs/ALPHARAVIS_OPEN_TASKS.md` | Mark "Promote context leases to Redis" as implemented |
| `docs/ALPHARAVIS_CHANGES.md` | Record the change, rationale, feature flag |
| `docs/ALPHARAVIS_USAGE_NOTES.md` | Document `ALPHARAVIS_CONTEXT_LEASE_BACKEND` usage |
| `docs/ALPHARAVIS_ARCHITECTURE.md` | Add note about Redis lease store for multi-worker scaling |

## Tests

1. **`test_lease_store_is_local_by_default`** — `_make_lease_store_from_env()` returns `LocalLeaseStore` when env unset
2. **`test_redis_lease_store_atomic_try_add`** — Two leases, capacity=1000, each needs 600 → second rejected
3. **`test_redis_lease_store_release`** — After release, tokens freed, new lease admitted
4. **`test_redis_lease_store_ttl_expiry`** — Lease with short TTL expires, tokens freed
5. **`test_redis_lease_store_fallback_on_error`** — Invalid Redis URL → falls back to `LocalLeaseStore`
6. **`test_local_lease_store_unchanged`** — Existing tests still pass with `LocalLeaseStore`

Redis tests use `fakeredis` or a real Redis if available (same pattern as other tests —
skip if no Redis, run if `REDIS_URL` set). Tests are optional — core logic is in
the Lua script and `LocalLeaseStore` which always runs.

## Verification

```bash
# Unit tests (always run)
pytest -q tests/test_llama_context_scheduler.py -v

# Optional: Redis integration tests
REDIS_URL=redis://localhost:6379 pytest -q tests/test_llama_context_scheduler.py -v -k redis

# AST parse
python -m py_compile ai_stack/context_budget/leases.py
python -m py_compile ai_stack/context_budget/scheduler.py

# Bridge smoke (no behavior change for local mode)
python scripts/alpharavis_setup.py bridge-smoke
```

## Risks / Tradeoffs

- **Redis dependency is lazy:** `redis.asyncio` only imported when `RedisLeaseStore`
  is constructed. If `redis` package is missing, falls back to `LocalLeaseStore`.
  No new hard dependency.
- **Lua script complexity:** Simple script (~15 lines). Race-free. Tested with edge cases.
- **TTL is safety net, not precision:** If TTL=600 and a worker hangs 10 minutes
  mid-request, its lease expires. The hung request still runs but new requests
  get admitted. Better than permanent leak. Operator can tune TTL.
- **Performance:** Redis call per lease acquire/release. For single-worker setups
  this is unnecessary overhead — which is why `local` is the default.
