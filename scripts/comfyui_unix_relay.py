#!/usr/bin/env python3
"""Tiny TCP-to-Unix-socket relay for local ComfyUI.

Use this when Docker bridge traffic to host ports is blocked but a container has
access to a bind-mounted directory. The relay runs on the host, listens on a
Unix domain socket in that shared directory, and forwards raw HTTP/WebSocket TCP
traffic to the host ComfyUI port.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import signal
from pathlib import Path


async def _pipe(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    try:
        while not reader.at_eof():
            data = await reader.read(64 * 1024)
            if not data:
                break
            writer.write(data)
            await writer.drain()
    except (BrokenPipeError, ConnectionResetError):
        pass
    finally:
        try:
            writer.close()
            await writer.wait_closed()
        except Exception:
            pass


async def _handle_client(
    client_reader: asyncio.StreamReader,
    client_writer: asyncio.StreamWriter,
    *,
    target_host: str,
    target_port: int,
) -> None:
    try:
        upstream_reader, upstream_writer = await asyncio.open_connection(target_host, target_port)
    except Exception:
        client_writer.close()
        await client_writer.wait_closed()
        return
    await asyncio.gather(
        _pipe(client_reader, upstream_writer),
        _pipe(upstream_reader, client_writer),
    )


async def main() -> int:
    parser = argparse.ArgumentParser(description="Relay a Unix socket to a local ComfyUI TCP endpoint.")
    parser.add_argument("--socket", required=True, help="Unix socket path, e.g. media-data/comfyui.sock")
    parser.add_argument("--target-host", default="127.0.0.1")
    parser.add_argument("--target-port", type=int, default=8188)
    parser.add_argument("--mode", default="666", help="chmod mode for the socket; default: 666")
    args = parser.parse_args()

    socket_path = Path(args.socket).expanduser().resolve()
    socket_path.parent.mkdir(parents=True, exist_ok=True)
    if socket_path.exists():
        socket_path.unlink()

    server = await asyncio.start_unix_server(
        lambda r, w: _handle_client(r, w, target_host=args.target_host, target_port=args.target_port),
        path=str(socket_path),
    )
    os.chmod(socket_path, int(args.mode, 8))

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop.set)
        except NotImplementedError:
            pass

    print(f"comfyui unix relay listening on {socket_path} -> {args.target_host}:{args.target_port}", flush=True)
    async with server:
        await stop.wait()
    server.close()
    await server.wait_closed()
    try:
        socket_path.unlink()
    except FileNotFoundError:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
