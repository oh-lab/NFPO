# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import ipaddress
import os
import socket
import fcntl
from typing import Optional


def is_ipv4(ip_str: str) -> bool:
    """
    Check if the given string is an IPv4 address

    Args:
        ip_str: The IP address string to check

    Returns:
        bool: Returns True if it's an IPv4 address, False otherwise
    """
    try:
        ipaddress.IPv4Address(ip_str)
        return True
    except ipaddress.AddressValueError:
        return False


def is_ipv6(ip_str: str) -> bool:
    """
    Check if the given string is an IPv6 address

    Args:
        ip_str: The IP address string to check

    Returns:
        bool: Returns True if it's an IPv6 address, False otherwise
    """
    try:
        ipaddress.IPv6Address(ip_str)
        return True
    except ipaddress.AddressValueError:
        return False


def is_valid_ipv6_address(address: str) -> bool:
    try:
        ipaddress.IPv6Address(address)
        return True
    except ValueError:
        return False


def _parse_port_range(port_range: str | tuple[int, int]) -> tuple[int, int]:
    if isinstance(port_range, tuple):
        start, end = port_range
    else:
        separators = (":", "-", ",")
        for separator in separators:
            if separator in port_range:
                start_str, end_str = port_range.split(separator, 1)
                start, end = int(start_str), int(end_str)
                break
        else:
            raise ValueError(
                "Port range must use one of ':', '-' or ',' as a separator, "
                f"got {port_range!r}."
            )

    if start <= 0 or end <= start:
        raise ValueError(f"Invalid port range: {port_range!r}")
    return start, end


def _iter_ports(start: int, end: int, last_port: int):
    next_port = last_port + 1 if start <= last_port < end else start
    yield from range(next_port, end)
    yield from range(start, next_port)


def get_exclusive_port(
    address: str,
    port_range: str | tuple[int, int] = "45000:55000",
    state_file: Optional[str] = None,
) -> tuple[int, socket.socket]:
    """Reserve a unique local TCP port without SO_REUSEPORT.

    This is intended for multi-process launchers that need to reserve ports
    across several concurrently starting workers on the same node.
    """
    start, end = _parse_port_range(port_range)
    if state_file is None:
        state_file = f"/tmp/verl-exclusive-port-{os.getuid()}.txt"

    state_dir = os.path.dirname(state_file)
    if state_dir:
        os.makedirs(state_dir, exist_ok=True)

    family = socket.AF_INET6 if is_valid_ipv6_address(address) else socket.AF_INET

    with open(state_file, "a+", encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.seek(0)
        raw_value = f.read().strip()
        last_port = int(raw_value) if raw_value else start - 1

        for port in _iter_ports(start, end, last_port):
            sock = socket.socket(family=family, type=socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind((address, port))
            except OSError:
                sock.close()
                continue

            f.seek(0)
            f.truncate()
            f.write(str(port))
            f.flush()
            os.fsync(f.fileno())
            return port, sock

    raise RuntimeError(f"Could not reserve a port in range [{start}, {end}) for {address}")


def get_exclusive_port_block(
    address: str,
    block_size: int,
    port_range: str | tuple[int, int] = "47000:55000",
    state_file: Optional[str] = None,
) -> tuple[int, list[socket.socket]]:
    """Reserve a contiguous block of local TCP ports."""
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")

    start, end = _parse_port_range(port_range)
    if state_file is None:
        state_file = f"/tmp/verl-exclusive-port-block-{os.getuid()}.txt"

    state_dir = os.path.dirname(state_file)
    if state_dir:
        os.makedirs(state_dir, exist_ok=True)

    family = socket.AF_INET6 if is_valid_ipv6_address(address) else socket.AF_INET
    last_base = end - block_size

    with open(state_file, "a+", encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.seek(0)
        raw_value = f.read().strip()
        last_port = int(raw_value) if raw_value else start - 1

        for base_port in _iter_ports(start, last_base + 1, last_port):
            socks: list[socket.socket] = []
            try:
                for offset in range(block_size):
                    sock = socket.socket(family=family, type=socket.SOCK_STREAM)
                    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    sock.bind((address, base_port + offset))
                    socks.append(sock)
            except OSError:
                for sock in socks:
                    sock.close()
                continue

            f.seek(0)
            f.truncate()
            f.write(str(base_port + block_size - 1))
            f.flush()
            os.fsync(f.fileno())
            return base_port, socks

    raise RuntimeError(
        f"Could not reserve a contiguous block of {block_size} ports in range [{start}, {end}) for {address}"
    )


def get_free_port(address: str) -> tuple[int, socket.socket]:
    family = socket.AF_INET
    if is_valid_ipv6_address(address):
        family = socket.AF_INET6

    sock = socket.socket(family=family, type=socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
    sock.bind((address, 0))

    port = sock.getsockname()[1]
    return port, sock
