"""Start Streamlink, working around Windows blocking pycryptodome's compiled modules.

Windows Smart App Control / Application Control can refuse to load pycryptodome's unsigned .pyd files
(error 0x11c7). Streamlink imports pycryptodome at start-up, but only uses it for AES-encrypted HLS and a
few non-Twitch sites; Twitch streams aren't encrypted. If the real module can't load, we put a
placeholder in its place that raises a clear error only if encryption is ever actually needed.

Usage: python -m clipper.streamlink_runner <streamlink arguments>
"""

from __future__ import annotations

import os
import sys
import types

BLOCKED_MESSAGE = ("this stream is encrypted, which needs pycryptodome, but Windows blocked "
                   "pycryptodome's files from loading")


class _Unavailable:
    """Stands in for a pycryptodome class: plain constants work, anything that would encrypt raises."""

    def __init__(self, name: str, **constants):
        self._name = name
        self.__dict__.update(constants)

    def __getattr__(self, attr):
        raise RuntimeError(f"{self._name}.{attr}: {BLOCKED_MESSAGE}")


def _unavailable_function(name: str):
    def fn(*args, **kwargs):
        raise RuntimeError(f"{name}: {BLOCKED_MESSAGE}")
    return fn


def install_crypto_placeholder():
    names = {
        "Crypto.Cipher": {"AES": _Unavailable("AES", block_size=16, key_size=(16, 24, 32), MODE_ECB=1, MODE_CBC=2,
                                               MODE_CFB=3, MODE_OFB=5, MODE_CTR=6), "PKCS1_v1_5": _Unavailable("PKCS1_v1_5")},
        "Crypto.Hash": {"MD5": _Unavailable("MD5"), "SHA256": _Unavailable("SHA256")},
        "Crypto.PublicKey": {"RSA": _Unavailable("RSA")},
        "Crypto.Util.Padding": {"pad": _unavailable_function("pad"), "unpad": _unavailable_function("unpad")},
        "Crypto.Util": {},
        "Crypto": {},
    }
    for mod_name in ("Crypto", "Crypto.Cipher", "Crypto.Hash", "Crypto.PublicKey", "Crypto.Util",
                     "Crypto.Util.Padding"):
        mod = types.ModuleType(mod_name)
        mod.__path__ = []  # mark as a package so submodule imports resolve from sys.modules
        for attr, value in names[mod_name].items():
            setattr(mod, attr, value)
        sys.modules[mod_name] = mod
    for parent, child in (("Crypto", "Cipher"), ("Crypto", "Hash"), ("Crypto", "PublicKey"),
                          ("Crypto", "Util"), ("Crypto.Util", "Padding")):
        setattr(sys.modules[parent], child, sys.modules[f"{parent}.{child}"])


def main():
    force = os.environ.get("CLIPPER_FORCE_CRYPTO_PLACEHOLDER") == "1"  # for tests
    try:
        if force:
            raise OSError("forced for testing")
        from Crypto.Cipher import AES  # noqa: F401
    except (OSError, ImportError) as e:
        for name in [m for m in sys.modules if m == "Crypto" or m.startswith("Crypto.")]:
            del sys.modules[name]  # drop the half-imported real package
        install_crypto_placeholder()
        print(f"[clipper] pycryptodome can't load ({str(e)[:120]}); continuing without it "
              f"(Twitch streams don't need it)", file=sys.stderr)
    from streamlink_cli.main import main as streamlink_main
    streamlink_main()


if __name__ == "__main__":
    main()
