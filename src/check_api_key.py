"""
Check an API key is reachable and working, before relying on it.

A rejected key and a mistyped key produce the same 403, so this separates them:
it reports where the key was found, whether it looks malformed, and then asks the
provider which models that key can actually use. If the models call succeeds, the
key is fine and any later failure is about the model name, not the credential.

The key itself is NEVER printed - only its length and first four characters, which
is enough to spot a truncated paste without putting the secret on screen or into a
terminal scrollback that might get shared.

    python src/check_api_key.py                     # checks GROQ_API_KEY
    python src/check_api_key.py ANTHROPIC_API_KEY
"""

import importlib.util
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

PROVIDERS = {
    "GROQ_API_KEY": ("https://api.groq.com/openai/v1/models", "Groq"),
    "ANTHROPIC_API_KEY": (None, "Anthropic"),
}


def main() -> None:
    var = sys.argv[1] if len(sys.argv) > 1 else "GROQ_API_KEY"
    url, provider = PROVIDERS.get(var, (None, var))

    spec = importlib.util.spec_from_file_location(
        "narrator", ROOT / "src" / "15_narrate_findings.py")
    narrator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(narrator)

    print(f"\n{'=' * 72}\nAPI KEY CHECK - {var}\n{'=' * 72}")

    # ---- 1. where is it, if anywhere? ------------------------------------
    in_env = bool(os.environ.get(var))
    env_file = ROOT / ".env"
    in_file = env_file.exists() and var in env_file.read_text(encoding="utf-8")
    print(f"  environment variable : {'found' if in_env else 'not set'}")
    print(f"  .env file            : "
          f"{'found' if in_file else 'not present' if not env_file.exists() else 'file exists, key not in it'}")

    key = narrator.read_api_key(var)
    if not key:
        print("\n  NO KEY FOUND. Easiest fix - create a file called .env in")
        print(f"  {ROOT}")
        print(f"  containing one line:\n\n      {var}=gsk_your_key_here\n")
        print("  No quotes, no spaces around the '='. .env is gitignored, so it")
        print("  cannot reach GitHub by accident.")
        raise SystemExit(1)

    print(f"\n  key resolved         : length={len(key)}, starts {key[:4]!r}")
    if len(key) < 20:
        print("  WARNING: that is short for an API key - check it was pasted whole.")

    if url is None:
        print(f"\n  No live check available for {provider} here; the key is present.")
        return

    # ---- 2. does the provider accept it? ---------------------------------
    print(f"\n  asking {provider} which models this key can use ...")
    req = urllib.request.Request(url, headers={
        "Authorization": f"Bearer {key}",
        # see src/15: the default urllib UA is Cloudflare-blocked
        "User-Agent": "hospital-bottleneck-detective/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")[:300]
        print(f"\n  REJECTED - HTTP {exc.code}")
        if exc.code in (401, 403):
            if "1010" in body or "<html" in body.lower():
                print("  Cloudflare blocked the REQUEST - the key was never checked.")
                print("  This is a client-fingerprint block, not a credential fault.")
            else:
                print("  The provider rejected the key itself. Check it was pasted")
                print("  whole, and has not been deleted or regenerated.")
        print(f"\n  provider said: {body}")
        raise SystemExit(1)
    except Exception as exc:                                   # noqa: BLE001
        print(f"\n  Could not reach {provider}: {type(exc).__name__}: {exc}")
        print("  That is a network problem, not a key problem.")
        raise SystemExit(1)

    models = sorted(m["id"] for m in data.get("data", []))
    print(f"\n  KEY WORKS - {len(models)} models available to it.\n")
    interesting = [m for m in models
                   if any(t in m for t in ("llama", "gpt-oss", "qwen", "mixtral"))]
    for m in interesting[:14]:
        print(f"    {m}")
    if not interesting:
        for m in models[:14]:
            print(f"    {m}")

    print("\n  Use one of the names above with --model. If a model name is wrong")
    print("  you get a 404, which is a different error from this one.")


if __name__ == "__main__":
    main()
