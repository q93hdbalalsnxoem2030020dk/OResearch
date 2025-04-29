#!/usr/bin/env bash
set -euo pipefail

# Ensure wget, python3, pip3
if ! command -v wget &>/dev/null; then
  if command -v apt-get &>/dev/null; then
    apt-get update && apt-get install -y wget python3-pip
  elif command -v yum &>/dev/null; then
    yum install -y wget python3-pip
  else
    echo "No supported package manager for wget/pip" >&2
    exit 1
  fi
fi

# Ensure pip3 points to pip if missing
if ! command -v pip3 &>/dev/null; then
  ln -sf "$(command -v pip)" /usr/bin/pip3
fi

# Python deps
packages=(requests beautifulsoup4 wikipedia spacy torch numpy scikit-learn colorama)
for pkg in "${packages[@]}"; do
  if ! pip3 show "$pkg" &>/dev/null; then
    pip3 install "$pkg"
  else
    echo "✔ $pkg"
  fi
done

# spaCy model
en_model="en_core_web_sm"
python3 - <<PYCHECK 2>/dev/null
import spacy
spacy.load("$en_model")
PYCHECK
if [ $? -ne 0 ]; then
  python3 -m spacy download "$en_model"
else
  echo "✔ spaCy model $en_model"
fi

# Prompt & download
read -rp "This will download offline articles (~500 MB). Continue? (yes/no) " ans
case "${ans,,}" in
  y* )
    mkdir -p .data
    file=".data/knowledge_base.json"
    if [[ -f "$file" ]]; then
      echo "✔ $file already exists"
      exit 0
    fi

    echo "Resolving real MediaFire link…"
    python3 - <<'PYCODE'
import requests
from bs4 import BeautifulSoup
mf_page = "https://www.mediafire.com/file/op39n9w0sbc74yh/knowledge_base.json/file"
r = requests.get(mf_page)
r.raise_for_status()
soup = BeautifulSoup(r.text, "html.parser")
btn = soup.find("a", id="downloadButton")
if not btn:
    print("ERROR: could not find download button", flush=True)
    exit(1)
dl_url = btn["href"]
print(f"Downloading from: {dl_url}", flush=True)
with requests.get(dl_url, stream=True) as resp:
    resp.raise_for_status()
    with open(".data/knowledge_base.json", "wb") as out:
        for chunk in resp.iter_content(chunk_size=8192):
            out.write(chunk)
print("Download complete.", flush=True)
PYCODE
    ;;
  * )
    echo "Aborted."
    ;;
esac
