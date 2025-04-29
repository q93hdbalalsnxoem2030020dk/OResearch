#!/usr/bin/env bash
set -euo pipefail

if ! command -v wget &>/dev/null; then
  if command -v apt-get &>/dev/null; then
    apt-get update && apt-get install -y wget python3-pip xdg-utils
  elif command -v yum &>/dev/null; then
    yum install -y wget python3-pip xdg-utils
  else
    echo "No supported package manager found for wget/pip/xdg-utils" >&2
    exit 1
  fi
fi

if ! command -v pip3 &>/dev/null; then
  ln -sf "$(command -v pip)" /usr/bin/pip3
fi

packages=(requests beautifulsoup4 wikipedia spacy torch numpy scikit-learn colorama)
for pkg in "${packages[@]}"; do
  if ! pip3 show "$pkg" &>/dev/null; then
    pip3 install "$pkg"
  else
    echo "✔ $pkg already installed"
  fi
done

model="en_core_web_sm"
if ! python3 -c "import spacy; spacy.load('$model')" &>/dev/null; then
  python3 -m spacy download "$model"
else
  echo "✔ spaCy model $model is already installed"
fi

mkdir -p .data
file=".data/knowledge_base.json"
url="https://www.mediafire.com/file/op39n9w0sbc74yh/knowledge_base.json/file"

if [[ ! -f "$file" ]]; then
  echo
  echo "----------------------------------------------------------"
  echo "The file 'knowledge_base.json' (~500MB) is required."
  echo "Please download it manually from:"
  echo "  $url"
  echo
  echo "Then place it here:"
  echo "  $(realpath .data)"
  echo "----------------------------------------------------------"
  echo

  opened=0
  if command -v xdg-open &>/dev/null; then
    xdg-open "$url" >/dev/null 2>&1 && opened=1
  elif command -v open &>/dev/null; then
    open "$url" >/dev/null 2>&1 && opened=1
  fi

  if [[ $opened -eq 0 ]]; then
    echo "Could not open the URL automatically. Please open this manually:"
    echo "  $url"
  fi

  while [[ ! -f "$file" ]]; do
    read -rp "Waiting for 'knowledge_base.json'... Press Enter once placed in .data/"
  done
  echo "✔ File detected."
else
  echo "✔ knowledge_base.json already exists"
fi
