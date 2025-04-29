#!/usr/bin/env bash
set -e
if ! command -v wget >/dev/null; then
  if command -v apt-get >/dev/null; then
    apt-get update && apt-get install -y wget python3-pip
  elif command -v yum >/dev/null; then
    yum install -y wget python3-pip
  else
    echo "No supported package manager found for wget/pip" && exit 1
  fi
fi

if ! command -v pip3 >/dev/null; then
  ln -s "$(command -v pip)" /usr/bin/pip3 || true
fi

packages=(requests wikipedia spacy torch numpy scikit-learn colorama)
for pkg in "${packages[@]}"; do
  if ! pip3 show "$pkg" >/dev/null; then
    pip3 install "$pkg"
  else
    echo "$pkg already installed"
  fi
done

# spaCy model
en_model="en_core_web_sm"
if ! python3 -c "import spacy; spacy.load('$en_model')" >/dev/null 2>&1; then
  python3 -m spacy download $en_model
else
  echo "spaCy model $en_model already installed"
fi

while true; do
  read -p "This will download offline articles (~500MB). Continue? (yes/no) " answer
  case "${answer,,}" in
    y* )
      mkdir -p "$PWD/.data"
      file="$PWD/.data/knowledge_base.json"
      if [ -f "$file" ]; then
        echo "knowledge_base.json already exists"
      else
        wget --trust-server-names --progress=bar:force -O "$file" "https://download1498.mediafire.com/6le73j64qkzg/op39n9w0sbc74yh/knowledge_base.json" 2>&1 | sed -u 's,\\r*\[^ ]\,\\r\\1,'
      fi
      break
      ;;
    n* )
      break
      ;;
    * )
      echo "Please answer yes or no."
      ;;
  esac
done
