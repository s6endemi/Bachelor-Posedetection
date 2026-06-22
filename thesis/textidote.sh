#!/usr/bin/env bash

# https://sylvainhalle.github.io/textidote/

if ! command -v textidote &> /dev/null
then
  echo "textidote could not be found.\nAbort."
  exit 1
fi

textidote \
--read-all \
--check en \
--output html \
content.tex \
> /tmp/textidote-report.html

xdg-open /tmp/textidote-report.html

