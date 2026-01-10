# SPDX-License-Identifier: GPL-3.0-or-later
#
# Copyright (C) 2026 Emir Kızılçim, Emir Turgut
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

import re
from langchain.schema import Document

import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger(__name__)


def clean_text(text):
    logs = []

    # Remove multiple spaces, newlines, tabs
    if re.search(r"\s{2,}", text):
        old_text = text
        text = re.sub(r"\s+", " ", text)
        if old_text != text:
            logs.append("Condensed multiple spaces and line breaks.")

    """
    # Remove URLs and emails
    if re.search(r"http\S+|www\S+|[\w\.-]+@[\w\.-]+", text):
        old_text = text
        text = re.sub(r"http\S+|www\S+|[\w\.-]+@[\w\.-]+", "", text)
        if old_text != text:
            logs.append("Removed URLs and email addresses.")
    """
    
    # Remove page headers/footers
    if re.search(r"Page \d+ of \d+", text, flags=re.IGNORECASE):
        old_text = text
        text = re.sub(r"Page \d+ of \d+", "", text, flags=re.IGNORECASE)
        if old_text != text:
            logs.append("Removed page headers like 'Page X of Y'.")

    # Remove non-printable characters
    cleaned = "".join(c for c in text if c.isprintable())
    if cleaned != text:
        logs.append("Removed non-printable characters.")
    text = cleaned

    # Remove excessive punctuation
    if re.search(r"[-=]{3,}", text):
        old_text = text
        text = re.sub(r"[-=]{3,}", "", text)
        if old_text != text:
            logs.append("Removed excessive punctuation sequences (---, ===, etc).")

    if logs:
        logger.info("Cleaning Log:")
        for log in logs:
            logger.info(" -%s", log)
    else:
        logger.info("No noise found in this text.")

    return text.strip()

def clean_documents(documents):
    cleaned_docs = []
    for doc in documents:
        cleaned_content = clean_text(doc.page_content)
        cleaned_doc = Document(page_content=cleaned_content, metadata=doc.metadata)
        cleaned_docs.append(cleaned_doc)
    return cleaned_docs