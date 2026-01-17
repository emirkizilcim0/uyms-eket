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

# SPDX-License-Identifier: GPL-3.0-or-later

from .answer import evaluate_answers, answer_checker
from .create import generate_quiz_from_context, parse_mcq, parse_open_ended
from .query import get_context_and_language
from .ingest import load_documents, split_text, save_to_chroma
from .clean import clean_text, clean_documents

__all__ = [
    "evaluate_answers",
    "answer_checker",
    "generate_quiz_from_context",
    "parse_mcq",
    "parse_open_ended",
    "get_context_and_language",
    "load_documents",
    "split_text",
    "save_to_chroma",
    "clean_text",
    "clean_documents",
]

__version__ = "0.1.1"
