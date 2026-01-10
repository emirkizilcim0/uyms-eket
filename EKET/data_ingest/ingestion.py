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

from EKET.ingest import main as ingest_data
from EKET.utils import get_config, Path

import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr  # Log to stderr to avoid interfering with JSON stdout
)

logger = logging.getLogger(__name__)

def data_ingestion():
    logger.info("Documents in data folder:")
    for f in Path(get_config()['DATA_DIR']).glob('*'):
        logger.info(f"- {f.name}")
    
    ingest_data()

    