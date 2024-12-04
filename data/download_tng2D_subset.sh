#!/bin/bash

FTPUSER=${FTPUSER:-"cai-ftps"}
FPTPASS=${FTPPASS:-""}

curl -v --ssl -u "$FTPUSER:$FTPPASS" ftp://srv-fs-t-030.zhaw.ch:21/cai-share/illustris/tng50-1.2D/240818_tng50-1_dm_99_gids.1000.2000.hdf5 --output 240818_tng50-1_dm_99_gids.1000.2000.hdf5

