#!/bin/bash

FTPUSER=${FTPUSER:-"cai-ftps"}
FPTPASS=${FTPPASS:-""}

curl -v --ssl -u "$FTPUSER:$FTPPASS" ftp://srv-fs-t-030.zhaw.ch:21/cai-share/xray.zip --output xray.zip
