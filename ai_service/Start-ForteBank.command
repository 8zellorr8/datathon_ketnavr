#!/bin/sh
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd) || exit 1
sh "$SCRIPT_DIR/Start-ForteBank.sh" "$@"
FORTE_EXIT_CODE=$?
if [ "$FORTE_EXIT_CODE" -ne 0 ]; then
    printf '\n%s\n' 'ForteBank could not start. Read the error above. Press Enter to close.'
    read -r FORTE_REPLY
fi
exit "$FORTE_EXIT_CODE"
