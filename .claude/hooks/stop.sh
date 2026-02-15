#!/bin/bash

set -eu

cd "$(dirname "$0")"

CLAUDE_CODE_FEEDBACK_EXIT_CODE=2

source ./common.sh

if ! check_command mise; then
	echo "mise command not found. Please install mise to use this hook."
	exit 1
fi

mise exec -- just fmt && mise exec -- just lint
status=$?

if [ $status -ne 0 ]; then
	echo "Formatting or linting failed. Please fix the issues above."
	exit $CLAUDE_CODE_FEEDBACK_EXIT_CODE
fi

if [ -n "${CLAUDE_CODE_REMOTE:-}" ]; then
	mise exec -- just test
	status=$?
	if [ $status -ne 0 ]; then
		echo "Tests failed. Please fix the issues above."
		exit $CLAUDE_CODE_FEEDBACK_EXIT_CODE
	fi
fi

if [ -z "${SLACK_WEBHOOK_URL:-}" ]; then
	echo "SLACK_WEBHOOK_URL is not set. Skipping Slack notification."
else
	echo "Sending Slack notification..."
	message="PROJECT: Revaital Nexus\nBranch: $(git rev-parse --abbrev-ref HEAD)\nStatus: Completed Successfully"
	curl -X POST -H 'Content-type: application/json' --data "{\"text\":\"$message\"}" "$SLACK_WEBHOOK_URL"
fi
