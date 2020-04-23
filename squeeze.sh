#!/usr/bin/env bash
while read -r
do
	# embed referenced scripts
	[[ $REPLY == *\<script* ]] && [[ $REPLY == *src=* ]] && {
		SRC=${REPLY#*src=\"}
		SRC=${SRC%%\"*}
		[ -r "$SRC" ] && {
			echo '<script>'
			java -jar closure-compiler/*jar < "$SRC" | fgrep -v '$jscomp'
			echo '</script>'
			continue
		}
	}
	# skip comments
	REPLY=${REPLY%%//*}
	# skip indent
	REPLY=${REPLY##*$'\t'}
	# skip empty lines
	[ "$REPLY" ] || continue
	echo "$REPLY"
done
