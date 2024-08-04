# Make code in src directory available
export PYTHONPATH=$$PYTHONPATH:src

width_clause = $(if $(width),-w $(width),)

# Run the clipper module main function
clipper:
	@if [ -z "$(code)" ]; then \
		echo "\nRequired variable 'code' is not set.\n"; \
		exit 1; \
	fi

	@ mkdir -p target/$(code)
	python -m jwk.clipper -v dataset/$(code).webm -t dataset/$(code).csv -o target/$(code) --name $(code) $(width_clause)


# Run the posefitter module main function
posefitter:
	@if [ -z "$(code)" ]; then \
		echo "\nRequired variable 'code' is not set.\n"; \
		exit 1; \
	fi

	@ mkdir -p target/$(code)-posed

	for vid in target/$(code)/*.mp4; do \
		python -m jwk.posefitter -v $$vid -o target/$(code)-posed; \
	done
