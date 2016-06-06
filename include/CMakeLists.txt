##
# Canary core library external headers.

file(GLOB_RECURSE ALL_CANARY_EXTERNAL_HEADERS "*.h")

# Checks code style.
add_custom_target(check_h "${CPPLINT_TOOL}" ${ALL_CANARY_EXTERNAL_HEADERS} COMMENT "Checks coding style.")

# Formats codes in place.
add_custom_target(format_h "${FORMAT_TOOL}" ${ALL_CANARY_EXTERNAL_HEADERS} COMMENT "Formats all source codes.")
