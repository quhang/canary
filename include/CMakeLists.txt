file(GLOB_RECURSE ALL_EXTERNAL_HEADERS "*.h")

# Checks code style.
add_custom_target(check-h ${CMAKE_SOURCE_DIR}/tools/cpplint.py ${ALL_EXTERNAL_HEADERS} COMMENT "Check coding style.")

# Format codes in place.
add_custom_target(format-h clang-format-3.6 --style=Google -i ${ALL_EXTERNAL_HEADERS} COMMENT "Reformat all source codes.")
