cmake_minimum_required(VERSION 3.15)
project(embedlet VERSION 1.0.0 LANGUAGES C)

# Options
option(EMBEDLET_BUILD_TESTS "Build tests" ON)
option(EMBEDLET_BUILD_EXAMPLES "Build examples" ON)

# Set C standard
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Common compiler flags
set(EMBEDLET_COMPILE_OPTIONS)
set(EMBEDLET_COMPILE_DEFINITIONS)
set(EMBEDLET_LINK_LIBRARIES)

# Platform-specific settings
if(WIN32)
    list(APPEND EMBEDLET_COMPILE_DEFINITIONS _CRT_SECURE_NO_WARNINGS)
else()
    # POSIX systems need pthread and math library
    list(APPEND EMBEDLET_LINK_LIBRARIES m pthread)
endif()

# Compiler-specific flags
if(MSVC)
    # Disable /RTC1 in Release-like configs (fixes D8016 incompatibility)
    add_compile_options("$<$<CONFIG:Release,RelWithDebInfo,MinSizeRel>:/RTC1->")

    # Optimization via generator expressions (multi-config friendly)
    add_compile_options("$<$<CONFIG:Release,RelWithDebInfo>:/O2>")
    add_compile_options("$<$<CONFIG:MinSizeRel>:/O1>")
else()
    # GCC/Clang
    list(APPEND EMBEDLET_COMPILE_OPTIONS -Wall -Wextra)
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        list(APPEND EMBEDLET_COMPILE_OPTIONS -O3)
    else()
        list(APPEND EMBEDLET_COMPILE_OPTIONS -g)
    endif()
endif()

# Interface library for common settings
add_library(embedlet_common INTERFACE)
target_include_directories(embedlet_common INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_compile_options(embedlet_common INTERFACE ${EMBEDLET_COMPILE_OPTIONS})
target_compile_definitions(embedlet_common INTERFACE
    ${EMBEDLET_COMPILE_DEFINITIONS}
    EMBEDLET_IMPLEMENTATION
)
target_link_libraries(embedlet_common INTERFACE ${EMBEDLET_LINK_LIBRARIES})

# Tests
if(EMBEDLET_BUILD_TESTS)
    add_executable(test_embedlet tests/test_embedlet.c)
    target_link_libraries(test_embedlet PRIVATE embedlet_common)

    # Suppress unused variable warnings in tests
    if(NOT MSVC)
        target_compile_options(test_embedlet PRIVATE
            -Wno-unused-variable
            -Wno-unused-but-set-variable
        )
    endif()

    message(STATUS "Building tests: test_embedlet")
endif()

# Examples
if(EMBEDLET_BUILD_EXAMPLES)
    add_executable(example examples/example.c)
    target_link_libraries(example PRIVATE embedlet_common)

    message(STATUS "Building examples: example")
endif()

# Installation (optional)
install(FILES include/embedlet.h DESTINATION include)

if(EMBEDLET_BUILD_TESTS)
    install(TARGETS test_embedlet RUNTIME DESTINATION bin OPTIONAL)
endif()

if(EMBEDLET_BUILD_EXAMPLES)
    install(TARGETS example RUNTIME DESTINATION bin OPTIONAL)
endif()

# Print configuration summary
message(STATUS "")
message(STATUS "===========================================")
message(STATUS "  Embedlet Build Configuration")
message(STATUS "===========================================")
message(STATUS "  Build type:     ${CMAKE_BUILD_TYPE}")
message(STATUS "  Compiler:       ${CMAKE_C_COMPILER_ID} ${CMAKE_C_COMPILER_VERSION}")
message(STATUS "  Architecture:   ${CMAKE_SYSTEM_PROCESSOR} (${CMAKE_SIZEOF_VOID_P}x8-bit)")
message(STATUS "  Build tests:    ${EMBEDLET_BUILD_TESTS}")
message(STATUS "  Build examples: ${EMBEDLET_BUILD_EXAMPLES}")
message(STATUS "===========================================")
message(STATUS "")
