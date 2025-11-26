@rem scripts/build.bat
@rem Build script for Windows (MSVC and MinGW64)
@rem Usage: build.bat [msvc|mingw] [debug|release]

@echo off
setlocal enabledelayedexpansion

cd /d "%~dp0\.."

set "COMPILER=%~1"
set "CONFIG=%~2"

if "%COMPILER%"=="" set "COMPILER=msvc"
if "%CONFIG%"=="" set "CONFIG=release"

set "INCLUDE_DIR=include"
set "EXAMPLES_DIR=examples"
set "TESTS_DIR=tests"
set "BUILD_DIR=build\%COMPILER%\%CONFIG%"

if not exist "%BUILD_DIR%" mkdir "%BUILD_DIR%"

echo.
echo ============================================
echo  Building embedlet (%COMPILER% / %CONFIG%)
echo ============================================
echo.

if /i "%COMPILER%"=="msvc" goto :build_msvc
if /i "%COMPILER%"=="mingw" goto :build_mingw
echo Unknown compiler: %COMPILER%
echo Usage: build.bat [msvc^|mingw] [debug^|release]
exit /b 1

:build_msvc
echo Using MSVC...

rem Check for Visual Studio environment
where cl >nul 2>&1
if errorlevel 1 (
    echo Error: cl.exe not found. Run from Developer Command Prompt
    echo   or run: "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
    exit /b 1
)

rem Set compiler flags
set "COMMON_FLAGS=/nologo /W4 /I%INCLUDE_DIR% /D_CRT_SECURE_NO_WARNINGS"

if /i "%CONFIG%"=="debug" (
    set "OPT_FLAGS=/Od /Zi /DDEBUG"
) else (
    set "OPT_FLAGS=/O2 /DNDEBUG"
)

rem SIMD flags - AVX2 is widely supported, AVX-512 requires specific CPUs
set "SIMD_FLAGS=/arch:AVX2"

rem Build test executable
echo Building test_embedlet.exe...
cl %COMMON_FLAGS% %OPT_FLAGS% %SIMD_FLAGS% ^
    /Fe:"%BUILD_DIR%\test_embedlet.exe" ^
    "%TESTS_DIR%\test_embedlet.c" ^
    /link /INCREMENTAL:NO
if errorlevel 1 (
    echo Failed to build test_embedlet.exe
    exit /b 1
)

rem Build example executable
echo Building example.exe...
cl %COMMON_FLAGS% %OPT_FLAGS% %SIMD_FLAGS% ^
    /Fe:"%BUILD_DIR%\example.exe" ^
    "%EXAMPLES_DIR%\example.c" ^
    /link /INCREMENTAL:NO
if errorlevel 1 (
    echo Failed to build example.exe
    exit /b 1
)

rem Clean up object files
del *.obj 2>nul

goto :done

:build_mingw
echo Using MinGW64...

rem Check for MinGW
where gcc >nul 2>&1
if errorlevel 1 (
    echo Error: gcc not found. Add MinGW64 bin directory to PATH
    exit /b 1
)

rem Set compiler flags
set "COMMON_FLAGS=-Wall -Wextra -I%INCLUDE_DIR% -std=c11"

if /i "%CONFIG%"=="debug" (
    set "OPT_FLAGS=-O0 -g -DDEBUG"
) else (
    set "OPT_FLAGS=-O2 -DNDEBUG"
)

rem SIMD flags
set "SIMD_FLAGS=-mavx2 -mfma"

rem Libraries (MinGW needs pthread even on Windows for this implementation)
set "LIBS=-lm"

rem Build test executable
echo Building test_embedlet.exe...
gcc %COMMON_FLAGS% %OPT_FLAGS% %SIMD_FLAGS% ^
    -o "%BUILD_DIR%\test_embedlet.exe" ^
    "%TESTS_DIR%\test_embedlet.c" ^
    %LIBS%
if errorlevel 1 (
    echo Failed to build test_embedlet.exe
    exit /b 1
)

rem Build example executable
echo Building example.exe...
gcc %COMMON_FLAGS% %OPT_FLAGS% %SIMD_FLAGS% ^
    -o "%BUILD_DIR%\example.exe" ^
    "%EXAMPLES_DIR%\example.c" ^
    %LIBS%
if errorlevel 1 (
    echo Failed to build example.exe
    exit /b 1
)

goto :done

:done
echo.
echo ============================================
echo  Build complete!
echo ============================================
echo.
echo Executables in: %BUILD_DIR%\
echo.
echo To run tests:
echo   cd sample_data
echo   ..\%BUILD_DIR%\test_embedlet.exe
echo.
echo To run example:
echo   cd sample_data
echo   ..\%BUILD_DIR%\example.exe
echo.

endlocal
exit /b 0