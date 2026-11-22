@echo off
setlocal
rem Anime Denoise - build and deploy helper.
rem   build.bat            build + install the debug APK on the connected phone
rem   build.bat release    build the minified release APK
rem   build.bat selftest   install, then run the on-device pixel check via adb
rem   build.bat log        follow the app log

set ROOT=%~dp0
set JAVA_HOME=F:\0-dsh-build-tools\jdk-17.0.21+9
set ANDROID_HOME=F:\Program\Android\SDK
set GRADLE=C:\Users\21294\.gradle\wrapper\dists\gradle-8.9-bin\gradle-8.9\bin\gradle.bat
set ADB=F:\platform-tools-latest-windows\platform-tools\adb.exe
set PATH=%JAVA_HOME%\bin;%PATH%

if /I "%~1"=="log" goto log
if /I "%~1"=="selftest" goto selftest
if /I "%~1"=="release" goto release
goto debug

:release
call "%GRADLE%" --no-daemon -p "%ROOT%.." :app:assembleRelease || exit /b 1
echo.
echo Release APK: %ROOT%..\app\build\outputs\apk\release\app-release.apk
goto :eof

:debug
call "%GRADLE%" --no-daemon -p "%ROOT%.." :app:assembleDebug || exit /b 1
echo.
"%ADB%" install -r -d "%ROOT%..\app\build\outputs\apk\debug\app-debug.apk" || exit /b 1
echo Installed. Launching...
"%ADB%" shell am start -n com.animedenoise/.MainActivity
goto :eof

:selftest
call "%GRADLE%" --no-daemon -p "%ROOT%.." :app:assembleDebug || exit /b 1
"%ADB%" install -r -d "%ROOT%..\app\build\outputs\apk\debug\app-debug.apk" || exit /b 1
"%ADB%" logcat -c
"%ADB%" shell am broadcast -a com.animedenoise.SELFTEST -n com.animedenoise/.SelfTestReceiver
echo Waiting for results...
timeout /t 3 /nobreak >nul
"%ADB%" logcat -s SelfTest:V
goto :eof

:log
"%ADB%" logcat -s SelfTest:V DenoiseVM:W OnnxSR:W Runtimes:I
goto :eof
