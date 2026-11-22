# ONNX Runtime reaches its JNI bridge and provider classes reflectively.
-keep class ai.onnxruntime.** { *; }
-dontwarn ai.onnxruntime.**

# Keep the model catalogue reachable from the UI and the self-test receiver from the manifest.
-keep class com.animedenoise.sr.** { *; }
-keepclassmembers class com.animedenoise.core.** { *; }
-keep class com.animedenoise.SelfTestReceiver { *; }
