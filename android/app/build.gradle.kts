plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("org.jetbrains.kotlin.plugin.compose")
}

android {
    namespace = "com.animedenoise"
    compileSdk = 36

    defaultConfig {
        applicationId = "com.animedenoise"
        minSdk = 26
        targetSdk = 35
        versionCode = 1
        versionName = "1.0"
        ndk {
            abiFilters += listOf("arm64-v8a")
        }
    }

    sourceSets {
        // The golden fixtures and the self-test receiver only exist in debug builds; a release APK
        // carries the models and nothing else.
        getByName("debug") {
            assets.srcDir("src/debug/assets")
        }
    }

    buildTypes {
        debug {
            versionNameSuffix = "-debug"
        }
        release {
            isMinifyEnabled = true
            isShrinkResources = true
            proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")
            // Signed with the debug key so adb install works without a keystore.
            signingConfig = signingConfigs.getByName("debug")
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions {
        jvmTarget = "17"
    }
    buildFeatures {
        compose = true
        buildConfig = true
    }
    packaging {
        resources {
            excludes += listOf("/META-INF/{AL2.0,LGPL2.1}", "META-INF/*.kotlin_module")
        }
        jniLibs {
            useLegacyPackaging = false
        }
    }
    androidResources {
        // Store the ONNX graphs uncompressed so they can be memory-mapped from the APK.
        noCompress += listOf("onnx", "onnx.data", "pte")
    }
}

dependencies {
    implementation("androidx.core:core-ktx:1.13.1")
    implementation("androidx.activity:activity-compose:1.9.3")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.8.7")
    implementation("androidx.lifecycle:lifecycle-viewmodel-compose:2.8.7")
    implementation("androidx.lifecycle:lifecycle-runtime-compose:2.8.7")
    implementation(platform("androidx.compose:compose-bom:2024.10.01"))
    implementation("androidx.compose.ui:ui")
    implementation("androidx.compose.ui:ui-graphics")
    implementation("androidx.compose.foundation:foundation")
    implementation("androidx.compose.material3:material3")
    implementation("androidx.compose.material:material-icons-extended")
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.20.0")
    // ExecuTorch: the only working accelerated backend on this hardware. XNNPACK is baked into the
    // .pte at export time, whereas ONNX Runtime ships neither XNNPACK nor NNAPI in its Android AAR.
    // ExecuTorch evaluated and rejected for the default pipeline: its Vulkan backend caps at
    // 128x128 tiles on Adreno (256/384 abort natively), and 198 tiles for a real 8.3 MP image
    // costs ~6 minutes vs 11-17 s for one whole-image ONNX CPU pass. See android/README.md.
    // implementation("org.pytorch:executorch-android-vulkan:1.5.1")

    // Desktop ORT for JVM unit tests: the Android AAR only carries Android natives, but the Java API
    // is identical, so end-to-end parity can be verified on the dev machine before touching a phone.
    testImplementation("com.microsoft.onnxruntime:onnxruntime:1.20.0")
    testImplementation("junit:junit:4.13.2")
    testImplementation("org.jetbrains.kotlin:kotlin-test-junit:2.0.21")
}
