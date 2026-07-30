plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.medapp"
    compileSdk = 34

    defaultConfig {
        applicationId = "com.medapp"
        minSdk = 26
        targetSdk = 34
        versionCode = 1
        versionName = "0.1.0"

        // Don't compress model files so they can be memory-mapped / streamed from assets
        androidResources {
            noCompress.add("onnx")
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
        debug {
            isDebuggable = true
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
        viewBinding = true
    }

    packaging {
        resources {
            excludes += "/META-INF/{AL2.0,LGPL2.1}"
        }
    }
}

dependencies {
    // Core Android / Kotlin
    implementation("androidx.core:core-ktx:1.13.1")
    implementation("androidx.appcompat:appcompat:1.7.0")
    implementation("com.google.android.material:material:1.12.0")
    implementation("androidx.constraintlayout:constraintlayout:2.1.4")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.8.1")
    implementation("androidx.activity:activity-ktx:1.9.0")
    implementation("androidx.fragment:fragment-ktx:1.8.0")
    implementation("androidx.recyclerview:recyclerview:1.3.2")

    // Coroutines for async model inference (keep UI thread free)
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.8.1")

    // ONNX Runtime for Android — runs converted MONAI/PyTorch models fully offline
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.18.0")

    // JSON parsing for the model registry config
    implementation("org.json:json:20240303")

    // Networking for downloading models from GitHub at runtime
    implementation("com.squareup.okhttp3:okhttp:4.12.0")

    // Image loading/display
    implementation("com.github.bumptech.glide:glide:4.16.0")

    // Testing
    testImplementation("junit:junit:4.13.2")
    androidTestImplementation("androidx.test.ext:junit:1.2.1")
    androidTestImplementation("androidx.test.espresso:espresso-core:3.6.1")
}
