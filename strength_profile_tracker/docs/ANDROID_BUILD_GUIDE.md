# REPPIT Android Build Guide

This guide explains how to build and sign the Android app for Google Play Store submission.

## Prerequisites

1. **Node.js** (v18+)
2. **Android Studio** with SDK tools
3. **Java JDK 17+** (for signing)
4. **Release keystore** (already created: `android/reppit-release-key.jks`)

## Current Configuration

| Setting | Value |
|---------|-------|
| App ID | com.reppit.app |
| App Name | REPPIT |
| Version Name | 1.0 |
| Version Code | 1 |
| Min SDK | 24 (Android 7.0) |
| Target SDK | 36 |
| Web Dir | out |
| Server URL | https://strengthprofiletracker-l5ybveppi-castroaruns-projects.vercel.app |

## Build Steps

### Step 1: Build the Next.js App

```bash
npm run build
```

This creates a static export in the `out/` directory.

### Step 2: Sync with Capacitor

```bash
npx cap sync android
```

This copies the web assets to the Android project and updates native dependencies.

### Step 3: Build the Android App Bundle (AAB)

Navigate to the android directory and build:

```bash
cd android
./gradlew bundleRelease
```

On Windows:
```powershell
cd android
.\gradlew.bat bundleRelease
```

The unsigned AAB will be at:
```
android/app/build/outputs/bundle/release/app-release.aab
```

### Step 4: Sign the AAB

Using jarsigner:

```bash
jarsigner -verbose -sigalg SHA256withRSA -digestalg SHA-256 \
  -keystore reppit-release-key.jks \
  app/build/outputs/bundle/release/app-release.aab \
  reppit-key
```

**Note:** You'll be prompted for the keystore password.

### Step 5: Verify the Signature

```bash
jarsigner -verify -verbose -certs app/build/outputs/bundle/release/app-release.aab
```

## Quick Build Script

Create a build script for convenience:

**build-release.sh (Linux/Mac):**
```bash
#!/bin/bash
echo "Building Next.js..."
npm run build

echo "Syncing Capacitor..."
npx cap sync android

echo "Building Android bundle..."
cd android
./gradlew bundleRelease

echo "Signing the bundle..."
jarsigner -verbose -sigalg SHA256withRSA -digestalg SHA-256 \
  -keystore reppit-release-key.jks \
  app/build/outputs/bundle/release/app-release.aab \
  reppit-key

echo "Done! Bundle is at: android/app/build/outputs/bundle/release/app-release.aab"
```

**build-release.ps1 (Windows PowerShell):**
```powershell
Write-Host "Building Next.js..."
npm run build

Write-Host "Syncing Capacitor..."
npx cap sync android

Write-Host "Building Android bundle..."
Set-Location android
.\gradlew.bat bundleRelease

Write-Host "Signing the bundle..."
jarsigner -verbose -sigalg SHA256withRSA -digestalg SHA-256 `
  -keystore reppit-release-key.jks `
  app\build\outputs\bundle\release\app-release.aab `
  reppit-key

Set-Location ..
Write-Host "Done! Bundle is at: android\app\build\outputs\bundle\release\app-release.aab"
```

## Uploading to Play Store

### Play Console Steps

1. Go to [Google Play Console](https://play.google.com/console)
2. Click **Create app**
3. Fill in:
   - App name: **REPPIT**
   - Default language: English
   - App or game: App
   - Free or paid: Free
   - Accept policies

4. **Store listing:**
   - Use content from [PLAY_STORE_LISTING.md](./PLAY_STORE_LISTING.md)
   - Add screenshots (at least 2)
   - Add feature graphic (1024x500)
   - Add app icon (512x512)

5. **App content:**
   - Privacy policy: Host [PRIVACY_POLICY.md](./PRIVACY_POLICY.md) as a webpage
   - Content rating: Complete the questionnaire
   - Target audience: 18+ (fitness app)
   - Data safety: Select "No data collected"

6. **Release:**
   - Go to Production > Create new release
   - Upload the signed AAB
   - Add release notes
   - Review and roll out

## Version Updates

When releasing a new version:

1. Update version in `android/app/build.gradle`:
   ```gradle
   versionCode 2  // Increment this
   versionName "1.1"  // Update as needed
   ```

2. Rebuild and upload following the same steps.

## Keystore Backup

**IMPORTANT:** Back up your keystore file (`reppit-release-key.jks`) and password securely. If lost, you cannot update your app on the Play Store.

Store the keystore in:
- Encrypted cloud storage
- Password manager
- Secure offline backup

## Troubleshooting

### Build fails with memory error
Add to `android/gradle.properties`:
```
org.gradle.jvmargs=-Xmx2048m
```

### SDK version mismatch
Update Android Studio and SDK tools to latest version.

### Signing verification fails
Ensure you're using the correct keystore alias (`reppit-key`).

## Asset Requirements

### App Icon
- Size: 512x512 pixels
- Format: PNG (32-bit with alpha)
- Location: `android/app/src/main/res/mipmap-xxxhdpi/ic_launcher.png`

### Screenshots
- Phone: 1080x1920 or 1440x2560
- Minimum: 2 screenshots
- Maximum: 8 screenshots
- Format: JPEG or PNG

### Feature Graphic
- Size: 1024x500 pixels
- Format: JPEG or PNG
- No text required (Google may crop)

## Checklist Before Submission

- [ ] Build completes without errors
- [ ] AAB is signed
- [ ] App icon is set (not default Capacitor icon)
- [ ] Screenshots captured
- [ ] Feature graphic created
- [ ] Privacy policy hosted online
- [ ] Play Store listing complete
- [ ] Content rating completed
- [ ] Data safety form filled
