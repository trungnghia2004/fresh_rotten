import React, { useMemo, useState } from "react";
import {
  ActivityIndicator,
  Alert,
  SafeAreaView,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  TouchableOpacity,
  View,
  Image,
} from "react-native";
import * as ImagePicker from "expo-image-picker";
import { CameraView, useCameraPermissions } from "expo-camera";
import * as FileSystem from "expo-file-system";

const DEFAULT_API = "http://127.0.0.1:8000";

export default function App() {
  const [apiBase, setApiBase] = useState(DEFAULT_API);
  const [imageUri, setImageUri] = useState(null);
  const [videoUri, setVideoUri] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [cameraVisible, setCameraVisible] = useState(false);
  const [cameraPermission, requestPermission] = useCameraPermissions();
  const [cameraRef, setCameraRef] = useState(null);

  const canUseCamera = useMemo(() => cameraPermission?.granted, [cameraPermission]);

  const showError = (msg) => Alert.alert("Error", msg);

  async function pickImage() {
    const res = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      quality: 0.9,
    });
    if (!res.canceled) {
      setImageUri(res.assets[0].uri);
      setVideoUri(null);
      setResult(null);
    }
  }

  async function pickVideo() {
    const res = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Videos,
      quality: 0.9,
    });
    if (!res.canceled) {
      setVideoUri(res.assets[0].uri);
      setImageUri(null);
      setResult(null);
    }
  }

  async function uploadImage(uri) {
    setLoading(true);
    try {
      const form = new FormData();
      const file = {
        uri,
        name: "image.jpg",
        type: "image/jpeg",
      };
      form.append("file", file);

      const res = await fetch(`${apiBase}/predict_image`, {
        method: "POST",
        body: form,
        headers: { "Content-Type": "multipart/form-data" },
      });
      const data = await res.json();
      setResult(data);
    } catch (e) {
      showError("Cannot upload image. Check API base URL.");
    } finally {
      setLoading(false);
    }
  }

  async function uploadVideo(uri) {
    setLoading(true);
    try {
      const form = new FormData();
      const file = {
        uri,
        name: "video.mp4",
        type: "video/mp4",
      };
      form.append("file", file);

      const res = await fetch(`${apiBase}/predict_video`, {
        method: "POST",
        body: form,
        headers: { "Content-Type": "multipart/form-data" },
      });
      const data = await res.json();
      setResult(data);
    } catch (e) {
      showError("Cannot upload video. Check API base URL.");
    } finally {
      setLoading(false);
    }
  }

  async function captureAndPredict() {
    if (!cameraRef) return;
    setLoading(true);
    try {
      const photo = await cameraRef.takePictureAsync({ quality: 0.7, base64: true });
      const base64 = photo.base64 ? `data:image/jpeg;base64,${photo.base64}` : null;
      if (!base64) {
        showError("Cannot read captured image.");
        return;
      }
      const form = new FormData();
      form.append("image_base64", base64);
      const res = await fetch(`${apiBase}/predict_camera`, {
        method: "POST",
        body: form,
      });
      const data = await res.json();
      setResult(data);
    } catch (e) {
      showError("Cannot capture or predict.");
    } finally {
      setLoading(false);
    }
  }

  async function requestCam() {
    const res = await requestPermission();
    if (!res.granted) {
      showError("Camera permission not granted.");
    } else {
      setCameraVisible(true);
    }
  }

  return (
    <SafeAreaView style={styles.container}>
      <ScrollView contentContainerStyle={styles.content}>
        <Text style={styles.title}>Fresh / Rotten Classifier</Text>
        <Text style={styles.subtitle}>Upload image, video or capture from camera.</Text>

        <View style={styles.card}>
          <Text style={styles.label}>API Base URL</Text>
          <TextInput
            value={apiBase}
            onChangeText={setApiBase}
            style={styles.input}
            placeholder="http://192.168.1.10:8000"
            autoCapitalize="none"
          />
          <Text style={styles.hint}>If testing on phone, use your PC LAN IP.</Text>
        </View>

        <View style={styles.card}>
          <Text style={styles.section}>Image</Text>
          <View style={styles.row}>
            <TouchableOpacity style={styles.btn} onPress={pickImage}>
              <Text style={styles.btnText}>Pick Image</Text>
            </TouchableOpacity>
            <TouchableOpacity
              style={[styles.btn, !imageUri && styles.btnDisabled]}
              onPress={() => imageUri && uploadImage(imageUri)}
              disabled={!imageUri}
            >
              <Text style={styles.btnText}>Predict</Text>
            </TouchableOpacity>
          </View>
          {imageUri ? <Image source={{ uri: imageUri }} style={styles.preview} /> : null}
        </View>

        <View style={styles.card}>
          <Text style={styles.section}>Video</Text>
          <View style={styles.row}>
            <TouchableOpacity style={styles.btn} onPress={pickVideo}>
              <Text style={styles.btnText}>Pick Video</Text>
            </TouchableOpacity>
            <TouchableOpacity
              style={[styles.btn, !videoUri && styles.btnDisabled]}
              onPress={() => videoUri && uploadVideo(videoUri)}
              disabled={!videoUri}
            >
              <Text style={styles.btnText}>Predict</Text>
            </TouchableOpacity>
          </View>
          {videoUri ? <Text style={styles.hint}>Video selected.</Text> : null}
        </View>

        <View style={styles.card}>
          <Text style={styles.section}>Camera</Text>
          {!cameraVisible ? (
            <TouchableOpacity style={styles.btn} onPress={requestCam}>
              <Text style={styles.btnText}>Open Camera</Text>
            </TouchableOpacity>
          ) : (
            <View>
              <CameraView style={styles.camera} ref={(ref) => setCameraRef(ref)} />
              <TouchableOpacity style={[styles.btn, styles.capture]} onPress={captureAndPredict}>
                <Text style={styles.btnText}>Capture & Predict</Text>
              </TouchableOpacity>
            </View>
          )}
        </View>

        <View style={styles.card}>
          <Text style={styles.section}>Result</Text>
          {loading ? (
            <ActivityIndicator />
          ) : result ? (
            <Text style={styles.result}>
              Label: {result.label} | Confidence: {result.confidence}
              {result.votes ? ` | votes: f ${result.votes.fresh}, r ${result.votes.rotten}` : ""}
            </Text>
          ) : (
            <Text style={styles.hint}>No prediction yet.</Text>
          )}
          {result?.error ? <Text style={styles.error}>{result.error}</Text> : null}
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: "#f7f3eb" },
  content: { padding: 16 },
  title: { fontSize: 26, fontWeight: "700", color: "#3a7d44" },
  subtitle: { marginTop: 4, marginBottom: 16, color: "#555" },
  card: {
    backgroundColor: "#fff",
    borderRadius: 12,
    padding: 14,
    marginBottom: 12,
    borderWidth: 1,
    borderColor: "#e1ddd5",
  },
  section: { fontWeight: "700", marginBottom: 8, fontSize: 16 },
  label: { fontWeight: "600", marginBottom: 6 },
  input: {
    borderWidth: 1,
    borderColor: "#ddd",
    borderRadius: 8,
    padding: 8,
    backgroundColor: "#fafafa",
  },
  hint: { marginTop: 6, color: "#666" },
  row: { flexDirection: "row", gap: 8 },
  btn: {
    backgroundColor: "#3a7d44",
    paddingVertical: 10,
    paddingHorizontal: 12,
    borderRadius: 8,
    alignItems: "center",
    flex: 1,
  },
  btnDisabled: { opacity: 0.5 },
  btnText: { color: "#fff", fontWeight: "600" },
  preview: { marginTop: 10, width: "100%", height: 220, borderRadius: 8 },
  camera: { width: "100%", height: 320, borderRadius: 8, overflow: "hidden" },
  capture: { marginTop: 10 },
  result: { fontSize: 16, fontWeight: "600", color: "#1b5e20" },
  error: { color: "#b00020", marginTop: 6 },
});
