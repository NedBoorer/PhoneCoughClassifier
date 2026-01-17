import 'dart:io';
import 'package:flutter/material.dart';
import 'package:record/record.dart';
import 'package:path_provider/path_provider.dart';
import 'package:permission_handler/permission_handler.dart';
import 'api_service.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Cough Classifier',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.blue),
        useMaterial3: true,
      ),
      home: const HomePage(),
    );
  }
}

class HomePage extends StatefulWidget {
  const HomePage({super.key});

  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  final ApiService _api = ApiService();
  final AudioRecorder _recorder = AudioRecorder();
  
  bool _isRecording = false;
  bool _isProcessing = false;
  String? _recordedPath;
  Map<String, dynamic>? _result;
  String? _error;
  bool _apiConnected = false;

  @override
  void initState() {
    super.initState();
    _checkPermissions();
    _checkApi();
  }

  Future<void> _checkApi() async {
    bool connected = await _api.checkHealth();
    setState(() => _apiConnected = connected);
  }

  Future<void> _checkPermissions() async {
    await Permission.microphone.request();
    await Permission.storage.request(); // For Android < 10
  }

  Future<void> _startRecording() async {
    try {
      if (await _recorder.hasPermission()) {
        final directory = await getTemporaryDirectory();
        final path = '${directory.path}/recording.m4a';

        await _recorder.start(const RecordConfig(), path: path);
        setState(() {
          _isRecording = true;
          _recordedPath = null;
          _result = null;
          _error = null;
        });
      }
    } catch (e) {
      setState(() => _error = "Could not start recording: $e");
    }
  }

  Future<void> _stopRecording() async {
    try {
      final path = await _recorder.stop();
      setState(() {
        _isRecording = false;
        _recordedPath = path;
      });
    } catch (e) {
      setState(() => _error = "Could not stop recording: $e");
    }
  }

  Future<void> _analyzeRecording() async {
    if (_recordedPath == null) return;

    setState(() {
      _isProcessing = true;
      _error = null;
    });

    try {
      final result = await _api.classifyCough(_recordedPath!);
      setState(() => _result = result);
    } catch (e) {
      setState(() => _error = e.toString());
    } finally {
      setState(() => _isProcessing = false);
    }
  }

  @override
  void dispose() {
    _recorder.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Cough Classifier'),
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
        actions: [
          Container(
            margin: const EdgeInsets.only(right: 16),
            child: Icon(
              Icons.circle,
              color: _apiConnected ? Colors.green : Colors.red,
              size: 12,
            ),
          )
        ],
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(24.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            // Status Card
            Card(
              elevation: 0,
              color: Theme.of(context).colorScheme.surfaceVariant,
              child: Padding(
                padding: const EdgeInsets.all(16.0),
                child: Column(
                  children: [
                    Text(
                      _isRecording ? "Listening..." : "Ready to Screen",
                      style: Theme.of(context).textTheme.headlineSmall,
                    ),
                    const SizedBox(height: 8),
                    Text(
                      _isRecording 
                        ? "Cough clearly into the microphone" 
                        : "Tap the mic to start recording",
                      textAlign: TextAlign.center,
                    ),
                  ],
                ),
              ),
            ),
            
            const SizedBox(height: 40),

            // Recording Controls
            Center(
              child: GestureDetector(
                onTap: _isProcessing 
                  ? null 
                  : (_isRecording ? _stopRecording : _startRecording),
                child: Container(
                  width: 120,
                  height: 120,
                  decoration: BoxDecoration(
                    color: _isRecording ? Colors.red : Colors.blue,
                    shape: BoxShape.circle,
                    boxShadow: [
                      BoxShadow(
                        color: (_isRecording ? Colors.red : Colors.blue).withOpacity(0.3),
                        blurRadius: 20,
                        spreadRadius: 5,
                      )
                    ],
                  ),
                  child: Icon(
                    _isRecording ? Icons.stop : Icons.mic,
                    color: Colors.white,
                    size: 50,
                  ),
                ),
              ),
            ),

            const SizedBox(height: 40),

            // Analyze Button
            if (_recordedPath != null && !_isRecording && _result == null)
              ElevatedButton.icon(
                onPressed: _isProcessing ? null : _analyzeRecording,
                icon: _isProcessing 
                  ? const SizedBox(
                      width: 20, height: 20, 
                      child: CircularProgressIndicator(strokeWidth: 2)
                    ) 
                  : const Icon(Icons.analytics),
                label: Text(_isProcessing ? "Analyzing..." : "Analyze Recording"),
                style: ElevatedButton.styleFrom(
                  padding: const EdgeInsets.symmetric(vertical: 16),
                  textStyle: const TextStyle(fontSize: 18),
                ),
              ),

            // Results Section
            if (_error != null)
              Padding(
                padding: const EdgeInsets.only(top: 24),
                child: Text(
                  _error!,
                  style: const TextStyle(color: Colors.red),
                  textAlign: TextAlign.center,
                ),
              ),

            if (_result != null) ...[
              const SizedBox(height: 24),
              _buildResultCard(_result!),
            ],
          ],
        ),
      ),
    );
  }

  Widget _buildResultCard(Map<String, dynamic> result) {
    final severity = result['severity'] ?? 'unknown';
    final classification = result['classification'] ?? 'Unknown';
    final confidence = (result['confidence'] ?? 0.0) * 100;
    final recommendation = result['recommendation'] ?? '';

    Color color;
    switch (severity.toLowerCase()) {
      case 'urgent':
      case 'high':
        color = Colors.red;
        break;
      case 'moderate':
        color = Colors.orange;
        break;
      default:
        color = Colors.green;
    }

    return Card(
      color: color.withOpacity(0.1),
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(16),
        side: BorderSide(color: color, width: 2),
      ),
      child: Padding(
        padding: const EdgeInsets.all(20.0),
        child: Column(
          children: [
            Text(
              classification.toUpperCase(),
              style: TextStyle(
                color: color,
                fontSize: 24,
                fontWeight: FontWeight.bold,
              ),
            ),
            const SizedBox(height: 8),
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 4),
              decoration: BoxDecoration(
                color: color,
                borderRadius: BorderRadius.circular(20),
              ),
              child: Text(
                severity.toUpperCase(),
                style: const TextStyle(color: Colors.white, fontWeight: FontWeight.bold),
              ),
            ),
            const SizedBox(height: 16),
            Text(
              "${confidence.toStringAsFixed(1)}% Confidence",
              style: Theme.of(context).textTheme.bodySmall,
            ),
            const Divider(height: 32),
            Text(
              recommendation,
              textAlign: TextAlign.center,
              style: const TextStyle(fontSize: 16),
            ),
          ],
        ),
      ),
    );
  }
}
