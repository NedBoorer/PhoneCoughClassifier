import 'dart:io';
import 'package:http/http.dart' as http;
import 'dart:convert';

class ApiService {
  // REPLACE WITH YOUR PC's LOCAL IP if running on a real phone!
  // Android Emulator uses 10.0.2.2 for localhost.
  // iOS Simulator uses localhost.
  static const String _baseUrl = 'http://10.0.2.2:8000'; 
  // static const String _baseUrl = 'http://192.168.1.X:8000'; // For real device

  Future<Map<String, dynamic>> classifyCough(String filePath) async {
    final uri = Uri.parse('$_baseUrl/test/classify');
    
    var request = http.MultipartRequest('POST', uri);
    
    // Attach the file
    var file = await http.MultipartFile.fromPath(
      'audio_file', 
      filePath,
      filename: 'recording.m4a' // or .wav depending on recorder
    );
    
    request.files.add(file);

    try {
      final streamedResponse = await request.send();
      final response = await http.Response.fromStream(streamedResponse);

      if (response.statusCode == 200) {
        return jsonDecode(response.body);
      } else {
        throw Exception('Failed to analyze: ${response.body}');
      }
    } catch (e) {
      throw Exception('Connection error: $e');
    }
  }

  Future<bool> checkHealth() async {
    try {
      final response = await http.get(Uri.parse('$_baseUrl/health'));
      return response.statusCode == 200;
    } catch (e) {
      print('Health check failed: $e');
      return false;
    }
  }
}
