import SwiftUI
import AVFoundation

struct ContentView: View {
    @StateObject var audioRecorder = AudioRecorder()
    @State private var isProcessing = false
    @State private var result: ClassificationResult?
    @State private var errorMessage: String?
    
    private let apiService = APIService()
    
    var body: some View {
        NavigationView {
            VStack(spacing: 30) {
                
                // Status Card
                VStack(spacing: 10) {
                    Text(audioRecorder.isRecording ? "Listening..." : "Ready to Screen")
                        .font(.title2)
                        .fontWeight(.bold)
                    
                    Text(audioRecorder.isRecording ? "Cough clearly into microphone" : "Tap button to start")
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                }
                .padding()
                .frame(maxWidth: .infinity)
                .background(Color(UIColor.secondarySystemBackground))
                .cornerRadius(12)
                .padding(.horizontal)
                
                Spacer()
                
                // Record Button
                Button(action: {
                    if audioRecorder.isRecording {
                        audioRecorder.stopRecording()
                    } else {
                        audioRecorder.startRecording()
                        // Reset previous results
                        result = nil
                        errorMessage = nil
                    }
                }) {
                    ZStack {
                        Circle()
                            .fill(audioRecorder.isRecording ? Color.red : Color.blue)
                            .frame(width: 120, height: 120)
                            .shadow(color: (audioRecorder.isRecording ? Color.red : Color.blue).opacity(0.4), radius: 10, x: 0, y: 5)
                        
                        Image(systemName: audioRecorder.isRecording ? "stop.fill" : "mic.fill")
                            .font(.system(size: 50))
                            .foregroundColor(.white)
                    }
                }
                .disabled(isProcessing)
                
                Spacer()
                
                // Analyze Button
                if let _ = audioRecorder.recordingURL, !audioRecorder.isRecording && result == nil {
                    Button(action: analyzeRecording) {
                        HStack {
                            if isProcessing {
                                ProgressView()
                                    .progressViewStyle(CircularProgressViewStyle(tint: .white))
                            } else {
                                Image(systemName: "waveform.path.ecg")
                            }
                            Text(isProcessing ? "Analyzing..." : "Analyze Recording")
                        }
                        .frame(minWidth: 200)
                        .padding()
                        .background(Color.green)
                        .foregroundColor(.white)
                        .cornerRadius(30)
                    }
                    .disabled(isProcessing)
                }
                
                // Results View
                if let res = result {
                    ResultView(result: res)
                }
                
                if let error = errorMessage {
                    Text(error)
                        .foregroundColor(.red)
                        .font(.caption)
                        .padding()
                }
                
                Spacer()
            }
            .navigationTitle("Cough Classifier")
        }
    }
    
    func analyzeRecording() {
        guard let url = audioRecorder.recordingURL else { return }
        
        isProcessing = true
        errorMessage = nil
        
        Task {
            do {
                let classification = try await apiService.classifyCough(fileURL: url)
                DispatchQueue.main.async {
                    self.result = classification
                    self.isProcessing = false
                }
            } catch {
                DispatchQueue.main.async {
                    self.errorMessage = "Analysis failed: \(error.localizedDescription)"
                    self.isProcessing = false
                }
            }
        }
    }
}

struct ResultView: View {
    let result: ClassificationResult
    
    var color: Color {
        switch result.severity.lowercased() {
        case "urgent", "high": return .red
        case "moderate": return .orange
        default: return .green
        }
    }
    
    var body: some View {
        VStack(spacing: 15) {
            Text(result.classification.uppercased())
                .font(.title)
                .fontWeight(.black)
                .foregroundColor(color)
            
            Text(result.severity.uppercased())
                .font(.headline)
                .padding(.horizontal, 12)
                .padding(.vertical, 5)
                .background(color)
                .foregroundColor(.white)
                .cornerRadius(8)
            
            Text("\(Int(result.confidence * 100))% Confidence")
                .font(.caption)
                .foregroundColor(.secondary)
            
            Divider()
            
            Text(result.recommendation)
                .multilineTextAlignment(.center)
                .font(.body)
                .padding(.horizontal)
        }
        .padding()
        .background(color.opacity(0.1))
        .cornerRadius(16)
        .overlay(
            RoundedRectangle(cornerRadius: 16)
                .stroke(color, lineWidth: 2)
        )
        .padding(.horizontal)
    }
}
