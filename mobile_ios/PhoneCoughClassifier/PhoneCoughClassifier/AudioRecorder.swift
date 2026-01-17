import Foundation
import AVFoundation

class AudioRecorder: NSObject, ObservableObject {
    var audioRecorder: AVAudioRecorder?
    @Published var isRecording = false
    @Published var recordingURL: URL?
    
    override init() {
        super.init()
        requestPermission()
    }
    
    func requestPermission() {
        AVAudioSession.sharedInstance().requestRecordPermission { allowed in
            if !allowed {
                print("Microphone permission denied")
            }
        }
    }
    
    func startRecording() {
        let recordingSession = AVAudioSession.sharedInstance()
        
        do {
            try recordingSession.setCategory(.playAndRecord, mode: .default)
            try recordingSession.setActive(true)
            
            let docPath = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            let audioFilename = docPath.appendingPathComponent("recording.wav")
            
            let settings: [String: Any] = [
                AVFormatIDKey: Int(kAudioFormatLinearPCM),
                AVSampleRateKey: 16000,
                AVNumberOfChannelsKey: 1,
                AVEncoderAudioQualityKey: AVAudioQuality.high.rawValue
            ]
            
            audioRecorder = try AVAudioRecorder(url: audioFilename, settings: settings)
            audioRecorder?.record()
            
            DispatchQueue.main.async {
                self.isRecording = true
                self.recordingURL = nil
            }
            
        } catch {
            print("Could not start recording: \(error)")
        }
    }
    
    func stopRecording() {
        audioRecorder?.stop()
        DispatchQueue.main.async {
            self.isRecording = false
            self.recordingURL = self.audioRecorder?.url
        }
    }
}
