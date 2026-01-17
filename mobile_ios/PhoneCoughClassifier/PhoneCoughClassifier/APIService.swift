import Foundation

struct ClassificationResult: Codable {
    let classification: String
    let confidence: Double
    let severity: String
    let recommendation: String
}

class APIService {
    // CHANGE THIS TO YOUR MAC'S LOCAL IP ADDRESS (e.g., http://192.168.1.5:8000)
    // "localhost" will NOT work on a real iPhone, only mostly on Simulator.
    static let baseURL = "http://localhost:8000"
    
    func classifyCough(fileURL: URL) async throws -> ClassificationResult {
        let url = URL(string: "\(APIService.baseURL)/test/classify")!
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        
        let boundary = UUID().uuidString
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")
        
        let data = try Data(contentsOf: fileURL)
        request.httpBody = createBody(with: data, boundary: boundary)
        
        let (responseData, response) = try await URLSession.shared.data(for: request)
        
        guard let httpResponse = response as? HTTPURLResponse, httpResponse.statusCode == 200 else {
            throw NSError(domain: "APIService", code: 0, userInfo: [NSLocalizedDescriptionKey: "Server error or invalid response"])
        }
        
        let decoder = JSONDecoder()
        return try decoder.decode(ClassificationResult.self, from: responseData)
    }
    
    private func createBody(with data: Data, boundary: String) -> Data {
        var body = Data()
        
        body.append("--\(boundary)\r\n".data(using: .utf8)!)
        body.append("Content-Disposition: form-data; name=\"audio_file\"; filename=\"recording.wav\"\r\n".data(using: .utf8)!)
        body.append("Content-Type: audio/wav\r\n\r\n".data(using: .utf8)!)
        body.append(data)
        body.append("\r\n".data(using: .utf8)!)
        body.append("--\(boundary)--\r\n".data(using: .utf8)!)
        
        return body
    }
}
