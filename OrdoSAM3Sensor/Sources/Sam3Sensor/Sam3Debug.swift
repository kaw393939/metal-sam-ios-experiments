import Foundation

/// Lightweight runtime debug flags controlled via environment variables.
///
/// Default: everything off.
///
/// Enable examples:
/// - `SAM3_DEBUG=1` (enables all debug categories)
/// - `SAM3_DEBUG=rope` (enables only RoPE-related logs)
/// - `SAM3_DEBUG=rope,encoder`
/// - `SAM3_DEBUG_ROPE=1`
public enum Sam3Debug {
    private static func envBool(_ key: String) -> Bool {
        guard let value = ProcessInfo.processInfo.environment[key]?.lowercased() else { return false }
        return value == "1" || value == "true" || value == "yes" || value == "on"
    }

    private static func envList(_ key: String) -> Set<String> {
        guard let value = ProcessInfo.processInfo.environment[key]?.lowercased() else { return [] }
        let parts = value
            .split(separator: ",")
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
        return Set(parts)
    }

    private static var globalEnabled: Bool {
        envBool("SAM3_DEBUG") || envBool("SAM3_DEBUG_ALL")
    }

    private static func categoryEnabled(_ category: String, extraKey: String) -> Bool {
        if globalEnabled { return true }
        if envBool(extraKey) { return true }
        let list = envList("SAM3_DEBUG")
        return list.contains("all") || list.contains(category.lowercased())
    }

    public static var rope: Bool {
        categoryEnabled("rope", extraKey: "SAM3_DEBUG_ROPE")
    }

    public static var encoder: Bool {
        categoryEnabled("encoder", extraKey: "SAM3_DEBUG_ENCODER")
    }
}
