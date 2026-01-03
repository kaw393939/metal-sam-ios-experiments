import Foundation

public enum Sam3Log {
    /// Enable debug logging by setting environment variable `SAM3_DEBUG_LOGS=1`.
    public static var isEnabled: Bool = {
        ProcessInfo.processInfo.environment["SAM3_DEBUG_LOGS"] == "1"
    }()

    /// Enable one-line timing breakdowns by setting `SAM3_STAGE_TIMING=1`.
    /// This is intended for performance work; keep off by default.
    public static var isStageTimingEnabled: Bool = {
        ProcessInfo.processInfo.environment["SAM3_STAGE_TIMING"] == "1"
    }()

    @inline(__always)
    public static func debug(_ message: @autoclosure () -> String) {
        guard isEnabled else { return }
        print(message())
    }

    @inline(__always)
    public static func stageTiming(_ message: @autoclosure () -> String) {
        guard isStageTimingEnabled else { return }
        print(message())
    }
}
