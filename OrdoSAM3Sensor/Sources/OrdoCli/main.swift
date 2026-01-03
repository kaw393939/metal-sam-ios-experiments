import Foundation
import Metal
import Sam3Sensor

enum CliError: Error {
	case invalidArguments(String)
	case unsupportedPlatform(String)
}

func usage() {
	print("""
OrdoCli

Commands:
  bench-attn [--b B] [--h H] [--n N] [--d D] [--warmup W] [--iters I]
	Benchmarks attention (reference softmax vs SDPA) and prints mean/p50/p90 in ms.

  eval-iou --pred <pred_dir> --gt <gt_dir> [--threshold T]
	Computes mean IoU by matching files by basename in the two directories.

Notes:
  - SDPA benchmark requires macOS 15+.
""")
}

func argValue(_ args: [String], _ name: String) -> String? {
	guard let idx = args.firstIndex(of: name), idx + 1 < args.count else { return nil }
	return args[idx + 1]
}

func parseInt(_ s: String?, _ name: String, default def: Int) throws -> Int {
	guard let s else { return def }
	guard let v = Int(s) else { throw CliError.invalidArguments("Expected int for \(name)") }
	return v
}

func parseUInt8(_ s: String?, _ name: String, default def: UInt8) throws -> UInt8 {
	guard let s else { return def }
	guard let v = Int(s), (0...255).contains(v) else { throw CliError.invalidArguments("Expected 0..255 for \(name)") }
	return UInt8(v)
}

do {
	let args = Array(CommandLine.arguments.dropFirst())
	guard let cmd = args.first else {
		usage()
		exit(0)
	}

	switch cmd {
	case "bench-attn":
		guard #available(macOS 15.0, *) else {
			throw CliError.unsupportedPlatform("bench-attn requires macOS 15+")
		}
		guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
			throw BenchmarkingError.noMetalDevice
		}

		let b = try parseInt(argValue(args, "--b"), "--b", default: 1)
		let h = try parseInt(argValue(args, "--h"), "--h", default: 16)
		let n = try parseInt(argValue(args, "--n"), "--n", default: 256)
		let d = try parseInt(argValue(args, "--d"), "--d", default: 64)
		let warmup = try parseInt(argValue(args, "--warmup"), "--warmup", default: 25)
		let iters = try parseInt(argValue(args, "--iters"), "--iters", default: 200)

		let ref = try AttentionBenchmark.run(
			device: device,
			queue: queue,
			batch: b,
			heads: h,
			seqLen: n,
			dimPerHead: d,
			implementation: .referenceSoftmax,
			warmup: warmup,
			iterations: iters
		)

		let sdpa = try AttentionBenchmark.run(
			device: device,
			queue: queue,
			batch: b,
			heads: h,
			seqLen: n,
			dimPerHead: d,
			implementation: .scaledDotProduct,
			warmup: warmup,
			iterations: iters
		)

		print("Attention benchmark [B=\(b) H=\(h) N=\(n) D=\(d)]")
		print(String(format: "Reference  mean=%.3fms  p50=%.3fms  p90=%.3fms", ref.meanMs, ref.p50Ms, ref.p90Ms))
		print(String(format: "SDPA       mean=%.3fms  p50=%.3fms  p90=%.3fms", sdpa.meanMs, sdpa.p50Ms, sdpa.p90Ms))

	case "eval-iou":
		let predDirStr = argValue(args, "--pred")
		let gtDirStr = argValue(args, "--gt")
		guard let predDirStr, let gtDirStr else {
			throw CliError.invalidArguments("eval-iou requires --pred and --gt")
		}

		let threshold = try parseUInt8(argValue(args, "--threshold"), "--threshold", default: 128)
		let predDir = URL(fileURLWithPath: predDirStr, isDirectory: true)
		let gtDir = URL(fileURLWithPath: gtDirStr, isDirectory: true)

		let fm = FileManager.default
		let predFiles = (try fm.contentsOfDirectory(at: predDir, includingPropertiesForKeys: nil))
			.filter { ["png", "jpg", "jpeg"].contains($0.pathExtension.lowercased()) }
		let gtFiles = (try fm.contentsOfDirectory(at: gtDir, includingPropertiesForKeys: nil))
			.filter { ["png", "jpg", "jpeg"].contains($0.pathExtension.lowercased()) }

		var gtByStem: [String: URL] = [:]
		gtByStem.reserveCapacity(gtFiles.count)
		for u in gtFiles {
			gtByStem[u.deletingPathExtension().lastPathComponent] = u
		}

		var ious: [Double] = []
		ious.reserveCapacity(predFiles.count)

		for predURL in predFiles {
			let stem = predURL.deletingPathExtension().lastPathComponent
			guard let gtURL = gtByStem[stem] else { continue }

			let predImg = try IoUMetrics.loadCGImage(from: predURL)
			let gtImg = try IoUMetrics.loadCGImage(from: gtURL)
			let score = try IoUMetrics.iou(pred: predImg, gt: gtImg, threshold: threshold)
			ious.append(score.iou)
		}

		if ious.isEmpty {
			print("No matching image pairs found (by basename).")
			exit(2)
		}

		let mean = ious.reduce(0, +) / Double(ious.count)
		print(String(format: "Mean IoU: %.4f over %d pairs (threshold=%d)", mean, ious.count, Int(threshold)))

	case "-h", "--help", "help":
		usage()
	default:
		usage()
		throw CliError.invalidArguments("Unknown command: \(cmd)")
	}
} catch {
	fputs("Error: \(error)\n", stderr)
	usage()
	exit(1)
}
