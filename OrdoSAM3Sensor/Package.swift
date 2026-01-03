// swift-tools-version: 5.9

import PackageDescription

let package = Package(
    name: "Sam3Sensor",
    platforms: [
        .macOS(.v14),
        .iOS(.v17)
    ],
    products: [
        .library(
            name: "Sam3Sensor",
            targets: ["Sam3Sensor"]
        ),
        .executable(
            name: "LoadWeightsApp",
            targets: ["LoadWeightsApp"]
        )
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.6.0")
    ],
    targets: [
        .target(
            name: "Sam3Sensor",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift")
            ],
            resources: [
                .process("Resources"),
                .process("Metal")
            ]
        ),
//        .executableTarget(
//            name: "OrdoCli",
//            dependencies: ["Sam3Sensor"]
//        ),
        .executableTarget(
            name: "LoadWeightsApp",
            dependencies: ["Sam3Sensor"]
        ),
        .testTarget(
            name: "Sam3SensorTests",
            dependencies: ["Sam3Sensor"]
        ),
    ]
)
