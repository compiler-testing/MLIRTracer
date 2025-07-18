module {
  func.func @main(%arg0: tensor<30x79xi64>, %arg1: tensor<49x51x53x41xi1>, %arg2: tensor<78x41x61xf32>, %arg3: tensor<1x1x61xf32>) -> (tensor<30x79xi1>, tensor<78x41x61xf32>, tensor<49x51x53x1xi1>) {
    %0 = tosa.identity %arg0 : (tensor<30x79xi64>) -> tensor<30x79xi64>
    %1 = tosa.identity %0 : (tensor<30x79xi64>) -> tensor<30x79xi64>
    %2 = tosa.clamp %1 {min_val = -39 : i64, max_val = 44 : i64} : (tensor<30x79xi64>) -> tensor<30x79xi64>
    %3 = tosa.reduce_any %arg1 {axis = 3 : i32} : (tensor<49x51x53x41xi1>) -> tensor<49x51x53x1xi1>
    %4 = tosa.greater %2, %0 : (tensor<30x79xi64>, tensor<30x79xi64>) -> tensor<30x79xi1>
    %5 = tosa.bitwise_and %3, %3 : (tensor<49x51x53x1xi1>, tensor<49x51x53x1xi1>) -> tensor<49x51x53x1xi1>
    %6 = tosa.pow %arg2, %arg3 : (tensor<78x41x61xf32>, tensor<1x1x61xf32>) -> tensor<78x41x61xf32>
    %7 = tosa.bitwise_xor %5, %5 : (tensor<49x51x53x1xi1>, tensor<49x51x53x1xi1>) -> tensor<49x51x53x1xi1>
    return %4, %6, %7 : tensor<30x79xi1>, tensor<78x41x61xf32>, tensor<49x51x53x1xi1>
  }
}
