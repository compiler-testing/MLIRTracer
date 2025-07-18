module {
  func.func @main(%arg0: tensor<100x3x27x12xi16>, %arg1: tensor<92x94x8x74xf32>, %arg2: tensor<13xi1>) -> (tensor<100x3x27x12xi16>, tensor<92x94x8x74xf32>, tensor<2xi1>) {
    %0 = tosa.clamp %arg0 {min_val = -3 : i16, max_val = 98 : i16} : (tensor<100x3x27x12xi16>) -> tensor<100x3x27x12xi16>
    %1 = tosa.reciprocal %arg1 : (tensor<92x94x8x74xf32>) -> tensor<92x94x8x74xf32>
    %2 = tosa.reduce_all %arg2 {axis = 0 : i32} : (tensor<13xi1>) -> tensor<1xi1>
    %3 = tosa.concat %2, %2 {axis = 0 : i32} : (tensor<1xi1>, tensor<1xi1>) -> tensor<2xi1>
    return %0, %1, %3 : tensor<100x3x27x12xi16>, tensor<92x94x8x74xf32>, tensor<2xi1>
  }
}
