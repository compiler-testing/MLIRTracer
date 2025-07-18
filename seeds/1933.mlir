module {
  func.func @main(%arg0: tensor<100x30x51x27x57xf32>, %arg1: tensor<35x53x87x7x3x81xi16>, %arg2: tensor<1x1x1x1x3x1xi16>, %arg3: tensor<41x73x9xi32>, %arg4: tensor<25x82x79x92xi1>, %arg5: tensor<1x1x1x92xi1>) -> (tensor<41x73x1xi32>, tensor<100x30x51x27x57xf32>, tensor<25x82x79x92xi1>, tensor<35x53x87x7x3x81xi16>) {
    %0 = tosa.reciprocal %arg0 : (tensor<100x30x51x27x57xf32>) -> tensor<100x30x51x27x57xf32>
    %1 = tosa.bitwise_or %arg1, %arg2 : (tensor<35x53x87x7x3x81xi16>, tensor<1x1x1x1x3x1xi16>) -> tensor<35x53x87x7x3x81xi16>
    %2 = tosa.reduce_min %arg3 {axis = 2 : i32} : (tensor<41x73x9xi32>) -> tensor<41x73x1xi32>
    %3 = tosa.clamp %1 {min_val = 11 : i16, max_val = 51 : i16} : (tensor<35x53x87x7x3x81xi16>) -> tensor<35x53x87x7x3x81xi16>
    %4 = tosa.minimum %0, %0 : (tensor<100x30x51x27x57xf32>, tensor<100x30x51x27x57xf32>) -> tensor<100x30x51x27x57xf32>
    %5 = tosa.logical_xor %arg4, %arg5 : (tensor<25x82x79x92xi1>, tensor<1x1x1x92xi1>) -> tensor<25x82x79x92xi1>
    %6 = tosa.clz %3 : (tensor<35x53x87x7x3x81xi16>) -> tensor<35x53x87x7x3x81xi16>
    return %2, %4, %5, %6 : tensor<41x73x1xi32>, tensor<100x30x51x27x57xf32>, tensor<25x82x79x92xi1>, tensor<35x53x87x7x3x81xi16>
  }
}
