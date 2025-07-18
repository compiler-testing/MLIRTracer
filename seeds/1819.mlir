module {
  func.func @main(%arg0: tensor<22x4x3x27x65x73xi16>, %arg1: tensor<39xi1>, %arg2: tensor<54xf32>) -> (tensor<22x4x3x27x65x73xi16>, tensor<1xi1>, tensor<54xf32>) {
    %0 = tosa.clamp %arg0 {min_val = -14 : i16, max_val = 107 : i16} : (tensor<22x4x3x27x65x73xi16>) -> tensor<22x4x3x27x65x73xi16>
    %1 = tosa.reduce_all %arg1 {axis = 0 : i32} : (tensor<39xi1>) -> tensor<1xi1>
    %2 = tosa.reciprocal %arg2 : (tensor<54xf32>) -> tensor<54xf32>
    return %0, %1, %2 : tensor<22x4x3x27x65x73xi16>, tensor<1xi1>, tensor<54xf32>
  }
}
