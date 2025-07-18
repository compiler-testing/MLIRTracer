module {
  func.func @main(%arg0: tensor<93x15x31x97x61xi16>, %arg1: tensor<90x42x47x9xi8>, %arg2: tensor<59x68x4x89x4x62xf32>, %arg3: tensor<88x33x83xi1>) -> (tensor<93x15x31x97x61xi16>, tensor<59x68x4x89x4x62xi1>, tensor<1x42x47xi32>, tensor<1x33x83xi1>, tensor<59x68x4x89x4x62xf32>, tensor<1x33x1xi1>, tensor<1x33x83xi1>, tensor<59x68x4x89x4x62xf32>) {
    %in_zp_0 = "tosa.const"() <{values = dense<0> : tensor<1xi16>}> : () -> tensor<1xi16>
    %out_zp_0 = "tosa.const"() <{values = dense<0> : tensor<1xi16>}> : () -> tensor<1xi16>
    %0 = tosa.negate %arg0, %in_zp_0, %out_zp_0 : (tensor<93x15x31x97x61xi16>, tensor<1xi16>, tensor<1xi16>) -> tensor<93x15x31x97x61xi16>
    %1 = tosa.reduce_max %arg1 {axis = 0 : i32} : (tensor<90x42x47x9xi8>) -> tensor<1x42x47x9xi8>
    %2 = tosa.abs %0 : (tensor<93x15x31x97x61xi16>) -> tensor<93x15x31x97x61xi16>
    %3 = tosa.log %arg2 : (tensor<59x68x4x89x4x62xf32>) -> tensor<59x68x4x89x4x62xf32>
    %4 = tosa.argmax %1 {axis = 3 : i32} : (tensor<1x42x47x9xi8>) -> tensor<1x42x47xi32>
    %5 = tosa.logical_left_shift %4, %4 : (tensor<1x42x47xi32>, tensor<1x42x47xi32>) -> tensor<1x42x47xi32>
    %6 = tosa.equal %3, %3 : (tensor<59x68x4x89x4x62xf32>, tensor<59x68x4x89x4x62xf32>) -> tensor<59x68x4x89x4x62xi1>
    %7 = tosa.reduce_any %arg3 {axis = 0 : i32} : (tensor<88x33x83xi1>) -> tensor<1x33x83xi1>
    %8 = tosa.intdiv %5, %4 : (tensor<1x42x47xi32>, tensor<1x42x47xi32>) -> tensor<1x42x47xi32>
    %9 = tosa.reverse %7 {axis = 2 : i32} : (tensor<1x33x83xi1>) -> tensor<1x33x83xi1>
    %10 = tosa.reciprocal %3 : (tensor<59x68x4x89x4x62xf32>) -> tensor<59x68x4x89x4x62xf32>
    %11 = tosa.logical_not %7 : (tensor<1x33x83xi1>) -> tensor<1x33x83xi1>
    %12 = tosa.reduce_min %11 {axis = 2 : i32} : (tensor<1x33x83xi1>) -> tensor<1x33x1xi1>
    %13 = tosa.reduce_sum %11 {axis = 0 : i32} : (tensor<1x33x83xi1>) -> tensor<1x33x83xi1>
    %14 = tosa.logical_left_shift %13, %13 : (tensor<1x33x83xi1>, tensor<1x33x83xi1>) -> tensor<1x33x83xi1>
    %15 = tosa.tanh %3 : (tensor<59x68x4x89x4x62xf32>) -> tensor<59x68x4x89x4x62xf32>
    return %2, %6, %8, %9, %10, %12, %14, %15 : tensor<93x15x31x97x61xi16>, tensor<59x68x4x89x4x62xi1>, tensor<1x42x47xi32>, tensor<1x33x83xi1>, tensor<59x68x4x89x4x62xf32>, tensor<1x33x1xi1>, tensor<1x33x83xi1>, tensor<59x68x4x89x4x62xf32>
  }
}
