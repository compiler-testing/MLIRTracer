module {
  func.func @main(%arg0: tensor<54x59x7xi8>, %arg1: tensor<60x14x70x89xf32>, %arg2: tensor<60x14x70x1xf32>, %arg3: tensor<75x5xi1>) -> (tensor<54x59x7xi8>, tensor<60x14x70x89xf32>, tensor<1x5xi1>, tensor<60x14x70x89xf32>) {
    %in_zp_0 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %out_zp_0 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %0 = tosa.negate %arg0, %in_zp_0, %out_zp_0 : (tensor<54x59x7xi8>, tensor<1xi8>, tensor<1xi8>) -> tensor<54x59x7xi8>
    %1 = tosa.bitwise_and %0, %0 : (tensor<54x59x7xi8>, tensor<54x59x7xi8>) -> tensor<54x59x7xi8>
    %2 = tosa.pow %arg1, %arg2 : (tensor<60x14x70x89xf32>, tensor<60x14x70x1xf32>) -> tensor<60x14x70x89xf32>
    %3 = tosa.pow %2, %2 : (tensor<60x14x70x89xf32>, tensor<60x14x70x89xf32>) -> tensor<60x14x70x89xf32>
    %4 = tosa.reverse %2 {axis = 3 : i32} : (tensor<60x14x70x89xf32>) -> tensor<60x14x70x89xf32>
    %5 = tosa.abs %3 : (tensor<60x14x70x89xf32>) -> tensor<60x14x70x89xf32>
    %6 = tosa.reduce_any %arg3 {axis = 0 : i32} : (tensor<75x5xi1>) -> tensor<1x5xi1>
    %7 = tosa.reciprocal %4 : (tensor<60x14x70x89xf32>) -> tensor<60x14x70x89xf32>
    return %1, %5, %6, %7 : tensor<54x59x7xi8>, tensor<60x14x70x89xf32>, tensor<1x5xi1>, tensor<60x14x70x89xf32>
  }
}
