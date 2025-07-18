module {
  func.func @main(%arg0: tensor<39x77xi1>, %arg1: tensor<39x1xi1>, %arg2: tensor<i8>, %arg3: tensor<i8>, %arg4: tensor<33x57x87xf32>) -> (tensor<i1>, tensor<33x57x174xf32>, tensor<1x1xi1>) {
    %0 = tosa.logical_or %arg0, %arg1 : (tensor<39x77xi1>, tensor<39x1xi1>) -> tensor<39x77xi1>
    %1 = tosa.reduce_sum %0 {axis = 1 : i32} : (tensor<39x77xi1>) -> tensor<39x1xi1>
    %2 = tosa.reduce_sum %1 {axis = 0 : i32} : (tensor<39x1xi1>) -> tensor<1x1xi1>
    %3 = tosa.greater_equal %arg2, %arg3 : (tensor<i8>, tensor<i8>) -> tensor<i1>
    %4 = tosa.arithmetic_right_shift %2, %2 {round = false} : (tensor<1x1xi1>, tensor<1x1xi1>) -> tensor<1x1xi1>
    %5 = tosa.ceil %arg4 : (tensor<33x57x87xf32>) -> tensor<33x57x87xf32>
    %6 = tosa.add %5, %5 : (tensor<33x57x87xf32>, tensor<33x57x87xf32>) -> tensor<33x57x87xf32>
    %7 = tosa.concat %6, %5 {axis = 2 : i32} : (tensor<33x57x87xf32>, tensor<33x57x87xf32>) -> tensor<33x57x174xf32>
    %in_zp_8 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %out_zp_8 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %8 = tosa.negate %7, %in_zp_8, %out_zp_8 : (tensor<33x57x174xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<33x57x174xf32>
    %9 = tosa.clz %4 : (tensor<1x1xi1>) -> tensor<1x1xi1>
    %10 = tosa.logical_or %9, %2 : (tensor<1x1xi1>, tensor<1x1xi1>) -> tensor<1x1xi1>
    return %3, %8, %10 : tensor<i1>, tensor<33x57x174xf32>, tensor<1x1xi1>
  }
}
