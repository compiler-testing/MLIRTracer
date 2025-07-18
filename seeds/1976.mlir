module {
  func.func @main(%arg0: tensor<62xi16>, %arg1: tensor<82xi1>, %arg2: tensor<30x38x44x50x73xi64>, %arg3: tensor<30x38x44x1x1xi64>, %arg4: tensor<f32>, %arg5: tensor<f32>) -> (tensor<30x38x44x50x73xi1>, tensor<62xi16>, tensor<1xi1>, tensor<f32>, tensor<f32>, tensor<1xi1>) {
    %t_0 = tosa.const_shape {values = dense<[ 1 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %0 = tosa.tile %arg0, %t_0 : (tensor<62xi16>, !tosa.shape<1>) -> tensor<62xi16>
    %1 = tosa.identity %0 : (tensor<62xi16>) -> tensor<62xi16>
    %2 = tosa.reduce_all %arg1 {axis = 0 : i32} : (tensor<82xi1>) -> tensor<1xi1>
    %3 = tosa.greater_equal %arg2, %arg3 : (tensor<30x38x44x50x73xi64>, tensor<30x38x44x1x1xi64>) -> tensor<30x38x44x50x73xi1>
    %in_zp_4 = "tosa.const"() <{values = dense<0> : tensor<1xi16>}> : () -> tensor<1xi16>
    %out_zp_4 = "tosa.const"() <{values = dense<0> : tensor<1xi16>}> : () -> tensor<1xi16>
    %4 = tosa.negate %1, %in_zp_4, %out_zp_4 : (tensor<62xi16>, tensor<1xi16>, tensor<1xi16>) -> tensor<62xi16>
    %5 = tosa.pow %arg4, %arg5 : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %6 = tosa.bitwise_and %2, %2 : (tensor<1xi1>, tensor<1xi1>) -> tensor<1xi1>
    %7 = tosa.clz %6 : (tensor<1xi1>) -> tensor<1xi1>
    %8 = tosa.ceil %5 : (tensor<f32>) -> tensor<f32>
    %9 = tosa.bitwise_and %7, %6 : (tensor<1xi1>, tensor<1xi1>) -> tensor<1xi1>
    %10 = tosa.sigmoid %5 : (tensor<f32>) -> tensor<f32>
    %11 = tosa.floor %8 : (tensor<f32>) -> tensor<f32>
    %12 = tosa.reduce_product %6 {axis = 0 : i32} : (tensor<1xi1>) -> tensor<1xi1>
    return %3, %4, %9, %10, %11, %12 : tensor<30x38x44x50x73xi1>, tensor<62xi16>, tensor<1xi1>, tensor<f32>, tensor<f32>, tensor<1xi1>
  }
}
