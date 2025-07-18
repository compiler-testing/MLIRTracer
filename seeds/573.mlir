module {
  func.func @main(%arg0: tensor<37x55x76xi1>, %arg1: tensor<37x76x8xi1>, %arg2: tensor<79x10x89x24x58xi64>, %arg3: tensor<1x10x1x24x1xi64>, %arg4: tensor<86xf32>) -> (tensor<79x10x89x24x58xi1>, tensor<1x55x8xi1>, tensor<79x10x89x24x58xi64>, tensor<79x10x89x24x58xi1>, tensor<86xf32>) {
    %a_zp_0 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %b_zp_0 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %0 = tosa.matmul %arg0, %arg1, %a_zp_0, %b_zp_0 : (tensor<37x55x76xi1>, tensor<37x76x8xi1>, tensor<1xi1>, tensor<1xi1>) -> tensor<37x55x8xi1>
    %1 = tosa.maximum %arg2, %arg3 : (tensor<79x10x89x24x58xi64>, tensor<1x10x1x24x1xi64>) -> tensor<79x10x89x24x58xi64>
    %2 = tosa.maximum %1, %1 : (tensor<79x10x89x24x58xi64>, tensor<79x10x89x24x58xi64>) -> tensor<79x10x89x24x58xi64>
    %3 = tosa.logical_right_shift %2, %2 : (tensor<79x10x89x24x58xi64>, tensor<79x10x89x24x58xi64>) -> tensor<79x10x89x24x58xi64>
    %4 = tosa.equal %1, %3 : (tensor<79x10x89x24x58xi64>, tensor<79x10x89x24x58xi64>) -> tensor<79x10x89x24x58xi1>
    %5 = tosa.logical_xor %4, %4 : (tensor<79x10x89x24x58xi1>, tensor<79x10x89x24x58xi1>) -> tensor<79x10x89x24x58xi1>
    %6 = tosa.equal %3, %3 : (tensor<79x10x89x24x58xi64>, tensor<79x10x89x24x58xi64>) -> tensor<79x10x89x24x58xi1>
    %7 = tosa.reduce_product %0 {axis = 0 : i32} : (tensor<37x55x8xi1>) -> tensor<1x55x8xi1>
    %8 = tosa.maximum %1, %3 : (tensor<79x10x89x24x58xi64>, tensor<79x10x89x24x58xi64>) -> tensor<79x10x89x24x58xi64>
    %9 = tosa.logical_right_shift %5, %4 : (tensor<79x10x89x24x58xi1>, tensor<79x10x89x24x58xi1>) -> tensor<79x10x89x24x58xi1>
    %10 = tosa.sigmoid %arg4 : (tensor<86xf32>) -> tensor<86xf32>
    return %6, %7, %8, %9, %10 : tensor<79x10x89x24x58xi1>, tensor<1x55x8xi1>, tensor<79x10x89x24x58xi64>, tensor<79x10x89x24x58xi1>, tensor<86xf32>
  }
}
