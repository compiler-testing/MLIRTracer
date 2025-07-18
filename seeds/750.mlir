module {
  func.func @main(%arg0: tensor<50x78x35x29xi1>, %arg1: tensor<9x91x88x62xf32>, %arg2: tensor<1x91x88x1xf32>) -> (tensor<1x78x35x29xi1>, tensor<9x91x88x62xf32>) {
    %0 = tosa.reduce_product %arg0 {axis = 0 : i32} : (tensor<50x78x35x29xi1>) -> tensor<1x78x35x29xi1>
    %1 = tosa.pow %arg1, %arg2 : (tensor<9x91x88x62xf32>, tensor<1x91x88x1xf32>) -> tensor<9x91x88x62xf32>
    %in_zp_2 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %out_zp_2 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %2 = tosa.negate %0, %in_zp_2, %out_zp_2 : (tensor<1x78x35x29xi1>, tensor<1xi1>, tensor<1xi1>) -> tensor<1x78x35x29xi1>
    %3 = tosa.minimum %1, %1 : (tensor<9x91x88x62xf32>, tensor<9x91x88x62xf32>) -> tensor<9x91x88x62xf32>
    return %2, %3 : tensor<1x78x35x29xi1>, tensor<9x91x88x62xf32>
  }
}
