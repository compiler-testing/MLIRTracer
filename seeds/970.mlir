module {
  func.func @main(%arg0: tensor<8x77x24xi1>, %arg1: tensor<8x24x100xi1>, %arg2: tensor<83x39x88x26xf32>, %arg3: tensor<1x39x88x1xf32>) -> (tensor<8x1x100xi1>, tensor<83x39x88x26xf32>) {
    %a_zp_0 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %b_zp_0 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %0 = tosa.matmul %arg0, %arg1, %a_zp_0, %b_zp_0 : (tensor<8x77x24xi1>, tensor<8x24x100xi1>, tensor<1xi1>, tensor<1xi1>) -> tensor<8x77x100xi1>
    %1 = tosa.bitwise_xor %0, %0 : (tensor<8x77x100xi1>, tensor<8x77x100xi1>) -> tensor<8x77x100xi1>
    %2 = tosa.bitwise_and %1, %1 : (tensor<8x77x100xi1>, tensor<8x77x100xi1>) -> tensor<8x77x100xi1>
    %3 = tosa.reduce_product %2 {axis = 1 : i32} : (tensor<8x77x100xi1>) -> tensor<8x1x100xi1>
    %4 = tosa.maximum %arg2, %arg3 : (tensor<83x39x88x26xf32>, tensor<1x39x88x1xf32>) -> tensor<83x39x88x26xf32>
    return %3, %4 : tensor<8x1x100xi1>, tensor<83x39x88x26xf32>
  }
}
