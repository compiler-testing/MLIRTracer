module {
  func.func @main(%arg0: tensor<44x87x65xi32>, %arg1: tensor<44x65x12xi32>, %arg2: tensor<3x63x23x12x23x99xf32>) -> (tensor<3x63x23x12x23x99xf32>, tensor<44x1x12xi32>) {
    %a_zp_0 = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %b_zp_0 = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %0 = tosa.matmul %arg0, %arg1, %a_zp_0, %b_zp_0 : (tensor<44x87x65xi32>, tensor<44x65x12xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<44x87x12xi32>
    %1 = tosa.reciprocal %arg2 : (tensor<3x63x23x12x23x99xf32>) -> tensor<3x63x23x12x23x99xf32>
    %2 = tosa.reduce_product %0 {axis = 1 : i32} : (tensor<44x87x12xi32>) -> tensor<44x1x12xi32>
    return %1, %2 : tensor<3x63x23x12x23x99xf32>, tensor<44x1x12xi32>
  }
}
