module {
  func.func @main(%arg0: tensor<26x58x96xf32>, %arg1: tensor<26x94xi32>) -> (tensor<144768x1xf32>, tensor<26x94xi32>) {
    %r_0 = tosa.const_shape {values = dense<[ 58, 2496 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %0 = tosa.reshape %arg0, %r_0 : (tensor<26x58x96xf32>, !tosa.shape<2>) -> tensor<58x2496xf32>
    %1 = tosa.bitwise_not %arg1 : (tensor<26x94xi32>) -> tensor<26x94xi32>
    %2 = tosa.floor %0 : (tensor<58x2496xf32>) -> tensor<58x2496xf32>
    %r_3 = tosa.const_shape {values = dense<[ 144768, 1 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %3 = tosa.reshape %2, %r_3 : (tensor<58x2496xf32>, !tosa.shape<2>) -> tensor<144768x1xf32>
    %in_zp_4 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %out_zp_4 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %4 = tosa.negate %3, %in_zp_4, %out_zp_4 : (tensor<144768x1xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<144768x1xf32>
    %5 = tosa.abs %1 : (tensor<26x94xi32>) -> tensor<26x94xi32>
    return %4, %5 : tensor<144768x1xf32>, tensor<26x94xi32>
  }
}
