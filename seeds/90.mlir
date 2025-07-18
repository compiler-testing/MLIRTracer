module {
  func.func @main(%arg0: tensor<50x12x2x14xf32>) -> tensor<1x3360xf32> {
    %0 = tosa.exp %arg0 : (tensor<50x12x2x14xf32>) -> tensor<50x12x2x14xf32>
    %r_1 = tosa.const_shape {values = dense<[ 5, 3360 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %1 = tosa.reshape %0, %r_1 : (tensor<50x12x2x14xf32>, !tosa.shape<2>) -> tensor<5x3360xf32>
    %2 = tosa.reduce_min %1 {axis = 0 : i32} : (tensor<5x3360xf32>) -> tensor<1x3360xf32>
    return %2 : tensor<1x3360xf32>
  }
}
