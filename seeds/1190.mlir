module {
  func.func @main(%arg0: tensor<i8>, %arg1: tensor<i8>, %arg2: tensor<3x94x60xf32>, %arg3: tensor<1x94x60xf32>) -> (tensor<1x1xi1>, tensor<3x1x60xf32>) {
    %0 = tosa.equal %arg0, %arg1 : (tensor<i8>, tensor<i8>) -> tensor<i1>
    %1 = tosa.minimum %arg2, %arg3 : (tensor<3x94x60xf32>, tensor<1x94x60xf32>) -> tensor<3x94x60xf32>
    %r_2 = tosa.const_shape {values = dense<[ 1, 1 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %2 = tosa.reshape %0, %r_2 : (tensor<i1>, !tosa.shape<2>) -> tensor<1x1xi1>
    %3 = tosa.identity %1 : (tensor<3x94x60xf32>) -> tensor<3x94x60xf32>
    %4 = tosa.reduce_max %2 {axis = 0 : i32} : (tensor<1x1xi1>) -> tensor<1x1xi1>
    %5 = tosa.pow %3, %3 : (tensor<3x94x60xf32>, tensor<3x94x60xf32>) -> tensor<3x94x60xf32>
    %6 = tosa.reduce_sum %5 {axis = 1 : i32} : (tensor<3x94x60xf32>) -> tensor<3x1x60xf32>
    return %4, %6 : tensor<1x1xi1>, tensor<3x1x60xf32>
  }
}
