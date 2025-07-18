module {
  func.func @main(%arg0: tensor<45x44x26xi32>, %arg1: tensor<1x1x26xi32>, %arg2: tensor<82x24xf32>) -> (tensor<1x44x26xi32>, tensor<164x12xf32>) {
    %0 = tosa.maximum %arg0, %arg1 : (tensor<45x44x26xi32>, tensor<1x1x26xi32>) -> tensor<45x44x26xi32>
    %1 = tosa.reduce_sum %0 {axis = 0 : i32} : (tensor<45x44x26xi32>) -> tensor<1x44x26xi32>
    %2 = tosa.sigmoid %arg2 : (tensor<82x24xf32>) -> tensor<82x24xf32>
    %r_3 = tosa.const_shape {values = dense<[ 164, 12 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %3 = tosa.reshape %2, %r_3 : (tensor<82x24xf32>, !tosa.shape<2>) -> tensor<164x12xf32>
    return %1, %3 : tensor<1x44x26xi32>, tensor<164x12xf32>
  }
}
