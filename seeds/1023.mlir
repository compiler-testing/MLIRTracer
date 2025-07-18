module {
  func.func @main(%arg0: tensor<87x48x70x44x99x17xi1>, %arg1: tensor<39x66x55xi64>) -> (tensor<87x48x70x44x99x17xi1>, tensor<39x1xi32>, tensor<3x3x4xi64>) {
    %0 = tosa.logical_not %arg0 : (tensor<87x48x70x44x99x17xi1>) -> tensor<87x48x70x44x99x17xi1>
    %1 = tosa.reduce_sum %arg1 {axis = 2 : i32} : (tensor<39x66x55xi64>) -> tensor<39x66x1xi64>
    %2 = tosa.argmax %1 {axis = 1 : i32} : (tensor<39x66x1xi64>) -> tensor<39x1xi32>
    %s_3_start = tosa.const_shape {values = dense<[ 10, 35, 0 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %s_3_size = tosa.const_shape {values = dense<[ 3, 3, 4 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %3 = tosa.slice %1, %s_3_start, %s_3_size : (tensor<39x66x1xi64>, !tosa.shape<3>, !tosa.shape<3>) -> tensor<3x3x4xi64>
    return %0, %2, %3 : tensor<87x48x70x44x99x17xi1>, tensor<39x1xi32>, tensor<3x3x4xi64>
  }
}
