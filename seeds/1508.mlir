module {
  func.func @main(%arg0: tensor<66x93x87xf32>, %arg1: tensor<i64>, %arg2: tensor<i64>, %arg3: tensor<67xi1>) -> (tensor<178002x3xf32>, tensor<i64>, tensor<1xi1>) {
    %r_0 = tosa.const_shape {values = dense<[ 178002, 3 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %0 = tosa.reshape %arg0, %r_0 : (tensor<66x93x87xf32>, !tosa.shape<2>) -> tensor<178002x3xf32>
    %1 = tosa.logical_left_shift %arg1, %arg2 : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %2 = tosa.reduce_any %arg3 {axis = 0 : i32} : (tensor<67xi1>) -> tensor<1xi1>
    return %0, %1, %2 : tensor<178002x3xf32>, tensor<i64>, tensor<1xi1>
  }
}
