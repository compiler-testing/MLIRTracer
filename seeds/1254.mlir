module {
  func.func @main(%arg0: tensor<21x66x88xi1>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> (tensor<42x132x264xi1>, tensor<i1>) {
    %t_0 = tosa.const_shape {values = dense<[ 2, 2, 3 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %0 = tosa.tile %arg0, %t_0 : (tensor<21x66x88xi1>, !tosa.shape<3>) -> tensor<42x132x264xi1>
    %1 = tosa.greater_equal %arg1, %arg2 : (tensor<i64>, tensor<i64>) -> tensor<i1>
    return %0, %1 : tensor<42x132x264xi1>, tensor<i1>
  }
}
