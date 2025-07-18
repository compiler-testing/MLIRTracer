module {
  func.func @main(%arg0: tensor<88x28xi64>) -> tensor<88x1xi64> {
    %0 = tosa.reduce_min %arg0 {axis = 1 : i32} : (tensor<88x28xi64>) -> tensor<88x1xi64>
    %1 = tosa.sub %0, %0 : (tensor<88x1xi64>, tensor<88x1xi64>) -> tensor<88x1xi64>
    %2 = tosa.clamp %1 {min_val = 53 : i64, max_val = 66 : i64} : (tensor<88x1xi64>) -> tensor<88x1xi64>
    %t_3 = tosa.const_shape {values = dense<[ 1, 1 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %3 = tosa.tile %2, %t_3 : (tensor<88x1xi64>, !tosa.shape<2>) -> tensor<88x1xi64>
    return %3 : tensor<88x1xi64>
  }
}
