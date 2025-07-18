module {
  func.func @main(%arg0: tensor<i64>, %arg1: tensor<i64>, %arg2: tensor<45xf32>, %arg3: tensor<45xf32>) -> (tensor<i64>, tensor<45xf32>) {
    %0 = tosa.bitwise_and %arg0, %arg1 : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %1 = tosa.pow %arg2, %arg3 : (tensor<45xf32>, tensor<45xf32>) -> tensor<45xf32>
    %2 = tosa.floor %1 : (tensor<45xf32>) -> tensor<45xf32>
    %3 = tosa.logical_right_shift %0, %0 : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %t_4 = tosa.const_shape {values = dense<[ 1 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %4 = tosa.tile %2, %t_4 : (tensor<45xf32>, !tosa.shape<1>) -> tensor<45xf32>
    %5 = tosa.clamp %4 {min_val = -8.000000e+00 : f32, max_val = 5.000000e+01 : f32} : (tensor<45xf32>) -> tensor<45xf32>
    %6 = tosa.logical_right_shift %3, %3 : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %7 = tosa.reciprocal %1 : (tensor<45xf32>) -> tensor<45xf32>
    %8 = tosa.add %7, %5 : (tensor<45xf32>, tensor<45xf32>) -> tensor<45xf32>
    return %6, %8 : tensor<i64>, tensor<45xf32>
  }
}
