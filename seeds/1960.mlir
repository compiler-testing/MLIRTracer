module {
  func.func @main(%arg0: tensor<72x31x37x61xf32>, %arg1: tensor<31x99x15xi64>, %arg2: tensor<31x1x1xi64>, %arg3: tensor<75xi1>, %arg4: tensor<75xi1>) -> (tensor<31x99x15xi64>, tensor<75xi1>, tensor<1x9176xf32>) {
    %0 = tosa.ceil %arg0 : (tensor<72x31x37x61xf32>) -> tensor<72x31x37x61xf32>
    %1 = tosa.logical_right_shift %arg1, %arg2 : (tensor<31x99x15xi64>, tensor<31x1x1xi64>) -> tensor<31x99x15xi64>
    %r_2 = tosa.const_shape {values = dense<[ 549, 9176 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %2 = tosa.reshape %0, %r_2 : (tensor<72x31x37x61xf32>, !tosa.shape<2>) -> tensor<549x9176xf32>
    %3 = tosa.logical_and %arg3, %arg4 : (tensor<75xi1>, tensor<75xi1>) -> tensor<75xi1>
    %4 = tosa.reduce_sum %2 {axis = 0 : i32} : (tensor<549x9176xf32>) -> tensor<1x9176xf32>
    return %1, %3, %4 : tensor<31x99x15xi64>, tensor<75xi1>, tensor<1x9176xf32>
  }
}
