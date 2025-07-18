module {
  func.func @main(%arg0: tensor<87xi32>, %arg1: tensor<87xi32>, %arg2: tensor<18x19xf32>) -> (tensor<87xi32>, tensor<36x6xi1>, tensor<18x1xf32>) {
    %0 = tosa.bitwise_and %arg0, %arg1 : (tensor<87xi32>, tensor<87xi32>) -> tensor<87xi32>
    %1 = tosa.log %arg2 : (tensor<18x19xf32>) -> tensor<18x19xf32>
    %2 = tosa.reduce_min %1 {axis = 1 : i32} : (tensor<18x19xf32>) -> tensor<18x1xf32>
    %t_3 = tosa.const_shape {values = dense<[ 1, 2 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %3 = tosa.tile %2, %t_3 : (tensor<18x1xf32>, !tosa.shape<2>) -> tensor<18x2xf32>
    %4 = tosa.pow %3, %3 : (tensor<18x2xf32>, tensor<18x2xf32>) -> tensor<18x2xf32>
    %t_5 = tosa.const_shape {values = dense<[ 2, 3 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %5 = tosa.tile %4, %t_5 : (tensor<18x2xf32>, !tosa.shape<2>) -> tensor<36x6xf32>
    %6 = tosa.greater_equal %5, %5 : (tensor<36x6xf32>, tensor<36x6xf32>) -> tensor<36x6xi1>
    %7 = tosa.add %3, %3 : (tensor<18x2xf32>, tensor<18x2xf32>) -> tensor<18x2xf32>
    %8 = tosa.reduce_min %7 {axis = 1 : i32} : (tensor<18x2xf32>) -> tensor<18x1xf32>
    return %0, %6, %8 : tensor<87xi32>, tensor<36x6xi1>, tensor<18x1xf32>
  }
}
