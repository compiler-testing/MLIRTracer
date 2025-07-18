module {
  func.func @main(%arg0: tensor<i1>, %arg1: tensor<71x44x13xi1>, %arg2: tensor<57x46x85x5x79x60xf32>, %arg3: tensor<100x27x24x78x92xi32>, %arg4: tensor<1x1x1x1x1xi32>) -> (tensor<142x132x1xi1>, tensor<i1>, tensor<57x46x85x5x79x60xi1>, tensor<1x132x13xi1>, tensor<100x27x24x78x92xi32>, tensor<57x46x85x5x79x60xf32>) {
    %0 = tosa.logical_not %arg0 : (tensor<i1>) -> tensor<i1>
    %t_1 = tosa.const_shape {values = dense<[ 2, 3, 1 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %1 = tosa.tile %arg1, %t_1 : (tensor<71x44x13xi1>, !tosa.shape<3>) -> tensor<142x132x13xi1>
    %2 = tosa.reduce_all %1 {axis = 2 : i32} : (tensor<142x132x13xi1>) -> tensor<142x132x1xi1>
    %3 = tosa.reciprocal %arg2 : (tensor<57x46x85x5x79x60xf32>) -> tensor<57x46x85x5x79x60xf32>
    %4 = tosa.exp %3 : (tensor<57x46x85x5x79x60xf32>) -> tensor<57x46x85x5x79x60xf32>
    %5 = tosa.pow %3, %3 : (tensor<57x46x85x5x79x60xf32>, tensor<57x46x85x5x79x60xf32>) -> tensor<57x46x85x5x79x60xf32>
    %6 = tosa.logical_xor %0, %0 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %7 = tosa.greater_equal %5, %3 : (tensor<57x46x85x5x79x60xf32>, tensor<57x46x85x5x79x60xf32>) -> tensor<57x46x85x5x79x60xi1>
    %8 = tosa.reduce_any %1 {axis = 0 : i32} : (tensor<142x132x13xi1>) -> tensor<1x132x13xi1>
    %9 = tosa.pow %5, %5 : (tensor<57x46x85x5x79x60xf32>, tensor<57x46x85x5x79x60xf32>) -> tensor<57x46x85x5x79x60xf32>
    %10 = tosa.tanh %4 : (tensor<57x46x85x5x79x60xf32>) -> tensor<57x46x85x5x79x60xf32>
    %11 = tosa.exp %9 : (tensor<57x46x85x5x79x60xf32>) -> tensor<57x46x85x5x79x60xf32>
    %12 = tosa.intdiv %arg3, %arg4 : (tensor<100x27x24x78x92xi32>, tensor<1x1x1x1x1xi32>) -> tensor<100x27x24x78x92xi32>
    %13 = tosa.reciprocal %11 : (tensor<57x46x85x5x79x60xf32>) -> tensor<57x46x85x5x79x60xf32>
    %14 = tosa.minimum %10, %13 : (tensor<57x46x85x5x79x60xf32>, tensor<57x46x85x5x79x60xf32>) -> tensor<57x46x85x5x79x60xf32>
    return %2, %6, %7, %8, %12, %14 : tensor<142x132x1xi1>, tensor<i1>, tensor<57x46x85x5x79x60xi1>, tensor<1x132x13xi1>, tensor<100x27x24x78x92xi32>, tensor<57x46x85x5x79x60xf32>
  }
}
