module {
  func.func @main(%arg0: tensor<79x69x6xi1>, %arg1: tensor<88x19xi32>, %arg2: tensor<88x1xi32>, %arg3: tensor<85x49x84x27xi32>, %arg4: tensor<85x49x84x27xi32>, %arg5: tensor<66x99x95x23x91xf32>, %arg6: tensor<1x1x95x1x1xf32>) -> (tensor<85x49x84x27xi32>, tensor<88x19xi1>, tensor<6x11x2x3x10xf32>, tensor<79x138x1xi1>) {
    %t_0 = tosa.const_shape {values = dense<[ 1, 2, 3 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %0 = tosa.tile %arg0, %t_0 : (tensor<79x69x6xi1>, !tosa.shape<3>) -> tensor<79x138x18xi1>
    %1 = tosa.greater %arg1, %arg2 : (tensor<88x19xi32>, tensor<88x1xi32>) -> tensor<88x19xi1>
    %2 = tosa.intdiv %arg3, %arg4 : (tensor<85x49x84x27xi32>, tensor<85x49x84x27xi32>) -> tensor<85x49x84x27xi32>
    %3 = tosa.arithmetic_right_shift %0, %0 {round = false} : (tensor<79x138x18xi1>, tensor<79x138x18xi1>) -> tensor<79x138x18xi1>
    %4 = tosa.pow %arg5, %arg6 : (tensor<66x99x95x23x91xf32>, tensor<1x1x95x1x1xf32>) -> tensor<66x99x95x23x91xf32>
    %5 = tosa.logical_and %1, %1 : (tensor<88x19xi1>, tensor<88x19xi1>) -> tensor<88x19xi1>
    %s_6_start = tosa.const_shape {values = dense<[ 33, 63, 38, 20, 20 ]> : tensor<5xindex>} : () -> !tosa.shape<5>
    %s_6_size = tosa.const_shape {values = dense<[ 6, 11, 2, 3, 10 ]> : tensor<5xindex>} : () -> !tosa.shape<5>
    %6 = tosa.slice %4, %s_6_start, %s_6_size : (tensor<66x99x95x23x91xf32>, !tosa.shape<5>, !tosa.shape<5>) -> tensor<6x11x2x3x10xf32>
    %7 = tosa.reduce_all %3 {axis = 2 : i32} : (tensor<79x138x18xi1>) -> tensor<79x138x1xi1>
    return %2, %5, %6, %7 : tensor<85x49x84x27xi32>, tensor<88x19xi1>, tensor<6x11x2x3x10xf32>, tensor<79x138x1xi1>
  }
}
