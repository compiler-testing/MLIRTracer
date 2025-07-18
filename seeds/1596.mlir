module {
  func.func @main(%arg0: tensor<36xi32>, %arg1: tensor<88x69x1x57xf32>, %arg2: tensor<i1>, %arg3: tensor<i1>) -> (tensor<8xi32>, tensor<i1>, tensor<88x69x1x57xi1>, tensor<176x1x57xi32>) {
    %0 = tosa.reduce_max %arg0 {axis = 0 : i32} : (tensor<36xi32>) -> tensor<1xi32>
    %1 = tosa.ceil %arg1 : (tensor<88x69x1x57xf32>) -> tensor<88x69x1x57xf32>
    %s_2_start = tosa.const_shape {values = dense<[ 0 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %s_2_size = tosa.const_shape {values = dense<[ 8 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %2 = tosa.slice %0, %s_2_start, %s_2_size : (tensor<1xi32>, !tosa.shape<1>, !tosa.shape<1>) -> tensor<8xi32>
    %3 = tosa.sigmoid %1 : (tensor<88x69x1x57xf32>) -> tensor<88x69x1x57xf32>
    %4 = tosa.logical_or %arg2, %arg3 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %5 = tosa.argmax %1 {axis = 1 : i32} : (tensor<88x69x1x57xf32>) -> tensor<88x1x57xi32>
    %6 = tosa.greater_equal %3, %3 : (tensor<88x69x1x57xf32>, tensor<88x69x1x57xf32>) -> tensor<88x69x1x57xi1>
    %7 = tosa.concat %5, %5 {axis = 0 : i32} : (tensor<88x1x57xi32>, tensor<88x1x57xi32>) -> tensor<176x1x57xi32>
    return %2, %4, %6, %7 : tensor<8xi32>, tensor<i1>, tensor<88x69x1x57xi1>, tensor<176x1x57xi32>
  }
}
