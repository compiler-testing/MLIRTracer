module {
  func.func @main(%arg0: tensor<83xf32>) -> (tensor<6xf32>, tensor<i1>) {
    %0 = tosa.rsqrt %arg0 : (tensor<83xf32>) -> tensor<83xf32>
    %1 = tosa.argmax %0 {axis = 0 : i32} : (tensor<83xf32>) -> tensor<i32>
    %s_2_start = tosa.const_shape {values = dense<[ 19 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %s_2_size = tosa.const_shape {values = dense<[ 6 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %2 = tosa.slice %0, %s_2_start, %s_2_size : (tensor<83xf32>, !tosa.shape<1>, !tosa.shape<1>) -> tensor<6xf32>
    %3 = tosa.equal %1, %1 : (tensor<i32>, tensor<i32>) -> tensor<i1>
    return %2, %3 : tensor<6xf32>, tensor<i1>
  }
}
