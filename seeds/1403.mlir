module {
  func.func @main(%arg0: tensor<28xf32>) -> tensor<1xf32> {
    %0 = tosa.rsqrt %arg0 : (tensor<28xf32>) -> tensor<28xf32>
    %1 = tosa.sigmoid %0 : (tensor<28xf32>) -> tensor<28xf32>
    %2 = tosa.rsqrt %1 : (tensor<28xf32>) -> tensor<28xf32>
    %3 = tosa.minimum %2, %1 : (tensor<28xf32>, tensor<28xf32>) -> tensor<28xf32>
    %t_4 = tosa.const_shape {values = dense<[ 2 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %4 = tosa.tile %3, %t_4 : (tensor<28xf32>, !tosa.shape<1>) -> tensor<56xf32>
    %5 = tosa.concat %4, %4 {axis = 0 : i32} : (tensor<56xf32>, tensor<56xf32>) -> tensor<112xf32>
    %6 = tosa.reduce_product %5 {axis = 0 : i32} : (tensor<112xf32>) -> tensor<1xf32>
    return %6 : tensor<1xf32>
  }
}
