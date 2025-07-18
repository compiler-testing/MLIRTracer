module {
  func.func @main(%arg0: tensor<25xf32>) -> tensor<4xi1> {
    %0 = tosa.reduce_product %arg0 {axis = 0 : i32} : (tensor<25xf32>) -> tensor<1xf32>
    %1 = tosa.reduce_product %0 {axis = 0 : i32} : (tensor<1xf32>) -> tensor<1xf32>
    %2 = tosa.exp %1 : (tensor<1xf32>) -> tensor<1xf32>
    %3 = tosa.reduce_min %2 {axis = 0 : i32} : (tensor<1xf32>) -> tensor<1xf32>
    %s_4_start = tosa.const_shape {values = dense<[ 0 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %s_4_size = tosa.const_shape {values = dense<[ 4 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %4 = tosa.slice %3, %s_4_start, %s_4_size : (tensor<1xf32>, !tosa.shape<1>, !tosa.shape<1>) -> tensor<4xf32>
    %5 = tosa.equal %4, %4 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
    return %5 : tensor<4xi1>
  }
}
