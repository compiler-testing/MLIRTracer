module {
  func.func @main(%arg0: tensor<87x8xi16>, %arg1: tensor<30x93x50xi32>, %arg2: tensor<30x93x50xi32>, %arg3: tensor<f32>) -> (tensor<174x8xi16>, tensor<f32>, tensor<30x1x50xi1>) {
    %0 = tosa.clz %arg0 : (tensor<87x8xi16>) -> tensor<87x8xi16>
    %1 = tosa.concat %0, %0 {axis = 0 : i32} : (tensor<87x8xi16>, tensor<87x8xi16>) -> tensor<174x8xi16>
    %2 = tosa.equal %arg1, %arg2 : (tensor<30x93x50xi32>, tensor<30x93x50xi32>) -> tensor<30x93x50xi1>
    %t_3 = tosa.const_shape {values = dense<[ 1, 1 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %3 = tosa.tile %1, %t_3 : (tensor<174x8xi16>, !tosa.shape<2>) -> tensor<174x8xi16>
    %4 = tosa.log %arg3 : (tensor<f32>) -> tensor<f32>
    %5 = tosa.log %4 : (tensor<f32>) -> tensor<f32>
    %6 = tosa.reduce_sum %2 {axis = 1 : i32} : (tensor<30x93x50xi1>) -> tensor<30x1x50xi1>
    return %3, %5, %6 : tensor<174x8xi16>, tensor<f32>, tensor<30x1x50xi1>
  }
}
