module {
  func.func @main(%arg0: tensor<71x11x56xi8>, %arg1: tensor<71x51x56xi8>, %arg2: tensor<f32>) -> (tensor<213x372x168xi8>, tensor<f32>) {
    %0 = tosa.concat %arg0, %arg1 {axis = 1 : i32} : (tensor<71x11x56xi8>, tensor<71x51x56xi8>) -> tensor<71x62x56xi8>
    %t_1 = tosa.const_shape {values = dense<[ 3, 3, 3 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %1 = tosa.tile %0, %t_1 : (tensor<71x62x56xi8>, !tosa.shape<3>) -> tensor<213x186x168xi8>
    %2 = tosa.concat %1, %1 {axis = 1 : i32} : (tensor<213x186x168xi8>, tensor<213x186x168xi8>) -> tensor<213x372x168xi8>
    %3 = tosa.abs %2 : (tensor<213x372x168xi8>) -> tensor<213x372x168xi8>
    %4 = tosa.log %arg2 : (tensor<f32>) -> tensor<f32>
    %5 = tosa.exp %4 : (tensor<f32>) -> tensor<f32>
    return %3, %5 : tensor<213x372x168xi8>, tensor<f32>
  }
}
