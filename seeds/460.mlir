module {
  func.func @main(%arg0: tensor<60x58x72x64x75x99xf32>, %arg1: tensor<97x16x49xi1>) -> (tensor<60x58x72x64x75x99xi1>, tensor<97x2x49xi1>) {
    %0 = tosa.floor %arg0 : (tensor<60x58x72x64x75x99xf32>) -> tensor<60x58x72x64x75x99xf32>
    %1 = tosa.maximum %0, %0 : (tensor<60x58x72x64x75x99xf32>, tensor<60x58x72x64x75x99xf32>) -> tensor<60x58x72x64x75x99xf32>
    %2 = tosa.equal %1, %1 : (tensor<60x58x72x64x75x99xf32>, tensor<60x58x72x64x75x99xf32>) -> tensor<60x58x72x64x75x99xi1>
    %3 = tosa.reduce_all %arg1 {axis = 1 : i32} : (tensor<97x16x49xi1>) -> tensor<97x1x49xi1>
    %t_4 = tosa.const_shape {values = dense<[ 1, 2, 1 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %4 = tosa.tile %3, %t_4 : (tensor<97x1x49xi1>, !tosa.shape<3>) -> tensor<97x2x49xi1>
    return %2, %4 : tensor<60x58x72x64x75x99xi1>, tensor<97x2x49xi1>
  }
}
