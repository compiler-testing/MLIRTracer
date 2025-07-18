module {
  func.func @main(%arg0: tensor<29x100xi1>, %arg1: tensor<44x48x46xf32>) -> (tensor<29x1xi1>, tensor<88x48x1xf32>) {
    %0 = tosa.reduce_all %arg0 {axis = 1 : i32} : (tensor<29x100xi1>) -> tensor<29x1xi1>
    %1 = tosa.sigmoid %arg1 : (tensor<44x48x46xf32>) -> tensor<44x48x46xf32>
    %t_2 = tosa.const_shape {values = dense<[ 2, 1, 3 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %2 = tosa.tile %1, %t_2 : (tensor<44x48x46xf32>, !tosa.shape<3>) -> tensor<88x48x138xf32>
    %3 = tosa.reduce_max %2 {axis = 2 : i32} : (tensor<88x48x138xf32>) -> tensor<88x48x1xf32>
    return %0, %3 : tensor<29x1xi1>, tensor<88x48x1xf32>
  }
}
