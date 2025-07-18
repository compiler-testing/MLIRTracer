module {
  func.func @main(%arg0: tensor<52x81x90xi8>, %arg1: tensor<1x81x90xi8>) -> tensor<156x1x90xi1> {
    %0 = tosa.greater_equal %arg0, %arg1 : (tensor<52x81x90xi8>, tensor<1x81x90xi8>) -> tensor<52x81x90xi1>
    %1 = tosa.reduce_all %0 {axis = 1 : i32} : (tensor<52x81x90xi1>) -> tensor<52x1x90xi1>
    %2 = tosa.add %1, %1 : (tensor<52x1x90xi1>, tensor<52x1x90xi1>) -> tensor<52x1x90xi1>
    %t_3 = tosa.const_shape {values = dense<[ 3, 1, 1 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %3 = tosa.tile %2, %t_3 : (tensor<52x1x90xi1>, !tosa.shape<3>) -> tensor<156x1x90xi1>
    %4 = tosa.identity %3 : (tensor<156x1x90xi1>) -> tensor<156x1x90xi1>
    return %4 : tensor<156x1x90xi1>
  }
}
