module {
  func.func @main(%arg0: tensor<67x91x90x89x38x70xi1>, %arg1: tensor<1x91x90x89x1x1xi1>) -> tensor<9x5x2x5x10x2xi1> {
    %0 = tosa.logical_right_shift %arg0, %arg1 : (tensor<67x91x90x89x38x70xi1>, tensor<1x91x90x89x1x1xi1>) -> tensor<67x91x90x89x38x70xi1>
    %s_1_start = tosa.const_shape {values = dense<[ 33, 67, 7, 63, 28, 24 ]> : tensor<6xindex>} : () -> !tosa.shape<6>
    %s_1_size = tosa.const_shape {values = dense<[ 9, 5, 2, 5, 10, 2 ]> : tensor<6xindex>} : () -> !tosa.shape<6>
    %1 = tosa.slice %0, %s_1_start, %s_1_size : (tensor<67x91x90x89x38x70xi1>, !tosa.shape<6>, !tosa.shape<6>) -> tensor<9x5x2x5x10x2xi1>
    return %1 : tensor<9x5x2x5x10x2xi1>
  }
}
