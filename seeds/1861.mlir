module {
  func.func @main(%arg0: tensor<28xi1>, %arg1: tensor<94x23x88x23x20xf32>) -> (tensor<28xi1>, tensor<94x23x88x23x20xf32>) {
    %t_0 = tosa.const_shape {values = dense<[ 1 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %0 = tosa.tile %arg0, %t_0 : (tensor<28xi1>, !tosa.shape<1>) -> tensor<28xi1>
    %1 = tosa.add %0, %0 : (tensor<28xi1>, tensor<28xi1>) -> tensor<28xi1>
    %2 = tosa.logical_and %1, %0 : (tensor<28xi1>, tensor<28xi1>) -> tensor<28xi1>
    %3 = tosa.rsqrt %arg1 : (tensor<94x23x88x23x20xf32>) -> tensor<94x23x88x23x20xf32>
    %4 = tosa.floor %3 : (tensor<94x23x88x23x20xf32>) -> tensor<94x23x88x23x20xf32>
    return %2, %4 : tensor<28xi1>, tensor<94x23x88x23x20xf32>
  }
}
