module {
  func.func @main(%arg0: tensor<87x7x1x60x10xi1>, %arg1: tensor<94x2xi16>, %arg2: tensor<5x92xi64>, %arg3: tensor<5x92xi64>) -> (tensor<282x6xi16>, tensor<5x92xi1>, tensor<87x7x1x60x10xi1>) {
    %0 = tosa.bitwise_not %arg0 : (tensor<87x7x1x60x10xi1>) -> tensor<87x7x1x60x10xi1>
    %t_1 = tosa.const_shape {values = dense<[ 3, 3 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %1 = tosa.tile %arg1, %t_1 : (tensor<94x2xi16>, !tosa.shape<2>) -> tensor<282x6xi16>
    %2 = tosa.logical_not %0 : (tensor<87x7x1x60x10xi1>) -> tensor<87x7x1x60x10xi1>
    %3 = tosa.greater_equal %arg2, %arg3 : (tensor<5x92xi64>, tensor<5x92xi64>) -> tensor<5x92xi1>
    %4 = tosa.logical_and %2, %0 : (tensor<87x7x1x60x10xi1>, tensor<87x7x1x60x10xi1>) -> tensor<87x7x1x60x10xi1>
    return %1, %3, %4 : tensor<282x6xi16>, tensor<5x92xi1>, tensor<87x7x1x60x10xi1>
  }
}
